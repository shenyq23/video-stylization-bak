"""
FFmpeg minterpolate-based FRUC (Frame Rate Up-Conversion) interpolator.

Drop-in replacement for KFRUCInterpolator that uses FFmpeg's built-in
minterpolate filter (MEMC: Motion Estimation + Motion Compensation) instead
of the proprietary libkfruc.so.

Algorithm: EPZS (Enhanced Predictive Zonal Search) + AOBMC
           (Adaptive Overlapped Block Motion Compensation)

Dependency: ffmpeg binary — resolved automatically in this priority order:
  1. System ffmpeg (PATH)
  2. imageio-ffmpeg bundled binary (already in requirements.txt, zero extra install)

No .so / C-source compilation required.
"""

import subprocess
import shutil
import logging
import numpy as np
from typing import List, Optional


def _find_ffmpeg() -> str:
    """Resolve ffmpeg binary: system PATH first, then imageio-ffmpeg bundle."""
    # 1. System ffmpeg
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    # 2. imageio-ffmpeg bundled binary (already a requirement in requirements.txt)
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    raise RuntimeError(
        "ffmpeg not found. It is bundled via imageio-ffmpeg (already in "
        "requirements.txt). Run: pip install imageio-ffmpeg"
    )


class FFmpegFRUCInterpolator:
    """
    Frame interpolation via FFmpeg minterpolate filter.

    API is compatible with KFRUCInterpolator:
        interpolator.initialize(input_fps, interpolate_rate, width, height)
        video_out = interpolator.interpolate_chunk(video_np)  # [T,H,W,3] f32
        remaining = interpolator.flush()                       # always []
        interpolator.cleanup()

    For streaming compatibility, process_stream() and flush() are also provided.
    """

    # minterpolate quality presets
    PRESETS = {
        "fast":     "mi_mode=mci:me=hexbs:mc_mode=obmc:scd=none",
        "balanced": "mi_mode=mci:me=epzs:mc_mode=aobmc:scd=fdiff",
        "best":     "mi_mode=mci:me=epzs:mc_mode=aobmc:vsbmc=1:scd=fdiff",
    }

    def __init__(self, ffmpeg_bin: str = None, preset: str = "balanced"):
        self.logger = logging.getLogger("FFmpegFRUCInterpolator")

        # Resolve ffmpeg binary: PATH → imageio-ffmpeg bundle (no extra install)
        if ffmpeg_bin is None:
            ffmpeg_bin = _find_ffmpeg()
        self.ffmpeg_bin = ffmpeg_bin

        if preset not in self.PRESETS:
            raise ValueError(f"preset must be one of {list(self.PRESETS.keys())}")
        self.preset = preset

        self.input_fps: Optional[float] = None
        self.interpolate_rate: Optional[int] = None
        self.output_fps: Optional[float] = None
        self.width: int = 0
        self.height: int = 0

        # For streaming API compatibility
        self._stream_buffer: List[np.ndarray] = []

        self.logger.info(f"FFmpegFRUCInterpolator ready: {self.ffmpeg_bin} (preset={preset})")

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def initialize(
        self,
        input_fps: float = 6.0,
        interpolate_rate: int = 2,
        width: int = 832,
        height: int = 480,
    ) -> None:
        """
        Configure the interpolator. Must be called before any interpolation.

        Args:
            input_fps:        Frame rate of the input video (e.g. 4.0, 6.0).
            interpolate_rate: Up-conversion multiplier (2 → 2x, 4 → 4x, 8 → 8x).
            width:            Frame width in pixels.
            height:           Frame height in pixels.
        """
        self.input_fps = float(input_fps)
        self.interpolate_rate = int(interpolate_rate)
        self.output_fps = self.input_fps * self.interpolate_rate
        self.width = int(width)
        self.height = int(height)
        self._stream_buffer = []

        self.logger.info(
            f"Initialized: {self.input_fps}fps → {self.output_fps}fps "
            f"({self.interpolate_rate}x), {self.width}x{self.height}, preset={self.preset}"
        )

    def interpolate_chunk(
        self,
        frames: np.ndarray,
        tail_frame: np.ndarray = None,
    ) -> np.ndarray:
        """
        Interpolate a batch of frames with guaranteed output count.

        Args:
            frames:     [T, H, W, 3] float32 [0,1] — the chunk to interpolate.
            tail_frame: optional [H, W, 3] float32 — first frame of the NEXT chunk,
                        used as a lookahead so minterpolate can generate complete
                        inter-frame motion estimates at the tail of this chunk.
                        When provided, the returned array has exactly
                        T * interpolate_rate frames (tail transition frames are
                        stripped).  When None, minterpolate may produce fewer frames
                        at the tail; the output is zero-padded to T * interpolate_rate.

        Returns:
            [T * interpolate_rate, H, W, 3] float32 [0,1].
        """
        self._check_initialized()

        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"frames must be [T,H,W,3], got {frames.shape}")

        T, H, W = frames.shape[:3]
        want = T * self.interpolate_rate  # exact target output count

        if T == 0:
            return frames

        if T == 1:
            return np.repeat(frames, self.interpolate_rate, axis=0)

        # Attach lookahead tail frame so minterpolate has a reference beyond the
        # last input frame and won't truncate the final interpolated interval.
        if tail_frame is not None:
            tail = tail_frame[np.newaxis]  # [1, H, W, 3]
            feed = np.concatenate([frames, tail], axis=0)  # [T+1, H, W, 3]
        else:
            feed = frames  # [T, H, W, 3]

        # Convert float32 [0,1] → uint8 [0,255]
        frames_u8 = (feed.clip(0.0, 1.0) * 255.0).astype(np.uint8)
        raw_in = frames_u8.tobytes()

        vf = f"minterpolate=fps={self.output_fps}:{self.PRESETS[self.preset]}"

        cmd = [
            self.ffmpeg_bin, "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{W}x{H}",
            "-r", str(self.input_fps),
            "-i", "pipe:0",
            "-vf", vf,
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "pipe:1",
        ]

        try:
            result = subprocess.run(cmd, input=raw_in, capture_output=True, timeout=60)
        except subprocess.TimeoutExpired:
            self.logger.error("ffmpeg minterpolate timed out, falling back to duplication")
            return self._duplicate_fallback(frames, want)
        except Exception as e:
            self.logger.error(f"ffmpeg subprocess error: {e}, falling back to duplication")
            return self._duplicate_fallback(frames, want)

        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace")[-500:]
            self.logger.warning(f"ffmpeg exited {result.returncode}: ...{stderr}")
            return self._duplicate_fallback(frames, want)

        # Parse rawvideo output
        frame_bytes = W * H * 3
        raw_out = result.stdout
        n_out = len(raw_out) // frame_bytes

        if n_out == 0:
            self.logger.warning("ffmpeg produced 0 output frames, falling back")
            return self._duplicate_fallback(frames, want)

        out_u8 = np.frombuffer(raw_out[: n_out * frame_bytes], dtype=np.uint8)
        out_u8 = out_u8.reshape(n_out, H, W, 3)
        out_f32 = out_u8.astype(np.float32) / 255.0

        # Keep only the first `want` frames (strips the tail_frame transition if present).
        # If ffmpeg still produced fewer than `want` frames (rare edge case), pad with
        # the last frame so the caller always gets exactly T * interpolate_rate frames.
        if n_out >= want:
            out_f32 = out_f32[:want]
        else:
            pad = np.repeat(out_f32[[-1]], want - n_out, axis=0)
            out_f32 = np.concatenate([out_f32, pad], axis=0)
            self.logger.debug(
                f"interpolate_chunk: padded {n_out} → {want} frames "
                f"(ffmpeg produced fewer frames than expected)"
            )

        self.logger.debug(f"interpolate_chunk: {T} → {out_f32.shape[0]} frames (target={want})")
        return out_f32

    # -----------------------------------------------------------------------
    # Streaming API (compatible with KFRUCInterpolator)
    # -----------------------------------------------------------------------

    def process_stream(self, new_frame: np.ndarray) -> List[np.ndarray]:
        """
        Streaming interface: buffer a single frame.
        Returns [] always; call flush() to get all interpolated frames.

        Note: for efficiency, frames are processed in batch in flush().
        """
        self._check_initialized()
        self._stream_buffer.append(new_frame.copy())
        return []

    def flush(self) -> List[np.ndarray]:
        """
        Process all buffered frames and return the interpolated result.
        Clears internal buffer after processing.

        Returns:
            List of interpolated frames (float32 [0,1] HxWx3), or [] if buffer empty.
        """
        if not self._stream_buffer:
            return []

        frames = np.stack(self._stream_buffer, axis=0)  # [T, H, W, 3]
        self._stream_buffer = []

        out = self.interpolate_chunk(frames)
        return list(out)  # List[ndarray(H,W,3)]

    def cleanup(self) -> None:
        self._stream_buffer = []
        self.logger.info("FFmpegFRUCInterpolator cleaned up")

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _check_initialized(self) -> None:
        if self.input_fps is None:
            raise RuntimeError("FFmpegFRUCInterpolator not initialized. Call initialize() first.")

    def _duplicate_fallback(self, frames: np.ndarray, want: int) -> np.ndarray:
        """Nearest-neighbour frame duplication as a safe fallback, padded/trimmed to `want`."""
        self.logger.warning(
            f"Using duplication fallback: {len(frames)} → {want} frames"
        )
        repeated = np.repeat(frames, self.interpolate_rate, axis=0)
        if len(repeated) >= want:
            return repeated[:want]
        pad = np.repeat(repeated[[-1]], want - len(repeated), axis=0)
        return np.concatenate([repeated, pad], axis=0)

    @property
    def is_available(self) -> bool:
        """True if ffmpeg binary is found and supports minterpolate."""
        try:
            r = subprocess.run(
                [self.ffmpeg_bin, "-filters"],
                capture_output=True, timeout=5
            )
            return b"minterpolate" in r.stdout
        except Exception:
            return False

def _check_available() -> bool:
    try:
        ffmpeg = _find_ffmpeg()
        r = subprocess.run([ffmpeg, "-filters"], capture_output=True, timeout=5)
        return b"minterpolate" in r.stdout
    except Exception:
        return False


FFMPEG_FRUC_AVAILABLE = _check_available()


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    print(f"FFMPEG_FRUC_AVAILABLE = {FFMPEG_FRUC_AVAILABLE}")

    interp = FFmpegFRUCInterpolator(preset="balanced")
    interp.initialize(input_fps=4.0, interpolate_rate=2, width=128, height=72)

    # Simulate a chunk of 5 gradient frames
    frames = np.stack([
        np.full((72, 128, 3), i / 4.0, dtype=np.float32) for i in range(5)
    ])
    print(f"Input:  {frames.shape}  ({frames.min():.2f}–{frames.max():.2f})")
    out = interp.interpolate_chunk(frames)
    print(f"Output: {out.shape}  ({out.min():.2f}–{out.max():.2f})")
    assert out.shape[0] >= frames.shape[0], "Expected at least as many output frames"

    # Streaming API test
    for f in frames:
        interp.process_stream(f)
    stream_out = interp.flush()
    print(f"Stream flush: {len(stream_out)} frames")

    interp.cleanup()
    print("Done ✓")

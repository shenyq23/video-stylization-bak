"""
Single GPU Inference Pipeline - Optimized Parallel Execution with Simulated Producer FPS

This file implements an optimized producer-consumer pattern using separate CUDA
streams to achieve true parallelism between VAE encoding and DiT inference.
It adds a real-time simulation by throttling the producer to match a
specified generation FPS (--fps_generate), mimicking a live input source like a camera.

**Design Principles:**
1.  **CUDA Streams**:
    -   A `producer_stream` is dedicated to VAE encoding.
    -   A `consumer_stream` is dedicated to DiT inference and VAE decoding.
2.  **CUDA Events for Synchronization and Timing**:
    -   `torch.cuda.Event` is used for efficient, non-blocking, GPU-side synchronization
        between streams and for precise GPU execution timing.
3.  **Producer FPS Simulation (fps_generate)**:
    -   The `--fps_generate` argument simulates a fixed-rate input source.
    -   The producer calculates the time required to "receive" a new chunk of frames
        (e.g., 4 frames at 30 FPS = 133ms).
    -   After its VAE encoding task for a chunk is finished on the GPU (verified via
        `event.synchronize()`), the producer thread will `time.sleep()` if it
        finished faster than the target interval. If it's slower, it logs a lag warning.
4.  **Detailed Overlap Logging**:
    -   Timestamps clearly show when tasks are submitted (CPU-side) and completed
        (GPU-side), visualizing the computational overlap and any introduced sleep time.
5.  **Logical Equivalence**: The core numerical logic remains a 1:1 replication of
    the original serial code to ensure bit-for-bit identical output.
"""
import sys
sys.path.append("../")
sys.path.append("../deps/gmflow")
sys.path.append("../StreamDiffusionV2")

from causvid.models.wan.causal_stream_inference import CausalStreamInferencePipeline
from causvid.models.wan.wan_base.modules import TAEHV, StreamingTAEHV
from diffusers.utils import export_to_video
from causvid.data import TextDataset
from omegaconf import OmegaConf
import argparse
import torch
import os
import time
import numpy as np
import logging
import threading
import queue
import traceback
import cv2
import json
import urllib.request
from typing import Optional

import random
import collections
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange
from gmflow.geometry import flow_warp as universal_flow_warp

from deps.sige3d.torch_kernels.backend import set_kernel_backend
from utils.vae_utils.mem_stats import collect_scatter_cache_modules, feat_map_nbytes, format_bytes, scatter_cache_nbytes
from utils.optical_wrapper import GMFlowWrapper, RAFTFlowWrapper, OcclusionComputation, X265MVWrapper, X265MVWrapper, OcclusionComputation
from utils.vae_utils.mask_utils import (
    build_gather_block_masks,
    dilate_mask,
    downsample_mask,
    reduce_mask,
    resolve_mask_for_res,
)

LOG_HANDLERS = None
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATEFMT = '%Y-%m-%d %H:%M:%S'


def configure_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    handlers = LOG_HANDLERS or [logging.StreamHandler(sys.stdout)]
    if not logger.handlers:
        formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)
        for handler in handlers:
            handler.setFormatter(formatter)
            logger.addHandler(handler)
    return logger


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class CameraStream:
    """Real-time camera capture in a background thread.

    The capture loop reads frames from V4L2 as fast as the camera delivers them and
    appends them to a fixed-size ring buffer. The producer consumes by always taking
    the most recent ``chunk_size + 1`` frames; any frames left between two consumes
    are counted as dropped.
    """

    def __init__(self,
                 device_path: str = "/dev/video0",
                 src_w: int = 848,
                 src_h: int = 480,
                 src_fps: int = 30,
                 target_w: int = 832,
                 target_h: int = 480,
                 dtype: torch.dtype = torch.float16,
                 buffer_size: int = 64):
        self.logger = logging.getLogger("CameraStream")
        # Force V4L2 backend on Linux; the default (CAP_ANY) sometimes ignores SET_FMT.
        self.cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera device: {device_path}")
        # FOURCC must be set BEFORE width/height: many drivers lock to the default
        # YUYV bandwidth limit (640x480) if the format change comes after sizing.
        fourcc_mjpg = cv2.VideoWriter_fourcc(*'MJPG')
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc_mjpg)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, src_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, src_h)
        self.cap.set(cv2.CAP_PROP_FPS, src_fps)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        actual_fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        try:
            fourcc_str = actual_fourcc.to_bytes(4, "little").decode(errors="replace")
        except Exception:
            fourcc_str = str(actual_fourcc)
        self.logger.info(
            f"Camera initial negotiation: FOURCC={fourcc_str}, {actual_w}x{actual_h} @ {actual_fps:.1f}fps"
        )

        # Some V4L2 drivers need a second pass with FOURCC re-applied AFTER size.
        if (actual_w, actual_h) != (src_w, src_h):
            self.logger.warning(
                f"Driver did not honor {src_w}x{src_h} on first pass (got {actual_w}x{actual_h}); retrying."
            )
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, src_w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, src_h)
            self.cap.set(cv2.CAP_PROP_FOURCC, fourcc_mjpg)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, src_w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, src_h)
            self.cap.set(cv2.CAP_PROP_FPS, src_fps)
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.logger.info(f"After retry: {actual_w}x{actual_h} @ {actual_fps:.1f}fps")

        if target_w > actual_w or target_h > actual_h:
            raise RuntimeError(
                f"Camera native size {actual_w}x{actual_h} smaller than target {target_w}x{target_h}. "
                f"Pick one of the supported sizes from `v4l2-ctl --list-formats-ext -d {device_path}` "
                f"and pass --camera_src_w/--camera_src_h accordingly."
            )

        self.target_w = target_w
        self.target_h = target_h
        self.crop_x = max((actual_w - target_w) // 2, 0)
        self.crop_y = max((actual_h - target_h) // 2, 0)
        self.dtype = dtype
        self.actual_fps = actual_fps
        self.src_fps = src_fps
        self.logger.info(
            f"Camera opened ({device_path}): native {actual_w}x{actual_h} @ {actual_fps:.1f}fps, "
            f"crop to {target_w}x{target_h}"
        )

        self._cond = threading.Condition()
        self._buffer = collections.deque(maxlen=buffer_size)
        self._next_frame_id = 0
        self._stopped = False
        self._thread = None
        self._last_consumed_id = -1

        self.frames_captured = 0
        self.frames_consumed = 0
        self.frames_dropped = 0

    def start(self):
        self._thread = threading.Thread(target=self._capture_loop, name="CameraCapture", daemon=True)
        self._thread.start()
        return self

    def _preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame = frame[self.crop_y:self.crop_y + self.target_h,
                      self.crop_x:self.crop_x + self.target_w, :]
        tensor = torch.from_numpy(frame).permute(2, 0, 1).contiguous().float()
        tensor = tensor / 127.5 - 1.0
        return tensor.to(self.dtype)

    def _capture_loop(self):
        try:
            while True:
                with self._cond:
                    if self._stopped:
                        break
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.warning("Camera read returned no frame; ending capture loop.")
                    break
                tensor = self._preprocess(frame)
                with self._cond:
                    self._buffer.append((self._next_frame_id, tensor))
                    self._next_frame_id += 1
                    self.frames_captured += 1
                    self._cond.notify_all()
        finally:
            with self._cond:
                self._stopped = True
                self._cond.notify_all()

    def take_initial(self, n: int, stride: int = 1):
        """Block until enough frames for strided selection are buffered.

        With stride>1, selects every stride-th frame from the latest available.
        Requires at least ``(n-1)*stride + 1`` frames in the buffer.
        Returns ``(tensor (1,3,n,H,W), ids)`` or None on stop.
        """
        min_needed = 1 + (n - 1) * stride
        with self._cond:
            while not self._stopped and len(self._buffer) < min_needed:
                self._cond.wait()
            if len(self._buffer) < min_needed:
                return None
            all_frames = list(self._buffer)
            # n frames from end stepping stride back, then reverse to chronological order
            selected = [all_frames[-(1 + i * stride)] for i in range(n - 1, -1, -1)]
            ids = [f[0] for f in selected]
            tensors = torch.stack([f[1] for f in selected], dim=1).unsqueeze(0).contiguous()
            self.frames_dropped += ids[0]  # frames before first selected were never consumed
            self.frames_consumed += n
            self._last_consumed_id = ids[-1]
            return tensors, ids

    def take_chunk_with_lookback(self, chunk_size: int, stride: int = 1):
        """Block until enough new frames for a strided chunk are available.

        Selects ``chunk_size+1`` frames spaced ``stride`` apart from the latest end of the buffer
        (``tensors[:,:,0]`` = lookback; ``tensors[:,:,1:]`` = chunk).
        Requires ``chunk_size * stride + 1`` frames in buffer with at least one new frame
        beyond ``last_consumed_id``.
        Returns ``(tensor (1,3,chunk_size+1,H,W), ids, dropped_this_chunk)`` or None on stop.
        """
        n = chunk_size + 1  # lookback + chunk frames
        min_needed = 1 + chunk_size * stride
        with self._cond:
            while not self._stopped:
                if len(self._buffer) >= min_needed and self._buffer[-1][0] > self._last_consumed_id:
                    break
                self._cond.wait()
            if len(self._buffer) < min_needed or self._buffer[-1][0] <= self._last_consumed_id:
                return None
            all_frames = list(self._buffer)
            # n frames from end stepping stride back, then reverse to chronological order
            selected = [all_frames[-(1 + i * stride)] for i in range(n - 1, -1, -1)]
            ids = [f[0] for f in selected]
            tensors = torch.stack([f[1] for f in selected], dim=1).unsqueeze(0).contiguous()
            chunk_start_id = ids[1]  # first chunk frame (not lookback)
            if self._last_consumed_id >= 0:
                dropped = max(0, chunk_start_id - self._last_consumed_id - 1)
            else:
                dropped = chunk_start_id
            self.frames_dropped += dropped
            self.frames_consumed += chunk_size
            self._last_consumed_id = ids[-1]
            return tensors, ids, dropped

    def stop(self):
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if self.cap.isOpened():
            self.cap.release()

    def is_stopped(self) -> bool:
        with self._cond:
            return self._stopped

    def get_stats(self) -> dict:
        cap = self.frames_captured
        cons = self.frames_consumed
        dropped = max(0, cap - cons)
        return {
            "captured": cap,
            "consumed": cons,
            "dropped": dropped,
            "drop_rate": dropped / cap if cap > 0 else 0.0,
        }


class TAEHVDiffusersWrapper(torch.nn.Module):
    def __init__(self, checkpoint_path: str, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.dtype = dtype
        self.taehv = TAEHV(checkpoint_path=checkpoint_path).to(self.dtype)
        self.streaming_encoder = StreamingTAEHV(self.taehv)
        self.streaming_decoder = StreamingTAEHV(self.taehv)
        self.config = DotDict(scaling_factor=1.0)

    def stream_encode(self, video: torch.Tensor, mask: torch.Tensor, flow: torch.Tensor, is_nocache: bool) -> torch.Tensor:
        del mask, flow
        video = (video * 0.5 + 0.5).clamp(0, 1)
        video = video.permute(0, 2, 1, 3, 4).contiguous()
        latents = []
        latent = self.streaming_encoder.encode(video)
        while latent is not None:
            latents.append(latent)
            latent = self.streaming_encoder.encode()
        if not latents:
            raise RuntimeError("StreamingTAEHV encoder produced no latents for the input chunk.")
        latents = torch.cat(latents, dim=1)
        return latents.permute(0, 2, 1, 3, 4).contiguous()

    def stream_decode_to_pixel(self, latent: torch.Tensor, mask: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        del mask, flow
        latent = latent.to(dtype=self.dtype)
        frames = []
        for latent_frame in latent.unbind(1):
            frame = self.streaming_decoder.decode(latent_frame.unsqueeze(1))
            if frame is not None:
                frames.append(frame)
            while True:
                next_frame = self.streaming_decoder.decode()
                if next_frame is None:
                    break
                frames.append(next_frame)
        if not frames:
            return None
        video = torch.cat(frames, dim=1)
        return video.mul(2).sub(1)


def build_taehv_vae(device: torch.device, dtype: torch.dtype = torch.float16) -> TAEHVDiffusersWrapper:
    taehv_checkpoint_path = os.path.join("./wan_models", "taew2_1.pth")
    if not os.path.exists(taehv_checkpoint_path):
        logger = logging.getLogger("TAEHV")
        logger.info("taew2_1.pth not found at %s, downloading...", taehv_checkpoint_path)
        os.makedirs(os.path.dirname(taehv_checkpoint_path), exist_ok=True)
        download_url = "https://github.com/madebyollin/taehv/raw/main/taew2_1.pth"
        urllib.request.urlretrieve(download_url, taehv_checkpoint_path)
        logger.info("Downloaded taew2_1.pth to %s", taehv_checkpoint_path)

    vae = TAEHVDiffusersWrapper(checkpoint_path=taehv_checkpoint_path, dtype=dtype)
    vae.eval()
    vae.requires_grad_(False)
    vae.to(device=device, dtype=dtype)
    return vae


def normalize_map(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # x: [B,1,H,W]
    x_min = x.amin(dim=(2, 3), keepdim=True)
    x_max = x.amax(dim=(2, 3), keepdim=True)
    return (x - x_min) / (x_max - x_min + eps)

def box_blur_map(x: torch.Tensor, kernel_size: int = 11) -> torch.Tensor:
    pad = kernel_size // 2
    return F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)

def fuse_occ_maps(
    occ_geom: torch.Tensor,      # [B,1,H,W], warp error / geometry occ
    occ_motion: torch.Tensor,    # [B,1,H,W], relative motion magnitude
    gamma: float = 0.6,          # soften geometry gate
    alpha: float = 0.7,          # weight of local diffusion
    blur_ks: int = 11,
    motion_power: float = 1.2,   # sharpen motion peak slightly
    gate_floor: float = 0.3,     # keep some softness after blur
    eps: float = 1e-6,
) -> torch.Tensor:

    occ_geom_n = normalize_map(occ_geom, eps=eps)
    occ_motion_n = normalize_map(occ_motion, eps=eps)

    # 1) foreground soft gate: constrain to bird interior but not too harsh
    soft_gate = occ_geom_n.clamp_min(0.0).pow(gamma)

    # 2) sharpen motion peak slightly so head stands out more
    motion_peak = occ_motion_n.clamp_min(0.0).pow(motion_power)

    # 3) seed = strong motion inside foreground
    seed = soft_gate * motion_peak

    # 4) local diffusion -> make region connected
    seed_blur = box_blur_map(seed, kernel_size=blur_ks)

    # 5) combine point evidence + connected support
    fused = (1.0 - alpha) * seed + alpha * seed_blur

    # 6) gate again to prevent blur leaking to background
    fused = fused * (gate_floor + (1.0 - gate_floor) * soft_gate)

    # 7) optional final normalization for stable top-k behavior
    fused = normalize_map(fused, eps=eps)

    return fused

class OpticalFlowCalculator:
    def __init__(self,
                 flow_model_type: str,
                 device: torch.device,
                 x265_params: dict = None,
                 occlusion_method: str = 'quantile',
                 top_k_percentage: float=(0.1,0.1),
                 morph_kernel_size: int = 7,
                 conn_comp_threshold_quantile: float = 0.75
                 ):
        self.device = device
        self.logger = logging.getLogger("OpticalFlowCalculator")
        self.x265_params = x265_params or {}
        self.flow_model_type = flow_model_type
        self.occlusion_method = occlusion_method
        self.top_k_percentage = top_k_percentage
        self.morph_kernel_size = morph_kernel_size
        self.conn_comp_threshold_quantile = conn_comp_threshold_quantile

        self.logger.info(f"Using occlusion mask generation method: '{self.occlusion_method}'")

        if not flow_model_type or flow_model_type.lower() == 'none':
            self.model = None
            self.logger.info("Optical flow calculation is disabled.")
            return

        self.logger.info(f"Initializing optical flow model: {flow_model_type}")
        FlowModel = {"gmflow": GMFlowWrapper, "raft": RAFTFlowWrapper,
                     "x265": X265MVWrapper}.get(flow_model_type.lower())
        if FlowModel is None:
            raise ValueError(f"Unsupported flow model type: {flow_model_type}")

        if flow_model_type == "x265":
            self.model = FlowModel(str(self.device), native_x265=True)
        else:
            self.model = FlowModel(str(self.device))

        if self.flow_model_type.lower() == 'x265':
            self.logger.info("Using 'luminosity' occlusion for x265.")
            self.occlusion_computer = OcclusionComputation(use_luminosity=True)
        else:
            self.logger.info("Using 'geometry' occlusion for DL models.")
            self.occlusion_computer = OcclusionComputation(use_geometry=True)

        self.bwd_occ_avg=None
        # from ultralytics import YOLO
        # self.seg_model = YOLO("yolo26n-seg.pt")
        # self.seg_model.to(device)

    def compute_binary_occlusion_mask(self, raw_occ_map: torch.Tensor) -> torch.Tensor:
        B, _, H, W = raw_occ_map.shape
        final_masks = []

        for i in range(B):
            single_occ_map = raw_occ_map[i, 0]

            if self.occlusion_method in ['exact','gather_block']:
                num_elements = single_occ_map.numel()
                k = int(num_elements * self.top_k_percentage[1])

                # 确保 k 至少为 1 (如果百分比 > 0)，且不超过总元素数
                k = max(1, min(k, num_elements)) if self.top_k_percentage[1] > 0 else 0

                if k == 0:
                    binary_mask = torch.zeros_like(single_occ_map, dtype=torch.bool)
                else:
                    # 展平张量并找到前 k 大的值的索引
                    flat_map = single_occ_map.flatten()
                    _, topk_indices = torch.topk(flat_map, k)

                    # 创建一个新的布尔掩码，并将 topk 索引位置设为 True
                    binary_mask_flat = torch.zeros_like(flat_map, dtype=torch.bool)
                    binary_mask_flat.scatter_(0, topk_indices, True)

                    # 恢复为原始的 2D 形状
                    binary_mask = binary_mask_flat.view(H, W)

            elif self.occlusion_method == 'quantile':
                threshold = torch.quantile(single_occ_map, 1.0 - self.top_k_percentage[1])
                binary_mask = (single_occ_map >= threshold)
            elif self.occlusion_method == 'morphological':
                initial_quantile = max(0.5, 1.0 - self.top_k_percentage[1] * 2)
                threshold = torch.quantile(single_occ_map, initial_quantile)
                noisy_mask_np = (single_occ_map > threshold).cpu().numpy().astype(np.uint8)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_kernel_size, self.morph_kernel_size))
                cleaned_mask_np = cv2.morphologyEx(noisy_mask_np, cv2.MORPH_OPEN, kernel)
                binary_mask = torch.from_numpy(cleaned_mask_np).to(self.device, dtype=torch.bool)
            elif self.occlusion_method == 'connected_components':
                threshold = torch.quantile(single_occ_map, self.conn_comp_threshold_quantile)
                binary_mask_np = (single_occ_map > threshold).cpu().numpy().astype(np.uint8)
                num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(binary_mask_np, connectivity=8)

                if num_labels <= 1:
                    final_masks.append(torch.zeros_like(single_occ_map, dtype=torch.bool))
                    continue

                region_scores = []
                for label in range(1, num_labels):
                    area = stats[label, cv2.CC_STAT_AREA]
                    if area < self.morph_kernel_size * self.morph_kernel_size:
                        continue
                    region_mask_np = (labels_im == label)
                    score = single_occ_map[torch.from_numpy(region_mask_np).to(self.device)].mean().item()
                    region_scores.append({'id': label, 'score': score, 'area': area})

                region_scores.sort(key=lambda x: x['score'], reverse=True)
                final_mask_np = np.zeros((H, W), dtype=bool)
                target_area = H * W * self.top_k_percentage[1]
                covered_area = 0
                for region in region_scores:
                    if covered_area >= target_area:
                        break
                    region_mask_np = (labels_im == region['id'])
                    final_mask_np[region_mask_np] = True
                    covered_area += region['area']

                binary_mask = torch.from_numpy(final_mask_np).to(self.device)
            elif self.occlusion_method == "gather_block":
                binary_mask=single_occ_map.to(torch.float32).contiguous()
                # raise RuntimeError(
                #     "gather_block uses raw residual map directly and should not call compute_binary_occlusion_mask()."
                # )
            else:
                raise ValueError(f"Unsupported occlusion method: {self.occlusion_method}")

            final_masks.append(binary_mask)

        return torch.stack(final_masks, dim=0).unsqueeze(1)

    def get_foreground_mask(self, frame_tensor: torch.Tensor) -> torch.Tensor:
        """
        输入: [1, 3, H, W] 范围 [-1, 1] 的张量
        输出: [1, 1, H, W] 范围 [0, 1] 的二值掩码
        """
        img_for_seg = (frame_tensor * 0.5 + 0.5).clamp(0, 1)
        results = self.seg_model.predict(img_for_seg, conf=0.25, verbose=True)

        _, _, H, W = frame_tensor.shape
        mask_out = torch.zeros((1, 1, H, W), device=self.device)

        if results[0].masks is not None:
            combined_mask = torch.any(results[0].masks.data, dim=0).float()
            mask_out = F.interpolate(combined_mask.unsqueeze(0).unsqueeze(0),
                                     size=(H, W), mode='nearest')

        return mask_out

    def calculate_flow(self, ref_frame: torch.Tensor, current_frame: torch.Tensor) -> tuple | None:
        if self.model is None:
            return None

        if (self.flow_model_type == "x265"):
            # print(self.x265_params)
            fwd_flow, bwd_flow = self.model.compute_flow_from_tensors(ref_frame, current_frame, **self.x265_params)
            # print(bwd_flow.shape)
            # fwd_flow=torch.ones((1,2,480,832),dtype=torch.float32,device=ref_frame.device)
            # bwd_flow=torch.ones((1,2,480,832),dtype=torch.float32,device=ref_frame.device)

        else:
            fwd_flow, bwd_flow = self.model.compute_flow_from_tensors(ref_frame, current_frame)

        ####修改1. 把occ改成flow的模长

        _, bwd_occ_geom = self.occlusion_computer(ref_frame, current_frame, fwd_flow, bwd_flow)
        if bwd_occ_geom.dim() == 3:
            bwd_occ_geom = bwd_occ_geom.unsqueeze(1)

        # 方法2: 相对运动模长图 (优点: 强度精确，高亮主要运动)
        # 我将其重命名为 bwd_occ_motion
        global_motion = bwd_flow.mean(dim=(2, 3), keepdim=True)
        relative_flow = bwd_flow - global_motion
        bwd_occ_motion = torch.norm(relative_flow, p=2, dim=1, keepdim=True)

        bwd_occ = 0*bwd_occ_geom+1*bwd_occ_motion

        # # bwd_occ = fuse_occ_maps(
        # #     occ_geom=bwd_occ_geom,
        # #     occ_motion=bwd_occ_motion,
        # #     gamma=0.6,
        # #     alpha=0.7,
        # #     blur_ks=11,
        # #     motion_power=1.2,
        # #     gate_floor=0.3,
        # # )
        # # torch.cuda.synchronize(ref_frame.device)
        # # start=time.time()
        # foreground_mask = self.get_foreground_mask(current_frame)
        # bwd_occ = bwd_occ * foreground_mask
        # # torch.cuda.synchronize(ref_frame.device)
        # # end=time.time()
        # # print(f"Segmentation and fusion took {end-start:.3f} seconds")

        if (self.bwd_occ_avg==None):
            self.bwd_occ_avg=bwd_occ
        else:
            self.bwd_occ_avg=0.5*self.bwd_occ_avg+0.5*bwd_occ
        return bwd_flow,self.bwd_occ_avg

        #

        # 最终的 bwd_occ 是一个浮点分布图，它结合了两种方法的优点
        # 后续的 compute_binary_occlusion_mask 会从这个更优的分布中选取 top_k_percentage
        return bwd_flow, bwd_occ


def tensor_to_np_img(tensor: torch.Tensor) -> np.ndarray:
    """Converts a [-1, 1] or [0, 1] image tensor to a [0, 255] uint8 RGB numpy array."""
    if tensor.min() < -0.1:
        tensor = (tensor * 0.5 + 0.5)
    tensor = tensor.clamp(0, 1)

    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)

    np_img = tensor.permute(1, 2, 0).contiguous().cpu().numpy()
    return (np_img * 255).astype(np.uint8)


def visualize_latent_to_image(latent: torch.Tensor) -> np.ndarray:
    """Visualizes a latent tensor by taking the mean across channels and normalizing."""
    if latent.dim() == 4:
        latent = latent.squeeze(0)

    latent_mean = latent.mean(dim=0)
    min_val, max_val = latent_mean.min(), latent_mean.max()
    if max_val > min_val:
        latent_norm = (latent_mean - min_val) / (max_val - min_val)
    else:
        latent_norm = torch.zeros_like(latent_mean)

    img_np = (latent_norm.float().cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)


def visualize_flow_to_rgb(flow: torch.Tensor, vector_stride: int = 20) -> np.ndarray:
    """
    Visualizes an optical flow tensor by drawing arrows on a black background.
    """
    if flow.dim() != 4 or flow.shape[1] != 2:
        raise ValueError("Input flow must be a [B, 2, H, W] tensor.")

    B, _, H, W = flow.shape
    flow_canvas = np.zeros((H, W, 3), dtype=np.uint8)
    flow_np = flow.squeeze(0).permute(1, 2, 0).cpu().numpy()
    arrow_color = (0, 255, 0)  # Green in BGR

    for y in range(vector_stride // 2, H, vector_stride):
        for x in range(vector_stride // 2, W, vector_stride):
            dx, dy = flow_np[y, x, :]
            start_point = (x, y)
            end_x = int(np.clip(round(x + dx), 0, W - 1))
            end_y = int(np.clip(round(y + dy), 0, H - 1))
            end_point = (end_x, end_y)
            cv2.arrowedLine(flow_canvas, start_point, end_point, arrow_color, 1, tipLength=0.3)

    return cv2.cvtColor(flow_canvas, cv2.COLOR_BGR2RGB)


def overlay_flow_on_image(image: np.ndarray, flow_viz: np.ndarray) -> np.ndarray:
    """
    Overlays the flow visualization (arrows) on top of a background image.

    Args:
        image (np.ndarray): The background RGB image (H, W, 3).
        flow_viz (np.ndarray): The flow visualization RGB image (H, W, 3) with arrows.

    Returns:
        np.ndarray: The combined image.
    """
    if image.shape != flow_viz.shape:
        h, w, _ = image.shape
        flow_viz = cv2.resize(flow_viz, (w, h), interpolation=cv2.INTER_NEAREST)

    # cv2.add performs saturated addition, which is perfect for this overlay effect.
    # It adds the green arrow color to the background image pixels.
    overlayed_image = cv2.add(image, cv2.cvtColor(flow_viz, cv2.COLOR_RGB2BGR))
    return overlayed_image
    return cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB)


def visualize_flow_with_source_overlay(
    source_image: np.ndarray,
    target_image: np.ndarray,
    flow_viz: np.ndarray,
    alpha: float = 0.4
) -> np.ndarray:
    """
    创建一个复合视觉效果：在目标图像上叠加光流箭头，然后再与半透明的源图像进行混合。

    这个视觉效果旨在取代简单的光流箭头可视化，以提供更丰富的上下文。
    最终图像的计算方式为: (target_image + flow_arrows) * (1-alpha) + source_image * alpha.

    Args:
        source_image (np.ndarray): 用于半透明叠加的源 RGB 图像 (H, W, 3)。
        target_image (np.ndarray): 作为背景的目标 RGB 图像 (H, W, 3)。
        flow_viz (np.ndarray): 在黑色背景上带有光流箭头的 RGB 视觉效果图 (H, W, 3)。
        alpha (float): 源图像叠加的透明度/权重。

    Returns:
        np.ndarray: 合成后的 RGB 图像。
    """
    # 确保所有输入图像的尺寸与目标图像一致
    h, w, _ = target_image.shape
    if source_image.shape[:2] != (h, w):
        source_image = cv2.resize(source_image, (w, h), interpolation=cv2.INTER_AREA)
    if flow_viz.shape[:2] != (h, w):
        flow_viz = cv2.resize(flow_viz, (w, h), interpolation=cv2.INTER_NEAREST)

    # --- OpenCV 操作需要 BGR 格式 ---
    source_bgr = cv2.cvtColor(source_image, cv2.COLOR_RGB2BGR)
    target_bgr = cv2.cvtColor(target_image, cv2.COLOR_RGB2BGR)
    flow_viz_bgr = cv2.cvtColor(flow_viz, cv2.COLOR_RGB2BGR)

    # 步骤 1: 创建基础叠加层 (目标图像 + 光流箭头)
    # cv2.add 执行饱和加法，非常适合添加绿色箭头。
    base_overlay_bgr = cv2.add(target_bgr, flow_viz_bgr)

    # 步骤 2: 在基础叠加层之上混合半透明的源图像
    # 公式为: dst = src1*alpha + src2*(1-alpha) + gamma
    composite_bgr = cv2.addWeighted(source_bgr, alpha, base_overlay_bgr, 1.0 - alpha, 0.0)

    # --- 将最终图像转换回 RGB 格式，以与其他可视化函数保持一致 ---
    composite_rgb = cv2.cvtColor(composite_bgr, cv2.COLOR_BGR2RGB)

    return composite_rgb


def load_mp4_as_tensor(
    video_path: str,
    max_frames: int = None,
    resize_hw: tuple[int, int] = None,
    normalize: bool = True,
    device: str = 'cuda:0',
) -> tuple[torch.Tensor, int]:  # <--- 修改: 更新返回类型提示
    assert os.path.exists(video_path), f"Video file not found: {video_path}"
    # <--- 修改: 捕获第三个返回值 info，其中包含元数据
    video, _, info = torchvision.io.read_video(video_path, output_format="TCHW", pts_unit="sec")

    # <--- 新增: 从元数据中获取视频的FPS，如果获取不到则提供一个默认值
    original_fps = info.get('video_fps', 16)

    if max_frames is not None:
        video = video[:max_frames]
    video = rearrange(video, "t c h w -> c t h w")
    if resize_hw is not None:
        c, t, h0, w0 = video.shape

        video = video.to(device) if torch.cuda.is_available() and str(device).startswith('cuda') else video
        video = torch.stack([TF.resize(video[:, i], resize_hw, antialias=True) for i in range(t)], dim=1)
    if video.dtype != torch.float32:
        video = video.float()
    if normalize:
        video = video / 127.5 - 1.0

    return video, original_fps  # <--- 修改: 返回视频张量和原始FPS


def compute_noise_scale_and_step(input_video_original: torch.Tensor, end_idx: int, chunk_size: int, noise_scale: float, init_noise_scale: float,
                                  frame_indices: list = None):
    """计算 noise scale 和 denoising step。

    Args:
        frame_indices: 当非 None 时，使用这些索引对应的帧计算 L2 距离（FRUC 模式下传入 hot_indices）。
                       为 None 时退回到原来的连续切片逻辑。
    """
    if frame_indices is not None and len(frame_indices) >= 2:
        # FRUC 模式：用实际送入 DiT 的帧（间隔帧）计算运动量
        frames = input_video_original[:, :, frame_indices]  # [B,C,T,H,W]
        l2_dist = (frames[:, :, 1:] - frames[:, :, :-1])**2
    else:
        l2_dist = (input_video_original[:, :, end_idx-chunk_size:end_idx] -
                   input_video_original[:, :, end_idx-chunk_size-1:end_idx-1])**2
    l2_dist = (torch.sqrt(l2_dist.mean(dim=(0, 1, 3, 4))).max()/0.2).clamp(0, 1)
    new_noise_scale = (init_noise_scale-0.1*l2_dist.item())*0.9+noise_scale*0.1
    current_step = int(1000*new_noise_scale)-100
    return new_noise_scale, current_step


def compute_noise_scale_and_step_chunk(chunk_with_prev: torch.Tensor, noise_scale: float, init_noise_scale: float):
    """Same noise-scale rule as compute_noise_scale_and_step, but takes ``chunk_with_prev`` of
    shape (B, C, chunk_size+1, H, W) where index 0 is the lookback frame and 1: is the chunk.
    """
    cur = chunk_with_prev[:, :, 1:]
    prev = chunk_with_prev[:, :, :-1]
    l2_dist = (cur - prev) ** 2
    l2_dist = (torch.sqrt(l2_dist.mean(dim=(0, 1, 3, 4))).max() / 0.2).clamp(0, 1)
    new_noise_scale = (init_noise_scale - 0.1 * l2_dist.item()) * 0.9 + noise_scale * 0.1
    current_step = int(1000 * new_noise_scale) - 100
    return new_noise_scale, current_step


# --- Import FRUC Interpolator (FFmpeg minterpolate, open-source replacement for kfruc) ---
try:
    from streamv2v.ffmpeg_fruc_interpolator import FFmpegFRUCInterpolator, FFMPEG_FRUC_AVAILABLE
    KFRUC_AVAILABLE = FFMPEG_FRUC_AVAILABLE
    if not FFMPEG_FRUC_AVAILABLE:
        logging.warning("[FRUC] ffmpeg found but minterpolate filter unavailable. Upgrade ffmpeg.")
except ImportError:
    KFRUC_AVAILABLE = False
    FFmpegFRUCInterpolator = None
    logging.warning("[FRUC] FFmpegFRUCInterpolator not available. Install ffmpeg to enable FRUC.")

# --- SingleGPUInferencePipeline class (Logging format updated) ---


class SingleGPUInferencePipeline:
    def __init__(self, config, device: torch.device, cache_min_downsample: int = 0, use_cached_text_embedding: bool = False):
        self.config = config
        self.device = device
        self.logger = logging.getLogger("SingleGPUInference")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            # Updated formatter to match target log
            formatter = logging.Formatter(
                '%(asctime)s,%(msecs)03d - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.info("Initializing CausalStreamInferencePipeline...")
        # <--- MODIFIED LINE: Pass the new argument to the pipeline --->
        self.pipeline = CausalStreamInferencePipeline(config, device=str(
            device), text_encoder_on_cpu=True, cache_min_downsample=cache_min_downsample, use_cached_text_embedding=use_cached_text_embedding)
        self.pipeline.to(device=str(device), dtype=torch.float16)
        self.vae_encoder = self.pipeline.vae
        self.vae_decoder = self.pipeline.vae
        self.logger.info("Single GPU inference pipeline manager initialized")

    def load_model(self, checkpoint_folder: str):
        ckpt_path = os.path.join(checkpoint_folder, "model.pt")
        self.logger.info(f"Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get('generator') or ckpt.get('generator_ema') or ckpt.get('state_dict') or ckpt
        try:
            self.pipeline.generator.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            self.logger.warning(f"Strict load_state_dict failed: {e}; retrying with strict=False")
            self.pipeline.generator.load_state_dict(state_dict, strict=False)

    def set_vae_backend(self, vae_backend: str):
        if vae_backend == "taehv":
            taehv = build_taehv_vae(self.device, dtype=torch.float16)
            self.vae_encoder = taehv
            self.vae_decoder = taehv
            self.pipeline.vae = taehv
            self.logger.info("Using TAEHV encoder + TAEHV decoder.")
            return

        if vae_backend == "wan-taehv":
            taehv = build_taehv_vae(self.device, dtype=torch.float16)
            self.vae_encoder = self.pipeline.vae
            self.vae_decoder = taehv
            self.pipeline.vae = taehv
            self.logger.info("Using WanVAE encoder + TAEHV decoder.")
            return

        self.logger.info("Using WanVAE encoder + WanVAE decoder.")

    def prepare_pipeline(self, text_prompts: list, noise: torch.Tensor, current_start: int, current_end: int):
        return self.pipeline.prepare(
            text_prompts=text_prompts, device=self.device, dtype=torch.float16,
            block_mode='input', noise=noise, current_start=current_start, current_end=current_end
        )

# --- Optimized ParallelInferenceOrchestrator (Logging format updated) ---


class ParallelInferenceOrchestrator:
    def __init__(self, pipeline_manager: SingleGPUInferencePipeline, enable_kfruc: bool = False, kfruc_rate: int = 2):
        self.pipeline_manager = pipeline_manager
        self.pipeline = pipeline_manager.pipeline
        self.device = pipeline_manager.device
        self.logger = configure_logger("ParallelOrchestrator")

        self.producer_stream = torch.cuda.Stream(device=self.device)
        self.consumer_stream = torch.cuda.Stream(device=self.device)

        self.data_queue = queue.Queue(maxsize=5)
        self.save_queue = queue.Queue()
        self.fruc_queue = queue.Queue()  # fruc_thread 输入队列（无界，避免背压阻塞 consumer/DiT）
        self.producer_thread = None
        self.saver_thread = None
        self.fruc_thread = None
        self.processed = 0
        self.producer_error = None
        self.stop_event = threading.Event()
        self.pipeline_ready_event = threading.Event()

        # FRUC (Frame Rate Up-Conversion) via FFmpeg minterpolate
        self.kfruc_rate = kfruc_rate
        self.kfruc = None
        if enable_kfruc and KFRUC_AVAILABLE:
            self.enable_kfruc = True
            self.kfruc = FFmpegFRUCInterpolator(preset="balanced")
            self.logger.info(f"[FRUC] Enabled with rate={kfruc_rate}x (FFmpeg minterpolate)")
        elif enable_kfruc and not KFRUC_AVAILABLE:
            self.enable_kfruc = False
            self.logger.warning("[FRUC] Requested but FFmpeg minterpolate not available. Disable FRUC.")
        else:
            self.enable_kfruc = False

    def _producer_task_wrapper(self, *args, **kwargs):
        try:
            self._producer_task(*args, **kwargs)
        except Exception as e:
            tb = traceback.format_exc()
            self.producer_error = (e, tb)
            self.logger.error("Producer thread failed.", exc_info=True)
            try:
                # Unblock consumer so it can surface the error.
                self._force_put_sentinel((None, self.producer_error, None, None, "ERROR"))
            except Exception:
                pass

    def _raise_if_producer_error(self, noisy_latents, current_step, flow_data, producer_done_event, chunk_id):
        if chunk_id == "ERROR":
            err, tb = current_step
            raise RuntimeError(f"Producer thread failed: {err}\n{tb}")

    def _get_from_queue_or_raise(self):
        while True:
            if self.producer_error is not None:
                err, tb = self.producer_error
                raise RuntimeError(f"Producer thread failed: {err}\n{tb}")
            try:
                item = self.data_queue.get(timeout=1)
            except queue.Empty:
                if self.producer_thread is not None and not self.producer_thread.is_alive():
                    raise RuntimeError("Producer thread exited without producing data.")
                continue
            self._raise_if_producer_error(*item)
            return item

    def _put_or_stop(self, item) -> bool:
        """Put on data_queue, polling stop_event so we can't hang on a full queue
        when the consumer has stopped. Returns False if stop fires before put."""
        while not self.stop_event.is_set():
            try:
                self.data_queue.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def _force_put_sentinel(self, item):
        """Always deliver the sentinel; if the queue is full, evict one item to
        make room. Used so the consumer can never miss the STOP signal."""
        deadline = time.time() + 5.0
        while time.time() < deadline:
            try:
                self.data_queue.put(item, timeout=0.2)
                return
            except queue.Full:
                try:
                    self.data_queue.get_nowait()
                except queue.Empty:
                    pass

    def _stream_encode(self, video: torch.Tensor, mask: torch.Tensor, flow: torch.Tensor, is_nocache: bool) -> torch.Tensor:
        return self.pipeline_manager.vae_encoder.stream_encode(video, mask, flow, is_nocache=is_nocache)

    def _stream_decode_to_pixel(self, latents: torch.Tensor, mask: torch.Tensor, flow: torch.Tensor):
        return self.pipeline_manager.vae_decoder.stream_decode_to_pixel(latents, mask, flow)
        self.prev_latent = None

    def _encode_chunk_and_build_flow(self,
                                     inp: torch.Tensor,
                                     ref_frame_tensor: torch.Tensor,
                                     cur_frame_tensor: torch.Tensor,
                                     flow_calculator: "OpticalFlowCalculator",
                                     mask_dilate: int,
                                     min_res: tuple,
                                     occlusion_method: str,
                                     top_k_percentage,
                                     is_nocache: bool):
        """Run flow + encoder mask build + VAE encode for one chunk and produce
        ``(latents, flow_data_for_dit)``. Shared between tensor and camera modes."""
        flow_pair = flow_calculator.calculate_flow(ref_frame_tensor, cur_frame_tensor)
        bwd_flow, bwd_occ = flow_pair

        if occlusion_method == "gather_block":
            masks_enc = build_gather_block_masks(
                bwd_occ.squeeze(0).squeeze(0), top_k_percentage=top_k_percentage[0]
            )
        else:
            mask_enc = dilate_mask(bwd_occ.squeeze(0).squeeze(0), int(mask_dilate))
            masks_enc = downsample_mask(
                mask_enc,
                min_res=tuple(min_res),
                dilation=int(mask_dilate),
            )

        latents = self._stream_encode(
            inp,
            mask=masks_enc,
            flow=bwd_flow.squeeze(0).permute(1, 2, 0).contiguous(),
            is_nocache=is_nocache,
        )
        latents = latents.transpose(2, 1).contiguous()

        latent_h, latent_w = latents.shape[-2:]
        target_h, target_w = latent_h, latent_w
        downsampled_flow = F.interpolate(bwd_flow, size=(target_h, target_w), mode='bilinear', align_corners=False)
        downsampled_flow *= (float(target_h) / bwd_flow.shape[2])
        downsampled_occ = F.interpolate(bwd_occ, size=(target_h, target_w), mode='bilinear', align_corners=False)
        latent_binary_mask = flow_calculator.compute_binary_occlusion_mask(downsampled_occ)

        downsampled_occ_half = F.interpolate(bwd_occ, size=(target_h // 2, target_w // 2), mode='bilinear', align_corners=False)
        latent_binary_mask_half = flow_calculator.compute_binary_occlusion_mask(downsampled_occ_half)
        flow_data = (downsampled_flow, latent_binary_mask, latent_binary_mask_half)
        return latents, flow_data

    def _producer_task(self, input_video_original: torch.Tensor,
                       flow_calculator: OpticalFlowCalculator,
                       num_chunks: int, chunk_size: int, noise_scale: float,
                       num_steps: int, fps_generate: int, mask_dilate: int, min_res: tuple[int, int],
                       occlusion_method: str, top_k_percentage: dict, is_nocache: bool,
                       camera_stream: "CameraStream" = None):
        self.logger.info("Producer thread started.")

        camera_mode = camera_stream is not None
        if camera_mode:
            self.logger.info("Producer: camera mode (real-time stream).")
        else:
            self.logger.info("Producer: tensor/file mode.")

        # When FRUC is enabled in file mode, use strided sampling.
        # In camera mode we always use stride=1 for minimal latency.
        fruc_stride = self.kfruc_rate if self.enable_kfruc else 1

        is_realtime_sim = (not camera_mode) and fps_generate > 0
        chunk_interval_seconds = 0
        if is_realtime_sim:
            chunk_interval_seconds = chunk_size / fps_generate
            self.logger.info(
                f"Real-time simulation enabled (file mode): Target Producer FPS={fps_generate}, "
                f"Chunk Size={chunk_size}, Target Interval={chunk_interval_seconds:.4f}s"
            )

        next_chunk_submit_time = time.time()

        with torch.cuda.stream(self.producer_stream):
            # --- 1. Cold start: encode the first 5 frames for prepare() ---
            prod_end_event = torch.cuda.Event(enable_timing=True)

            if camera_mode:
                taken = camera_stream.take_initial(5, stride=fruc_stride)
                if taken is None:
                    self.logger.error("Camera stream stopped before initial 5 frames were available.")
                    self._force_put_sentinel((None, None, None, None, "STOP"))
                    return
                init_chunk_cpu, init_ids = taken
                self.logger.info(
                    f"Producer: cold-start frames captured (ids={init_ids}); "
                    f"camera stats={camera_stream.get_stats()}"
                )
                inp = init_chunk_cpu.to(self.device, non_blocking=True)
                latents = self._stream_encode(inp, None, None, is_nocache)
                latents = latents.transpose(2, 1).contiguous()
                noise = torch.randn_like(latents)
                noisy_latents = noise * noise_scale + latents * (1 - noise_scale)
            else:
                # File mode: strided cold start
                cold_indices = [i * fruc_stride for i in range(5)]
                start_idx, end_idx = 0, cold_indices[-1] + fruc_stride

                if input_video_original is not None:
                    inp = input_video_original[:, :, cold_indices].to(self.device, non_blocking=True)
                    latents = self._stream_encode(inp, None, None, is_nocache)
                    latents = latents.transpose(2, 1).contiguous()
                    noise = torch.randn_like(latents)
                    noisy_latents = noise * noise_scale + latents * (1 - noise_scale)
                else:
                    noisy_latents = torch.randn(
                        1, 1 + self.pipeline.num_frame_per_block, 16,
                        self.pipeline.height, self.pipeline.width,
                        device=self.device, dtype=torch.float16,
                    )

            prod_end_event.record()
            if not self._put_or_stop((noisy_latents, None, None, prod_end_event, "Initial")):
                self._force_put_sentinel((None, None, None, None, "STOP"))
                return
            self.logger.info(
                f"Producer: Initial 5-frame data block placed in queue. "
                f"({time.time()-next_chunk_submit_time:.4f}s)"
            )

            # --- 2. Hot loop ---
            init_noise_scale = noise_scale
            chunk_counter = 0
            real_chunks_emitted = 0

            if not camera_mode:
                # File mode: track ref frame index and total chunk budget
                ref_frame_idx = cold_indices[-1]
                total_hot_chunks = num_chunks + num_steps - 1

            while True:
                # Termination check
                if camera_mode:
                    if self.stop_event.is_set() or camera_stream.is_stopped():
                        self.logger.info(
                            f"Producer: stop signal / camera ended. Real chunks emitted={real_chunks_emitted}."
                        )
                        break
                else:
                    if chunk_counter >= total_hot_chunks:
                        break
                    if is_realtime_sim:
                        next_chunk_submit_time += chunk_interval_seconds
                        sleep_needed = next_chunk_submit_time - time.time()
                        if sleep_needed > 0:
                            self.logger.warning(f"Producer is sleeping for {sleep_needed:.4f}s")
                            time.sleep(sleep_needed)
                        else:
                            self.logger.warning(
                                f"Producer is lagging behind real-time schedule by {-sleep_needed:.4f}s"
                            )

                chunk_counter += 1
                chunk_id = chunk_counter
                prod_end_event = torch.cuda.Event(enable_timing=True)
                flow_data = None
                current_step = None

                if camera_mode:
                    taken = camera_stream.take_chunk_with_lookback(chunk_size, stride=fruc_stride)
                    if taken is None:
                        self.logger.info("Producer: camera stream returned no more chunks; transitioning to flush.")
                        chunk_counter -= 1  # undo this slot
                        break
                    chunk_with_prev_cpu, ids, dropped = taken
                    self.logger.info(
                        f"Camera chunk {chunk_id}: ids={ids[1:]} (lookback id={ids[0]}), "
                        f"dropped_since_last={dropped}, total_stats={camera_stream.get_stats()}"
                    )
                    chunk_with_prev = chunk_with_prev_cpu.to(self.device, non_blocking=True)
                    inp = chunk_with_prev[:, :, 1:].contiguous()  # (1,3,chunk_size,H,W)
                    ref_frame_tensor = chunk_with_prev[:, :, 0].to(torch.float32)
                    cur_frame_tensor = chunk_with_prev[:, :, -1].to(torch.float32)

                    latents, flow_data = self._encode_chunk_and_build_flow(
                        inp, ref_frame_tensor, cur_frame_tensor, flow_calculator,
                        mask_dilate, min_res, occlusion_method, top_k_percentage, is_nocache,
                    )

                    noise_scale, current_step = compute_noise_scale_and_step_chunk(
                        chunk_with_prev, noise_scale, init_noise_scale
                    )
                    noise = torch.randn_like(latents)
                    noisy_latents = noise * noise_scale + latents * (1 - noise_scale)
                    real_chunks_emitted += 1

                else:
                    # File mode: strided hot chunk
                    hot_indices = [end_idx + j * fruc_stride for j in range(chunk_size)]
                    end_idx = hot_indices[-1] + fruc_stride

                    if input_video_original is not None and hot_indices[-1] < input_video_original.shape[2]:
                        inp = input_video_original[:, :, hot_indices].to(self.device)
                        cur_frame_idx = hot_indices[-1]

                        self.logger.info(
                            f"Chunk {chunk_id}: calculating flow Frame {ref_frame_idx} -> {cur_frame_idx}"
                        )
                        ref_frame_tensor = input_video_original[:, :, ref_frame_idx].to(self.device, torch.float32)
                        cur_frame_tensor = input_video_original[:, :, cur_frame_idx].to(self.device, torch.float32)

                        flow_pair = flow_calculator.calculate_flow(ref_frame_tensor, cur_frame_tensor)
                        bwd_flow, bwd_occ = flow_pair

                        if occlusion_method == "gather_block":
                            masks_enc = build_gather_block_masks(
                                bwd_occ.squeeze(0).squeeze(0), top_k_percentage=top_k_percentage[0]
                            )
                        else:
                            mask_enc = dilate_mask(bwd_occ.squeeze(0).squeeze(0), int(mask_dilate))
                            masks_enc = downsample_mask(
                                mask_enc,
                                min_res=tuple(min_res),
                                dilation=int(mask_dilate),
                            )

                        ref_frame_idx = cur_frame_idx

                        latents = self._stream_encode(
                            inp,
                            mask=masks_enc,
                            flow=bwd_flow.squeeze(0).permute(1, 2, 0).contiguous(),
                            is_nocache=is_nocache,
                        )
                        latents = latents.transpose(2, 1).contiguous()

                        noise_scale, current_step = compute_noise_scale_and_step(
                            input_video_original, end_idx, chunk_size, noise_scale, init_noise_scale,
                        )
                        noise = torch.randn_like(latents)
                        noisy_latents = noise * noise_scale + latents * (1 - noise_scale)

                        latent_h, latent_w = latents.shape[-2:]
                        target_h = latent_h
                        target_w = latent_w
                        downsampled_flow = F.interpolate(bwd_flow, size=(target_h, target_w), mode='bilinear', align_corners=False)
                        downsampled_flow *= (float(target_h) / bwd_flow.shape[2])
                        downsampled_occ = F.interpolate(bwd_occ, size=(target_h, target_w), mode='bilinear', align_corners=False)
                        latent_binary_mask = flow_calculator.compute_binary_occlusion_mask(downsampled_occ)

                        downsampled_occ_half = F.interpolate(bwd_occ, size=(target_h // 2, target_w // 2), mode='bilinear', align_corners=False)
                        latent_binary_mask_half = flow_calculator.compute_binary_occlusion_mask(downsampled_occ_half)
                        flow_data = (downsampled_flow, latent_binary_mask, latent_binary_mask_half)
                        real_chunks_emitted += 1

                    else:
                        # Tail flush in tensor mode
                        noisy_latents = torch.randn(
                            1, self.pipeline.num_frame_per_block, 16,
                            self.pipeline.height, self.pipeline.width,
                            device=self.device, dtype=torch.float16,
                        )
                        current_step = None

                prod_end_event.record()
                if not self._put_or_stop((noisy_latents, current_step, flow_data, prod_end_event, chunk_id)):
                    self.logger.info("Producer: stop_event set while enqueueing; exiting hot loop.")
                    break

                if camera_mode:
                    self.logger.info(
                        f"Producer: camera chunk {chunk_id} enqueued. Queue len={self.data_queue.qsize()}"
                    )
                else:
                    if chunk_id <= num_chunks:
                        self.logger.info(f"Producer: Real data chunk {chunk_id}/{num_chunks} placed in queue. ")
                        self.logger.info(f"Queue len: {self.data_queue.qsize()}")
                    else:
                        flush_chunk_id = chunk_id - num_chunks
                        total_flush_chunks = num_steps - 1
                        self.logger.info(
                            f"Producer: Flush chunk {flush_chunk_id}/{total_flush_chunks} placed in queue."
                        )

            # --- 3. Pipeline drain (camera mode only) ---
            if camera_mode:
                drain_chunks = max(0, num_steps - 1)
                self.logger.info(f"Producer: producing {drain_chunks} flush chunks to drain DiT pipeline.")
                for _ in range(drain_chunks):
                    chunk_counter += 1
                    prod_end_event = torch.cuda.Event(enable_timing=True)
                    noisy_latents = torch.randn(
                        1, self.pipeline.num_frame_per_block, 16,
                        self.pipeline.height, self.pipeline.width,
                        device=self.device, dtype=torch.float16,
                    )
                    prod_end_event.record()
                    if not self._put_or_stop((noisy_latents, None, None, prod_end_event, chunk_counter)):
                        break

        # Sentinel: always delivered, even if the queue is full at this moment.
        self._force_put_sentinel((None, None, None, None, "STOP"))
        self.logger.info("Producer thread finished. All data blocks produced; sentinel sent.")

    def _fruc_task(self):
        """FRUC 线程：从 fruc_queue 取帧，用 lookahead 方式插帧后送入 save_queue。

        关键设计：lookahead tail 模式
        ───────────────────────────────
        minterpolate 依赖前后帧做运动估计。仅输入当前 chunk（4帧）时，末帧没有后续
        参考，导致最后一个时间间隔的插帧被截断（4帧输入只输出 5 帧而非 8 帧）。

        解法：把下一个 chunk 的第一帧拼到当前 chunk 尾部（tail lookahead），让
        minterpolate 有完整的运动窗口，再把多余的最后 kfruc_rate 帧（属于下一
        chunk 的过渡区）截掉。每个 chunk 精确输出 chunk_size * kfruc_rate 帧。

        帧流示意（kfruc_rate=2, chunk_size=4, fruc_stride=2）：
          DiT 输出:  [f0,f2,f4,f6]        [f8,f10,f12,f14]     ...
          第1chunk:  frames=[f0,f2,f4,f6], tail=f8
                     输入 minterpolate: [f0,f2,f4,f6 | f8]（5帧）
                     输出 10 帧，取前 8 帧: [f0,f1, f2,f3, f4,f5, f6,f7]
          第2chunk:  frames=[f8,f10,f12,f14], tail=f16
                     输入 minterpolate: [f8,f10,f12,f14 | f16]（5帧）
                     输出 10 帧，取前 8 帧: [f8,f9, f10,f11, f12,f13, f14,f15]
          最后chunk: 无 tail → fallback 到末帧复制，仍精确输出 chunk_size*kfruc_rate 帧
        """
        assert self.kfruc is not None, "_fruc_task started without a valid kfruc instance"
        self.logger.info("FRUC thread started.")

        # lookahead buffer: 先缓存当前 chunk，等下一个 chunk 到来后才处理
        pending = None  # (video_tensor, index)

        def process_one(video_tensor, index, tail_frame_np):
            """插帧并送入 save_queue。tail_frame_np 为 None 时末尾用复制填充。"""
            video_np = video_tensor.cpu().float().numpy()  # [T, H, W, 3]
            t_fruc = time.perf_counter()
            try:
                video_np_out = self.kfruc.interpolate_chunk(video_np, tail_frame=tail_frame_np)
            except Exception as e:
                self.logger.error(f"[FRUC] interpolate_chunk failed: {e}, falling back")
                # fallback: 简单帧复制到目标帧数
                want = video_np.shape[0] * self.kfruc_rate
                video_np_out = np.repeat(video_np, self.kfruc_rate, axis=0)[:want]
            t_fruc = time.perf_counter() - t_fruc
            n_in, n_out = video_np.shape[0], video_np_out.shape[0]
            self.logger.info(f"[FRUC] {n_in} → {n_out} frames in {t_fruc:.3f}s")
            video_out = torch.from_numpy(video_np_out)
            self.save_queue.put((video_out.to('cpu', non_blocking=False), index))

        while True:
            item = self.fruc_queue.get()

            if item is None:  # sentinel: 队列结束
                self.logger.info("FRUC thread received termination signal.")
                if pending is not None:
                    # 最后一个 chunk 没有 lookahead tail，用 None（内部末帧复制填充）
                    video_t, idx = pending
                    process_one(video_t, idx, tail_frame_np=None)
                    pending = None
                break

            video, index = item

            if pending is not None:
                # 用当前 chunk 的第一帧作为上一个 chunk 的 lookahead tail
                prev_video, prev_index = pending
                tail_np = video.cpu().float().numpy()[0]  # [H, W, 3]
                process_one(prev_video, prev_index, tail_frame_np=tail_np)

            pending = (video, index)

        self.logger.info("FRUC thread finished.")

    def _saver_task(self, results_dict: dict):
        self.logger.info("Saver thread started.")
        last_save_time = time.time()
        chunk_size = 4
        iteration_times = []
        while True:
            # Get data from the save queue
            item = self.save_queue.get()

            # Sentinel value to signal termination
            if item is None:
                self.logger.info("Saver thread received termination signal.")
                break

            cpu_tensor_future, index = item

            # This line will block THIS (saver) thread until the specific
            # non-blocking transfer initiated by the consumer is complete.
            # The main consumer thread is NOT blocked.
            numpy_array = cpu_tensor_future.float().numpy()

            results_dict[index] = numpy_array
            self.logger.debug(f"Saver: Saved numpy array for index {index}.")
            current_time = time.time()
            iter_time = current_time - last_save_time
            last_save_time = current_time
            iteration_times.append(iter_time)
            iter_fps = chunk_size / iter_time
            self.logger.info(
                f"Saver: Render Video Chunk for iter {index}, Iter Time: {iter_time:.4f}s, FPS: {iter_fps:.4f}")

        if iteration_times:
            iteration_times = np.array(iteration_times)
            iteration_times = iteration_times[1:]
            avg_iter_time = np.mean(iteration_times)
            avg_fps = chunk_size / avg_iter_time
            self.logger.info(f"Average End-to-End FPS (Saver-side, after pipeline fill): {avg_fps:.4f}")
        self.logger.info("Saver thread finished.")

    def run_parallel_inference(
        self,
        input_video_original: torch.Tensor,
        flow_calculator: OpticalFlowCalculator,
        prompts: list,
        num_chunks: int,
        chunk_size: int,
        noise_scale: float,
        output_folder: str,
        fps: int,
        num_steps: int,
        fps_generate: int,
        mask_dilate: int,
        min_res: tuple[int, int],
        occlusion_method: str,
        top_k_percentage: int,
        # profile_consumer: bool = False,
        is_nocache: bool,
        camera_stream: "CameraStream" = None,
    ):
        # torch.cuda.synchronize(device=self.device)
        # mem_run_start = torch.cuda.memory_reserved(device=self.device)

        self.logger.info("Consumer started. Replicating original inference logic with detailed timing.")
        os.makedirs(output_folder, exist_ok=True)

        scatter_cache_modules = collect_scatter_cache_modules(self.pipeline_manager.vae_encoder)

        self.producer_thread = threading.Thread(
            target=self._producer_task_wrapper,
            kwargs=dict(
                input_video_original=input_video_original,
                flow_calculator=flow_calculator,
                num_chunks=num_chunks,
                chunk_size=chunk_size,
                noise_scale=noise_scale,
                num_steps=num_steps,
                fps_generate=fps_generate,
                mask_dilate=mask_dilate,
                min_res=min_res,
                occlusion_method=occlusion_method,
                top_k_percentage=top_k_percentage,
                is_nocache=is_nocache,
                camera_stream=camera_stream,
            ),
        )
        self.producer_thread.start()

        results = {}
        self.saver_thread = threading.Thread(target=self._saver_task, args=(results,))
        self.saver_thread.start()

        # FRUC 独立线程：只在启用时启动，并提前初始化
        if self.enable_kfruc and self.kfruc is not None and (input_video_original is not None or camera_stream is not None):
            if input_video_original is not None:
                h = (input_video_original.shape[3] // 32) * 32
                w = (input_video_original.shape[4] // 32) * 32
            else:
                # camera mode: use pipeline resolution
                h = (self.pipeline.height // 32) * 32
                w = (self.pipeline.width // 32) * 32
            # input_fps = 原始视频帧率（FRUC 输出帧率 = input_fps，保持一致）
            # interpolate_rate = kfruc_rate（每帧间距对应插出的帧数）
            self.kfruc.initialize(
                input_fps=float(fps) / self.kfruc_rate,   # DiT 输入帧率（间隔采样后）
                interpolate_rate=self.kfruc_rate,
                width=w, height=h,
            )
            self.fruc_thread = threading.Thread(target=self._fruc_task, daemon=True)
            self.fruc_thread.start()
            self.logger.info(
                f"[FRUC] Thread started: DiT input {fps/self.kfruc_rate:.1f}fps "
                f"-> output {fps:.1f}fps (keep same as source), {w}x{h}")

        # results, save_results = {}, 0
        iteration_times = []
        save_results = 0

        current_start = 0
        current_end = self.pipeline.frame_seq_length * 2

        try:
            # --- 3. Process the "Cold Start" data from the queue ---
            self.logger.info("Consumer: Waiting for initial data block...")
            initial_noisy_latents, current_step, flow_data, producer_done_event, chunk_id = self._get_from_queue_or_raise()

            with torch.cuda.stream(self.consumer_stream):
                self.consumer_stream.wait_event(producer_done_event)

                self.logger.info(f"Consumer: Got initial data block. ")

                denoised_pred = self.pipeline_manager.prepare_pipeline(
                    text_prompts=prompts,
                    noise=initial_noisy_latents,
                    current_start=current_start,
                    current_end=current_end
                )
                video = self._stream_decode_to_pixel(denoised_pred, None, None)
                if video is None:
                    raise RuntimeError("Streaming VAE decoder produced no frames for the initial block.")
                video = (video * 0.5 + 0.5).clamp(0, 1)
                video = video[0].permute(0, 2, 3, 1).contiguous()
                self.save_queue.put((video.to('cpu', non_blocking=False), save_results))
                save_results += 1
                self.logger.info("Consumer: Initial block processed and enqueued for saving.")
                self.pipeline_ready_event.set()  # JIT warmup done

                # video = (video * 0.5 + 0.5).clamp(0, 1)
                # video = video[0].permute(0, 2, 3, 1).contiguous()
                # results[save_results] = video.cpu().float().numpy()
                # save_results += 1
                # self.logger.info("Consumer: Initial block processed and saved.")

            # torch.cuda.synchronize(device=self.device)
            # mem_run_end = torch.cuda.memory_reserved(device=self.device)
            # print("GPU memory used by consumer during run(): ", (mem_run_end - mem_run_start)/1024/1024/1024, "GB","from",mem_run_start/1024/1024/1024,"GB to",mem_run_end/1024/1024/1024,"GB")
            # --- 4. Process "Hot Loop" data from the queue ---
            last_save_time = time.time()  # Initialize timer for first iteration
            while True:
                noisy_latents, current_step, flow_data, producer_done_event, chunk_id = self._get_from_queue_or_raise()
                if chunk_id == "STOP":
                    self.logger.info("Consumer: received STOP sentinel; ending hot loop.")
                    break
                # In file mode, also respect the original bounded termination
                if camera_stream is None and self.processed >= num_chunks + num_steps - 1:
                    break

                with torch.cuda.stream(self.consumer_stream):
                    self.consumer_stream.wait_event(producer_done_event)
                    self.logger.info(f"Consumer: Got data block {self.processed+1}.")
                    current_start = current_end
                    current_end += (chunk_size // 4) * self.pipeline.frame_seq_length

                    denoised_pred = self.pipeline.inference_stream(
                        noise=noisy_latents,
                        current_start=current_start,
                        current_end=current_end,
                        current_step=current_step,
                        latent_flow_data=flow_data,
                        # latent_flow_data=None,
                    )

                    video_out = None
                    if self.processed + 1 >= num_steps:
                        # Decoder sparse only at low resolutions; build masks once per chunk.
                        if flow_data is not None:
                            bwd_flow, bwd_occ,_ = flow_data
                            if occlusion_method == "gather_block":
                                masks_dec = build_gather_block_masks(bwd_occ, top_k_percentage[0])
                            else:
                                # Encoder masks
                                mask_dec = dilate_mask(bwd_occ, int(mask_dilate))
                                masks_dec = downsample_mask(
                                    mask_dec,
                                    min_res=tuple(min_res),
                                    dilation=int(mask_dilate),
                                )
                        # start_event = torch.cuda.Event(enable_timing=True)
                        # end_event = torch.cuda.Event(enable_timing=True)
                        # torch.cuda.synchronize()  # 保证前面操作完成
                        # start_event.record()

                        video_out = self._stream_decode_to_pixel(denoised_pred[[-1]], mask=None, flow=None)
                        # end_event.record()
                        # torch.cuda.synchronize()
                        # elapsed_time_ms = start_event.elapsed_time(end_event)
                        # print(f"stream_decode GPU time: {elapsed_time_ms:.3f} ms")

                    self.processed += 1

                    print(
                        # f"feat_map={format_bytes(feat_map_nbytes(self.pipeline_manager.vae_encoder.model))} "
                        f"scatter_total={format_bytes(scatter_cache_nbytes(scatter_cache_modules))}"
                    )
                    if video_out is not None:
                        video = (video_out * 0.5 + 0.5).clamp(0, 1)
                        video = video[0].permute(0, 2, 3, 1).contiguous()

                        if self.enable_kfruc and self.kfruc is not None:
                            # === FRUC: 异步送入 fruc_thread，不阻塞 consumer ===
                            self.fruc_queue.put((video, save_results))
                        else:
                            self.save_queue.put((video.to('cpu', non_blocking=False), save_results))

                        # --- Iteration Timing and Logging ---
                        current_time = time.time()
                        iter_time = current_time - last_save_time
                        last_save_time = current_time
                        iteration_times.append(iter_time)
                        iter_fps = chunk_size / iter_time

                        self.logger.info(
                            f"Consumer: Enqueued output for iter {save_results}, Iter Time: {iter_time:.4f}s, FPS: {iter_fps:.4f}")
                        save_results += 1

                    # torch.cuda.synchronize(device=self.device)
                    # mem_run_end2 = torch.cuda.memory_reserved(device=self.device)
                    # print(self.processed,"GPU memory used by consumer during run(): ", (mem_run_end2 - mem_run_end)/1024/1024/1024, "GB","from",mem_run_end/1024/1024/1024,"GB to",mem_run_end2/1024/1024/1024,"GB")

        finally:
            # Make sure producer is unblocked even if consumer crashed.
            self.stop_event.set()
            self.producer_thread.join()

            # fruc_thread 先收到 sentinel，处理完剩余帧后再让 saver 退出
            if self.fruc_thread is not None:
                self.fruc_queue.put(None)  # sentinel
                self.fruc_thread.join()
                self.logger.info("[FRUC] Thread joined.")

            # === FRUC Cleanup ===
            if self.enable_kfruc and self.kfruc is not None:
                try:
                    self.kfruc.cleanup()
                    self.logger.info("[FRUC] Interpolator cleaned up")
                except Exception as e:
                    self.logger.error(f"[FRUC] Cleanup failed: {e}")

            self.save_queue.put(None)  # Sentinel value
            self.saver_thread.join()

            self.logger.info("="*50)
            self.logger.info("Performance Summary")
            self.logger.info("="*50)

            # Ensure we have the correct number of frames
            video_list = [results[i] for i in range(save_results) if i in results]
            if video_list:
                video = np.concatenate(video_list, axis=0)
                print(f"Video shape before trimming: {video.shape}")
                if camera_stream is None and input_video_original is not None:
                    target_frames = input_video_original.shape[2]  # FRUC 保持原始帧数
                    video = video[:target_frames]
            else:
                self.logger.warning("No output frames were produced.")
                video = None

            # Final FPS summary
            if iteration_times:
                avg_iter_time = np.mean(iteration_times[1:]) if len(iteration_times) > 1 else iteration_times[0]
                avg_input_fps = chunk_size / avg_iter_time
                # DiT 实际产出的帧率（间隔采样时 = avg_input_fps * kfruc_rate）
                avg_output_fps = avg_input_fps * (self.kfruc_rate if self.enable_kfruc else 1)
                self.logger.info(
                    f"Average DiT_fps={avg_input_fps:.2f}"
                    + (f" (stride={self.kfruc_rate}, FRUC {self.kfruc_rate}x)" if self.enable_kfruc else "")
                    + f", pipeline_output_fps={avg_output_fps:.2f}"
                    + f", saved @ {fps:.1f}fps"
                )

            if camera_stream is not None:
                stats = camera_stream.get_stats()
                self.logger.info(
                    f"Camera frame stats: captured={stats['captured']}, consumed={stats['consumed']}, "
                    f"dropped={stats['dropped']}, drop_rate={stats['drop_rate']*100:.2f}%"
                )

            if video is not None:
                self.logger.info(f"Final video shape: {video.shape}")

                output_fps = fps  # 输出播放帧率与原始视频一致（FRUC 只补帧不提速）
                tag = "camera" if camera_stream is not None else "file"
                output_path = os.path.join(
                    output_folder,
                    f"output_{tag}_{occlusion_method}_vae_{top_k_percentage[0]}_dit_{top_k_percentage[1]}_steps_{num_steps}.mp4",
                )
                export_to_video(video, output_path, fps=output_fps)
                self.logger.info(f"Video saved to: {output_path} @ {output_fps} fps")

            self.logger.info("Parallel inference with timing completed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="Configuration file path")
    parser.add_argument("--checkpoint_folder", type=str, required=True, help="Checkpoint folder path")
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder path")
    parser.add_argument("--prompt_file_path", type=str, required=True, help="Prompt file path")
    parser.add_argument("--video_path", type=str, required=False, default=None, help="Input video path")
    parser.add_argument("--noise_scale", type=float, default=0.700, help="Noise scale")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--width", type=int, default=832, help="Video width")
    parser.add_argument("--fps", type=int, default=16, help="Output video fps")
    parser.add_argument("--fps_generate", type=int, default=30,
                        help="Target FPS for the producer (VAE encode) thread. Simulates a camera. If 0, runs as fast as possible. Default: 0.")
    parser.add_argument("--step", type=int, default=2, help="Step")
    parser.add_argument("--model_type", type=str, default="T2V-1.3B", help="Model type (e.g., T2V-1.3B)")
    parser.add_argument(
        "--vae_type",
        type=str.lower,
        default="wanvae",
        choices=["wanvae", "taehv", "wan-taehv"],
        help="VAE backend: wanvae, taehv, or wan-taehv (Wan encoder + TAEHV decoder).",
    )
    parser.add_argument("--max_frames", type=int, default=None, help="Video length (number of frames)")
    parser.add_argument("--flow_model", type=str, default="x265", choices=["gmflow", "raft", "x265", "none"], help="Optical flow model to use (from calflow). If None, flow is not calculated.")
    parser.add_argument("--x265_params", type=str, default='{"stage":"encode", "quiet":true}', help="x265 parameters as a JSON string. e.g., '{\"stage\": \"lookahead\"}'")
    parser.add_argument("--occlusion_method", type=str, default="quantile", choices=["exact", "quantile", "morphological", "connected_components", "gather_block"], help="Method to generate occlusion mask.")
    parser.add_argument("--vae_ratio", type=float, default=0.1, help="Top percentage of occlusion values to consider as masked.")
    parser.add_argument("--dit_ratio", type=float, default=0.1, help="Top percentage of occlusion values to consider as masked.")
    parser.add_argument("--use_cached_text_embedding", action="store_true", help="If set, load pre-computed text embeddings from 'cached_text_embedding.pt' instead of initializing the text encoder.")
    # FRUC (Frame Rate Up-Conversion) 插帧参数
    parser.add_argument("--enable_kfruc", action="store_true",
                        help="Enable FRUC frame interpolation via FFmpeg minterpolate (MEMC algorithm).")
    parser.add_argument("--kfruc_rate", type=int, default=2,
                        choices=[2, 4, 8], help="FRUC interpolation rate: 2x, 4x, or 8x. Default: 2.")
    parser.add_argument("--morph_kernel_size", type=int, default=7,
                        help="Kernel size for morphological opening operation.")
    parser.add_argument("--mask_dilate", type=int, default=6,
                        help="Dilation (pixels) applied to the base update mask before downsample_mask().")
    parser.add_argument("--min_res", nargs=2, type=int, default=(40, 40), metavar=("H", "W"),
                        help="Minimum resolution for downsampled masks passed to SIGE (GMFlow).")
    # parser.add_argument("--profile_consumer", action="store_true", help="Enable detailed per-iteration consumer timing breakdown.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--is_nocache", action="store_true", default=False,
                        help="If is_nocache is True, Wan VAE Encoder compute fully!")
    parser.add_argument("--max_chunks", type=int, default=None,
                        help="If set, limit the number of hot chunks processed (for debugging/benchmarking).")
    parser.add_argument(
        "--cache_min_downsample",
        type=float,
        default=0,
        help=(
            "If >0, enable decoder sparse compute for feature resolutions <= (H//ds, W//ds). "
            "Larger resolutions will run full compute without decoder scatter cache."
        ),
    )

    # ===== camera (real-time) mode =====
    parser.add_argument("--use_camera", action="store_true",
                        help="Use a live V4L2 camera as input. When set, --video_path is ignored.")
    parser.add_argument("--camera_device", type=str, default="/dev/video0",
                        help="V4L2 camera device path. Default /dev/video0.")
    parser.add_argument("--camera_src_w", type=int, default=848,
                        help="Camera native capture width.")
    parser.add_argument("--camera_src_h", type=int, default=480,
                        help="Camera native capture height.")
    parser.add_argument("--camera_src_fps", type=int, default=30,
                        help="Camera native capture FPS.")
    parser.add_argument("--max_seconds", type=float, default=0,
                        help="Stop after this many seconds of streaming (0 = run until Ctrl+C / camera ends).")

    def _kernel_backend(v: str) -> str:
        v = (v or "").strip().lower()
        if v in {"pytorch", "torch"}:
            return "pytorch"
        if v in {"cuda", "ext"}:
            return "cuda"
        raise argparse.ArgumentTypeError("Expected 'PyTorch' or 'CUDA'.")

    parser.add_argument(
        "--sige_kernels",
        type=_kernel_backend,
        default="cuda",
        help="SIGE gather/scatter kernel backend: PyTorch (default) or CUDA.",
    )
    args = parser.parse_args()
    set_seed(args.seed)

    set_kernel_backend(args.sige_kernels)

    torch.set_grad_enabled(False)

    os.makedirs(args.output_folder, exist_ok=True)
    log_file = os.path.join(args.output_folder, f"{args.occlusion_method}_{args.vae_ratio}_{args.dit_ratio}_run.log")
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
    ]
    global LOG_HANDLERS
    LOG_HANDLERS = handlers
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
        handlers=handlers,
    )

    log_file = os.path.join(args.output_folder, f"{args.occlusion_method}__{args.vae_ratio}_{args.dit_ratio}_run.log")
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
    ]
    # global LOG_HANDLERS
    # LOG_HANDLERS = handlers
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format=LOG_FORMAT,
    #     datefmt=LOG_DATEFMT,
    #     handlers=handlers,
    # )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    flow_calculator = None
    camera_stream = None

    ratio_list=(args.vae_ratio,args.dit_ratio)

    def _build_flow_calculator():
        if args.flow_model is None or args.flow_model.lower() == "none":
            return None
        logging.info(f"Preparing for optical flow calculation with model: {args.flow_model}")
        x265_params = json.loads(args.x265_params)
        return OpticalFlowCalculator(
            flow_model_type=args.flow_model,
            device=device,
            x265_params=x265_params,
            occlusion_method=args.occlusion_method,
            top_k_percentage=ratio_list,
        )

    if args.use_camera:
        ALIGNMENT = 32
        new_height = (args.height // ALIGNMENT) * ALIGNMENT
        new_width = (args.width // ALIGNMENT) * ALIGNMENT
        if new_height != args.height or new_width != args.width:
            logging.warning(
                f"Adjusting resolution from {args.height}x{args.width} to {new_height}x{new_width}."
            )
        args.height, args.width = new_height, new_width
        args.fps = args.camera_src_fps  # output mp4 fps tracks camera capture rate

        camera_stream = CameraStream(
            device_path=args.camera_device,
            src_w=args.camera_src_w,
            src_h=args.camera_src_h,
            src_fps=args.camera_src_fps,
            target_w=args.width,
            target_h=args.height,
            dtype=torch.float16,
        ).start()

        input_video_original = None
        # No predetermined frame count in real-time mode; producer runs until stop.
        t = 0
        flow_calculator = _build_flow_calculator()

        logging.info(
            f"Camera mode enabled: capture {args.camera_src_w}x{args.camera_src_h}@{args.camera_src_fps}fps, "
            f"target {args.width}x{args.height}, max_seconds={args.max_seconds}"
        )

    elif args.video_path is not None:
        ALIGNMENT = 32
        new_height = (args.height // ALIGNMENT) * ALIGNMENT
        new_width = (args.width // ALIGNMENT) * ALIGNMENT
        if new_height != args.height or new_width != args.width:
            logging.warning(f"Adjusting resolution from {args.height}x{args.width} to {new_height}x{new_width}.")
        resize_hw = (new_height, new_width)
        args.height, args.width = new_height, new_width
        input_video_original, original_fps = load_mp4_as_tensor(
            args.video_path, resize_hw=resize_hw, max_frames=args.max_frames)

        args.fps = original_fps

        input_video_original = input_video_original.unsqueeze(0)
        logging.info(f"Input video tensor shape: {input_video_original.shape}")
        t = input_video_original.shape[2]
        input_video_original = input_video_original.to(dtype=torch.float16)

        flow_calculator = _build_flow_calculator()
    else:
        input_video_original = None
        t = 0
        if args.fps_generate > 0:
            logging.warning(
                "--fps_generate is specified but neither --video_path nor --use_camera is set. "
                "The simulation will run but without video input."
            )

    config = OmegaConf.load(args.config_path)
    config = OmegaConf.merge(config, OmegaConf.create(vars(args)))

    denoising_map = {1: [700, 0], 2: [700, 500, 0], 3: [700, 600, 400, 0]}
    config.denoising_step_list = denoising_map.get(args.step, [700, 600, 500, 400, 0])

    chunk_size = 4
    # FRUC 启用时，Producer 隔 kfruc_rate 帧采样（file mode only）：
    # 冷启动占用原始帧索引 0..4*stride，每个 hot chunk 跨度 = chunk_size * stride
    fruc_stride = args.kfruc_rate if args.enable_kfruc else 1
    if camera_stream is not None:
        num_chunks = 0  # unused in camera mode; producer terminates via stop_event/sentinel
    else:
        cold_span = 5 * fruc_stride          # 冷启动占用的原始帧数
        hot_span  = chunk_size * fruc_stride  # 每个 hot chunk 对应的原始帧跨度
        num_chunks = (t - cold_span) // hot_span if t > cold_span else 0
        if t > cold_span and (t - cold_span) % hot_span != 0:
            num_chunks += 1
    if args.max_chunks is not None:
        num_chunks = min(num_chunks, args.max_chunks)
        logging.info(f"[max_chunks] Limited to {num_chunks} hot chunks ({5 + num_chunks * chunk_size} input frames)")

    pipeline_manager = SingleGPUInferencePipeline(
        config, device, args.cache_min_downsample, use_cached_text_embedding=args.use_cached_text_embedding)
    pipeline_manager.set_vae_backend(args.vae_type)
    pipeline_manager.load_model(args.checkpoint_folder)

    num_steps = len(pipeline_manager.pipeline.denoising_step_list)

    orchestrator = ParallelInferenceOrchestrator(
        pipeline_manager, enable_kfruc=args.enable_kfruc, kfruc_rate=args.kfruc_rate)

    dataset = TextDataset(args.prompt_file_path)
    prompts = [dataset[0]]

    # Watchdog thread: in camera mode, signal stop after --max_seconds (if > 0).
    watchdog_thread = None
    if camera_stream is not None and args.max_seconds and args.max_seconds > 0:
        def _watchdog():
            orchestrator.pipeline_ready_event.wait()  # wait for CUDA JIT warmup
            stopped = orchestrator.stop_event.wait(timeout=args.max_seconds)
            if not stopped:
                logging.info(f"Watchdog: --max_seconds={args.max_seconds} elapsed; signalling stop.")
                orchestrator.stop_event.set()
                camera_stream.stop()
        watchdog_thread = threading.Thread(target=_watchdog, name="MaxSecondsWatchdog", daemon=True)
        watchdog_thread.start()

    try:
        orchestrator.run_parallel_inference(
            input_video_original,
            flow_calculator,
            prompts,
            num_chunks,
            chunk_size,
            args.noise_scale,
            args.output_folder,
            args.fps,
            num_steps,
            args.fps_generate,
            args.mask_dilate,
            args.min_res,
            args.occlusion_method,
            top_k_percentage=ratio_list,
            # args.profile_consumer,
            is_nocache=args.is_nocache,
            camera_stream=camera_stream,
        )
    except KeyboardInterrupt:
        logging.warning("KeyboardInterrupt received; stopping streams.")
        orchestrator.stop_event.set()
        if camera_stream is not None:
            camera_stream.stop()
        raise
    except Exception as e:
        logging.error(f"Error occurred during inference: {e}", exc_info=True)
        orchestrator.stop_event.set()
        if camera_stream is not None:
            camera_stream.stop()
        raise
    finally:
        if camera_stream is not None:
            camera_stream.stop()
        if watchdog_thread is not None:
            watchdog_thread.join(timeout=1)


if __name__ == "__main__":
    torch.cuda.reset_peak_memory_stats()
    main()
    peak_mem = torch.cuda.max_memory_reserved()

    print(f"Peak GPU memory: {peak_mem / 1024**3:.2f} GB")

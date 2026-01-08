import os
import ctypes
import numpy as np
import cv2
import torch
from pathlib import Path
from ctypes import CDLL, CFUNCTYPE, Structure, POINTER
from ctypes import c_int, c_uint, c_uint8, c_uint32, c_uint64, c_longlong, c_float, c_double, c_void_p, c_char_p


# void (*callback)(int poc, int x, int y, int w, int h,
#                  float mvx, float mvy, int deltapoc, void* user_data)
MV_CALLBACK_FUNC = CFUNCTYPE(
    None,
    c_int,    # poc
    c_int,    # x
    c_int,    # y
    c_int,    # w
    c_int,    # h
    c_float,  # mvx
    c_float,  # mvy
    c_int,    # deltapoc
    c_void_p  # user_data
)


class x265_nal(Structure):
    """x265 NAL unit structure"""
    _fields_ = [
        ("type", c_uint32),
        ("sizeBytes", c_uint32),
        ("payload", POINTER(c_uint8))
    ]


class x265_picture(Structure):
    """
    Must match x265's x265_picture deps/x265/source/x265.h
    """
    _fields_ = [
        # Timestamps
        ("pts", c_longlong),      # int64_t pts
        ("dts", c_longlong),      # int64_t dts

        ("vbvEndFlag", c_int),
        ("userData", c_void_p),

        # Frame data planes (4 planes for RGBA support, we use 3 for I420: Y, U, V)
        ("planes", c_void_p * 4),
        ("stride", c_int * 4),

        # Picture parameters
        ("bitDepth", c_int),
        ("sliceType", c_int),
        ("poc", c_int),
        ("colorSpace", c_int),

        # Note: x265_picture has many more fields
        # but these are the critical ones for basic encoding
    ]


class MVCollector:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        # Flow array: [2, H, W] for (x, y) components
        self.flow = np.zeros((2, height, width), dtype=np.float32)
        self.callback_count = 0

    def callback(self, poc, x, y, w, h, mvx, mvy, deltapoc, user_data):
        if poc == 0:
            return

        # Ensure block is within bounds
        x_end = min(x + w, self.width)
        y_end = min(y + h, self.height)

        if x >= self.width or y >= self.height:
            return

        # mvx/mvy are in pixels precision
        self.flow[0, y:y_end, x:x_end] = mvx
        self.flow[1, y:y_end, x:x_end] = mvy

        self.callback_count += 1

    def get_flow(self, poc):
        return self.flow.copy()

    def reset(self):
        self.flow.fill(0)
        self.callback_count = 0


class X265NativeEncoder:
    """
    Low-level x265 encoder interface using ctypes.
    """

    def __init__(self, lib_path=None):
        self.param = None
        self.encoder = None
        self.lib = None

        if lib_path is None:
            project_root = Path(__file__).parent.parent
            lib_path = project_root / "deps" / "x265" / "build" / "linux-dynamic" / "libx265.so"
            lib_path = str(lib_path)

        if not os.path.exists(lib_path):
            raise FileNotFoundError(
                f"x265 library not found: {lib_path}\n"
            )

        self.lib = CDLL(lib_path)
        self._setup_function_signatures()

    def _setup_function_signatures(self):
        lib = self.lib

        # x265_param_* functions
        lib.x265_param_alloc.restype = c_void_p
        lib.x265_param_alloc.argtypes = []

        lib.x265_param_free.restype = None
        lib.x265_param_free.argtypes = [c_void_p]

        lib.x265_param_default.restype = None
        lib.x265_param_default.argtypes = [c_void_p]

        lib.x265_param_default_preset.restype = c_int
        lib.x265_param_default_preset.argtypes = [c_void_p, c_char_p, c_char_p]

        lib.x265_param_apply_profile.restype = c_int
        lib.x265_param_apply_profile.argtypes = [c_void_p, c_char_p]

        lib.x265_param_parse.restype = c_int
        lib.x265_param_parse.argtypes = [c_void_p, c_char_p, c_char_p]

        # callback
        lib.x265_param_set_mv_callback.restype = None
        lib.x265_param_set_mv_callback.argtypes = [c_void_p, MV_CALLBACK_FUNC, c_void_p]

        # x265_encoder_* functions
        lib.x265_encoder_open_215.restype = c_void_p
        lib.x265_encoder_open_215.argtypes = [c_void_p]

        lib.x265_encoder_close.restype = None
        lib.x265_encoder_close.argtypes = [c_void_p]

        # x265_picture_* functions
        lib.x265_picture_alloc.restype = c_void_p
        lib.x265_picture_alloc.argtypes = []

        lib.x265_picture_free.restype = None
        lib.x265_picture_free.argtypes = [c_void_p]

        lib.x265_picture_init.restype = None
        lib.x265_picture_init.argtypes = [c_void_p, c_void_p]

        # x265_encoder_encode function
        lib.x265_encoder_encode.restype = c_int
        lib.x265_encoder_encode.argtypes = [
            c_void_p,
            POINTER(c_void_p),
            POINTER(c_uint32),
            c_void_p,
            c_void_p
        ]

    def open_encoder(self, width, height, fps, preset="medium", tune=None,
                     callback=None, user_data=None, **params):
        # Allocate and initialize parameters
        self.param = self.lib.x265_param_alloc()
        if not self.param:
            raise RuntimeError("Failed to allocate x265 parameters")

        preset_bytes = preset.encode('utf-8')
        tune_bytes = tune.encode('utf-8') if tune else None
        ret = self.lib.x265_param_default_preset(self.param, preset_bytes, tune_bytes)
        if ret != 0:
            raise RuntimeError(f"Failed to set preset: {preset}")

        self.lib.x265_param_parse(self.param, b"input-res", f"{width}x{height}".encode())
        self.lib.x265_param_parse(self.param, b"fps", str(fps).encode())

        self.lib.x265_param_parse(self.param, b"print-motion-info", b"2")

        if callback:
            callback_func = MV_CALLBACK_FUNC(callback)
            self.lib.x265_param_set_mv_callback(self.param, callback_func, user_data)
            self._callback_func = callback_func

        for key, value in params.items():
            key_bytes = key.encode('utf-8')
            value_bytes = str(value).encode('utf-8')
            self.lib.x265_param_parse(self.param, key_bytes, value_bytes)

        self.encoder = self.lib.x265_encoder_open_215(self.param)
        if not self.encoder:
            raise RuntimeError("Failed to open x265 encoder")

        return True

    def close_encoder(self):
        if self.encoder:
            self.lib.x265_encoder_close(self.encoder)
            self.encoder = None
        if self.param:
            self.lib.x265_param_free(self.param)
            self.param = None

    def __del__(self):
        self.close_encoder()


class X265NativeWrapper:
    def __init__(self, device='cuda', lib_path=None):
        self.device = device
        self.lib_path = lib_path

    def compute_flow(self, frames, ref_frame_idx_list, **kwargs):
        # Parse parameters
        size_str = kwargs.get('size', f"{frames[0].shape[1]}x{frames[0].shape[0]}")
        width, height = map(int, size_str.split('x'))
        fps = kwargs.get('fps', 30)
        preset = kwargs.get('preset', 'medium')
        stage = kwargs.get('stage', 'lookahead')
        if stage not in ['lookahead', 'encode']:
            raise ValueError(f"Invalid stage: {stage}. Must be 'lookahead' or 'encode'")

        num_frames = len(frames)

        forward_flows = torch.zeros((num_frames, 2, height, width), device=self.device, dtype=torch.float32)
        backward_flows = torch.zeros((num_frames, 2, height, width), device=self.device, dtype=torch.float32)

        # loop frame pair
        for idx in range(num_frames):
            ref_idx_list = ref_frame_idx_list[idx] if isinstance(ref_frame_idx_list[idx], list) else [ref_frame_idx_list[idx]]

            for ref_idx in ref_idx_list:
                if ref_idx < 0 or ref_idx >= num_frames:
                    continue

                # Encode frame pair: [ref_frame, current_frame]
                # This produces motion from ref_frame -> current_frame
                flow = self._encode_frame_pair(
                    frames[ref_idx], frames[idx],
                    width, height, fps, preset, stage
                )

                flow_tensor = torch.from_numpy(flow).to(self.device)

                if ref_idx < idx:
                    # ref_idx -> idx: this is backward flow for frame idx
                    backward_flows[idx] = flow_tensor
                elif ref_idx > idx:
                    # ref_idx -> idx: this is forward flow for frame idx
                    forward_flows[idx] = flow_tensor

        return [forward_flows, backward_flows], None

    def _encode_frame_pair(self, frame0, frame1, width, height, fps, preset, stage='lookahead'):
        if frame0.shape[:2] != (height, width):
            frame0 = cv2.resize(frame0, (width, height))
        if frame1.shape[:2] != (height, width):
            frame1 = cv2.resize(frame1, (width, height))

        # Convert BGR to YUV I420
        yuv0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2YUV_I420)
        yuv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2YUV_I420)

        collector = MVCollector(width, height)

        encoder = X265NativeEncoder(self.lib_path)
        encoder.open_encoder(
            width, height, fps,
            preset=preset,
            tune='zerolatency',
            callback=collector.callback,
            frames=2
        )

        try:
            self._encode_yuv_frame(encoder, yuv0, width, height, 0)
            self._encode_yuv_frame(encoder, yuv1, width, height, 1)

            self._flush_encoder(encoder)
        finally:
            encoder.close_encoder()

        return collector.get_flow(1)

    def _encode_yuv_frame(self, encoder, yuv_data, width, height, poc):
        pic = encoder.lib.x265_picture_alloc()
        if not pic:
            raise RuntimeError("Failed to allocate x265_picture")

        try:
            encoder.lib.x265_picture_init(encoder.param, pic)

            pic_struct = x265_picture.from_address(pic)

            y_size = height * width
            uv_size = (height // 2) * (width // 2)

            yuv_data = np.ascontiguousarray(yuv_data)

            y_plane = yuv_data[:y_size]
            u_plane = yuv_data[y_size:y_size + uv_size]
            v_plane = yuv_data[y_size + uv_size:]

            pic_struct.planes[0] = y_plane.ctypes.data_as(c_void_p)
            pic_struct.planes[1] = u_plane.ctypes.data_as(c_void_p)
            pic_struct.planes[2] = v_plane.ctypes.data_as(c_void_p)
            pic_struct.stride[0] = width
            pic_struct.stride[1] = width // 2
            pic_struct.stride[2] = width // 2

            pic_struct.poc = poc

            pp_nal = c_void_p()
            pi_nal = c_uint32()
            ret = encoder.lib.x265_encoder_encode(
                encoder.encoder,
                ctypes.byref(pp_nal),
                ctypes.byref(pi_nal),
                pic,
                None  # pic_out
            )

            if ret < 0:
                raise RuntimeError(f"Encoding failed with code {ret}")

        finally:
            encoder.lib.x265_picture_free(pic)

    def _flush_encoder(self, encoder):
        pp_nal = c_void_p()
        pi_nal = c_uint32()

        while True:
            ret = encoder.lib.x265_encoder_encode(
                encoder.encoder,
                ctypes.byref(pp_nal),
                ctypes.byref(pi_nal),
                None,  # NULL picture means flush
                None
            )
            if ret <= 0:
                break

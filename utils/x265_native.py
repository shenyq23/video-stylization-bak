import os
import ctypes
import numpy as np
import cv2
import torch
from pathlib import Path
from ctypes import CDLL, Structure, POINTER
from ctypes import c_int, c_uint, c_uint8, c_uint32, c_uint64, c_longlong, c_float, c_void_p, c_char_p


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
    def __init__(self, width, height, enable_deltapoc=True):
        self.width = width
        self.height = height
        self.enable_deltapoc = enable_deltapoc

        # Pre-allocated buffers
        self.mvx_buffer = np.zeros((height, width), dtype=np.float32)
        self.mvy_buffer = np.zeros((height, width), dtype=np.float32)
        self.deltapoc_buffer = np.zeros((height, width), dtype=np.int32) if enable_deltapoc else None

        # Flow output array [2, H, W]
        self.flow = np.zeros((2, height, width), dtype=np.float32)

    def get_output_pointers(self):
        """Return C-compatible pointers to pre-allocated buffers for x265"""
        deltapoc_ptr = self.deltapoc_buffer.ctypes.data_as(POINTER(c_int)) if self.deltapoc_buffer is not None else None
        return (
            self.mvx_buffer.ctypes.data_as(POINTER(c_float)),
            self.mvy_buffer.ctypes.data_as(POINTER(c_float)),
            deltapoc_ptr
        )

    def get_flow(self, poc):
        """Return motion vectors (already filled by x265 via preallocate)"""
        self.flow[0] = self.mvx_buffer
        self.flow[1] = self.mvy_buffer
        return self.flow

    def reset(self):
        """Reset buffers for next frame"""
        self.mvx_buffer.fill(0)
        self.mvy_buffer.fill(0)
        if self.deltapoc_buffer is not None:
            self.deltapoc_buffer.fill(0)
        self.flow.fill(0)


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
            lib_path = project_root / "bin" / "libx265.so"
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

        # Pre-allocated output pointers
        lib.x265_param_set_output_ptrs.restype = None
        lib.x265_param_set_output_ptrs.argtypes = [
            c_void_p, POINTER(c_float), POINTER(c_float), POINTER(c_int), c_int, c_int
        ]

        # x265_encoder_* functions
        lib.x265_encoder_open_215.restype = c_void_p
        lib.x265_encoder_open_215.argtypes = [c_void_p]

        lib.x265_encoder_close.restype = None
        lib.x265_encoder_close.argtypes = [c_void_p]

        # x265_encoder_reset function
        lib.x265_encoder_reset.restype = c_int
        lib.x265_encoder_reset.argtypes = [c_void_p]

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

    def open_encoder(self, width, height, fps, preset="medium", tune=None, **params):
        # Allocate and initialize parameters
        self.param = self.lib.x265_param_alloc()
        if not self.param:
            raise RuntimeError("Failed to allocate x265 parameters")

        preset_bytes = preset.encode('utf-8')
        tune_bytes = tune.encode('utf-8') if tune else None
        ret = self.lib.x265_param_default_preset(self.param, preset_bytes, tune_bytes)
        if ret != 0:
            raise RuntimeError(f"Failed to set preset: {preset}")

        # basic params
        self.lib.x265_param_parse(self.param, b"input-res", f"{width}x{height}".encode())
        self.lib.x265_param_parse(self.param, b"fps", str(fps).encode())

        # pre-allocated buffers
        if hasattr(self, '_output_collector') and self._output_collector:
            mvx_ptr, mvy_ptr, deltapoc_ptr = self._output_collector.get_output_pointers()
            self.lib.x265_param_set_output_ptrs(
                self.param, mvx_ptr, mvy_ptr, deltapoc_ptr, width, height
            )
            stage = params.get('stage', 'unknown')
            print(f"  Zero-IO preallocated output: ENABLED (stage={stage})")
        else:
            print(f"  Warning: No output collector set, motion vectors will not be captured!")

        for key, value in params.items():
            if key == 'stage':
                continue
            key_bytes = key.encode('utf-8')
            value_bytes = str(value).encode('utf-8')
            self.lib.x265_param_parse(self.param, key_bytes, value_bytes)

        self.encoder = self.lib.x265_encoder_open_215(self.param)
        if not self.encoder:
            raise RuntimeError("Failed to open x265 encoder")

        return True

    def reset_encoder(self):
        if not self.encoder:
            raise RuntimeError("Encoder not open")
        ret = self.lib.x265_encoder_reset(self.encoder)
        if ret < 0:
            raise RuntimeError(f"Failed to reset encoder, return code: {ret}")
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

        # cached encoder
        self._encoder = None
        self._encoder_config = None
        self._collector = None

    def _get_or_create_encoder(self, width, height, fps, preset, stage, **extra_params):
        config = (width, height, fps, preset, stage, tuple(sorted(extra_params.items())))

        # reset if config changed
        if self._encoder is not None and self._encoder_config == config:
            self._encoder.reset_encoder()
            self._collector.reset()
            return self._encoder, self._collector

        # close old if exists
        if self._encoder is not None:
            self._encoder.close_encoder()

        # create new
        self._collector = MVCollector(width, height, enable_deltapoc=True)
        self._encoder = X265NativeEncoder(self.lib_path)
        self._encoder._output_collector = self._collector

        x265_params = {
            'stage': stage,
            'frames': 2,
        }

        if stage == 'lookahead':
            x265_params['print-motion-info'] = 1  # lookahead_flag=True
            x265_params['skip-lookahead-intra'] = 1
            x265_params['skip-lookahead-slicetype'] = 1
            x265_params['lookahead-threads'] = 8
            x265_params['lookahead-slices'] = 8
            x265_params['motion-only'] = 1
        else:  # encode
            x265_params['print-motion-info'] = 2  # encoding_flag=True

        enable_p_intra = extra_params.get('enable_p_intra', False)
        if not enable_p_intra:
            x265_params['no-p-intra'] = 1

        for key in ['ctu', 'crf']:
            if key in extra_params:
                x265_params[key] = extra_params[key]

        self._encoder.open_encoder(
            width, height, fps,
            preset=preset,
            tune=None,
            **x265_params
        )
        self._encoder_config = config

        return self._encoder, self._collector

    def close(self):
        if self._encoder is not None:
            self._encoder.close_encoder()
            self._encoder = None
            self._encoder_config = None
            self._collector = None

    def __del__(self):
        self.close()

    def compute_flow(self, frames, ref_frame_idx_list, **kwargs):
        import time

        # Parse parameters
        size_str = kwargs.get('size', f"{frames[0].shape[1]}x{frames[0].shape[0]}")
        width, height = map(int, size_str.split('x'))
        fps = kwargs.get('fps', 30)
        preset = kwargs.get('preset', 'medium')
        stage = kwargs.get('stage', 'lookahead')
        enable_profile = kwargs.get('profile', False)
        if stage not in ['lookahead', 'encode']:
            raise ValueError(f"Invalid stage: {stage}. Must be 'lookahead' or 'encode'")

        x265_params = {}
        for key in ['ctu', 'crf', 'enable_p_intra']:
            if key in kwargs:
                x265_params[key] = kwargs[key]

        num_frames = len(frames)

        # Initialize profiling data
        profile_data = None
        if enable_profile:
            profile_data = {
                'yuv_conversion': [],       # RGB→YUV conversion time
                'encoder_open': [],         # encoder open/reset time
                'struct_filling': [],       # x265_picture filling time
                'x265_encode_call': [],     # x265_encoder_encode() call time
                'encoder_flush': [],        # encoder flush time
                'flow_copy': [],            # flow np.copy time
                'tensor_conversion': [],    # numpy→torch + to(device) time
                'total_per_pair': []        # Total time per frame pair
            }

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
                t_pair_start = time.perf_counter() if enable_profile else None

                flow = self._encode_frame_pair(
                    frames[ref_idx], frames[idx],
                    width, height, fps, preset, stage,
                    profile_data=profile_data,
                    **x265_params
                )

                if enable_profile:
                    profile_data['total_per_pair'].append(time.perf_counter() - t_pair_start)

                # Tensor conversion timing
                t_tensor_start = time.perf_counter() if enable_profile else None
                flow_tensor = torch.from_numpy(flow).to(self.device)
                if enable_profile:
                    profile_data['tensor_conversion'].append(time.perf_counter() - t_tensor_start)

                if ref_idx < idx:
                    # ref_idx -> idx: this is backward flow for frame idx
                    backward_flows[idx] = flow_tensor
                elif ref_idx > idx:
                    # ref_idx -> idx: this is forward flow for frame idx
                    forward_flows[idx] = flow_tensor

        # Print profiling summary
        if enable_profile and profile_data['total_per_pair']:
            self._print_profile_summary(profile_data, stage, width, height)

        return [forward_flows, backward_flows], None

    def _print_profile_summary(self, profile_data, stage, width, height):
        print("\n" + "="*70)
        print("X265 Native Wrapper - Performance Profiling")
        print("="*70)
        print(f"Stage: {stage}")
        print(f"Resolution: {width}x{height}")
        print(f"Frame pairs processed: {len(profile_data['total_per_pair'])}")
        print("\nPer-frame-pair timing breakdown:")
        print(f"  1. YUV conversion:      {np.mean(profile_data['yuv_conversion'])*1000:7.2f}ms  (RGB→YUV I420)")
        print(f"  2. Encoder open/reset:  {np.mean(profile_data['encoder_open'])*1000:7.2f}ms  (first: open, rest: reset)")
        print(f"  3. Struct filling:      {np.mean(profile_data['struct_filling'])*1000:7.2f}ms  (x265_picture setup)")
        print(f"  4. X265 encode calls:   {np.mean(profile_data['x265_encode_call'])*1000:7.2f}ms  (2x x265_encoder_encode)")
        print(f"  5. Encoder flush:       {np.mean(profile_data['encoder_flush'])*1000:7.2f}ms  (x265_encoder_encode NULL)")
        print(f"  6. Flow copy:           {np.mean(profile_data['flow_copy'])*1000:7.2f}ms  (np.copy)")
        print(f"  7. Tensor conversion:   {np.mean(profile_data['tensor_conversion'])*1000:7.2f}ms  (numpy→torch + to(device))")
        print(f"  " + "-"*60)
        print(f"  Total per pair:         {np.mean(profile_data['total_per_pair'])*1000:7.2f}ms  ± {np.std(profile_data['total_per_pair'])*1000:.2f}ms")

        print(f"\nPerformance analysis:")
        total_mean = np.mean(profile_data['total_per_pair'])
        encode_time = np.mean(profile_data['x265_encode_call']) + np.mean(profile_data['encoder_flush'])
        encoder_open_reset = np.mean(profile_data['encoder_open'])
        x265_total = encode_time + encoder_open_reset
        overhead = (np.mean(profile_data['yuv_conversion']) +
                   np.mean(profile_data['struct_filling']) +
                   np.mean(profile_data['flow_copy']) +
                   np.mean(profile_data['tensor_conversion']))

        print(f"  X265 encoding (encode+flush):   {encode_time*1000:7.2f}ms ({encode_time/total_mean*100:.1f}%)")
        print(f"  Encoder open/reset:             {encoder_open_reset*1000:7.2f}ms ({encoder_open_reset/total_mean*100:.1f}%)")
        print(f"  Total X265 time:                {x265_total*1000:7.2f}ms ({x265_total/total_mean*100:.1f}%)")
        print(f"  Python/ctypes overhead:         {overhead*1000:7.2f}ms ({overhead/total_mean*100:.1f}%)")
        print("="*70 + "\n")

    def _encode_frame_pair(self, frame0, frame1, width, height, fps, preset, stage='lookahead', profile_data=None, **x265_params):
        import time

        # Resize frames if needed
        if frame0.shape[:2] != (height, width):
            frame0 = cv2.resize(frame0, (width, height))
        if frame1.shape[:2] != (height, width):
            frame1 = cv2.resize(frame1, (width, height))

        # 1. Convert BGR to YUV I420
        t_yuv_start = time.perf_counter() if profile_data is not None else None
        yuv0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2YUV_I420)
        yuv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2YUV_I420)
        if profile_data is not None:
            profile_data['yuv_conversion'].append(time.perf_counter() - t_yuv_start)

        # 2. Get or create encoder (reuse with reset)
        t_open_start = time.perf_counter() if profile_data is not None else None
        encoder, collector = self._get_or_create_encoder(width, height, fps, preset, stage, **x265_params)
        if profile_data is not None:
            profile_data['encoder_open'].append(time.perf_counter() - t_open_start)

        # 3. Encode both frames
        t_encode_start = time.perf_counter() if profile_data is not None else None
        self._encode_yuv_frame(encoder, yuv0, width, height, 0, profile_data)
        self._encode_yuv_frame(encoder, yuv1, width, height, 1, profile_data)
        if profile_data is not None:
            profile_data['x265_encode_call'].append(time.perf_counter() - t_encode_start)

        # 4. Flush encoder
        t_flush_start = time.perf_counter() if profile_data is not None else None
        self._flush_encoder(encoder)

        if profile_data is not None:
            profile_data['encoder_flush'].append(time.perf_counter() - t_flush_start)

        # 6. Get flow (copy)
        t_copy_start = time.perf_counter() if profile_data is not None else None
        flow = collector.get_flow(1)
        if profile_data is not None:
            profile_data['flow_copy'].append(time.perf_counter() - t_copy_start)

        return flow

    def _encode_yuv_frame(self, encoder, yuv_data, width, height, poc, profile_data=None):
        import time

        # 3a. Allocate and fill x265_picture struct
        t_struct_start = time.perf_counter() if profile_data is not None else None

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

            if profile_data is not None:
                profile_data['struct_filling'].append(time.perf_counter() - t_struct_start)

            # 3b. Call x265_encoder_encode
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

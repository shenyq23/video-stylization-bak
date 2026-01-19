import os
import ctypes
import numpy as np
import cv2
import torch
from pathlib import Path
from ctypes import CDLL, CFUNCTYPE, Structure, POINTER
from ctypes import c_int, c_uint, c_uint8, c_uint32, c_uint64, c_longlong, c_float, c_double, c_void_p, c_char_p


# Per-block callback
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

# Per-frame callback
MV_FRAME_CALLBACK_FUNC = CFUNCTYPE(
    None,
    c_int,              # poc
    c_int,              # width
    c_int,              # height
    c_int,              # block_size
    POINTER(c_float),   # mvx_array
    POINTER(c_float),   # mvy_array
    POINTER(c_int),     # deltapoc_array
    c_void_p            # user_data
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

        # PRE-ALLOCATED BUFFERS
        self.mvx_buffer = np.zeros((height, width), dtype=np.float32)
        self.mvy_buffer = np.zeros((height, width), dtype=np.float32)
        # self.deltapoc_buffer = np.zeros((height, width), dtype=np.int32)

        # callback to be deleted
        self.flow = np.zeros((2, height, width), dtype=np.float32)
        self.callback_count = 0

    def get_output_pointers(self):
        """Return C-compatible pointers to pre-allocated buffers for x265"""
        # NULL for deltapoc
        return (
            self.mvx_buffer.ctypes.data_as(POINTER(c_float)),
            self.mvy_buffer.ctypes.data_as(POINTER(c_float)),
            None
        )

    def block_callback(self, poc, x, y, w, h, mvx, mvy, deltapoc, user_data):
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

    def frame_callback(self, poc, width, height, block_size, mvx_ptr, mvy_ptr, deltapoc_ptr, user_data):
        if poc == 0:
            return

        if block_size == 1:
            # x265 already upsampled to full-resolution
            mvx_full = np.ctypeslib.as_array(mvx_ptr, shape=(height, width))
            mvy_full = np.ctypeslib.as_array(mvy_ptr, shape=(height, width))

            self.flow[0] = mvx_full
            self.flow[1] = mvy_full
            self.callback_count = height * width

        else:
            blocks_x = (width + block_size - 1) // block_size
            blocks_y = (height + block_size - 1) // block_size
            num_blocks = blocks_x * blocks_y

            # numpy view from C pointers
            mvx_arr = np.ctypeslib.as_array(mvx_ptr, shape=(num_blocks,))
            mvy_arr = np.ctypeslib.as_array(mvy_ptr, shape=(num_blocks,))

            mvx_2d = mvx_arr.reshape(blocks_y, blocks_x)
            mvy_2d = mvy_arr.reshape(blocks_y, blocks_x)

            mvx_full = np.repeat(np.repeat(mvx_2d, block_size, axis=0), block_size, axis=1)
            mvy_full = np.repeat(np.repeat(mvy_2d, block_size, axis=0), block_size, axis=1)

            # crop
            self.flow[0] = mvx_full[:self.height, :self.width]
            self.flow[1] = mvy_full[:self.height, :self.width]
            self.callback_count = num_blocks

    callback = block_callback

    def get_flow(self, poc):
        # already filled, return directly
        self.flow[0] = self.mvx_buffer
        self.flow[1] = self.mvy_buffer
        return self.flow

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
            # lib_path = project_root / "deps" / "x265" / "build" / "linux-dynamic" / "libx265.so"
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

        # per-block callback
        lib.x265_param_set_mv_callback.restype = None
        lib.x265_param_set_mv_callback.argtypes = [c_void_p, MV_CALLBACK_FUNC, c_void_p]

        # per-frame callback
        lib.x265_param_set_mv_frame_callback.restype = None
        lib.x265_param_set_mv_frame_callback.argtypes = [c_void_p, MV_FRAME_CALLBACK_FUNC, c_void_p]

        # pre-allocated output pointers
        lib.x265_param_set_output_ptrs.restype = None
        lib.x265_param_set_output_ptrs.argtypes = [
            c_void_p, POINTER(c_float), POINTER(c_float), POINTER(c_int), c_int, c_int
        ]

        # x265_encoder_* functions
        lib.x265_encoder_open_215.restype = c_void_p
        lib.x265_encoder_open_215.argtypes = [c_void_p]

        lib.x265_encoder_close.restype = None
        lib.x265_encoder_close.argtypes = [c_void_p]

        lib.x265_profiling_lookahead.restype = None
        lib.x265_profiling_lookahead.argtypes = [c_void_p]

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

    def open_encoder(self, width, height, fps, preset="medium", tune=None,
                     callback=None, frame_callback=None, user_data=None,
                     stage='lookahead', **params):
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

        print_motion_info = b"1" if stage == 'lookahead' else b"2"
        self.lib.x265_param_parse(self.param, b"print-motion-info", print_motion_info)

        if stage == 'lookahead':
            self.lib.x265_param_parse(self.param, b"skip-lookahead-intra", b"1")
            self.lib.x265_param_parse(self.param, b"skip-lookahead-slicetype", b"1")
            # self.lib.x265_param_parse(self.param, b"profile-lookahead", b"1")
            # self.lib.x265_param_parse(self.param, b"log-level", b"info")
            self.lib.x265_param_parse(self.param, b"lookahead-threads", b"8")
            self.lib.x265_param_parse(self.param, b"lookahead-slices", b"8")
            self.lib.x265_param_parse(self.param, b"motion-only", b"1")

        use_preallocated = False
        if hasattr(self, '_output_collector') and self._output_collector:
            try:
                mvx_ptr, mvy_ptr, deltapoc_ptr = self._output_collector.get_output_pointers()
                self.lib.x265_param_set_output_ptrs(
                    self.param, mvx_ptr, mvy_ptr, deltapoc_ptr, width, height
                )
                print(f"  Pre-allocated output: ENABLED (zero-copy, no callback!)")
                use_preallocated = True
            except AttributeError:
                print(f"  Falling back to callback mode...")
            except Exception as e:
                print(f"  Falling back to callback mode...")

        # set callback if not using pre-allocated
        if not use_preallocated:
            if frame_callback:
                cb_func = MV_FRAME_CALLBACK_FUNC(frame_callback)
                self.lib.x265_param_set_mv_frame_callback(self.param, cb_func, user_data)
                self._callback_func = cb_func
            elif callback:
                cb_func = MV_CALLBACK_FUNC(callback)
                self.lib.x265_param_set_mv_callback(self.param, cb_func, user_data)
                self._callback_func = cb_func

        for key, value in params.items():
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

    def _get_or_create_encoder(self, width, height, fps, preset, stage):
        config = (width, height, fps, preset, stage)

        # reset if config changed
        if self._encoder is not None and self._encoder_config == config:
            self._encoder.reset_encoder()
            self._collector.reset()
            return self._encoder, self._collector

        # close old if exists
        if self._encoder is not None:
            self._encoder.close_encoder()

        # create new
        self._collector = MVCollector(width, height)
        self._encoder = X265NativeEncoder(self.lib_path)
        self._encoder._output_collector = self._collector
        self._encoder.open_encoder(
            width, height, fps,
            preset=preset,
            tune='zerolatency',
            frame_callback=self._collector.frame_callback,
            stage=stage,
            frames=2
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
                    profile_data=profile_data
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

    def _encode_frame_pair(self, frame0, frame1, width, height, fps, preset, stage='lookahead', profile_data=None):
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
        encoder, collector = self._get_or_create_encoder(width, height, fps, preset, stage)
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

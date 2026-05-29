"""
gpu.py — per-module GPU profiler for StreamDiffusionV2 / MotionFlow.

Runs in single-stream serial mode (cuda.synchronize around every measurement),
so module timings have no producer/consumer overlap. End-to-end FPS must be
measured separately by inference.py; per-module numbers are then scaled to
match the parallel total (see methodology doc).

Two modes:
  --flow_model none     -> pure-baseline (full computation, no warp, no sparse)
  --flow_model x265     -> MotionFlow path (sparse VAE + sparse DiT + flow warp)

Outputs the same three coarse modules (VAE Encode / DiT Inference / VAE Decode)
plus, if instrumented inside the model, a detailed sub-module breakdown via the
`profile_timings` dict on the generator/encoder.
"""
import sys
sys.path.append("../StreamDiffusionV2")
sys.path.append("../")
sys.path.append("../deps/gmflow")

import argparse
import json
import logging
import os
import threading
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from einops import rearrange
from omegaconf import OmegaConf
try:
    from pynvml import (
        NVMLError, nvmlInit, nvmlShutdown,
        nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates,
    )
    _NVML_AVAILABLE = True
except ImportError:
    _NVML_AVAILABLE = False
    class NVMLError(Exception): pass
    def nvmlInit(): pass
    def nvmlShutdown(): pass
    def nvmlDeviceGetHandleByIndex(_i): return None
    def nvmlDeviceGetUtilizationRates(_h):
        class _U: gpu = 0
        return _U()

from causvid.data import TextDataset
from causvid.models.wan.causal_stream_inference import CausalStreamInferencePipeline
from utils.optical_wrapper import (
    GMFlowWrapper, RAFTFlowWrapper, X265MVWrapper, OcclusionComputation,
)
from utils.vae_utils.mask_utils import (
    build_gather_block_masks, dilate_mask, downsample_mask,
)
from deps.sige3d.torch_kernels.backend import set_kernel_backend

# Reuse the inference.py flow-calculator wrapper so flow/occlusion logic
# stays identical to the production pipeline.
from streamv2v.inference import OpticalFlowCalculator, build_taehv_vae

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ----------------------------------------------------------------------------
# Helpers (unchanged from prior gpu.py)
# ----------------------------------------------------------------------------

def load_mp4_as_tensor(video_path, max_frames=None, resize_hw=None, normalize=True):
    assert os.path.exists(video_path), f"Video file not found: {video_path}"
    video, _, _ = torchvision.io.read_video(video_path, output_format="TCHW", pts_unit="sec")
    if max_frames is not None:
        video = video[:max_frames]
    video = rearrange(video, "t c h w -> c t h w")
    if resize_hw is not None:
        c, t, h0, w0 = video.shape
        video = torch.stack([TF.resize(video[:, i], resize_hw, antialias=True) for i in range(t)], dim=1)
    if video.dtype != torch.float32:
        video = video.float()
    if normalize:
        video = video / 127.5 - 1.0
    return video


class GPUProfiler:
    """Background NVML sampler so we can report GPU util alongside latency."""
    def __init__(self, device_id=0):
        self.device_id = device_id
        self._stop_event = threading.Event()
        self._is_profiling_event = threading.Event()
        self.utilizations = []
        nvmlInit()
        self.handle = nvmlDeviceGetHandleByIndex(self.device_id)
        logging.info(f"NVML initialized for GPU {self.device_id}.")
        self.monitor_thread = threading.Thread(target=self._monitor_gpu, daemon=True)

    def _monitor_gpu(self):
        while not self._stop_event.is_set():
            if self._is_profiling_event.is_set():
                try:
                    util = nvmlDeviceGetUtilizationRates(self.handle)
                    self.utilizations.append(util.gpu)
                except NVMLError:
                    pass
            time.sleep(0.01)

    def start(self): self.monitor_thread.start()
    def start_profile(self):
        self.utilizations = []
        self._is_profiling_event.set()
    def stop_profile(self): self._is_profiling_event.clear()
    def get_summary(self):
        if not self.utilizations:
            return {"avg": 0, "max": 0, "min": 0, "samples": 0}
        return {"avg": sum(self.utilizations) / len(self.utilizations),
                "max": max(self.utilizations), "min": min(self.utilizations),
                "samples": len(self.utilizations)}
    def shutdown(self):
        self._stop_event.set()
        self.monitor_thread.join()
        try:
            nvmlShutdown()
        except NVMLError:
            pass


def profile_module_with_gpu_util(name, func, gpu_profiler, num_iterations=20, warmup=5):
    times = []
    gpu_utils_avg = []
    gpu_utils_max = []
    logging.info(f"[{name}] warmup...")
    for _ in range(warmup):
        func()
        torch.cuda.synchronize()
    logging.info(f"[{name}] profiling {num_iterations} iters...")
    for i in range(num_iterations):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        gpu_profiler.start_profile()
        torch.cuda.synchronize()
        start_event.record()
        func()
        end_event.record()
        torch.cuda.synchronize()
        gpu_profiler.stop_profile()
        elapsed_ms = start_event.elapsed_time(end_event)
        times.append(elapsed_ms)
        summary = gpu_profiler.get_summary()
        gpu_utils_avg.append(summary["avg"])
        gpu_utils_max.append(summary["max"])
        logging.info(f"[{name}] iter {i+1}/{num_iterations}: {elapsed_ms:.2f} ms | avg util {summary['avg']:.1f}% | peak {summary['max']:.1f}%")
    avg_time = sum(times) / len(times)
    avg_gpu_util = sum(gpu_utils_avg) / len(gpu_utils_avg)
    max_gpu_util = max(gpu_utils_max) if gpu_utils_max else 0
    print("\n" + "=" * 60)
    print(f"Module [{name}] summary")
    print("=" * 60)
    print(f"Avg per-call latency over {num_iterations} iters: {avg_time:.4f} ms")
    print(f"Avg GPU util:  {avg_gpu_util:.2f}%")
    print(f"Peak GPU util: {max_gpu_util:.2f}%")
    print("=" * 60 + "\n")
    return avg_time, avg_gpu_util, max_gpu_util


def profile_module_detailed(name, func, num_iterations=20, warmup=5):
    """Like profile_module_with_gpu_util but threads a `timings` dict through
    `func` so it can return sub-module breakdowns."""
    total_times = []
    detailed_timings_accumulator = defaultdict(list)

    logging.info(f"[{name}] (detailed) warmup...")
    for _ in range(warmup):
        timings = defaultdict(float)
        func(timings)
        torch.cuda.synchronize()

    logging.info(f"[{name}] (detailed) profiling...")
    for i in range(num_iterations):
        timings = defaultdict(float)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_event.record()
        func(timings)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        total_times.append(elapsed_ms)
        for key, value in timings.items():
            detailed_timings_accumulator[key].append(value)

    avg_total_time = sum(total_times) / len(total_times)
    avg_detailed_times = {k: sum(v) / len(v) for k, v in detailed_timings_accumulator.items()}

    print("\n" + "=" * 60)
    print(f"Module [{name}] detailed breakdown")
    print("=" * 60)
    print(f"{'Avg total':<25}: {avg_total_time:.4f} ms")
    print("-" * 60)
    for key, value in sorted(avg_detailed_times.items()):
        pct = (value / avg_total_time) * 100 if avg_total_time > 0 else 0
        print(f"{key:<25}: {value:.4f} ms ({pct:.1f}%)")
    manual_sum = sum(avg_detailed_times.values())
    print("-" * 60)
    print(f"{'Sub-module sum':<25}: {manual_sum:.4f} ms")
    print("=" * 60 + "\n")
    return avg_total_time, avg_detailed_times


# ----------------------------------------------------------------------------
# MotionFlow flow-data construction
# ----------------------------------------------------------------------------

def build_flow_data(flow_calculator, input_video_original, ref_frame_idx, cur_frame_idx,
                    latent_shape_hw, occlusion_method, top_k_percentage, mask_dilate, min_res):
    """Replicate the producer-side flow / mask / mask_half construction from
    inference.py, but as a single synchronous call. Returns
    (masks_enc, bwd_flow_for_vae, flow_data_for_dit).
    """
    device = input_video_original.device
    ref_frame = input_video_original[:, :, ref_frame_idx].to(device, torch.float32)
    cur_frame = input_video_original[:, :, cur_frame_idx].to(device, torch.float32)

    bwd_flow, bwd_occ = flow_calculator.calculate_flow(ref_frame, cur_frame)

    # VAE encoder mask
    if occlusion_method == "gather_block":
        masks_enc = build_gather_block_masks(
            bwd_occ.squeeze(0).squeeze(0),
            top_k_percentage=top_k_percentage[0],
            adaptive=flow_calculator.adaptive_sparsity,
            cdf_coverage=flow_calculator.cdf_coverage,
            r_min=flow_calculator.r_min,
            r_max=flow_calculator.r_max,
        )
    else:
        mask_enc = dilate_mask(bwd_occ.squeeze(0).squeeze(0), int(mask_dilate))
        masks_enc = downsample_mask(mask_enc, min_res=tuple(min_res), dilation=int(mask_dilate))

    bwd_flow_for_vae = bwd_flow.squeeze(0).permute(1, 2, 0).contiguous()

    # DiT-side flow data (downsampled to latent resolution)
    latent_h, latent_w = latent_shape_hw
    downsampled_flow = F.interpolate(bwd_flow, size=(latent_h, latent_w),
                                     mode='bilinear', align_corners=False)
    downsampled_flow *= (float(latent_h) / bwd_flow.shape[2])
    downsampled_occ = F.interpolate(bwd_occ, size=(latent_h, latent_w),
                                    mode='bilinear', align_corners=False)
    latent_binary_mask = flow_calculator.compute_binary_occlusion_mask(downsampled_occ)
    downsampled_occ_half = F.interpolate(bwd_occ, size=(latent_h // 2, latent_w // 2),
                                         mode='bilinear', align_corners=False)
    latent_binary_mask_half = flow_calculator.compute_binary_occlusion_mask(downsampled_occ_half)
    flow_data = (downsampled_flow, latent_binary_mask, latent_binary_mask_half)
    return masks_enc, bwd_flow_for_vae, flow_data


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    # Existing args
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--checkpoint_folder", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--prompt_file_path", required=True)
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--noise_scale", type=float, default=0.700)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--step", type=int, default=4)
    parser.add_argument("--model_type", type=str, default="T2V-1.3B")
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--force_sparse", action="store_true",
                        help="Override pipeline.generator.count to keep use_sparse=True during measurement "
                             "(bypasses the 1/5 dense refresh). Use only for pure-sparse breakdown.")

    # MotionFlow / flow path
    parser.add_argument("--flow_model", type=str, default="none",
                        choices=["gmflow", "raft", "x265", "none"])
    parser.add_argument("--x265_params", type=str, default='{"stage":"encode","quiet":true}')
    parser.add_argument("--top_k_percentage", type=float, nargs="+", default=[0.1, 0.1])
    parser.add_argument("--occlusion_method", type=str, default="gather_block",
                        choices=["exact", "gather_block", "quantile"],
                        help="gather_block keeps the raw float occ map for per-layer resolve "
                             "(production stylization default). exact/quantile go through dilate_mask "
                             "which requires a binary occ map.")
    parser.add_argument("--mask_dilate", type=int, default=1)
    parser.add_argument("--min_res", type=int, nargs=2, default=[60, 104])
    parser.add_argument("--use_cached_text_embedding", action="store_true",
                        default=True,
                        help="Load cached prompt embedding from text_cache/ instead of running the T5 "
                             "encoder. Default True so the heavy T5 weights are never required.")
    parser.add_argument("--no_cached_text_embedding", dest="use_cached_text_embedding",
                        action="store_false",
                        help="Force re-encoding via T5; needs models_t5_*.pth on disk.")
    # is_nocache must be True for the dense baseline (no mask/flow inputs) and
    # False for the MotionFlow sparse path. Default is decided after parsing
    # based on --flow_model. Use --is_nocache / --no_is_nocache to override.
    parser.add_argument("--is_nocache", dest="is_nocache", action="store_true", default=None)
    parser.add_argument("--no_is_nocache", dest="is_nocache", action="store_false")
    parser.add_argument("--vae_type", type=str, default="wanvae",
                        choices=["wanvae", "taehv", "wan-taehv"])
    # Adaptive sparsity (paper §3.4): CDF-based dynamic ratio.
    parser.add_argument("--adaptive_sparsity", action="store_true",
                        help="Override fixed top_k_percentage with CDF-based per-chunk ratio.")
    parser.add_argument("--cdf_coverage", type=float, default=0.7,
                        help="Target motion coverage rho (paper default 0.7).")
    parser.add_argument("--r_min", type=float, default=0.08,
                        help="Lower clamp on adaptive ratio (paper default 0.08).")
    parser.add_argument("--r_max", type=float, default=0.30,
                        help="Upper clamp on adaptive ratio (paper default 0.30).")
    parser.add_argument("--sige_backend", type=str, default="cuda",
                        help="set_kernel_backend value for sige3d (cuda or pytorch).")
    args = parser.parse_args()
    args.x265_params = json.loads(args.x265_params)

    # Decide is_nocache default based on flow_model: baseline=True, MotionFlow=False.
    if args.is_nocache is None:
        args.is_nocache = (args.flow_model.lower() == "none")
        logging.info(f"is_nocache default resolved to {args.is_nocache} "
                     f"(flow_model={args.flow_model})")

    torch.set_grad_enabled(False)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16  # matches paper: Wan 1.3B + float16
    device_id = int(args.device.split(":")[-1]) if "cuda" in args.device else 0

    set_kernel_backend(args.sige_backend)

    gpu_profiler = GPUProfiler(device_id=device_id)
    gpu_profiler.start()

    try:
        # --- 1. Build pipeline ---
        config = OmegaConf.load(args.config_path)
        # Older configs (e.g. wan_causal_dmd_v2v_causvid.yaml) predate fields
        # the pipeline now requires. Fill in conservative defaults so the
        # benchmark works across all yaml variants without manual edits.
        defaults = {"adapt_sink_threshold": 0.2}
        for k, v in defaults.items():
            if k not in config:
                config[k] = v
        config = OmegaConf.merge(config, OmegaConf.create({
            k: v for k, v in vars(args).items() if not isinstance(v, dict)
        }))
        denoising_map = {1: [700, 0], 2: [700, 500, 0], 3: [700, 600, 400, 0]}
        config.denoising_step_list = denoising_map.get(args.step, [700, 600, 500, 400, 0])

        logging.info("Initializing CausalStreamInferencePipeline...")
        pipeline = CausalStreamInferencePipeline(
            config, device=str(device), text_encoder_on_cpu=True,
            use_cached_text_embedding=args.use_cached_text_embedding,
        )
        pipeline.to(device=str(device), dtype=dtype)

        ckpt_path = os.path.join(args.checkpoint_folder, "model.pt")
        logging.info(f"Loading weights from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get('generator') or ckpt.get('generator_ema') or ckpt.get('state_dict') or ckpt
        try:
            pipeline.generator.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            logging.warning(f"strict=True load failed ({e}); retrying strict=False")
            pipeline.generator.load_state_dict(state_dict, strict=False)
        logging.info("Model loaded.")

        # --- VAE decoder swap (paper uses distilled TAEHV decoder) ---
        # wan-taehv: keep WanVAE encoder (so sparse encode + cache warp still
        # measured), swap only the decoder to TAEHV.
        # taehv: encoder + decoder both TAEHV (rarely useful — sparse encode
        # path is what we're profiling).
        vae_decoder = pipeline.vae
        if args.vae_type in ("taehv", "wan-taehv"):
            taehv = build_taehv_vae(device, dtype=dtype)
            vae_decoder = taehv
            if args.vae_type == "taehv":
                pipeline.vae = taehv
            logging.info(f"VAE decoder swapped to TAEHV (vae_type={args.vae_type}).")

        # --- 2. Load prompt + video ---
        dataset = TextDataset(args.prompt_file_path)
        prompts = [dataset[0]]

        ALIGNMENT = 16  # VAE 8x downsample x DiT 2x patchify = 16; keeps 720 exact
        new_height = (args.height // ALIGNMENT) * ALIGNMENT
        new_width = (args.width // ALIGNMENT) * ALIGNMENT
        if (new_height, new_width) != (args.height, args.width):
            logging.warning(f"Resolution adjusted for alignment: {args.height}x{args.width} -> {new_height}x{new_width}")

        logging.info(f"Loading video {args.video_path}")
        # Profiling only needs cold-start (5) + 6 warm-up chunks (24) + 1 inference
        # chunk. Cap loaded frames so the full clip doesn't sit on GPU — at 720p the
        # whole 254-frame tensor is ~1.4 GB and causes OOM during VAE encode.
        PROFILE_MAX_FRAMES = 48
        input_video_original = load_mp4_as_tensor(
            args.video_path, max_frames=PROFILE_MAX_FRAMES, resize_hw=(new_height, new_width)).unsqueeze(0)
        input_video_original = input_video_original.to(device=device, dtype=dtype)
        logging.info(f"Input video shape: {input_video_original.shape}")

        # --- 3. Flow calculator (MotionFlow only) ---
        motionflow = args.flow_model.lower() != "none"
        flow_calculator = None
        if motionflow:
            logging.info(f"MotionFlow ON: flow_model={args.flow_model}, "
                         f"top_k={args.top_k_percentage}, occlusion={args.occlusion_method}")
            flow_calculator = OpticalFlowCalculator(
                flow_model_type=args.flow_model,
                device=device,
                x265_params=args.x265_params,
                occlusion_method=args.occlusion_method,
                top_k_percentage=tuple(args.top_k_percentage),
                adaptive_sparsity=args.adaptive_sparsity,
                cdf_coverage=args.cdf_coverage,
                r_min=args.r_min,
                r_max=args.r_max,
            )
            if args.adaptive_sparsity:
                logging.info(f"Adaptive sparsity ON: rho={args.cdf_coverage}, "
                             f"r in [{args.r_min}, {args.r_max}]")

        # --- 4. Cold-start prepare (mirror inference.py pattern) ---
        chunk_size = 4
        initial_frames_count = 5
        if input_video_original.shape[2] < initial_frames_count + chunk_size * 6:
            raise ValueError("Need at least 5 + 24 frames for sparse warm-up.")

        initial_video_chunk = input_video_original[:, :, 0:initial_frames_count]
        # cold-start VAE encode uses dense path (mask=None, flow=None, is_nocache=args.is_nocache)
        latents_for_prepare = pipeline.vae.stream_encode(initial_video_chunk, None, None, args.is_nocache)
        latents_for_prepare = latents_for_prepare.transpose(2, 1).contiguous()
        noise = torch.randn_like(latents_for_prepare)
        noisy_latents_for_prepare = noise * args.noise_scale + latents_for_prepare * (1 - args.noise_scale)

        prepare_start = 0
        prepare_end = pipeline.frame_seq_length * 2
        denoised_pred_for_prepare = pipeline.prepare(
            text_prompts=prompts, device=device, dtype=dtype,
            block_mode='input', noise=noisy_latents_for_prepare,
            current_start=prepare_start, current_end=prepare_end,
        )
        _ = vae_decoder.stream_decode_to_pixel(denoised_pred_for_prepare, None, None)
        torch.cuda.synchronize()
        logging.info("Cold start done — pipeline state initialized.")

        # --- 5. Warm-up the sparse path so flow_guidance_cache is populated and
        #        scatter caches reflect MotionFlow steady-state. We run several
        #        real hot-loop chunks with flow data before timing anything.
        ref_frame_idx = initial_frames_count - 1
        cur_frame_idx = ref_frame_idx + chunk_size
        latent_hw = latents_for_prepare.shape[-2:]
        current_start_local = prepare_end
        current_end_local = current_start_local + (chunk_size // 4) * pipeline.frame_seq_length

        sparse_warmup_chunks = 6 if motionflow else 1
        last_inference_chunk = None
        last_flow_data = None
        last_masks_enc = None
        last_bwd_flow_vae = None
        for k in range(sparse_warmup_chunks):
            start_idx = initial_frames_count + k * chunk_size
            end_idx = start_idx + chunk_size
            if end_idx > input_video_original.shape[2]:
                logging.warning(f"Video too short for warm-up chunk {k}; stopping warm-up.")
                break
            inp = input_video_original[:, :, start_idx:end_idx]
            if motionflow:
                masks_enc, bwd_flow_vae, flow_data = build_flow_data(
                    flow_calculator, input_video_original,
                    ref_frame_idx=cur_frame_idx if k > 0 else ref_frame_idx,
                    cur_frame_idx=end_idx - 1,
                    latent_shape_hw=latent_hw,
                    occlusion_method=args.occlusion_method,
                    top_k_percentage=args.top_k_percentage,
                    mask_dilate=args.mask_dilate, min_res=args.min_res,
                )
                latents = pipeline.vae.stream_encode(inp, masks_enc, bwd_flow_vae, args.is_nocache)
            else:
                masks_enc, bwd_flow_vae, flow_data = None, None, None
                latents = pipeline.vae.stream_encode(inp, None, None, args.is_nocache)

            latents = latents.transpose(2, 1).contiguous()
            noise = torch.randn_like(latents)
            noisy_latents = noise * args.noise_scale + latents * (1 - args.noise_scale)
            _ = pipeline.inference_stream(
                noise=noisy_latents,
                current_start=current_start_local,
                current_end=current_end_local,
                current_step=None,
                latent_flow_data=flow_data,
            )
            current_start_local = current_end_local
            current_end_local += (chunk_size // 4) * pipeline.frame_seq_length
            cur_frame_idx = end_idx - 1
            last_inference_chunk = inp
            last_flow_data = flow_data
            last_masks_enc = masks_enc
            last_bwd_flow_vae = bwd_flow_vae
        torch.cuda.synchronize()
        logging.info(f"Sparse warm-up done ({sparse_warmup_chunks} chunks). "
                     f"flow_guidance_cache populated={getattr(pipeline.generator, 'flow_guidance_cache', None) is not None}, "
                     f"generator.count={getattr(pipeline.generator, 'count', None)}")

        # --- 6. Prepare the inputs for the steady-state hot loop ---
        if last_inference_chunk is None:
            raise RuntimeError("Warm-up loop did not run; check video length.")
        inference_video_chunk = last_inference_chunk
        latents_for_inference = pipeline.vae.stream_encode(
            inference_video_chunk,
            last_masks_enc if motionflow else None,
            last_bwd_flow_vae if motionflow else None,
            args.is_nocache,
        ).transpose(2, 1).contiguous()
        noise = torch.randn_like(latents_for_inference)
        noisy_latents_for_inference = noise * args.noise_scale + latents_for_inference * (1 - args.noise_scale)

        inference_start = current_start_local
        inference_end = inference_start + (chunk_size // 4) * pipeline.frame_seq_length
        denoised_latents_from_dit = pipeline.inference_stream(
            noise=noisy_latents_for_inference,
            current_start=inference_start, current_end=inference_end,
            current_step=None, latent_flow_data=last_flow_data,
        )
        latents_for_decode = denoised_latents_from_dit[[-1]]
        torch.cuda.synchronize()
        logging.info(f"Steady-state inputs ready. latent_for_decode shape={latents_for_decode.shape}")

        # --- 7. Sparse-only override (for clean sub-module measurement) ---
        def _force_sparse_pre():
            # Bypass the periodic dense refresh in CausalWanModel by holding
            # generator.count off any multiple of 5.
            if args.force_sparse and motionflow:
                pipeline.generator.count = 1

        # --- 8. Coarse module timings (VAE encode / DiT / VAE decode) ---
        def vae_encode_op():
            _force_sparse_pre()
            pipeline.vae.stream_encode(
                inference_video_chunk,
                last_masks_enc if motionflow else None,
                last_bwd_flow_vae if motionflow else None,
                args.is_nocache,
            )

        def dit_inference_op():
            _force_sparse_pre()
            pipeline.inference_stream(
                noise=noisy_latents_for_inference,
                current_start=inference_start, current_end=inference_end,
                current_step=None, latent_flow_data=last_flow_data,
            )

        def vae_decode_op():
            _force_sparse_pre()
            vae_decoder.stream_decode_to_pixel(latents_for_decode, None, None)

        profile_module_with_gpu_util("VAE Encode", vae_encode_op, gpu_profiler,
                                     num_iterations=args.num_iterations, warmup=args.warmup)
        profile_module_with_gpu_util("DiT Inference", dit_inference_op, gpu_profiler,
                                     num_iterations=args.num_iterations, warmup=args.warmup)
        profile_module_with_gpu_util("VAE Decode", vae_decode_op, gpu_profiler,
                                     num_iterations=args.num_iterations, warmup=args.warmup)

        # --- 9. Detailed sub-module breakdown (only meaningful once
        #        instrumentation lands in vae.py / causal_model.py) ---
        def vae_encode_op_detailed(timings):
            _force_sparse_pre()
            pipeline.vae.model.profile_timings = timings
            pipeline.vae.stream_encode(
                inference_video_chunk,
                last_masks_enc if motionflow else None,
                last_bwd_flow_vae if motionflow else None,
                args.is_nocache,
            )
            pipeline.vae.model.profile_timings = None

        def dit_inference_op_detailed(timings):
            _force_sparse_pre()
            pipeline.generator.model.profile_timings = timings
            pipeline.inference_stream(
                noise=noisy_latents_for_inference,
                current_start=inference_start, current_end=inference_end,
                current_step=None, latent_flow_data=last_flow_data,
            )
            pipeline.generator.model.profile_timings = None

        def vae_decode_op_detailed(timings):
            _force_sparse_pre()
            # Only WanVAE decoder has profile_timings hooks; TAEHV is monolithic.
            if args.vae_type == "wanvae":
                pipeline.vae.model.profile_timings = timings
                pipeline.vae.stream_decode_to_pixel(latents_for_decode, None, None)
                pipeline.vae.model.profile_timings = None
            else:
                vae_decoder.stream_decode_to_pixel(latents_for_decode, None, None)

        profile_module_detailed("VAE Encode (Detailed)", vae_encode_op_detailed,
                                num_iterations=args.num_iterations, warmup=args.warmup)
        profile_module_detailed("DiT Inference (Detailed)", dit_inference_op_detailed,
                                num_iterations=args.num_iterations, warmup=args.warmup)
        profile_module_detailed("VAE Decode (Detailed)", vae_decode_op_detailed,
                                num_iterations=args.num_iterations, warmup=args.warmup)

    finally:
        gpu_profiler.shutdown()


if __name__ == "__main__":
    main()

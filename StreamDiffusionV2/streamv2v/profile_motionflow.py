#!/usr/bin/env python3
"""Serial per-stage / per-submodule profiler for the MotionFlow pipeline.

Reproduces the two panels of the MotionFlow latency figure:

  * Figure (a) -- the MotionFlow bar: per-frame VAE-encode / Denoise / VAE-decode.
  * Figure (b) -- the two breakdown bars:
        VAE bar : encoder resolution stages (VAE 512/256/128/64) + Cache Warp.
        DiT bar : Self Attention / Cross Attention / Linear (FFN) / RoPE / Warp.

Why a separate script (and not inference.py)?
  inference.py runs a *parallel* producer/consumer pipeline on two CUDA streams,
  so VAE-encode and DiT kernels overlap and per-stage wall-clock time cannot be
  separated. This script drives the exact same pipeline (wan-taehv backend,
  gather_block occlusion, sparse encode + flow warp) but **serially on the
  default stream**, timing each stage in isolation with CUDA events. The
  per-submodule numbers come from the opt-in causvid.profiling.PROFILER, whose
  instrumentation lives in vae.py / causal_model.py and is a no-op unless enabled.

Run it with the same flags as the real inference command, e.g.:

  CUDA_VISIBLE_DEVICES=0 python3 streamv2v/profile_motionflow.py \
      --config_path configs/wan_causal_dmd_v2v.yaml \
      --checkpoint_folder ckpts/wan_causal_dmd_v2v \
      --output_folder outputs/profile \
      --prompt_file_path prompt.txt \
      --video_path ./evaluation/videos/bird.mp4 \
      --height 480 --width 832 --fps 16 --step 4 \
      --occlusion_method gather_block --use_cached_text_embedding \
      --vae_type wan-taehv --x265_params '{"stage": "lookahead"}' \
      --vae_ratio 0.1 --dit_ratio 0.1
"""

import os
import sys

# Make sure 'import inference' resolves to streamv2v/inference.py, and let its own
# top-level sys.path setup (../, ../deps/gmflow, ../StreamDiffusionV2) put causvid /
# utils / deps on the path. This mirrors how inference.py itself is launched
# (cwd = StreamDiffusionV2).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

import inference as inf  # noqa: E402  (also performs project sys.path setup)
from causvid.data import TextDataset  # noqa: E402
from causvid.profiling import PROFILER  # noqa: E402
from utils.vae_utils.mask_utils import build_gather_block_masks  # noqa: E402
from deps.sige3d.torch_kernels.backend import set_kernel_backend  # noqa: E402


# ----- figure-bucket ordering (for stable, readable output) -----------------
# VAE resolution-stage buckets are discovered at runtime (named "VAE <height>" by
# vae.py), so they reflect the actual input resolution, e.g. 480/240/120/60 for a
# 480x832 run, or 512/256/128/64 for the paper's square-512 setting.
VAE_RES_PREFIX = "VAE "
CACHE_WARP = "Cache Warp"
DIT_BUCKETS = ["Self Attention", "Cross Attention", "Linear", "RoPE", "Warp"]
# Catch-all so each figure-(b) column sums EXACTLY to its figure-(a) stage:
# Other = stage_total(measured end-to-end) - sum(named buckets).
OTHER = "Other"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # --- pipeline args (mirror streamv2v/inference.py) ---
    p.add_argument("--config_path", type=str, required=True)
    p.add_argument("--checkpoint_folder", type=str, required=True)
    p.add_argument("--output_folder", type=str, required=True)
    p.add_argument("--prompt_file_path", type=str, required=True)
    p.add_argument("--video_path", type=str, required=True)
    p.add_argument("--noise_scale", type=float, default=0.700)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--step", type=int, default=4)
    p.add_argument("--model_type", type=str, default="T2V-1.3B")
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--vae_type", type=str.lower, default="wan-taehv",
                   choices=["wanvae", "taehv", "wan-taehv"])
    p.add_argument("--flow_model", type=str, default="x265",
                   choices=["gmflow", "raft", "x265", "none"])
    p.add_argument("--x265_params", type=str, default='{"stage": "lookahead"}')
    p.add_argument("--occlusion_method", type=str, default="gather_block",
                   choices=["exact", "quantile", "morphological", "connected_components", "gather_block"])
    p.add_argument("--vae_ratio", type=float, default=0.1)
    p.add_argument("--dit_ratio", type=float, default=0.1)
    p.add_argument("--use_cached_text_embedding", action="store_true")
    p.add_argument("--mask_dilate", type=int, default=6)
    p.add_argument("--min_res", nargs=2, type=int, default=(40, 40), metavar=("H", "W"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--is_nocache", action="store_true", default=False)
    p.add_argument("--cache_min_downsample", type=float, default=0)
    p.add_argument("--sige_kernels", type=str, default="cuda", choices=["cuda", "pytorch"])
    p.add_argument("--device", type=str, default="cuda:0")
    # --- profiling args ---
    p.add_argument("--warmup", type=int, default=12, help="Hot chunks to run before measuring (reach sparse steady state).")
    p.add_argument("--measure", type=int, default=30, help="Hot chunks to measure per phase.")
    p.add_argument("--setmasks", choices=["conv", "separate"], default="conv",
                   help="Where the VAE 'Set Masks' (gather/scatter index build) time goes in figure (b): "
                        "'conv' distributes it across the resolution stages (matches the paper, Cache Warp = "
                        "flow warp only; default), 'separate' shows it as its own bucket to inspect magnitude.")
    p.add_argument("--breakdown_mode", choices=["dense", "sparse"], default="dense",
                   help="DiT path for figure (b): 'dense' (full-token, matches the paper's "
                        "Linear-dominated breakdown; default) or 'sparse' (the 0.1-ratio runtime path). "
                        "VAE always stays sparse so Cache Warp is kept either way.")
    p.add_argument("--output_json", type=str, default=None, help="Where to write the JSON results (default: <output_folder>/motionflow_profile.json).")
    # --- optional rescaling to real parallel throughput (see method note in README) ---
    p.add_argument("--parallel_chunk_ms", type=float, default=None,
                   help="Measured PARALLEL wall-clock time PER CHUNK (ms), from a real inference.py run "
                        "(saver log 'Iter Time'). If given, all buckets are rescaled by "
                        "parallel_chunk_ms / serial_total so the stacked total reflects the true pipelined speed.")
    p.add_argument("--parallel_fps", type=float, default=None,
                   help="Alternative to --parallel_chunk_ms: the 'Average End-to-End FPS' printed by inference.py "
                        "(per-frame). Converted internally as chunk_size / fps * 1000 ms per chunk.")
    return p.parse_args()


def main():
    args = parse_args()

    inf.set_seed(args.seed)
    set_kernel_backend("cuda" if args.sige_kernels == "cuda" else "pytorch")
    torch.set_grad_enabled(False)
    os.makedirs(args.output_folder, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    # ----- load video (32-aligned, exactly like inference.main) -------------
    ALIGNMENT = 32
    new_h = (args.height // ALIGNMENT) * ALIGNMENT
    new_w = (args.width // ALIGNMENT) * ALIGNMENT
    args.height, args.width = new_h, new_w  # must happen before pipeline build (frame_seq_length)
    input_video, original_fps = inf.load_mp4_as_tensor(
        args.video_path, resize_hw=(new_h, new_w), max_frames=args.max_frames, device=str(device)
    )
    input_video = input_video.unsqueeze(0).to(device=device, dtype=dtype)
    T = input_video.shape[2]
    print(f"[profile] video tensor: {tuple(input_video.shape)}  (T={T} frames, fps={original_fps})")
    if T < 9:
        raise ValueError(f"Need at least 9 frames for a cold start + one hot chunk, got {T}.")

    # ----- optical flow / occlusion calculator ------------------------------
    ratio_list = (args.vae_ratio, args.dit_ratio)
    x265_params = json.loads(args.x265_params)
    flow_calculator = inf.OpticalFlowCalculator(
        flow_model_type=args.flow_model, device=device, x265_params=x265_params,
        occlusion_method=args.occlusion_method, top_k_percentage=ratio_list,
    )

    # ----- config + pipeline (mirror inference.main) ------------------------
    config = OmegaConf.load(args.config_path)
    config = OmegaConf.merge(config, OmegaConf.create(vars(args)))
    denoising_map = {1: [700, 0], 2: [700, 500, 0], 3: [700, 600, 400, 0]}
    config.denoising_step_list = denoising_map.get(args.step, [700, 600, 500, 400, 0])

    pm = inf.SingleGPUInferencePipeline(
        config, device, args.cache_min_downsample, use_cached_text_embedding=args.use_cached_text_embedding
    )
    pm.set_vae_backend(args.vae_type)
    pm.load_model(args.checkpoint_folder)

    pipeline = pm.pipeline
    dit_model = pipeline.generator.model  # the WanModel (has .count, used for sparse cadence)
    frame_seq_length = pipeline.frame_seq_length
    num_steps = len(pipeline.denoising_step_list)
    prompts = [TextDataset(args.prompt_file_path)[0]]
    chunk_size = 4

    print(f"[profile] frame_seq_length={frame_seq_length}, denoise steps={num_steps}, "
          f"vae_type={args.vae_type}, occlusion={args.occlusion_method}, "
          f"vae_ratio={args.vae_ratio}, dit_ratio={args.dit_ratio}")

    # ----- frame helpers (cycle frames so we can run arbitrarily many chunks) -
    def sel(a, b):
        idx = torch.arange(a, b, device=device) % T
        return input_video.index_select(2, idx)

    def fr(i):
        return input_video[:, :, i % T]

    # noise-scale / timestep, robust to frame cycling (only affects the timestep
    # value fed to the DiT; FLOPs are unchanged, so content here is irrelevant).
    init_noise_scale = args.noise_scale

    def noise_scale_and_step(end_idx, noise_scale):
        a = torch.arange(end_idx - chunk_size, end_idx, device=device) % T
        b = torch.arange(end_idx - chunk_size - 1, end_idx - 1, device=device) % T
        cur = input_video.index_select(2, a)
        prv = input_video.index_select(2, b)
        l2 = (cur - prv) ** 2
        l2 = (torch.sqrt(l2.mean(dim=(0, 1, 3, 4))).max() / 0.2).clamp(0, 1)
        new_ns = (init_noise_scale - 0.1 * l2.item()) * 0.9 + noise_scale * 0.1
        return new_ns, int(1000 * new_ns) - 100

    # ===================== cold start (initialize all caches) ===============
    state = {"end_idx": 5, "ref_idx": 4, "noise_scale": args.noise_scale,
             "current_start": 0, "current_end": frame_seq_length * 2}

    inp = sel(0, 5)
    latents = pm.vae_encoder.stream_encode(inp, None, None, is_nocache=args.is_nocache)
    latents = latents.transpose(2, 1).contiguous()
    noise = torch.randn_like(latents)
    noisy = noise * state["noise_scale"] + latents * (1 - state["noise_scale"])
    denoised_pred = pm.prepare_pipeline(
        text_prompts=prompts, noise=noisy,
        current_start=state["current_start"], current_end=state["current_end"],
    )
    _ = pm.vae_decoder.stream_decode_to_pixel(denoised_pred, None, None)
    torch.cuda.synchronize()
    print("[profile] cold start done (pipeline caches initialized).")

    def new_event():
        return torch.cuda.Event(enable_timing=True)

    def run_chunk(time_stages: bool, dit_dense: bool = False):
        """Run one serial hot chunk. Returns dict with per-stage ms (if timed),
        the DiT sparse flag, and (when PROFILER is enabled) per-submodule ms.

        dit_dense=True forces the DiT to run its DENSE (full-token) path by passing
        latent_flow_data=None, so figure (b)'s DiT breakdown reflects the paper's
        dense module distribution (Linear-dominated). The VAE still runs sparse,
        so its breakdown keeps the Cache Warp bucket."""
        start_idx = state["end_idx"]
        state["end_idx"] = start_idx + chunk_size
        end_idx = state["end_idx"]
        cur_idx = end_idx - 1

        inp = sel(start_idx, end_idx)
        ref_frame = fr(state["ref_idx"]).to(torch.float32)
        cur_frame = fr(cur_idx).to(torch.float32)
        bwd_flow, bwd_occ = flow_calculator.calculate_flow(ref_frame, cur_frame)
        masks_enc = build_gather_block_masks(bwd_occ.squeeze(0).squeeze(0), top_k_percentage=ratio_list[0])
        state["ref_idx"] = cur_idx

        # advance kv-cache window
        state["current_start"] = state["current_end"]
        state["current_end"] += (chunk_size // 4) * frame_seq_length
        # In dense mode the DiT always takes the full-token path; otherwise it follows
        # the sparse cadence (use_sparse when count%5!=0 and flow guidance is available).
        sparse_flag = (not dit_dense) and (dit_model.count % 5 != 0)

        # CUDA-event timers for the three stages (figure a)
        if time_stages:
            e_enc0, e_enc1 = new_event(), new_event()
            e_dit0, e_dit1 = new_event(), new_event()
            e_dec0, e_dec1 = new_event(), new_event()

        # --- stage 1: VAE encode (WanVAE encoder, sparse + cache warp) ---
        if time_stages:
            e_enc0.record()
        latents = pm.vae_encoder.stream_encode(
            inp, mask=masks_enc,
            flow=bwd_flow.squeeze(0).permute(1, 2, 0).contiguous(),
            is_nocache=args.is_nocache,
        )
        if time_stages:
            e_enc1.record()
        latents = latents.transpose(2, 1).contiguous()

        # build latent-space flow guidance (untimed producer overhead)
        ns, current_step = noise_scale_and_step(end_idx, state["noise_scale"])
        state["noise_scale"] = ns
        noise = torch.randn_like(latents)
        noisy = noise * ns + latents * (1 - ns)

        latent_h, latent_w = latents.shape[-2:]
        downsampled_flow = F.interpolate(bwd_flow, size=(latent_h, latent_w), mode="bilinear", align_corners=False)
        downsampled_flow = downsampled_flow * (float(latent_h) / bwd_flow.shape[2])
        downsampled_occ = F.interpolate(bwd_occ, size=(latent_h, latent_w), mode="bilinear", align_corners=False)
        latent_mask = flow_calculator.compute_binary_occlusion_mask(downsampled_occ)
        downsampled_occ_half = F.interpolate(bwd_occ, size=(latent_h // 2, latent_w // 2), mode="bilinear", align_corners=False)
        latent_mask_half = flow_calculator.compute_binary_occlusion_mask(downsampled_occ_half)
        flow_data = (downsampled_flow, latent_mask, latent_mask_half)

        # --- stage 2: Denoise (DiT) ---
        if time_stages:
            e_dit0.record()
        denoised_pred = pipeline.inference_stream(
            noise=noisy, current_start=state["current_start"], current_end=state["current_end"],
            current_step=current_step, latent_flow_data=(None if dit_dense else flow_data),
        )
        if time_stages:
            e_dit1.record()

        # --- stage 3: VAE decode (TAEHV) ---
        if time_stages:
            e_dec0.record()
        _ = pm.vae_decoder.stream_decode_to_pixel(denoised_pred[[-1]], None, None)
        if time_stages:
            e_dec1.record()

        result = {"sparse": sparse_flag}
        if time_stages:
            torch.cuda.synchronize()
            result["vae_encode"] = e_enc0.elapsed_time(e_enc1)
            result["denoise"] = e_dit0.elapsed_time(e_dit1)
            result["vae_decode"] = e_dec0.elapsed_time(e_dec1)
        if PROFILER.enabled:
            result["submodules"] = PROFILER.collect_iter()
        return result

    # ===================== warmup ==========================================
    print(f"[profile] warmup: {args.warmup} chunks ...")
    for _ in range(args.warmup):
        run_chunk(time_stages=False)
    torch.cuda.synchronize()

    # ===================== unified measurement (figures a & b, same state) =
    # ONE loop both (i) times each stage end-to-end with CUDA events -> figure (a),
    # and (ii) records the per-submodule breakdown via PROFILER -> figure (b). So
    # each stage is measured in the SAME state for both panels. The VAE always runs
    # sparse (keeps Cache Warp); the DiT runs per --breakdown_mode. figure (b) then
    # gets an "Other" bucket = stage_total - sum(named buckets), so every column
    # sums EXACTLY to its figure (a) stage -- a complete breakdown, never larger
    # than the real stage time.
    dense_b = (args.breakdown_mode == "dense")
    print(f"[profile] measuring {args.measure} chunks "
          f"(VAE=sparse, DiT={'dense/full-token' if dense_b else 'sparse'}) ...")

    # extra warmup in the chosen DiT mode so the kv-cache settles into that path
    for _ in range(min(4, args.warmup)):
        run_chunk(time_stages=False, dit_dense=dense_b)
    torch.cuda.synchronize()

    PROFILER.reset()
    PROFILER.enabled = True
    enc_ms, dit_ms, dec_ms, subs = [], [], [], []
    for i in range(args.measure):
        r = run_chunk(time_stages=True, dit_dense=dense_b)
        enc_ms.append(r["vae_encode"])
        dit_ms.append(r["denoise"])
        dec_ms.append(r["vae_decode"])
        subs.append(r.get("submodules", {}))
    PROFILER.enabled = False

    def mean(xs):
        return float(np.mean(xs)) if xs else 0.0

    # stage totals measured end-to-end (these define figure a, same state as fig b)
    enc_mean, denoise_mean, decode_mean = mean(enc_ms), mean(dit_ms), mean(dec_ms)

    def bucket_mean(name):
        return float(np.mean([d.get(name, 0.0) for d in subs])) if subs else 0.0

    # discover VAE resolution buckets (high -> low res), Cache Warp last
    seen = set()
    for d in subs:
        seen.update(d.keys())

    def _res_of(key):
        try:
            return int(key[len(VAE_RES_PREFIX):])
        except ValueError:
            return -1

    res_keys = sorted((k for k in seen if k.startswith(VAE_RES_PREFIX)), key=_res_of, reverse=True)
    dit_named = list(DIT_BUCKETS)

    # figure (b) VAE: resolution conv stages + Set Masks + Cache Warp(flow) + Other.
    # "Set Masks" (building per-layer gather/scatter indices) is really part of each
    # conv stage's sparse setup. --setmasks controls where it goes:
    #   conv     -> distribute it across the conv stages (matches the paper, where
    #               Cache Warp is only the flow warp); default.
    #   separate -> show it as its own bucket (to inspect its raw magnitude).
    conv = {k: bucket_mean(k) for k in res_keys}
    setmasks_ms = bucket_mean("Set Masks")
    cache_warp_ms = bucket_mean(CACHE_WARP)  # flow_cache only (after the vae.py split)

    if args.setmasks == "conv":
        conv_sum = sum(conv.values())
        if conv_sum > 0:
            for k in res_keys:
                conv[k] += setmasks_ms * conv[k] / conv_sum
        else:
            cache_warp_ms += setmasks_ms  # nowhere to distribute; keep it visible
        figure_b_vae = dict(conv)
        figure_b_vae[CACHE_WARP] = cache_warp_ms
        vae_named = res_keys + [CACHE_WARP]
    else:  # separate
        figure_b_vae = dict(conv)
        figure_b_vae["Set Masks"] = setmasks_ms
        figure_b_vae[CACHE_WARP] = cache_warp_ms
        vae_named = res_keys + ["Set Masks", CACHE_WARP]

    figure_b_vae[OTHER] = max(0.0, enc_mean - sum(figure_b_vae.values()))
    figure_b_vae["total"] = enc_mean
    vae_order = vae_named + [OTHER]

    figure_b_dit = {name: bucket_mean(name) for name in dit_named}
    figure_b_dit[OTHER] = max(0.0, denoise_mean - sum(figure_b_dit.values()))
    figure_b_dit["total"] = denoise_mean
    dit_order = dit_named + [OTHER]

    # figure (a): stage totals; encode == VAE-bar total, denoise == DiT-bar total
    figure_a = {
        "vae_encode_ms": enc_mean,
        "denoise_ms": denoise_mean,
        "vae_decode_ms": decode_mean,
        "total_ms": enc_mean + denoise_mean + decode_mean,
        "measured_chunks": args.measure,
    }

    # ===================== report ==========================================
    def bar(title, items, total):
        print("\n" + "=" * 56)
        print(title)
        print("-" * 56)
        for name, ms in items:
            pct = (ms / total * 100) if total else 0
            print(f"  {name:<18}: {ms:7.3f} ms  ({pct:4.1f}%)")
        print("-" * 56)
        print(f"  {'TOTAL':<18}: {total:7.3f} ms")
        print("=" * 56)

    print("\n\n############  MotionFlow profiling results  ############")
    bar("Figure (a) -- MotionFlow per-frame stages",
        [("VAE encode", figure_a["vae_encode_ms"]),
         ("Denoise (DiT)", figure_a["denoise_ms"]),
         ("VAE decode", figure_a["vae_decode_ms"])],
        figure_a["total_ms"])
    bar("Figure (b) -- VAE encoder breakdown (sums to figure-a encode)",
        [(n, figure_b_vae[n]) for n in vae_order], figure_b_vae["total"])
    bar("Figure (b) -- DiT (Denoise) breakdown (sums to figure-a denoise)",
        [(n, figure_b_dit[n]) for n in dit_order], figure_b_dit["total"])

    # ===================== optional rescale to real parallel throughput ====
    # Serial profiling gives accurate *proportions* but inflates the per-chunk
    # total (no producer/consumer overlap). If the user supplies the real
    # parallel per-chunk wall-clock (from an inference.py run), rescale every
    # bucket by  scale = parallel_chunk_ms / serial_total_ms.  A single scale
    # keeps figures (a) and (b) self-consistent: the rescaled VAE-bar total
    # equals the rescaled encode segment, and the DiT-bar total equals denoise.
    parallel_chunk_ms = args.parallel_chunk_ms
    if parallel_chunk_ms is None and args.parallel_fps is not None and args.parallel_fps > 0:
        parallel_chunk_ms = chunk_size / args.parallel_fps * 1000.0

    scale = None
    figure_a_scaled = figure_b_vae_scaled = figure_b_dit_scaled = None
    if parallel_chunk_ms is not None:
        ser_total = figure_a["total_ms"]
        scale = (parallel_chunk_ms / ser_total) if ser_total > 0 else 1.0

        _NON_TIME = {"sparse_chunks", "measured_chunks"}

        def scaled(d):
            return {k: (v * scale if (isinstance(v, (int, float)) and k not in _NON_TIME) else v)
                    for k, v in d.items()}

        figure_a_scaled = scaled(figure_a)
        figure_b_vae_scaled = scaled(figure_b_vae)
        figure_b_dit_scaled = scaled(figure_b_dit)

        print("\n\n########  Rescaled to real parallel throughput  ########")
        print(f"  serial per-chunk total   : {ser_total:.3f} ms")
        print(f"  parallel per-chunk total : {parallel_chunk_ms:.3f} ms  (from "
              f"{'--parallel_chunk_ms' if args.parallel_chunk_ms is not None else '--parallel_fps'})")
        print(f"  scale factor             : {scale:.4f}x")
        bar("Figure (a) [rescaled] -- MotionFlow per-frame stages",
            [("VAE encode", figure_a_scaled["vae_encode_ms"]),
             ("Denoise (DiT)", figure_a_scaled["denoise_ms"]),
             ("VAE decode", figure_a_scaled["vae_decode_ms"])],
            figure_a_scaled["total_ms"])
        bar("Figure (b) [rescaled] -- VAE encoder breakdown",
            [(n, figure_b_vae_scaled[n]) for n in vae_order], figure_b_vae_scaled["total"])
        bar("Figure (b) [rescaled] -- DiT (Denoise) breakdown",
            [(n, figure_b_dit_scaled[n]) for n in dit_order], figure_b_dit_scaled["total"])
    else:
        print("\n[note] no --parallel_chunk_ms / --parallel_fps given: results are the "
              "raw SERIAL timings. Run inference.py once, read the saver 'Iter Time' (or "
              "'Average End-to-End FPS'), and pass it in to rescale to real pipelined speed.")

    results = {
        "figure_a": figure_a,
        "figure_b_vae": figure_b_vae,
        "figure_b_dit": figure_b_dit,
        "scale": scale,
        "parallel_chunk_ms": parallel_chunk_ms,
        "figure_a_scaled": figure_a_scaled,
        "figure_b_vae_scaled": figure_b_vae_scaled,
        "figure_b_dit_scaled": figure_b_dit_scaled,
        "meta": {
            "device": str(device),
            "video": args.video_path,
            "resolution": [new_h, new_w],
            "chunk_size": chunk_size,
            "frames": T,
            "frame_seq_length": frame_seq_length,
            "denoise_steps": num_steps,
            "vae_type": args.vae_type,
            "occlusion_method": args.occlusion_method,
            "vae_ratio": args.vae_ratio,
            "dit_ratio": args.dit_ratio,
            "warmup": args.warmup,
            "measure": args.measure,
            "breakdown_mode": args.breakdown_mode,
            "setmasks": args.setmasks,
            "set_masks_ms_per_chunk": setmasks_ms,
            "cache_warp_flow_ms_per_chunk": cache_warp_ms,
            "vae_breakdown_state": "sparse",
            "dit_breakdown_state": ("dense" if dense_b else "sparse"),
        },
    }
    out_json = args.output_json or os.path.join(args.output_folder, "motionflow_profile.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[profile] wrote {out_json}")
    print("[profile] to plot:  python3 streamv2v/plot_motionflow.py " + out_json)


if __name__ == "__main__":
    main()

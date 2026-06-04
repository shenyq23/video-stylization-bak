#!/usr/bin/env python3
"""Render the MotionFlow latency figure from a motionflow_profile.json file.

Usage:
    # plot the raw serial numbers
    python3 streamv2v/plot_motionflow.py motionflow_profile.json

    # rescale to the real parallel throughput measured from an inference.py run
    python3 streamv2v/plot_motionflow.py motionflow_profile.json --parallel_fps 26.2294
    python3 streamv2v/plot_motionflow.py motionflow_profile.json --parallel_chunk_ms 152.5

Scaling rule (single factor, keeps panels self-consistent):
    scale = parallel_chunk_ms / figure_a.total_ms
where parallel_chunk_ms = chunk_size / parallel_fps * 1000 (per-chunk wall clock).
Every bucket in both panels is multiplied by `scale`, so the rescaled VAE-bar
total equals the rescaled encode segment and the DiT-bar total equals denoise.

Priority for what gets drawn:
    1. --parallel_fps / --parallel_chunk_ms on the command line (rescale now), else
    2. figure_*_scaled already present in the JSON, else
    3. the raw serial figure_*.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# teal/orange/red/blue/purple, then grey reserved for the "Other" catch-all bucket
COLORS = ["#56b4a5", "#f4a040", "#ef6f5a", "#5a9bd4", "#b59ad0", "#999999"]
NON_TIME_KEYS = {"sparse_chunks", "measured_chunks"}  # figure_a fields that must NOT be scaled


def color_for(i, name):
    return "#999999" if name == "Other" else COLORS[i % len(COLORS)]


def scale_dict(d, scale):
    return {k: (v * scale if (isinstance(v, (int, float)) and k not in NON_TIME_KEYS) else v)
            for k, v in d.items()}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("json_path")
    p.add_argument("out_png", nargs="?", default=None)
    p.add_argument("--parallel_fps", type=float, default=None,
                   help="Per-frame 'Average End-to-End FPS' from inference.py; rescales to real parallel speed.")
    p.add_argument("--parallel_chunk_ms", type=float, default=None,
                   help="Per-chunk parallel wall-clock (ms) from inference.py (alternative to --parallel_fps).")
    p.add_argument("--chunk_size", type=int, default=None,
                   help="Override chunk size (default: meta.chunk_size or 4).")
    p.add_argument("--per_chunk", action="store_true",
                   help="Show per-chunk numbers. Default is per-frame (per-chunk / chunk_size), "
                        "which matches the paper's 'Per Frame' axes.")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.json_path) as f:
        data = json.load(f)
    out_png = args.out_png or os.path.splitext(args.json_path)[0] + ".png"

    chunk_size = args.chunk_size or data.get("meta", {}).get("chunk_size", 4)

    # ----- decide which numbers to draw (see docstring priority) -----
    parallel_chunk_ms = args.parallel_chunk_ms
    if parallel_chunk_ms is None and args.parallel_fps is not None and args.parallel_fps > 0:
        parallel_chunk_ms = chunk_size / args.parallel_fps * 1000.0

    if parallel_chunk_ms is not None:
        ser_total = data["figure_a"]["total_ms"]
        scale = (parallel_chunk_ms / ser_total) if ser_total > 0 else 1.0
        fa = scale_dict(data["figure_a"], scale)
        fb_vae = scale_dict(data["figure_b_vae"], scale)
        fb_dit = scale_dict(data["figure_b_dit"], scale)
        tag = f" [rescaled x{scale:.3f}]"
        print(f"rescaling: parallel={parallel_chunk_ms:.2f} ms/chunk, serial={ser_total:.2f} ms/chunk, scale={scale:.4f}")
    elif data.get("figure_a_scaled") is not None:
        fa, fb_vae, fb_dit = data["figure_a_scaled"], data["figure_b_vae_scaled"], data["figure_b_dit_scaled"]
        tag = f" [rescaled x{data.get('scale', 1.0):.3f}]"
    else:
        fa, fb_vae, fb_dit = data["figure_a"], data["figure_b_vae"], data["figure_b_dit"]
        tag = " [serial]"

    # ----- per-frame vs per-chunk display -----
    # All raw measurements are per-chunk; the paper reports per-frame, so divide by
    # chunk_size unless --per_chunk is given. (Scaling above is dimensionless.)
    div = 1 if args.per_chunk else chunk_size
    unit = "Per Chunk" if args.per_chunk else "Per Frame"
    fa = scale_dict(fa, 1.0 / div)
    fb_vae = scale_dict(fb_vae, 1.0 / div)
    fb_dit = scale_dict(fb_dit, 1.0 / div)

    # Buckets discovered from JSON order (VAE: high->low res, Cache Warp, Other;
    # DiT: the 5 named modules then Other). Both end with the grey Other bucket.
    vae_buckets = [k for k in fb_vae if k != "total"]
    dit_buckets = [k for k in fb_dit if k != "total"]

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 5))

    # ----- (a) MotionFlow stacked bar (seconds) -----
    seg = [("VAE encode", fa["vae_encode_ms"] / 1000.0, COLORS[0]),
           ("Denoise", fa["denoise_ms"] / 1000.0, COLORS[1]),
           ("VAE decode", fa["vae_decode_ms"] / 1000.0, COLORS[2])]
    bottom = 0.0
    for name, val, color in seg:
        axa.bar("MotionFlow", val, bottom=bottom, color=color, label=name, width=0.5)
        if val > 0:
            axa.text(0, bottom + val / 2, f"{val:.3f}", ha="center", va="center", color="white", fontweight="bold")
        bottom += val
    axa.text(0, bottom * 1.01, f"{bottom:.3f}s", ha="center", va="bottom", fontweight="bold")
    axa.set_ylabel(f"{unit} Latency (s)")
    axa.set_title("(a) Overall breakdown" + tag)
    axa.legend(loc="upper right", fontsize=8)

    # ----- (b) VAE + DiT stacked bars (ms) -----
    def stacked(ax_x, buckets, source):
        bottom = 0.0
        for i, name in enumerate(buckets):
            val = source.get(name, 0.0)
            axb.bar(ax_x, val, bottom=bottom, color=color_for(i, name), width=0.5)
            if val > 0.3:
                axb.text(ax_x, bottom + val / 2, f"{val:.1f}", ha="center", va="center", color="white", fontsize=8)
            bottom += val
        axb.text(ax_x, bottom * 1.01, f"{bottom:.2f}ms", ha="center", va="bottom", fontweight="bold")

    stacked("VAE", vae_buckets, fb_vae)
    stacked("DiT", dit_buckets, fb_dit)
    axb.set_ylabel(f"{unit} Module Latency (ms)")
    axb.set_title("(b) MotionFlow breakdown" + tag)
    n = max(len(vae_buckets), len(dit_buckets))
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_for(i, (vae_buckets[i] if i < len(vae_buckets) else "")))
               for i in range(n)]
    labels = [f"{(vae_buckets[i] if i < len(vae_buckets) else '-')}  /  "
              f"{(dit_buckets[i] if i < len(dit_buckets) else '-')}" for i in range(n)]
    axb.legend(handles, labels, fontsize=7, loc="upper left", title="VAE  /  DiT")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()

"""Reproduce MotionFlow paper Fig 6 on bird.mp4 @ 512x512, 4-step, single 4090.

Corrected method (matches the paper, no parallel->serial rescaling):
  * Measure each module's serial GPU time per *iteration* with gpu.py
    (single CUDA stream + cuda.synchronize).
  * One iteration processes a chunk of CHUNK_SIZE=4 pixel frames (Wan VAE does
    4x temporal compression: 4 frames -> 1 latent frame). The paper's axis is
    "Per Frame Latency", so per-frame = per-iteration / CHUNK_SIZE.
  * (a) = single MotionFlow bar, stacked VAE encode / Denoise(DiT) / VAE decode.
  * (b) = two bars: VAE encoder (resolution buckets + Cache Warp) and DiT
    (Self Attn / Cross Attn / Linear / RoPE / Warp). This is just the detailed
    decomposition of (a)'s VAE-encode and Denoise segments.

The paper's Fig 6 is on Celeb 512x512; we use bird.mp4 rescaled to 512x512 so
the VAE resolution buckets line up (512/256/128/64).

Numbers below are per-ITERATION (per 4-frame chunk) straight from gpu.py's
"(Detailed)" sections; the script divides by CHUNK_SIZE to get per-frame.
"""
import matplotlib.pyplot as plt
import numpy as np

CHUNK_SIZE = 4  # pixel frames per iteration (Wan VAE temporal compression)

# ---------------------------------------------------------------------------
# Per-iteration serial measurements from gpu.py (MotionFlow + TAEHV + adaptive,
# bird.mp4 @ 720x1280, 2-step). REPLACE with the actual run numbers.
# ---------------------------------------------------------------------------
# Coarse module totals (ms / iteration).
COARSE = {
    "VAE encode": 569.9562,   # gpu.py [VAE Encode] summary
    "Denoise":    212.4439,   # gpu.py [DiT Inference] summary
    "VAE decode":  21.7852,   # gpu.py [VAE Decode] summary
}

# VAE encoder detailed buckets (ms / iteration), from [VAE Encode (Detailed)].
# At 720x1280 the resolution stages are 720/360/180/90.
VAE_SUB = {
    "VAE 720":    347.5283,
    "VAE 360":     90.0618,
    "VAE 180":     25.4414,
    "VAE 90":      12.4945,
    "Cache Warp":  56.4707,
}

# DiT detailed sub-modules (ms / iteration), from [DiT Inference (Detailed)].
DIT_SUB = {
    "Linear":     88.7371,
    "Self Attn":  49.1650,
    "Cross Attn": 24.9976,
    "RoPE":       19.4597,
    "Warp":        6.5956,
}


def per_frame(d):
    return {k: (v / CHUNK_SIZE if v is not None else None) for k, v in d.items()}


def require(d, name):
    missing = [k for k, v in d.items() if v is None]
    if missing:
        raise SystemExit(
            f"[plot_breakdown] {name} still has unfilled (None) entries: {missing}.\n"
            f"Run gpu.py at 720x1280 and paste the per-iteration ms into this file."
        )


require(COARSE, "COARSE")
require(VAE_SUB, "VAE_SUB")
require(DIT_SUB, "DIT_SUB")

coarse_pf = per_frame(COARSE)
vae_pf = per_frame(VAE_SUB)
dit_pf = per_frame(DIT_SUB)

print("=== Per-frame (per-iter / %d) ===" % CHUNK_SIZE)
print("  Coarse:", {k: round(v, 2) for k, v in coarse_pf.items()},
      " total=%.2f ms" % sum(coarse_pf.values()))
print("  VAE   :", {k: round(v, 2) for k, v in vae_pf.items()},
      " sum=%.2f ms" % sum(vae_pf.values()))
print("  DiT   :", {k: round(v, 2) for k, v in dit_pf.items()},
      " sum=%.2f ms" % sum(dit_pf.values()))

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 5),
                                 gridspec_kw={"width_ratios": [0.8, 1.4]})

# ---- (a) single MotionFlow bar, stacked coarse modules ----
stack_order_a = ["VAE encode", "Denoise", "VAE decode"]
colors_a = {"VAE encode": "#7DC0A8", "Denoise": "#F2C68B", "VAE decode": "#E08383"}
bottom = 0.0
for label in stack_order_a:
    h = coarse_pf[label]
    ax_a.bar(0, h, 0.5, bottom=bottom, color=colors_a[label], label=label,
             edgecolor="white", linewidth=0.6)
    if h > 0.6:
        ax_a.text(0, bottom + h / 2, f"{h/1000:.3f}s", ha="center", va="center",
                  fontsize=9, color="white", fontweight="bold")
    bottom += h
total_a = sum(coarse_pf.values())
ax_a.text(0, total_a + total_a * 0.02, f"{total_a/1000:.3f}s",
          ha="center", va="bottom", fontsize=11, fontweight="bold")
ax_a.set_xticks([0])
ax_a.set_xticklabels(["MotionFlow"])
ax_a.set_xlim(-0.6, 0.6)
ax_a.set_ylim(0, total_a * 1.18)
ax_a.set_ylabel("Per Frame Latency (ms)")
ax_a.set_title("(a) Overall breakdown")
ax_a.legend(loc="upper right", frameon=False, fontsize=9)
ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)

# ---- (b) two bars: VAE (resolution buckets + Cache Warp) and DiT ----
columns = ["VAE", "DiT"]
column_parts = [vae_pf, dit_pf]
vae_palette = {"VAE 720": "#5DA899", "VAE 360": "#F2C68B", "VAE 180": "#E08383",
               "VAE 90": "#7191C4", "Cache Warp": "#B79FD1"}
dit_palette = {"Self Attn": "#5DA899", "Cross Attn": "#F2C68B", "Linear": "#E08383",
               "RoPE": "#7191C4", "Warp": "#B79FD1"}
# Fixed stacking order (bottom -> top), matching the paper's column legend.
vae_order = ["VAE 720", "VAE 360", "VAE 180", "VAE 90", "Cache Warp"]
dit_order = ["Self Attn", "Cross Attn", "Linear", "RoPE", "Warp"]
orders = [vae_order, dit_order]
palettes = [vae_palette, dit_palette]

xs_b = np.arange(len(columns))
for col_idx, (parts, order, palette) in enumerate(zip(column_parts, orders, palettes)):
    bottom = 0.0
    for label in order:
        val = parts[label]
        ax_b.bar(xs_b[col_idx], val, 0.5, bottom=bottom, color=palette[label],
                 edgecolor="white", linewidth=0.6)
        if val > 0.4:
            ax_b.text(xs_b[col_idx], bottom + val / 2, f"{val:.1f}",
                      ha="center", va="center", fontsize=9, color="white",
                      fontweight="bold")
        bottom += val
    ax_b.text(xs_b[col_idx], bottom + bottom * 0.02, f"{bottom:.2f}ms",
              ha="center", va="bottom", fontsize=11, fontweight="bold")

# legend: two groups
from matplotlib.patches import Patch
vae_handles = [Patch(facecolor=vae_palette[k], label=k) for k in vae_order]
dit_handles = [Patch(facecolor=dit_palette[k], label=k) for k in dit_order]
leg1 = ax_b.legend(handles=vae_handles, title="VAE", loc="upper left",
                   frameon=False, fontsize=8, title_fontsize=9)
ax_b.add_artist(leg1)
ax_b.legend(handles=dit_handles, title="DiT", loc="upper center",
            frameon=False, fontsize=8, title_fontsize=9)

ax_b.set_xticks(xs_b)
ax_b.set_xticklabels(columns)
ax_b.set_ylabel("Per Frame Module Latency (ms)")
ax_b.set_title("(b) MotionFlow breakdown")
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)
ax_b.set_ylim(0, max(sum(vae_pf.values()), sum(dit_pf.values())) * 1.25)

plt.suptitle("MotionFlow reproduction  —  Wan 1.3B / bird.mp4 @ 720x1280 / 2-step / 4090",
             y=1.02, fontsize=11)
plt.tight_layout()
out_path = "breakdown.png"
plt.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"\nSaved figure to {out_path}")

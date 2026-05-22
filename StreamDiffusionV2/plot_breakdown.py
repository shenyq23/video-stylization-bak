"""Plot Fig-6-style breakdown for the local reproduction.

(a) Overall per-frame latency, stacked by VAE encode / Denoise / VAE decode,
    one bar per method, scaled from serial gpu.py to parallel inference.py FPS.
(b) MotionFlow module breakdown: VAE encode (5 sub-buckets), VAE decode (4
    sub-buckets), DiT (5 sub-buckets), each stacked.

Numbers come from the run on bird.mp4 at 480x832, 4-step denoising. To
regenerate, re-run the commands at the top of gpu.py and streamv2v/inference.py
and replace the values in DATA below.
"""
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Measurements (replace with new runs as needed; all values are milliseconds)
# ---------------------------------------------------------------------------
# Serial coarse times from gpu.py (single CUDA stream + cuda.synchronize).
# Mixed mode (matches paper Fig 6 setup):
#   - SDV2 (full)  : default WanVAE encoder + WanVAE decoder (un-optimized baseline)
#   - MotionFlow   : sparse WanVAE encoder + distilled TAEHV decoder (optimization stack)
# Swapping decoder on the baseline would be unfair — paper compares each method
# with the decoder it actually ships with.
SERIAL = {
    "SDV2 (full)": {"VAE encode": 169.01, "Denoise": 300.91, "VAE decode": 286.85},
    "MotionFlow":  {"VAE encode":  82.98, "Denoise": 196.27, "VAE decode":  10.30},
}

# Parallel end-to-end FPS from streamv2v/inference.py (saver-side, after pipeline
# fill). Same vae_type per method as the serial measurements above.
PARALLEL_FPS = {
    "SDV2 (full)": 5.2618,
    "MotionFlow":  12.9055,
}

# MotionFlow sub-module breakdowns from gpu.py "(Detailed)" sections.
# TAEHV decode is monolithic — no sub-buckets, just plotted as a single bar.
VAE_ENC_SUB = {  # ms
    "Enc/480":     29.62,
    "Cache Warp":  28.17,
    "Enc/240":     14.96,
    "Enc/60":       4.85,
    "Enc/120":      4.49,
}
VAE_DEC_SUB = {
    "TAEHV": 10.30,
}
DIT_SUB = {
    "Linear":      70.40,
    "Self Attn":   32.63,
    "Cross Attn":  20.82,
    "RoPE":        14.58,
    "Warp":         5.18,
}

# ---------------------------------------------------------------------------
# Scaling: parallel per-frame total = sum of scaled coarse modules.
# ---------------------------------------------------------------------------
def scale_method(method):
    serial = SERIAL[method]
    serial_sum = sum(serial.values())
    parallel_ms = 1000.0 / PARALLEL_FPS[method]
    ratio = parallel_ms / serial_sum
    return {k: v * ratio for k, v in serial.items()}, ratio, parallel_ms

scaled = {m: scale_method(m) for m in SERIAL}
print("=== Per-method scaling ===")
for m, (parts, r, total) in scaled.items():
    print(f"  {m:14s}  ratio={r:.4f}  parallel/frame={total:.2f} ms  parts={parts}")

# Build sub-module stacks for MotionFlow (b).
def distribute(scaled_total_ms, sub_ms_dict):
    sub_sum = sum(sub_ms_dict.values())
    return {k: scaled_total_ms * v / sub_sum for k, v in sub_ms_dict.items()}

mf_scaled, _, mf_total = scaled["MotionFlow"]
mf_vae_enc_parts = distribute(mf_scaled["VAE encode"], VAE_ENC_SUB)
mf_dit_parts     = distribute(mf_scaled["Denoise"],    DIT_SUB)

print("\n=== MotionFlow breakdown (scaled, ms) ===")
print("  VAE encode:", {k: round(v, 2) for k, v in mf_vae_enc_parts.items()})
print("  DiT:       ", {k: round(v, 2) for k, v in mf_dit_parts.items()})

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.3]})

# ---- (a) overall stacked bars ----
methods = list(SERIAL.keys())
stack_colors_a = {
    "VAE encode": "#7DC0A8",  # teal
    "Denoise":    "#F2C68B",  # orange
    "VAE decode": "#E08383",  # coral
}
stack_order_a = ["VAE encode", "Denoise", "VAE decode"]
bar_xs = np.arange(len(methods))
bottoms = np.zeros(len(methods))
for label in stack_order_a:
    heights = np.array([scaled[m][0][label] for m in methods])
    ax_a.bar(bar_xs, heights, 0.55, bottom=bottoms, color=stack_colors_a[label], label=label, edgecolor="white", linewidth=0.6)
    for x, h, b in zip(bar_xs, heights, bottoms):
        ax_a.text(x, b + h / 2, f"{h:.1f}", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    bottoms += heights

# total annotation on top
for x, m in zip(bar_xs, methods):
    total_ms = sum(scaled[m][0].values())
    ax_a.text(x, total_ms + 4, f"{total_ms:.1f} ms", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax_a.set_xticks(bar_xs)
ax_a.set_xticklabels(methods)
ax_a.set_ylabel("Per-frame latency (ms, parallel)")
ax_a.set_title("(a) Overall breakdown")
ax_a.legend(loc="upper right", frameon=False)
ax_a.spines["top"].set_visible(False)
ax_a.spines["right"].set_visible(False)
ax_a.set_ylim(0, max(sum(scaled[m][0].values()) for m in methods) * 1.18)

# ---- (b) MotionFlow stacked breakdown: VAE enc / DiT only (matches paper Fig 6) ----
# The paper's (b) excludes the decoder column entirely — decoder is not sparsified,
# it's just replaced wholesale with TinyDec, so it has no internal structure worth
# decomposing. We follow the same layout.
columns = ["VAE", "DiT"]
column_parts = [mf_vae_enc_parts, mf_dit_parts]

# distinct palettes for the two columns
vae_enc_palette = ["#5DA899", "#82C8B5", "#A8DCCE", "#CFEBE0", "#E6F3EE"]
dit_palette     = ["#7191C4", "#9DB1D2", "#C0CCDF", "#DDE0EA", "#F0F1F4"]
palettes = [vae_enc_palette, dit_palette]

# Sort each column descending for clearer stacking
sorted_parts = [dict(sorted(p.items(), key=lambda kv: -kv[1])) for p in column_parts]
xs_b = np.arange(len(columns))
bottoms_b = np.zeros(len(columns))

for col_idx, (parts, palette) in enumerate(zip(sorted_parts, palettes)):
    bottom = 0.0
    for i, (label, val) in enumerate(parts.items()):
        color = palette[i % len(palette)]
        ax_b.bar(xs_b[col_idx], val, 0.55, bottom=bottom, color=color, edgecolor="white", linewidth=0.6)
        # label inside if tall enough
        if val > 1.5:
            ax_b.text(xs_b[col_idx], bottom + val / 2, f"{label}\n{val:.2f}",
                      ha="center", va="center", fontsize=8, color="black")
        else:
            # small slice: side annotation
            ax_b.annotate(f"{label} {val:.2f}",
                          xy=(xs_b[col_idx] + 0.30, bottom + val / 2),
                          xytext=(xs_b[col_idx] + 0.55, bottom + val / 2),
                          ha="left", va="center", fontsize=7,
                          arrowprops=dict(arrowstyle="-", lw=0.5, color="gray"))
        bottom += val
    ax_b.text(xs_b[col_idx], bottom + 1.5, f"{bottom:.2f} ms",
              ha="center", va="bottom", fontsize=10, fontweight="bold")

ax_b.set_xticks(xs_b)
ax_b.set_xticklabels(columns)
ax_b.set_ylabel("Per-frame module latency (ms, parallel-scaled)")
ax_b.set_title("(b) MotionFlow breakdown")
ax_b.spines["top"].set_visible(False)
ax_b.spines["right"].set_visible(False)
ax_b.set_ylim(0, max(sum(p.values()) for p in sorted_parts) * 1.18)

plt.suptitle(
    f"Wan 1.3B / 480×832 (SDV2 bird.mp4) / 4-step / 4090 single GPU  —  "
    f"SDV2 (full) {1000/PARALLEL_FPS['SDV2 (full)']:.1f} ms/frame, "
    f"MotionFlow {1000/PARALLEL_FPS['MotionFlow']:.1f} ms/frame  "
    f"(reproduces Table 1 SDV2 row, not Fig 6 / Celeb 512²)",
    y=1.02, fontsize=10)
plt.tight_layout()

out_path = "breakdown.png"
plt.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"\nSaved figure to {out_path}")

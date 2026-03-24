#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSETS_DIR="${SCRIPT_DIR}/assets"
BASE_OUTPUT_DIR="${SCRIPT_DIR}/outputs/assets"

mkdir -p "${BASE_OUTPUT_DIR}"

shopt -s nullglob
mp4_files=("${ASSETS_DIR}"/*.mp4)
shopt -u nullglob

if [ "${#mp4_files[@]}" -eq 0 ]; then
  echo "No .mp4 files found in ${ASSETS_DIR}"
  exit 1
fi

for mp4_path in "${mp4_files[@]}"; do
  base_name="$(basename "${mp4_path}" .mp4)"
  prompt_path="${ASSETS_DIR}/${base_name}.txt"

  if [ ! -f "${prompt_path}" ]; then
    echo "Missing prompt txt for ${mp4_path}, skipping."
    continue
  fi

  output_dir="${BASE_OUTPUT_DIR}${base_name}"
  mkdir -p "${output_dir}"

  python3 "${SCRIPT_DIR}/streamv2v/inference.py" \
    --config_path "${SCRIPT_DIR}/configs/wan_causal_dmd_v2v.yaml" \
    --checkpoint_folder "/media/cephfs/video/VideoUsers/thu2025/zhurui11/StreamDiffusionV2/ckpts/wan_causal_dmd_v2v/" \
    --output_folder "${output_dir}" \
    --prompt_file_path "${prompt_path}" \
    --video_path "${mp4_path}" \
    --height 480 \
    --width 832 \
    --fps 16 \
    --step 4 \
    --flow_model x265 \
    --top_k_percentage 0.1 \
    --occlusion_method gather_block \
    --vae_type wanvae
done

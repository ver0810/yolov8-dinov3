#!/usr/bin/env bash
# Clone Meta DINOv3 source for local backbone loading.
# Weights: download manually after Meta access approval.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/third_party" "$ROOT/third_party/dinov3_weights"

if [[ ! -d "$ROOT/third_party/dinov3/dinov3" ]]; then
  git clone --depth 1 https://github.com/facebookresearch/dinov3.git "$ROOT/third_party/dinov3"
else
  echo "OK: third_party/dinov3 already present"
fi

cat <<EOF
DINOv3 source: $ROOT/third_party/dinov3

Download weights (after Meta license approval):
  https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/
Use wget, place under:
  $ROOT/third_party/dinov3_weights/

Expected filenames (examples):
  dinov3_vits16_pretrain_lvd1689m-*.pth
  dinov3_vitb16_pretrain_lvd1689m-*.pth
  dinov3_vitl16_pretrain_lvd1689m-*.pth

Or:
  export DINOV3_WEIGHTS=/path/to/your.pth

Code entry: third_party/ultralytics/ultralytics/nn/modules/dinov3.py
EOF

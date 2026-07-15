#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
THIRD_PARTY="$PROJECT_ROOT/third_party"

DINOV3_REPO="https://github.com/facebookresearch/dinov3.git"
DINOV3_DIR="$THIRD_PARTY/dinov3"

echo "=== DINOv3 Setup ==="

if [ -d "$DINOV3_DIR" ]; then
    echo "DINOv3 already cloned at $DINOV3_DIR"
else
    echo "Cloning DINOv3 from $DINOV3_REPO ..."
    mkdir -p "$THIRD_PARTY"
    git clone --depth 1 "$DINOV3_REPO" "$DINOV3_DIR"
    echo "Cloned to $DINOV3_DIR"
fi

WEIGHT_DIR="$THIRD_PARTY/dinov3_weights"
mkdir -p "$WEIGHT_DIR"

echo ""
echo "=== Weight Download ==="
echo "DINOv3 weights can be downloaded via:"
echo ""
echo "  Option 1: huggingface-cli (recommended, supports HF mirror)"
echo "    huggingface-cli download facebook/dinov3-vits16-pretrain-lvd1689m \\"
echo "      --local-dir $WEIGHT_DIR/vits16"
echo "    huggingface-cli download facebook/dinov3-vitb16-pretrain-lvd1689m \\"
echo "      --local-dir $WEIGHT_DIR/vitb16"
echo ""
echo "  Option 2: Direct from Meta (requires network access to dl.fbaipublicfiles.com)"
echo "    See: $DINOV3_DIR/README.md for download links"
echo ""
echo "=== Done ==="
echo "Next step: pixi run sanity"

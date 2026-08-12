#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CONFIGS=(
    "configs/exp_world_s_v2.yaml"
)

TOTAL=${#CONFIGS[@]}
echo "=== YOLO-World baseline (${TOTAL} experiments) ==="

for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    idx=$((i + 1))
    echo ""
    echo "============================================"
    echo "  [${idx}/${TOTAL}] ${cfg}"
    echo "============================================"
    uv run python scripts/train.py --config "${cfg}" || {
        echo "ERROR: ${cfg} failed with exit code $?"
        exit 1
    }
done

echo ""
echo "=== All ${TOTAL} experiments completed ==="

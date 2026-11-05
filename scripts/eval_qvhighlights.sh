#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
SPLIT="${SPLIT:-val}"
case "$SPLIT" in
    val) CONFIG="configs/qvhighlights/eval.yaml" ;;
    test) CONFIG="configs/qvhighlights/test.yaml" ;;
    *) echo "SPLIT must be val or test, got: $SPLIT" >&2; exit 2 ;;
esac
exec "${PYTHON:-python}" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE:-8}" \
    evaluate.py --cfg-path "$CONFIG" "$@"

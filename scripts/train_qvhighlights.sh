#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
exec "${PYTHON:-python}" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE:-8}" \
    train.py --cfg-path configs/qvhighlights/train.yaml "$@"

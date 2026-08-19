#!/usr/bin/env bash
# Build the CUDA training environment for the MgO LR experiment.
#
#   ./workflows/training/setup_gpu_env.sh /path/to/venv_cuda
#
# Installs a CUDA PyTorch and the MACE-H dependencies into a fresh venv, then
# verifies the GPU is actually reachable. Idempotent only in the sense that it
# refuses to touch an existing directory -- delete it and re-run.
#
# Note there is no torch_scatter here. MACE-H aggregates with
# `torch_geometric.utils.scatter`, which is pure PyTorch, so no compiled
# extension has to be matched to the torch build and no `nvcc` is needed.
# See tests/unit/test_scatter.py.
set -euo pipefail

VENV=${1:?usage: setup_gpu_env.sh /path/to/venv}
PYTHON=${PYTHON:-python3.14}
# cu129 covers Blackwell (sm_120, e.g. RTX 5090). Older cards work too; pick a
# smaller CUDA build only if you need to match a fixed driver.
TORCH_SPEC=${TORCH_SPEC:-torch==2.13.0+cu129}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu129}

if [ -e "$VENV" ]; then
    echo "$VENV already exists; remove it first" >&2
    exit 1
fi

echo "== creating venv at $VENV with $PYTHON"
"$PYTHON" -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip

echo "== installing $TORCH_SPEC from $TORCH_INDEX"
"$VENV/bin/pip" install "$TORCH_SPEC" --index-url "$TORCH_INDEX"

echo "== installing MACE-H dependencies"
"$VENV/bin/pip" install \
    "torch-geometric==2.8.0.post1" \
    "e3nn==0.6.0" \
    "pymatgen==2026.5.4" \
    "numpy==2.5.1" \
    "scipy==1.18.0" \
    "matplotlib==3.11.1" \
    "seekpath==2.2.1" \
    "pytest==9.0.2" \
    "h5py==3.16.0" \
    "PyYAML==6.0.3" \
    "tqdm==4.70.0" \
    "pathos==0.3.5" \
    "tensorboard==2.21.0" \
    "wandb==0.28.1"

echo "== verifying"
"$VENV/bin/python" - <<'PY'
import torch
print(f"torch          {torch.__version__}")
print(f"cuda build     {torch.version.cuda}")
print(f"cuda available {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "no CUDA device visible to torch"
props = torch.cuda.get_device_properties(0)
print(f"device 0       {props.name}, {props.total_memory / 2**30:.1f} GiB, "
      f"sm_{props.major}{props.minor}")
x = torch.randn(4096, 4096, device="cuda")
torch.cuda.synchronize()
print(f"matmul check   {float((x @ x).sum()):.4e}")

# the aggregation MACE-H depends on, on the GPU
from torch_geometric.utils import scatter
src = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device="cuda")
idx = torch.tensor([0, 0, 1], device="cuda")
got = scatter(src, idx, dim=0, dim_size=2, reduce="add")
assert got.tolist() == [[4.0, 6.0], [5.0, 6.0]], got
print("scatter check  OK (torch_geometric, no torch_scatter needed)")

import e3nn, pymatgen.core, h5py, matplotlib, seekpath, torch_geometric
print(f"e3nn           {e3nn.__version__}")
print(f"torch_geometric{torch_geometric.__version__}")
print(f"matplotlib     {matplotlib.__version__}")
print(f"seekpath       {seekpath.__version__}")
PY

cat <<EOF

Environment ready: $VENV

Next:
  export MACEH_DATA_ROOT=/path/to/run
  export MACEH_RUNS_ROOT=/path/to/generated-runs
  $VENV/bin/python -m workflows.training.freeze_provenance --verify
  $VENV/bin/python -m workflows.training.check_split_wiring
  $VENV/bin/python -m workflows.training.smoke_onebatch
EOF

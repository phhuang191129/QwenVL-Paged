#!/usr/bin/env bash
# One-shot environment setup for the GPU box. Run this first, so paid minutes go
# to compiling and debugging the kernel rather than dependency resolution.
# Safe to re-run: uv skips anything already installed.
#
# The system interpreter is not usable here. Amazon Linux 2023 ships Python 3.9
# with no pip, and recent torch publishes no 3.9 wheels, so this builds the same
# uv-managed 3.12 venv that docs/roadmap-phase2.md section 3 specifies.
set -euo pipefail

cd "$(dirname "$0")/../.."

# Versions validated against an NVIDIA L4 (sm_89), driver 595.91.07. Loosen the
# pins if a different GPU needs a different CUDA build, but record what ran:
# the kernel's numerics depend on Triton's frontend, which moves between releases.
TORCH_VERSION=2.13.0
NUMPY_VERSION=2.5.2

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv venv --allow-existing --python 3.12 .venv
uv pip install --python .venv/bin/python "numpy==${NUMPY_VERSION}" "torch==${TORCH_VERSION}"

.venv/bin/python -c "
import torch
assert torch.cuda.is_available(), 'no CUDA device visible'
print('torch', torch.__version__, '| CUDA', torch.version.cuda, '| GPU', torch.cuda.get_device_name(0))
import triton
print('triton', triton.__version__)
"

# The fixtures gate the kernel, so prove they are self-sufficient before running
# it. A fixture bug here would otherwise be misread as a kernel bug.
.venv/bin/python tools/golden_export/verify_fixtures.py fixtures

echo "setup complete. Run: .venv/bin/python tools/triton_kernel/run_fixture.py fixtures"

#!/usr/bin/env bash
# scripts/setup_env.sh
# Build two isolated uv-managed environments for the H200x8 LLM-quant course:
#   envs/quant/.venv  -> quantization env A (llm-compressor oneshot: FP8 / AWQ / SmoothQuant)
#   envs/deploy/.venv -> deploy env B (vLLM serve only)
#
# Why two envs: vLLM pins its own transformers/torch; llm-compressor and lm-eval
# want a different transformers. Putting them in one env makes pip/uv fight over
# the version. uv's `uv sync --directory <project>` builds each env from its own
# pyproject.toml + uv.lock, so the two transformers never collide.
#
# Design notes (verified against uv 0.8.x, 2025-2026):
#   * `uv venv` alone makes a BLANK venv with only seed packages — we need
#     `uv sync` afterwards to actually install from pyproject.toml / uv.lock.
#   * `uv sync --directory envs/quant` runs as if cwd were envs/quant: it reads
#     that folder's pyproject.toml and creates/updates envs/quant/.venv.
#   * Reproducibility comes from uv.lock, NOT pyproject.toml. Commit uv.lock for
#     both envs; `uv sync` then installs the exact pinned graph on any H200 node.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---- prerequisites -----------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "[setup_env] uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi
echo "[setup_env] uv version: $(uv --version)"

# Speed up HF downloads (huggingface-hub[hf-transfer] must be installed; it is,
# in both envs). Safe to leave on; hf-transfer self-disables on non-supporting hosts.
export HF_HUB_ENABLE_HF_TRANSFER=1

PYTHON_SPEC="${PYTHON_SPEC:-3.11}"   # override with PYTHON_SPEC=3.10 if your base differs
echo "[setup_env] Python target: $PYTHON_SPEC"

# ---- 1. quant env A ----------------------------------------------------------
echo
echo "=== [1/2] Building quant env A: envs/quant ==="
# Create the venv with a fixed Python so both envs are reproducible across nodes.
uv venv --python "$PYTHON_SPEC" envs/quant/.venv
# Materialize the lock (idempotent) and install the pinned graph into envs/quant/.venv.
uv lock --directory envs/quant
uv sync --directory envs/quant --frozen || uv sync --directory envs/quant
# lm-eval core no longer bundles model backends; add the vLLM eval backend into
# the quant env (where evaluation runs). Kept out of pyproject.toml because the
# [vllm] extra is an eval-time concern, not a quant-time one.
#
# IMPORTANT (verified against uv 0.8.x): use `uv pip install --python <venv-python>`
# to bind the install to this project's venv. Do NOT use `uv pip --directory ... install`:
# `--directory` only changes uv's cwd / project root for `uv pip`; it does NOT pin
# the target interpreter, so the extra can land in the wrong environment (or fail
# to find one). `uv pip` only honors an active venv (activation) OR an explicit
# `--python <path>` — it does NOT auto-discover the project's .venv.
uv pip install --python envs/quant/.venv/bin/python 'lm_eval[vllm]'

# ---- 2. deploy env B ---------------------------------------------------------
echo
echo "=== [2/2] Building deploy env B: envs/deploy ==="
uv venv --python "$PYTHON_SPEC" envs/deploy/.venv
uv lock --directory envs/deploy
uv sync --directory envs/deploy --frozen || uv sync --directory envs/deploy

# ---- 3. environment checks ---------------------------------------------------
echo
echo "=== Environment checks ==="
echo "--- nvidia-smi (driver / GPU) ---"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
else
  echo "[setup_env] WARN: nvidia-smi not on PATH (are you on an H200 node?)" >&2
fi

echo
echo "--- nvcc (CUDA toolkit) ---"
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version
else
  echo "[setup_env] WARN: nvcc not found. vLLM ships CUDA 12.8 runtime in the wheel,"
  echo "             so nvcc is only needed if you rebuild vLLM from source." >&2
fi

check_env () {
  local env_name="$1"
  local venv_dir="$2"
  echo
  echo "--- $env_name  ($venv_dir) ---"
  # Call the venv's python directly. NOTE: do NOT use `uv run` here — `uv run`
  # re-syncs the project on every invocation, which for a vLLM env means a
  # multi-minute reinstall. The sync already happened above; we just read versions.
  "$venv_dir/.venv/bin/python" - <<'PY'
import importlib.metadata as m
pkgs = ["vllm", "llmcompressor", "compressed-tensors", "transformers",
        "torch", "lm_eval", "accelerate", "huggingface_hub"]
width = max(len(p) for p in pkgs)
for p in pkgs:
    try:
        print(f"  {p:<{width}}  {m.version(p)}")
    except m.PackageNotFoundError:
        print(f"  {p:<{width}}  (not installed)")
PY
}

check_env "quant env A"  envs/quant
check_env "deploy env B" envs/deploy

echo
echo "[setup_env] Done."
echo "  Two ways to use each env:"
echo "    A) activate:   source envs/quant/.venv/bin/activate   && python recipes/smoothquant.py"
echo "    B) one-shot:   uv run --directory envs/quant python recipes/smoothquant.py"
echo "  Serve (deploy):  uv run --directory envs/deploy vllm serve <path> --tensor-parallel-size 8"
echo "  NOTE: 'uv run --directory' checks uv.lock is in sync before running; if the env is"
echo "        already synced (which this script did), the check is a no-op. It only re-installs"
echo "        if pyproject.toml/uv.lock changed since the last 'uv sync'."

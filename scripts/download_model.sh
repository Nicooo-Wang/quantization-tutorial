#!/usr/bin/env bash
# scripts/download_model.sh
# Download the baseline checkpoint (Qwen2.5-7B) to a shared path on an H200x8
# single node, so every GPU / every env sees the same weights.
#
# Uses `huggingface-cli download` (from huggingface-hub). Verified CLI shape
# (2025-2026):
#     huggingface-cli download <repo-id> --local-dir <dir>
#   - Qwen2.5-7B base + Qwen2.5-7B-Instruct are NOT gated -> no token strictly
#     required, but a token raises rate limits and is mandatory if you also pull
#     gated repos. We read it from $HF_TOKEN (the documented variable).
#   - Set HF_HUB_ENABLE_HF_TRANSFER=1 for ~5-10x faster downloads (needs
#     hf-transfer, which huggingface-hub[hf-transfer] pulls in — it is in both
#     pyproject.toml files).
#
# We pick the Instruct variant as the default because the course also does chat
# / tool-call evaluation; override with MODEL_REPO to grab the base model.
# (Base vs Instruct: the base model has NOT been instruction-tuned, so it scores
# lower on instruction-following tasks like GSM8K 5-shot / MMLU 5-shot — the
# course's unified baseline is Instruct for evaluation comparability.)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- config (override via env) ----------------------------------------------
MODEL_REPO="${MODEL_REPO:-Qwen/Qwen2.5-7B-Instruct}"
# Shared path: single H200 node, all 8 GPUs see the same filesystem. Override to
# an NFS / shared volume if your node mounts one.
MODEL_DIR="${MODEL_DIR:-${REPO_ROOT}/models/${MODEL_REPO#*/}}"

# ---- HF token ----------------------------------------------------------------
# Not required for non-gated Qwen2.5-7B, but strongly recommended to avoid
# anonymous rate limits on a fresh pull. Read explicitly so the failure is loud.
if [ -z "${HF_TOKEN:-}" ]; then
  echo "[download_model] NOTE: HF_TOKEN is unset. Qwen2.5-7B is NOT gated, so the"
  echo "                   download will still work, but anonymous rate limits apply."
  echo "                   Get a token at https://huggingface.co/settings/tokens and:"
  echo "                     export HF_TOKEN=hf_xxx   (or put it in ~/.cache/huggingface/token)"
else
  echo "[download_model] HF_TOKEN is set (len=${#HF_TOKEN})."
fi
export HF_HUB_ENABLE_HF_TRANSFER=1   # fast path; harmless if hf-transfer missing

# ---- which env provides huggingface-cli? -------------------------------------
# Prefer the deploy env (it has huggingface-hub[hf-transfer] and is the lighter
# one to stand up). Fall back to the quant env, then to system hf-cli.
HF_CLI=""
for cand in "$REPO_ROOT/envs/deploy/.venv" "$REPO_ROOT/envs/quant/.venv"; do
  if [ -x "$cand/bin/huggingface-cli" ]; then HF_CLI="$cand/bin/huggingface-cli"; break; fi
done
if [ -z "$HF_CLI" ]; then
  if command -v huggingface-cli >/dev/null 2>&1; then
    HF_CLI="$(command -v huggingface-cli)"
    echo "[download_model] Using system huggingface-cli at $HF_CLI" >&2
  else
    echo "[download_model] huggingface-cli not found. Build an env first:" >&2
    echo "                   ./scripts/setup_env.sh   (then re-run this script)" >&2
    exit 1
  fi
fi
echo "[download_model] using: $HF_CLI"

# ---- download ----------------------------------------------------------------
echo
echo "[download_model] $MODEL_REPO -> $MODEL_DIR"
mkdir -p "$MODEL_DIR"
# --local-dir: materialize a real folder (not a cache-only snapshot).
# No --resume without flags: hf-cli resumes automatically on re-run.
"$HF_CLI" download "$MODEL_REPO" --local-dir "$MODEL_DIR"

# ---- verify the files a loader actually needs are present --------------------
echo
echo "[download_model] verifying checkpoint files present:"
ok=1
for f in config.json tokenizer_config.json generation_config.json; do
  if [ -f "$MODEL_DIR/$f" ]; then
    echo "  [ok] $f"
  else
    echo "  [MISSING] $f" >&2
    ok=0
  fi
done
shopt -s nullglob
safetensors=( "$MODEL_DIR"/*.safetensors )
if [ "${#safetensors[@]}" -gt 0 ]; then
  echo "  [ok] ${#safetensors[@]} *.safetensors shard(s)"
else
  echo "  [MISSING] *.safetensors weights" >&2
  ok=0
fi
shopt -u nullglob

if [ "$ok" -ne 1 ]; then
  echo "[download_model] FAILED: checkpoint incomplete. Re-run; downloads resume." >&2
  exit 1
fi

# ---- verify architecture (declarative-deploy precondition) -------------------
echo
echo "[download_model] verifying architecture (declarative-deploy precondition):"
python3 - "$MODEL_DIR/config.json" <<'PY' 2>/dev/null || true
import json, sys
cfg = json.load(open(sys.argv[1]))
archs = cfg.get("architectures", [])
print("  architectures:", archs)
if "Qwen2ForCausalLM" in archs:
    print("  OK: Qwen2ForCausalLM is native to transformers/vLLM -> no --trust-remote-code needed")
else:
    print(f"  WARN: unexpected architectures {archs} (may need trust_remote_code or be a multimodal variant)")
PY

echo
echo "[download_model] Done."
echo "  Path: $MODEL_DIR"
echo "  Use:  set MODEL_DIR above as the load path in recipes and as <model> in 'vllm serve'."
echo "        vLLM reads config.json/tokenizer_config.json/generation_config.json automatically"
echo "        (Qwen2.5 is a native transformers architecture -> no --trust-remote-code needed)."

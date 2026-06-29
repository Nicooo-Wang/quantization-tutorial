#!/usr/bin/env bash
# course/m3-tuning-eval/scripts/download_model.sh
# 下 M3 所需基线模型（Qwen2.5-7B-Instruct + 0.5B）到本模块根 ./models/（双子项目共享）。
# 复用共享缓存免重下（同卷硬链，跨卷拷贝）：
#   MODEL_CACHE=<repo>/models bash scripts/download_model.sh
set -euo pipefail

MODULE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # scripts/.. = m3 根
MODELS_DIR="${MODULE}/models"
mkdir -p "$MODELS_DIR"

REPOS=("Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-0.5B-Instruct")

if [ -n "${HF_TOKEN:-}" ]; then echo "[download_model] HF_TOKEN set (len=${#HF_TOKEN})."; fi
export HF_HUB_ENABLE_HF_TRANSFER=1   # 装了 hf-transfer 时走快路径

# M3 双 env：hf CLI 在子项目 .venv（quant env 装了 huggingface-hub + hf-transfer）；m3 根无 .venv
HF_CLI=""
if [ -x "${MODULE}/steps/quant/.venv/bin/hf" ]; then HF_CLI="${MODULE}/steps/quant/.venv/bin/hf"
elif [ -x "${MODULE}/steps/vllm/.venv/bin/hf" ]; then HF_CLI="${MODULE}/steps/vllm/.venv/bin/hf"
elif command -v hf >/dev/null 2>&1; then HF_CLI="$(command -v hf)"
else echo "[download_model] 未找到 hf CLI。先在子项目内 uv sync（steps/quant 或 steps/vllm）。" >&2; exit 1; fi
echo "[download_model] using: $HF_CLI"

CACHE="${MODEL_CACHE:-}"
for repo in "${REPOS[@]}"; do
  name="${repo#*/}"
  dst="${MODELS_DIR}/${name}"
  if [ -e "${dst}/config.json" ]; then
    echo "[download_model] 已存在，跳过：${dst}"
  elif [ -n "$CACHE" ] && [ -d "${CACHE}/${name}" ]; then
    echo "[download_model] 复用缓存 ${CACHE}/${name} -> ${dst}（硬链，跨卷回退拷贝）"
    cp -rl "${CACHE}/${name}" "$dst" 2>/dev/null || cp -r "${CACHE}/${name}" "$dst"
  else
    echo "[download_model] 下载 ${repo} -> ${dst}"
    "$HF_CLI" download "$repo" --local-dir "$dst"
  fi
  for f in config.json tokenizer_config.json; do
    [ -f "${dst}/${f}" ] || { echo "[download_model] FAILED: 缺 ${dst}/${f}" >&2; exit 1; }
  done
  echo "  ✓ ${name}: $(du -sh "$dst" 2>/dev/null | cut -f1)"
done
echo "[download_model] Done. 模型在 ${MODELS_DIR}/"

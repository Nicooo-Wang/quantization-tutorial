#!/usr/bin/env bash
# 下 Qwen2.5-7B-Instruct + Qwen2.5-0.5B-Instruct 到本模块根 ./models/
# - 0.5B 兜底：FP16 即可 vllm 部署，M4 逻辑层（L2）独立可跑，不依赖任何量化产物
# - 7B：作 FP16 压测基线（s3）+ 供 M2/M3 量化引用（M4 L3 跨模块读 M2/M3 out/ 的量化产物）
set -euo pipefail
MODULE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODULE_ROOT}/models"
mkdir -p "${MODEL_DIR}"

# hf CLI 优先本模块 venv（单 env，模块根 .venv）；否则系统 hf / huggingface-cli
if [[ -x "${MODULE_ROOT}/.venv/bin/hf" ]]; then
  HF="${MODULE_ROOT}/.venv/bin/hf"
else
  HF="$(command -v hf || command -v huggingface-cli || true)"
fi
if [[ -z "${HF}" ]]; then
  echo "[error] 找不到 hf CLI——先 cd ${MODULE_ROOT} && uv sync" >&2
  exit 1
fi
echo "用 hf CLI: ${HF}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

# 复用缓存（同卷硬链，免重下）：MODEL_CACHE=<repo>/models bash scripts/download_model.sh
if [[ -n "${MODEL_CACHE:-}" && -d "${MODEL_CACHE}" ]]; then
  for m in Qwen2.5-7B-Instruct Qwen2.5-0.5B-Instruct; do
    src="${MODEL_CACHE}/$(basename ${m})"
    dst="${MODEL_DIR}/$(basename ${m})"
    if [[ -d "${src}" && ! -e "${dst}" ]]; then
      ln "${src}" "${dst}" 2>/dev/null || cp -r "${src}" "${dst}"
      echo "复用缓存: ${dst}"
    fi
  done
fi

for m in Qwen2.5-7B-Instruct Qwen2.5-0.5B-Instruct; do
  dst="${MODEL_DIR}/$(basename ${m})"
  if [[ ! -d "${dst}" ]]; then
    echo "下载 ${m} → ${dst}"
    "${HF}" download "${m}" --local-dir "${dst}"
  else
    echo "已存在: ${dst}"
  fi
done
echo "完成。models/: $(ls "${MODEL_DIR}")"

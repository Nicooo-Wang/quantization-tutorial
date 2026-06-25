# M2 — H200×8 端到端量化流水线（FP8 / AWQ / SmoothQuant）

对应 [OUTLINE.md](../../../OUTLINE.md) 模块 2。本课用 llm-compressor 把 Qwen2.5-7B-Instruct 量化为三种格式，产出可被 vLLM 加载的模型。

## Setup（本目录自包含，全部命令在此完成）

1. **建 env**（在本目录下，一次性）：
   ```bash
   cd course/m2-quant-pipeline
   uv sync            # 读 uv.lock，建 .venv 并装好依赖（torch cu128 / llmcompressor / transformers v5 …）
   ```
2. **拉基线模型**（7B + 0.5B，下到本目录 `./models/`；本模块自包含）：
   ```bash
   bash scripts/download_model.sh
   # 已在别处下过？复用免重下（同卷硬链）：
   # MODEL_CACHE=<repo>/models bash scripts/download_model.sh
   ```
3. **校准数据**（AWQ / SmoothQuant 需要；FP8 不需要）：每个需要校准的 notebook 首部都有这 cell，按提示跑即可（在下面的 env 里）。

## Steps（每个都是一个自包含 notebook，含填空 + ipytest + tiny 验证 + H200 执行）

- `steps/s1_fp8_quant.ipynb` — FP8 W8A8（`QuantizationModifier(scheme="FP8_DYNAMIC")`，无需校准，H200 首选）
- `steps/s2_awq_quant.ipynb` — AWQ W4A16（llm-compressor 统一路径；附 AutoAWQ 遗留对照）
- `steps/s3_smoothquant.ipynb` — SmoothQuant W8A8（两段式：SmoothQuant + GPTQ），末尾含三方法产物对比 finale

跑法：在本目录下 `uv run jupyter lab` 打开 notebook；按序填空 → 跑 ipytest 测试 cell → 跑 tiny 验证 cell → 跑 H200 执行 cell。

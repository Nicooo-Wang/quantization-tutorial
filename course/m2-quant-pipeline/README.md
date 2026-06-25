# M2 — H200×8 端到端量化流水线（FP8 / AWQ / SmoothQuant）

对应 [OUTLINE.md](../../../OUTLINE.md) 模块 2。本课用 llm-compressor 把 Qwen2.5-7B-Instruct 量化为三种格式，产出可被 vLLM 加载的模型。

## Setup 前置（开始 steps 前完成）

1. **环境**（一次性，两个隔离 uv env）：
   ```bash
   bash scripts/setup_env.sh
   ```
   产出 `envs/quant/.venv`（量化用）与 `envs/deploy/.venv`（部署用）。本模块全程用 `envs/quant`。
2. **拉基线模型**（H200×8 单节点，8 卡共享同一文件系统，无需每卡各下一份）：
   ```bash
   HF_TOKEN=xxxx bash scripts/download_model.sh
   # 默认下到 <repo>/models/Qwen2.5-7B-Instruct（脚本从 env 读 MODEL_REPO / MODEL_DIR，不接受位置参数；
   #  想换路径：MODEL_DIR=/your/path HF_TOKEN=xxxx bash scripts/download_model.sh）
   ```
3. **校准数据**（AWQ / SmoothQuant 需要；FP8 不需要）：
   在 quant env 里跑（每个需要校准的 notebook 首部都有这 cell）：
   ```python
   from datasets import load_dataset
   from transformers import AutoTokenizer
   tok = AutoTokenizer.from_pretrained("models/Qwen2.5-7B-Instruct")  # 默认下载路径，repo 根相对（notebook 从 repo 根启动）
   ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").shuffle(seed=42).select(range(512))
   calib = [tok(d["text"], return_tensors="pt").input_ids[0][:2048] for d in ds if d["text"].strip()]
   ```

## Steps（每个都是一个自包含 notebook，含填空 + ipytest + tiny 验证 + H200 执行）

- `steps/s1_fp8_quant.ipynb` — FP8 W8A8（`QuantizationModifier(scheme="FP8_DYNAMIC")`，无需校准，H200 首选）
- `steps/s2_awq_quant.ipynb` — AWQ W4A16（llm-compressor 统一路径；附 AutoAWQ 遗留对照）
- `steps/s3_smoothquant.ipynb` — SmoothQuant W8A8（两段式：SmoothQuant + GPTQ），末尾含三方法产物对比 finale

跑法：`uv run --directory envs/quant jupyter lab` 打开 notebook；按序填空 → 跑 ipytest 测试 cell → 跑 tiny 验证 cell → 跑 H200 执行 cell。

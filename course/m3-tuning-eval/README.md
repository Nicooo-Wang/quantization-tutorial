# M3 — 精度调优与评测（layer fallback 核心，SmoothQuant 范例）

对应 [OUTLINE.md](../../../OUTLINE.md) 模块 3。本课灵魂模块：以 SmoothQuant W8A8 为范例，走通
「敏感层识别 → ignore 回退 → mixed-precision → Pareto 调优 → 多维评测」全流程，产出「最少回退 → 最大精度」结果。
（同一流程适用于 AWQ / FP8。）

## 为什么有两个子项目（双 env）

量化要 `llmcompressor`（transformers v5），下游评测要 `vLLM`（wheel 自带一套锁定 transformers）——
一个 env 装不下（版本互锁）。本模块在 `steps/` 下划两个子目录，**各是一个独立 uv 项目**（自带 env + notebook）：

- `steps/quant/` — s1–s6（量化 + PPL）。env 含 llmcompressor + transformers v5 + matplotlib。
- `steps/vllm/` — s7（下游评测 + 性能/显存）。env 含 vLLM + lm_eval。

`models/`、`out/` 在本模块根共享，两子项目都读写。

## Setup（在本模块目录下完成）

1. **建两个 env**（各 uv sync；vLLM 子项目 wheel 很大，首次需几分钟）：
   ```bash
   cd course/m3-tuning-eval/steps/quant && uv sync    # 量化 env
   cd course/m3-tuning-eval/steps/vllm && uv sync     # vLLM env
   ```
2. **给 vLLM env 装 lm_eval[vllm] extra**（坑：`uv pip` 不认项目 venv，必须显式 `--python`）：
   ```bash
   cd course/m3-tuning-eval/steps/vllm
   uv pip install --python ./.venv/bin/python 'lm_eval[vllm]'
   ```
   ⚠ **重要坑（已实测）**：`lm_eval` extra **不进 `uv.lock`**（与 vLLM wheel 自带的 transformers 可能互锁，故旁路）。**每次 `uv sync`（步骤 1）都会把它当作偏离 lock 的包卸载掉**——若 s7 报 `No module named 'lm_eval'`，重跑上面命令（幂等）。正确顺序：**先 `uv sync`，再装 extra**；之后只要再 sync 过，就要重装。vLLM env 的冒烟/检查一律用 `./.venv/bin/python`，不要用 `uv run`（会触发 sync）。
3. **拉基线模型**（7B + 0.5B 下到本模块根 `./models/`，两子项目共享）：
   ```bash
   cd course/m3-tuning-eval
   bash scripts/download_model.sh
   # 已在别处下过？复用免重下（同卷硬链）：
   # MODEL_CACHE=<repo>/models bash scripts/download_model.sh
   ```

## Steps

- `steps/quant/s1_sensitive_layers.ipynb` — 3.1 敏感层识别（profiling + 逐层量化测 PPL）
- `steps/quant/s2_what_not_to_quantize.ipynb` — 3.2 不该量化什么（工业经验法则）
- `steps/quant/s3_ignore_syntax.ipynb` — 3.3 ignore 语法（精确 + `re:` 正则）
- `steps/quant/s4_mixed_precision.ipynb` — 3.4 mixed-precision config_groups
- `steps/quant/s5_pareto_tuning.ipynb` — 3.5 Pareto 调优（逐步 ignore → PPL 恢复曲线 → 拐点）
- `steps/quant/s6_group_size.ipynb` — 3.6 group_size 调参
- `steps/vllm/s7_evaluation.ipynb` — 3.7 评测方法论（下游 lm_eval + 性能/显存；PPL 数据从 quant 产物读）

## 跑法

- s1–s6：`cd course/m3-tuning-eval && uv run --directory steps/quant jupyter lab`
- s7：`cd course/m3-tuning-eval && uv run --directory steps/vllm jupyter lab`

按序填空 → 跑 ipytest 测试 cell（L1）→ 跑 tiny 验证（L2）→ 跑 H200 执行（L3）。

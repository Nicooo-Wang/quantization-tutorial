# M4 — vLLM 声明式部署与端到端闭环

对应 [OUTLINE.md](../../../OUTLINE.md) 模块 4。课程终点：把 M2/M3 产出的 compressed-tensors 量化模型部署到 H200×8 的 vLLM，走通「加载识别 → 多卡部署 → 压测 → 报错排查 → 端到端闭环」。引擎只用 vLLM，**声明式部署（无需改模型代码）**。

## Setup（双前置）

### 前置 1：建本模块 env（单 env）

```bash
cd course/m4-deploy-loop && uv sync
```
- env = vllm + ipytest + nbconvert + jupyter（vLLM wheel 自带 torch/transformers，**不另装 torch**）。
- **不装 lm_eval**（M4 用 `vllm bench serve` + `/metrics` 压测，不跑下游评测——评测属 M3 s7）；**不装 huggingface_hub**（Hub 发布已砍，s5 仅提示不实操）。env 干净无 extra 旁路坑。
- 冒烟/检查用 `./.venv/bin/python` 比 `uv run` 稳（`uv run` 触发 sync 检查，vllm env 首次可能数分钟）。

### 前置 2：拉模型 + （L3 真部署需）跨模块量化产物

```bash
cd course/m4-deploy-loop
bash scripts/download_model.sh    # 下 7B + 0.5B 到 ./models/
# 已在别处下过？复用免重下（同卷硬链）：
# MODEL_CACHE=<repo>/models bash scripts/download_model.sh
```
- **0.5B 兜底**：s1–s5 的 L2 流程验证用 0.5B（FP16 即可 vllm 部署），**不依赖任何量化产物**——M4 逻辑层独立可跑。
- **7B 量化产物（L3 真部署）**：s1–s5 的 L3 真部署需要 M2/M3 产出的 7B 量化模型。notebook 通过 `REPO_COURSE = MODULE_ROOT.parent` 引用兄弟模块 out/：
  - `course/m2-quant-pipeline/out/qwen7b-fp8`（及 `qwen7b-awq` / `qwen7b-smoothquant`）
  - `course/m3-tuning-eval/out/`（调优后 mixed-precision 产物）
  - **需先跑完 M2（FP8/AWQ/SmoothQuant）+ M3（调优）**。缺产物时 notebook 友好报错并降级到 0.5B 路径。
- **FP16 基线**（s3 压测对比）：download 拉的 7B 本身。

## Steps

- `steps/s1_load_quantized.ipynb` — 4.1 加载识别 + quantization_config 构造 + 声明式边界
- `steps/s2_multigpu_deploy.ipynb` — 4.2 H200×8 多卡 TP 部署
- `steps/s3_benchmark.ipynb` — 4.3 性能压测（bench + /metrics）
- `steps/s4_error_cheatsheet.ipynb` — 4.4 常见报错诊断
- `steps/s5_e2e_loop.ipynb` — 4.6 端到端闭环 mini 项目 + 4.7 QAT/NVFP4 前瞻（finale）

## 跑法

`cd course/m4-deploy-loop && uv run jupyter lab`，按序填空 → 跑 ipytest 测试 cell（L1）→ config 结构验证（L2）→ GPU vllm 真跑（L3，需 GPU）。

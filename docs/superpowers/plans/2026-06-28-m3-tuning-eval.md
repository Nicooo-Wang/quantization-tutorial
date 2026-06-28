# M3 精度调优与评测（layer fallback 核心）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `course/m3-tuning-eval/` 下产出 7 个自包含 notebook（s1–s7，对应 OUTLINE 3.1–3.7），以 SmoothQuant W8A8 为范例走通「敏感层识别 → ignore 回退 → mixed-precision → Pareto 调优 → 多维评测」全流程，并通过 dev-module.js 全流程（5 阶段）双重验收。

**Architecture:** M3 是模块内**双 env**模块——`steps/quant/`（s1–s6，llmcompressor + transformers v5 + matplotlib）与 `steps/vllm/`（s7，vLLM + lm_eval）各是一个独立 uv 子项目，因为 llmcompressor（要 transformers v5）与 vLLM（wheel 自带锁定 transformers）一个 env 装不下。`models/`、`out/` 在 m3 根共享。脚手架（env/scripts/README）由本 plan 的 T1–T3 先建；notebook 内容（教学法 markdown + 填空 + L1/L2/L3）由 T5 的 dev-module.js workflow 的 dev-agent 按 spec 写（不在本 plan 逐 cell 写）；T4 适配 dev-module.js 让它能跑双 env。

**Tech Stack:** torch 2.11.0+cu128（cu128 index）、transformers v5（5.10.1 实测，llmcompressor 0.12 必须）、llmcompressor 0.12.0、compressed-tensors 0.17.1、vLLM PyPI 0.x（≥0.11）、lm_eval[vllm] extra、matplotlib（Pareto/敏感度图）、ipytest/nbval/nbconvert/jupyter、CUDA 12.8 / 驱动 570、Python 3.11、uv 0.8.x。

## Global Constraints

（来自 spec §2/§4 + OUTLINE 版本基线；每个 task 的需求隐含包含本节）

- **分支**：从 `main` 开 feature 分支 `m3-tuning-eval` 开发；不在 main 直接提交。当前会话可能在 `m2-pedagogy-redesign`——先 `git checkout main && git pull` 再开分支。
- **双 env 不跨 env**：`steps/quant/` 与 `steps/vllm/` 各是独立 uv 项目，**不跨 env 调 subprocess**。s1–s6 在 quant 子项目跑、s7 在 vllm 子项目跑；s7 的 PPL 数据从 quant 产物（`out/`）读，不在 s7 重跑 PPL、不跨 env。
- **M3 自包含**：不跨模块读 `course/m2-quant-pipeline/out/`。s7 finale 用 M3 自身量化链路产物（FP16 基线 + SmoothQuant 全量/调优后）。三方法横向对比作为方法论延伸——dev-agent 可选地在 quant env 重产 FP8（一步 cast，快），不强制 AWQ（慢）。
- **版本基线（pyproject 下限，实测锁定见上）**：`torch>=2.10,<2.12` + cu128 index；`transformers>=5.0`；`llmcompressor>=0.9.0`；`compressed-tensors>=0.10`；`vllm>=0.11`（PyPI 0.x，**不是** CalVer `>=25.0`）。Python `>=3.10,<3.13`（锁定 3.11）。
- **不另装 torch（vllm 子项目）**：vLLM wheel 自带 CUDA 12.8 配对 torch/transformers；vllm 子项目 pyproject **不**单独列 torch（OUTLINE 坑#2）。
- **lm_eval[vllm] extra 坑**：核心包不含 model backend，必须 `cd course/m3-tuning-eval/steps/vllm && uv pip install --python ./.venv/bin/python 'lm_eval[vllm]'`。`uv pip` 只认激活 env / `--python <path>`，**不认项目 venv**（OUTLINE 坑#4）。该 extra **不进 uv.lock**（uv pip install 装的），README 须说明学员手动装。
- **环境检查不用 uv run（vllm env）**：OUTLINE 坑#3——`uv run` 每次触发 sync 检查，vLLM env（+ 不在 lock 的 lm_eval extra）可能触发数分钟重装。vllm env 的冒烟/检查一律用 `./.venv/bin/python -c "..."`。
- **提交规则**：notebook commit 前 `jupyter nbconvert --clear-output --inplace` + 删 cell `metadata.execution`（与 M1/M2 一致）；改 pyproject 后必须重生成 `uv.lock` 并一并 commit。
- **编辑范围**：T1–T3 只建 `course/m3-tuning-eval/` 下文件；T4 改 `workflows/dev-module.js`（向后兼容）；T5 的 dev-agent 只改 `course/m3-tuning-eval/steps/**/*.ipynb`，严禁改 pyproject/uv.lock/scripts/README/workflows。
- **教学法（套 M2 验证过的 lesson）**：每 notebook 顶部「## 学完应能讲清」清单（3–5 问）+ 摸一摸 cell（实例化打印对象）+ 端到端 why + 填空 why 引导（每个填空 docstring 加「为什么这么设计」段）。代码逻辑（填空实现 + L2/L3）由 dev-agent 写 + reviewer 执行验证 + 学员代理理解探测。
- **_find_module_root 适配（M3 关键）**：notebook 在 `steps/quant/` 或 `steps/vllm/` 子目录，模块根判据用**「`scripts/` + `steps/`」**（子项目有 pyproject.toml 但无 steps/，故不能用 CONV 原版的 pyproject 判据，否则误命中子项目）。T4 把这段代码注入 dev-agent prompt。

---

## File Structure

```
course/m3-tuning-eval/
  README.md                          # T1：双 env setup（两子项目 sync + lm_eval[vllm] 坑 + 模型共享）
  .gitignore                         # T1：忽略 models/ out/ steps/*/.venv/ steps/*/__pycache__/
  scripts/
    download_model.sh                # T1：下 7B+0.5B 到 m3 根 ./models/（两子项目共享；hf CLI 在 steps/quant/.venv）
  models/                            # 共享（gitignore）；download_model.sh 下此
  out/                               # 共享产物（调优中间模型、PPL 表、评测结果；gitignore）
  steps/
    quant/                           # T2：量化+PPL 子项目（独立 uv 项目，s1-s6）
      pyproject.toml                 # llmcompressor + transformers v5 + datasets + matplotlib + torch cu128 + ipytest/nbval/nbconvert/jupyter
      uv.lock                        # T2 uv sync 生成，commit
      .venv/                         # gitignore
      s1_sensitive_layers.ipynb ... s6_group_size.ipynb   # T5 dev-agent 写（6 个）
    vllm/                            # T3：downstream 评测子项目（独立 uv 项目，s7）
      pyproject.toml                 # vllm>=0.11 + ipytest/nbconvert/jupyter（vLLM wheel 自带 torch/transformers）
      uv.lock                        # T3 uv sync 生成，commit
      .venv/                         # gitignore（lm_eval[vllm] extra 装此，不进 lock）
      s7_evaluation.ipynb            # T5 dev-agent 写
workflows/dev-module.js              # T4：加 ENVS 双 env 支持（向后兼容 M1/M2/M4）
```

每文件职责：`scripts/download_model.sh` 模型唯一入口；`steps/quant/pyproject.toml` + `steps/vllm/pyproject.toml` 各 env 唯一事实来源（uv.lock 是复现性来源）；`workflows/dev-module.js` 全模块通用开发 workflow（T4 加双 env 分支）。

---

### Task 1: M3 模块根脚手架（README + .gitignore + download_model.sh）

**Files:**
- Create: `course/m3-tuning-eval/README.md`
- Create: `course/m3-tuning-eval/.gitignore`
- Create: `course/m3-tuning-eval/scripts/download_model.sh`（+ `chmod +x`）

**Interfaces:**
- Produces: `download_model.sh`（hf CLI 优先 `steps/quant/.venv/bin/hf`）、共享 `models/`（被 T2 冒烟后下模型 + T5 notebook 读）、`out/`（被 T5 notebook 写 PPL 表/调优产物）。

- [ ] **Step 1: 从 main 开 feature 分支**

```bash
git checkout main && git pull
git checkout -b m3-tuning-eval
```
Expected: 在 `m3-tuning-eval` 分支（`git branch --show-current` 打印 `m3-tuning-eval`）。

- [ ] **Step 2: 建目录结构**

```bash
mkdir -p course/m3-tuning-eval/scripts course/m3-tuning-eval/steps/quant course/m3-tuning-eval/steps/vllm
```
Expected: 三个目录创建（`ls course/m3-tuning-eval` 见 `scripts steps`）。

- [ ] **Step 3: 写 `.gitignore`**

Create `course/m3-tuning-eval/.gitignore`：
```gitignore
# 共享产物（两子项目都写）
models/
out/

# 双子项目各自的 venv（不进 repo；lm_eval[vllm] extra 装在 vllm/.venv，不锁）
steps/*/.venv/
steps/*/__pycache__/

# ipython / jupyter
.ipynb_checkpoints/
```

- [ ] **Step 4: 写 `scripts/download_model.sh`**

Create `course/m3-tuning-eval/scripts/download_model.sh`（M2 模板 + M3 适配：hf CLI 在子项目 `steps/quant/.venv`，因为 m3 根无 .venv）：
```bash
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
```
Then: `chmod +x course/m3-tuning-eval/scripts/download_model.sh`

- [ ] **Step 5: 写 `README.md`**

Create `course/m3-tuning-eval/README.md`：
```markdown
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
```

- [ ] **Step 6: 验证脚本可执行 + 语法检查**

Run: `bash -n course/m3-tuning-eval/scripts/download_model.sh && echo "syntax ok"`
Expected: 打印 `syntax ok`（`-n` 只解析语法不执行；此时 steps/quant/.venv 还没建，真跑会在 T2 后）。
Run: `ls -l course/m3-tuning-eval/scripts/download_model.sh`
Expected: 权限含 `x`（可执行）。

- [ ] **Step 7: Commit**

```bash
git add course/m3-tuning-eval/README.md course/m3-tuning-eval/.gitignore course/m3-tuning-eval/scripts/download_model.sh
git commit -m "feat(m3): module-root scaffold (README + .gitignore + download_model.sh for dual-env subprojects)"
```

---

### Task 2: steps/quant 子项目（pyproject + uv.lock + 冒烟）

**Files:**
- Create: `course/m3-tuning-eval/steps/quant/pyproject.toml`
- Generated: `course/m3-tuning-eval/steps/quant/uv.lock`（uv sync 生成，commit）
- Generated: `course/m3-tuning-eval/steps/quant/.venv/`（gitignore）

**Interfaces:**
- Consumes: Task 1 的目录结构。
- Produces: `steps/quant/.venv`（hf CLI 供 download_model.sh；import llmcompressor/transformers/matplotlib 供 T5 notebook + reviewer 执行验证 s1–s6）。

- [ ] **Step 1: 写 `steps/quant/pyproject.toml`**

Create `course/m3-tuning-eval/steps/quant/pyproject.toml`（M2 模板 + matplotlib；M3 量化+PPL 子项目）：
```toml
[project]
name = "m3-tuning-eval-quant"
version = "0.1.0"
description = "M3 量化+PPL env（llmcompressor layer fallback 重新量化 + PPL forward + matplotlib Pareto/敏感度图）。模块内双子项目之一。"
requires-python = ">=3.10,<3.13"
# torch 必须是 CUDA 12.8 (cu128) build 以匹配本节点驱动 (570.x / CUDA 12.8)：
# PyPI 默认 torch 是 cu130，在 12.8 驱动上 CUDA 初始化失败。经 cu128 index 取 torch 2.11.0+cu128。
dependencies = [
    "torch>=2.10,<2.12",       # cu128 build（见 [tool.uv]）；llmcompressor 0.12 要 torch [2.10,2.12]
    "llmcompressor>=0.9.0",    # 0.12 要 transformers v5；>=0.7 支持 mixed-precision config_groups
    "compressed-tensors>=0.10",
    "accelerate>=1.0",         # oneshot device_map / 多卡加载
    "datasets>=2.20",          # 校准集（wikitext-2）
    "transformers>=5.0",       # llmcompressor 0.12 requires transformers v5
    "safetensors>=0.4",
    "sentencepiece>=0.2",
    "huggingface-hub>=0.26",   # hf CLI（download_model.sh 用子项目 .venv 的 hf）
    "hf-transfer>=0.1",        # 快速下载；脚本设 HF_HUB_ENABLE_HF_TRANSFER=1
    "matplotlib>=3.8",         # M3 新增（M2 不需要）：Pareto 曲线 + 逐层敏感度图
    "ipytest>=0.14",           # notebook 内跑 pytest（L1）
    "nbval>=0.11",             # pytest --nbval 输出回归
    "nbconvert>=7.0",          # --execute / --clear-output
    "jupyter>=1.0",            # 学员交互跑 notebook
]

[tool.uv]
# torch/triton 必须从 cu128 index 取以匹配 570.x 驱动
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
triton = { index = "pytorch-cu128" }
```

- [ ] **Step 2: uv sync 生成 uv.lock + .venv**

Run: `cd course/m3-tuning-eval/steps/quant && uv sync`
Expected: 解析依赖、下 torch cu128（约 2.5GB）+ llmcompressor 全量传递依赖，建 `.venv` 与 `uv.lock`。首次较慢（数分钟）。退出码 0。
> 若报 CUDA/torch 相关错，确认 `[tool.uv.sources]` 的 cu128 index 正确（M2 同款）。

- [ ] **Step 3: 冒烟——关键 import 能过**

Run: `cd course/m3-tuning-eval/steps/quant && uv run python -c "import llmcompressor,transformers,matplotlib,compressed_tensors,accelerate,datasets; print('ok', llmcompressor.__version__, transformers.__version__)"`
Expected: 打印 `ok 0.12.x 5.x`（quant env 在 uv.lock 内，用 `uv run` 不触发额外 sync；若仍触发重装，改用 `./.venv/bin/python -c "..."`）。

- [ ] **Step 4: 下模型（download_model.sh 真跑，hf CLI 现在在 steps/quant/.venv）**

Run: `cd course/m3-tuning-eval && bash scripts/download_model.sh`
Expected: 打印 `[download_model] using: .../steps/quant/.venv/bin/hf`，下 7B + 0.5B 到 `./models/`，末尾 `Done. 模型在 .../models/`。
> 已在别处下过则 `MODEL_CACHE=<repo>/models bash scripts/download_model.sh` 复用。

- [ ] **Step 5: Commit pyproject + uv.lock（不 commit .venv）**

```bash
git add course/m3-tuning-eval/steps/quant/pyproject.toml course/m3-tuning-eval/steps/quant/uv.lock
git commit -m "feat(m3): steps/quant subproject env (llmcompressor v5 + matplotlib + torch cu128)"
```
Verify: `git status course/m3-tuning-eval/steps/quant/` —— `.venv/` 不在待提交（.gitignore 生效）。

---

### Task 3: steps/vllm 子项目（pyproject + uv.lock + lm_eval[vllm] + 冒烟）

**Files:**
- Create: `course/m3-tuning-eval/steps/vllm/pyproject.toml`
- Generated: `course/m3-tuning-eval/steps/vllm/uv.lock`（commit）
- Generated: `course/m3-tuning-eval/steps/vllm/.venv/`（gitignore；lm_eval[vllm] extra 装此，**不进 lock**）

**Interfaces:**
- Consumes: Task 1 目录结构。
- Produces: `steps/vllm/.venv`（vllm + lm_eval；import vllm/lm_eval 供 T5 notebook s7 + reviewer 执行验证 s7）。

- [ ] **Step 1: 写 `steps/vllm/pyproject.toml`**

Create `course/m3-tuning-eval/steps/vllm/pyproject.toml`（vLLM wheel 自带 torch/transformers，不另装；lm_eval extra 不进 lock）：
```toml
[project]
name = "m3-tuning-eval-vllm"
version = "0.1.0"
description = "M3 vLLM 评测 env（downstream lm_eval + 性能/显存）。vLLM wheel 自带并锁定 CUDA12.8 配对的 torch/transformers，不另装。模块内双子项目之一。"
requires-python = ">=3.10,<3.13"
# 注意：lm_eval[vllm] extra 不进本 uv.lock —— 它由 uv pip install 装（见 README + OUTLINE 坑#4），
# 学员/CI 手动 `uv pip install --python ./.venv/bin/python 'lm_eval[vllm]'`。原因：extra 与 vllm wheel
# 自带 transformers 可能互锁，放 uv.lock 会触发版本冲突；OUTLINE 附录 B open question #5 待实测统一。
dependencies = [
    "vllm>=0.11",            # PyPI 0.x（非 CalVer）；wheel 自带 torch/transformers，不要另装 torch（OUTLINE 坑#2）
    "ipytest>=0.14",         # notebook 内 pytest（s7 L1）
    "nbconvert>=7.0",        # --execute / --clear-output
    "jupyter>=1.0",          # 学员交互跑 s7
]
```

- [ ] **Step 2: uv sync 生成 uv.lock + .venv**

Run: `cd course/m3-tuning-eval/steps/vllm && uv sync`
Expected: 下 vLLM wheel（很大，含 torch + transformers + CUDA 库，首次可能 5–15 分钟）建 `.venv` 与 `uv.lock`。退出码 0。

- [ ] **Step 3: 冒烟——vllm import 能过（用 .venv/bin/python，不走 uv run）**

Run: `cd course/m3-tuning-eval/steps/vllm && ./.venv/bin/python -c "import vllm; print('ok', vllm.__version__)"`
Expected: 打印 `ok 0.x`（OUTLINE 坑#3：vllm env 检查用 `.venv/bin/python`，不用 `uv run` 免触发数分钟重装）。

- [ ] **Step 4: 装 lm_eval[vllm] extra（坑：显式 --python）**

Run:
```bash
cd course/m3-tuning-eval/steps/vllm
uv pip install --python ./.venv/bin/python 'lm_eval[vllm]'
```
Expected: 装 lm_eval + vllm backend extra 到 `./.venv`。退出码 0。
> `uv pip` 未激活时只认 `--python <path>`、不认项目 venv（OUTLINE 坑#4）；漏 `--python` 会装错地方或报错。

- [ ] **Step 5: 冒烟——lm_eval import + vllm backend 可用**

Run: `cd course/m3-tuning-eval/steps/vllm && ./.venv/bin/python -c "import lm_eval; from lm_eval.models.vllm_causallms import VLLM; print('ok', lm_eval.__version__)"`
Expected: 打印 `ok 0.4.x`（确认 lm_eval[vllm] extra 装好、vllm backend 可 import）。
> 若 `from lm_eval.models.vllm_causallms import VLLM` 报 ImportError，说明 extra 没装到位——回 Step 4 重装。

- [ ] **Step 6: Commit pyproject + uv.lock（lm_eval extra 不进 lock，不 commit）**

```bash
git add course/m3-tuning-eval/steps/vllm/pyproject.toml course/m3-tuning-eval/steps/vllm/uv.lock
git commit -m "feat(m3): steps/vllm subproject env (vllm + lm_eval[vllm] extra via uv pip)"
```

---

### Task 4: dev-module.js 双 env 适配（ENVS 支持，向后兼容）

> **背景**：dev-module.js 当前写死单 env——gate 冒烟 `cd course/<module> && uv sync`（M3 根无 pyproject 会失败）、reviewer 执行验证 `cd course/<module> && uv run nbconvert steps/<nb>`（M3 notebook 在子目录 + 根无 pyproject）、EDIT_SCOPE glob `steps/*.ipynb` 不匹配子目录。本 task 加可选 `args.envs`（子项目→env 映射），**无 envs 时（M1/M2/M4）行为完全不变**。

**Files:**
- Modify: `workflows/dev-module.js`（5 处：ARGS 解析 + EDIT_SCOPE + dev prompt + reviewer prompt + gate prompt）

**Interfaces:**
- Consumes: Tasks 1–3 建好的 `steps/quant/` + `steps/vllm/`（.venv + uv.lock）。
- Produces: 改造后的 dev-module.js 支持 `args.envs`，供 T5 调用。

- [ ] **Step 1: ARGS 后加 ENVS 派生 + ENV_HELP/SMOKE_CMD 辅助字符串**

Modify `workflows/dev-module.js`。在 `const MODULE_PATH = ...`（约第 22 行）之后、`const MAX_REVIEW_ROUNDS`（约第 23 行）之前插入：

old_string（定位锚，原文）：
```js
const MODULE_PATH = `course/${MODULE}`
const MAX_REVIEW_ROUNDS = 3
```
new_string：
```js
const MODULE_PATH = `course/${MODULE}`
// M3 双 env：子项目→env 映射。无 envs（M1/M2/M4）时退化为单根 env（向后兼容，行为不变）。
// envs 项形如 { dir: "steps/quant", smoke: "import llmcompressor,transformers,matplotlib" }
const ENVS = (ARGS.envs && ARGS.envs.length) ? ARGS.envs : [{ dir: '', smoke: "print('ok')" }]
const MULTI_ENV = ENVS.length > 1 || (ENVS.length === 1 && ENVS[0].dir)
// 多 env 时注入 dev/reviewer/gate prompt 的运行说明（单 env 时为空，prompt 行为不变）
const ENV_HELP = MULTI_ENV ? `

【本模块是多 env（子项目隔离，重要）】notebook 在 ${MODULE_PATH}/steps/ 的**子目录**里，每个子目录是独立 uv 项目（自带 .venv）：
${ENVS.map(e => `  - ${e.dir}/（uv 项目根 ${MODULE_PATH}/${e.dir}）`).join('\n')}

运行/执行规则：
- 每个 notebook 按其所在子目录跑：\`uv run --directory ${MODULE_PATH}/<子目录> <cmd>\`，或直接 \`${MODULE_PATH}/<子目录>/.venv/bin/python\`。
- 模块根 ${MODULE_PATH} **没有 pyproject.toml**——绝不在模块根跑 \`uv sync\`（报 no project）；只在各子目录 uv sync。
- setup cell 的 _find_module_root 用「scripts/ + steps/」判据找模块根（子项目自己有 pyproject.toml 但无 steps/，故别用 pyproject 判据）：
  \`\`\`python
  import pathlib
  def _find_module_root(start):
      p = pathlib.Path(start).resolve()
      for cand in [p, *p.parents]:
          if (cand / "scripts").is_dir() and (cand / "steps").is_dir():
              return cand
      raise RuntimeError("找不到模块根（含 scripts/ + steps/ 的目录）")
  MODULE_ROOT = _find_module_root(pathlib.Path.cwd())
  MODEL_DIR      = MODULE_ROOT / "models" / "Qwen2.5-7B-Instruct"
  TINY_MODEL_DIR = MODULE_ROOT / "models" / "Qwen2.5-0.5B-Instruct"
  OUT_ROOT       = MODULE_ROOT / "out"; OUT_ROOT.mkdir(parents=True, exist_ok=True)
  \`\`\`
- models/、out/ 在模块根共享（不在子目录），由上面的 MODULE_ROOT 解析。
- vLLM 子项目（若含）：lm_eval[vllm] extra 不在 uv.lock——执行其 notebook 前先 \`cd ${MODULE_PATH}/<vllm子目录> && uv pip install --python ./.venv/bin/python 'lm_eval[vllm]'\`（幂等）；vllm env 的冒烟/检查一律用 \`./.venv/bin/python\`（不用 uv run，会触发数分钟重装）。
` : ''
// gate 冒烟命令：多 env 遍历各子目录 sync + .venv/bin/python 冒烟；单 env 维持原行为
const SMOKE_CMD = MULTI_ENV
  ? ENVS.map(e => `cd ${MODULE_PATH}/${e.dir} && uv sync && ./.venv/bin/python -c "${e.smoke || "print('ok')"}"`).join(' && ')
  : `cd course/${MODULE} && uv sync && uv run python -c "print('ok')"`
const MAX_REVIEW_ROUNDS = 3
```

- [ ] **Step 2: EDIT_SCOPE glob 宽化（允许子目录 notebook）**

old_string：
```js
const EDIT_SCOPE = `编辑范围：只能创建/修改 ${MODULE_PATH}/steps/*.ipynb。严禁改 pyproject.toml、uv.lock、scripts/、workflows/、README.md、NOTEBOOK_CONVENTIONS.md、顶层任何文件。遇到 env/依赖/CLI 问题写进 findings 报告，不要自己改。`
```
new_string：
```js
const EDIT_SCOPE = `编辑范围：只能创建/修改 ${MODULE_PATH}/steps/ 下的 .ipynb（含子目录，如 steps/quant/、steps/vllm/）。严禁改任何 pyproject.toml、uv.lock、scripts/、workflows/、README.md、NOTEBOOK_CONVENTIONS.md、顶层任何文件。遇到 env/依赖/CLI 问题写进 findings 报告，不要自己改。`
```

- [ ] **Step 3: dev-agent prompt 末尾注入 ENV_HELP**

old_string（dev prompt 结尾，约第 101–102 行）：
```js
提交前 \`nbconvert --clear-output\` + 删 cell metadata.execution。
${EDIT_SCOPE}
按 schema 报告。`, { label: '开发专家', phase: '开发', schema: DEV_REPORT })
```
new_string：
```js
提交前 \`nbconvert --clear-output\` + 删 cell metadata.execution。
${EDIT_SCOPE}${ENV_HELP}
按 schema 报告。`, { label: '开发专家', phase: '开发', schema: DEV_REPORT })
```

- [ ] **Step 4: reviewer prompt 注入 ENV_HELP + 执行验证按子目录**

old_string（reviewer prompt 执行验证段，约第 112–117 行）：
```js
**执行验证（代码跑通 gate，必做）**：对每个 notebook，把参考实现注入填空（来源：dev 报告 referenceImpls；**若为空/skipDev（skipDev 模式，notebook 已存在），从 notebook 的 ipytest 测试语义反推每个填空的正确实现注入**），跑：
  cd ${MODULE_PATH} && uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 steps/<nb>.ipynb
L1(ipytest)+L2(tiny) 必过；L3(真模型) 有 GPU 必过、无 GPU 记 skip（算过）。报 execVerification（每 notebook l1/l2/l3 + passed + allPassed）。
verdict=pass 仅当：无 critical/major findings **AND** execVerification.allPassed=true。
按 schema 报 verdict/execVerification/findings/summary。
${EDIT_SCOPE}（reviewer 只审 + 跑验证，**不改 notebook**——审改分离；改由 dev 在下一 agent 做）`,
```
new_string：
```js
**执行验证（代码跑通 gate，必做）**：对每个 notebook，把参考实现注入填空（来源：dev 报告 referenceImpls；**若为空/skipDev（skipDev 模式，notebook 已存在），从 notebook 的 ipytest 测试语义反推每个填空的正确实现注入**），跑 nbconvert 全量执行：
${MULTI_ENV
  ? `  多 env：按 notebook 所在子目录跑——\`uv run --directory ${MODULE_PATH}/<子目录> jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 <子目录>/<nb>.ipynb\`。含 vLLM 子项目的先 \`cd ${MODULE_PATH}/<vllm子目录> && uv pip install --python ./.venv/bin/python 'lm_eval[vllm]'\` 再跑（幂等）。`
  : `  cd ${MODULE_PATH} && uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 steps/<nb>.ipynb`}
L1(ipytest)+L2(tiny) 必过；L3(真模型) 有 GPU 必过、无 GPU 记 skip（算过）。报 execVerification（每 notebook l1/l2/l3 + passed + allPassed）。
verdict=pass 仅当：无 critical/major findings **AND** execVerification.allPassed=true。
按 schema 报 verdict/execVerification/findings/summary。
${EDIT_SCOPE}${ENV_HELP}（reviewer 只审 + 跑验证，**不改 notebook**——审改分离；改由 dev 在下一 agent 做）`,
```

- [ ] **Step 5: dev 修 prompt 注入 ENV_HELP**

old_string（reviewer loop 内 dev 修 prompt，约第 122–125 行）：
```js
findings：${JSON.stringify(review)}。改掉所有 critical/major（合理采纳 minor）。改完对受影响 notebook 注入参考实现重跑 L1/L2 验证（\`cd ${MODULE_PATH} && uv sync && uv run jupyter nbconvert --execute...\`）。
${EDIT_SCOPE}`,
```
new_string：
```js
findings：${JSON.stringify(review)}。改掉所有 critical/major（合理采纳 minor）。改完对受影响 notebook 注入参考实现重跑 L1/L2 验证（\`uv run --directory ${MODULE_PATH}/<子目录> jupyter nbconvert --execute...\`）。
${EDIT_SCOPE}${ENV_HELP}`,
```

- [ ] **Step 6: gate prompt 冒烟用 SMOKE_CMD（多 env 遍历）**

old_string（gate prompt 冒烟段，约第 158–159 行）：
```js
4. 冒烟：\`cd course/${MODULE} && uv sync && uv run python -c "print('ok')"\`。
按 schema 报：violations=只列 steps/ 外被改文件路径（无则 []，说明写进 notes）；notes；reverted；smokeOk。`, { label: '完整性闸门', phase: '完整性闸门', schema: GATE })
```
new_string：
```js
4. 冒烟：\`${SMOKE_CMD}\`。
按 schema 报：violations=只列 steps/ 外被改文件路径（无则 []，说明写进 notes）；notes；reverted；smokeOk。`, { label: '完整性闸门', phase: '完整性闸门', schema: GATE })
```

- [ ] **Step 7: 语法检查 workflow 脚本**

Run: `node --check workflows/dev-module.js`
Expected: 无输出（语法正确）。`echo $?` 打印 `0`。
> 若报 SyntaxError，检查模板字符串里的反引号/`${}`嵌套（Step 4 的 `${MULTI_ENV ? ... : ...}` 在模板字符串内）。

- [ ] **Step 8: 回归——对 M2（单 env）跑 skipDev 确认向后兼容没破坏**

Run（确认 ENVS 单 env 默认路径行为不变）：
```bash
# 调用 workflow，M2 不传 envs → 走单 env 兼容分支
# （controller 执行：Workflow({scriptPath:"workflows/dev-module.js", args:{module:"m2-quant-pipeline", skipDev:true}})）
```
Expected: acceptance 全绿（`reviewerPassed && execVerified && learnerSatisfied && scaffoldClean`）——证明 ENVS 默认单 env 路径未破坏 M2。
> 若 M2 回归失败（之前全绿），说明 Step 1–6 改动伤了单 env 路径——对比 diff 找问题（最常见：ENV_HELP/SMOKE_CMD 在单 env 下应为空串/原命令，确认 `MULTI_ENV` 判定正确）。

- [ ] **Step 9: Commit dev-module.js**

```bash
git add workflows/dev-module.js
git commit -m "feat(workflow): dev-module.js dual-env support (args.envs, backward-compatible for single-env modules)"
```

---

### Task 5: dev-module.js 全流程产出 7 notebook

> **说明**：notebook 内容（每 s1–s7 的教学法 markdown + 填空 + L1/L2/L3）**由 workflow 的 dev-agent 写**，不在本 plan 逐 cell 写——spec §5 表（候选填空 + 教学目标）+ OUTLINE §模块3（3.1–3.7 要点 + 版本敏感 API）+ NOTEBOOK_CONVENTIONS（cell 顺序 + 填空/ipytest 模板）+ M2 s3_smoothquant.ipynb（教学法范例）是 dev-agent 的需求来源。本 task 只负责正确调用 workflow + 收集验收。

**Files:**
- Generated（dev-agent 写）: `course/m3-tuning-eval/steps/quant/s{1..6}_*.ipynb` + `course/m3-tuning-eval/steps/vllm/s7_evaluation.ipynb`

**Interfaces:**
- Consumes: Tasks 1–3 脚手架（env + 模型）、Task 4 改造后的 dev-module.js、spec 路径。
- Produces: 7 个 notebook（含教学目标清单 + 摸一摸 + 端到端 why + 填空 why + 参考实现 + L1/L2/L3）。

- [ ] **Step 1: 调用 dev-module.js 全流程（skipDev=false，传 envs）**

controller 执行：
```js
Workflow({
  scriptPath: "workflows/dev-module.js",
  args: {
    module: "m3-tuning-eval",
    skipDev: false,
    spec: "docs/superpowers/specs/2026-06-28-m3-tuning-eval-design.md",
    envs: [
      { dir: "steps/quant", smoke: "import llmcompressor,transformers,matplotlib" },
      { dir: "steps/vllm",  smoke: "import vllm" }
    ]
  }
})
```
Expected: workflow 跑完 5 阶段（开发 → reviewer 审核 loop → 学员代理 loop → 闸门 → 定稿），返回 `{ acceptance }`。耗时较长（dev 写 7 notebook + reviewer 双 env 执行验证 + 学员代理理解探测）。

- [ ] **Step 2: 核对 acceptance 全绿**

workflow 返回的 `acceptance` 须全 `true`：
```js
{
  module: "m3-tuning-eval",
  reviewerPassed: true,     // reviewer verdict=pass（无 critical/major findings）
  execVerified: true,       // reviewer 执行验证 allPassed（每 notebook L1/L2 过、L3 有 GPU 过/无 GPU skip）
  learnerSatisfied: true,   // 学员代理：所有教学目标题「只靠讲解」能答
  scaffoldClean: true       // 闸门：steps/ 外无污染 + 双 env 冒烟过
}
```
> 若任一为 false：读 workflow 各 agent 报告（devReport/reviewerResult/learnerResult/gate）定位——reviewer/execVerified 失败=代码或 env 问题（dev 修）；learnerSatisfied 失败=教学法缺口（dev 补讲解）；scaffoldClean 失败=脚手架被污染或冒烟失败（gate 还原 + 查 env）。

- [ ] **Step 3: 核对 7 notebook 齐全 + 教学法齐**

Run（确认产出）：
```bash
ls course/m3-tuning-eval/steps/quant/s*.ipynb   # 应见 s1-s6 六个
ls course/m3-tuning-eval/steps/vllm/s7_evaluation.ipynb
```
Expected: 6 + 1 = 7 个 notebook 存在。
抽查教学法（每 notebook 顶部有「## 学完应能讲清」清单 + 摸一摸 cell）：
```bash
grep -l "学完应能讲清" course/m3-tuning-eval/steps/quant/*.ipynb course/m3-tuning-eval/steps/vllm/*.ipynb | wc -l
```
Expected: `7`（每个 notebook 都有教学目标清单）。

- [ ] **Step 4: 核对 notebook 无输出（提交规则）**

Run：
```bash
for nb in course/m3-tuning-eval/steps/quant/*.ipynb course/m3-tuning-eval/steps/vllm/*.ipynb; do
  uv run --directory course/m3-tuning-eval/steps/quant jupyter nbconvert --clear-output --inplace "$nb" 2>/dev/null || true
done
```
> 注：dev-agent 应已 clear-output，本步是保险（reviewer 执行验证会注入输出，提交前必须清）。清完后 `git diff --stat` 应只见 notebook 源码（无 outputs 段）。
若 dev-agent 未清，本步修。

- [ ] **Step 5: Commit 7 notebook**

```bash
git add course/m3-tuning-eval/steps/quant/*.ipynb course/m3-tuning-eval/steps/vllm/*.ipynb
git commit -m "feat(m3): 7 notebooks via dev-module workflow (s1-s6 quant + s7 vllm, M2 pedagogy applied)"
```

---

### Task 6: 验收 + merge 准备

**Files:** 无新建（验收现有产物）。

**Interfaces:**
- Consumes: Task 5 的 7 notebook + Tasks 1–3 脚手架。

- [ ] **Step 1: 双 env 冒烟（独立于 workflow 再验一次）**

Run：
```bash
cd course/m3-tuning-eval/steps/quant && uv sync && ./.venv/bin/python -c "import llmcompressor,transformers,matplotlib; print('quant ok')"
cd course/m3-tuning-eval/steps/vllm  && uv sync && ./.venv/bin/python -c "import vllm,lm_eval; print('vllm ok')"
```
Expected: 打印 `quant ok` 与 `vllm ok`。

- [ ] **Step 2: 抽查 s1/s5 的 L3（真 7B SmoothQuant）**

> 这两节是 M3 核心实证：s1 见敏感层（profiling 输出 per-channel 幅值分布）、s5 见 Pareto 曲线（逐步 ignore → PPL 恢复）。有 GPU 时执行验证应已跑过；这里确认产物落 `out/`。
Run：
```bash
ls course/m3-tuning-eval/out/   # 应见调优中间模型 + PPL 表/json（s1 敏感层扫描、s5 Pareto 数据）
```
Expected: `out/` 有 s1/s5 L3 产物（敏感层排序结果、Pareto (回退层数, PPL) 数据）。无 GPU 则跳过（依赖 L3 skip）。

- [ ] **Step 3: scaffold 未污染（闸门独立复核）**

Run：
```bash
git diff --name-only main...HEAD -- ':!course/m3-tuning-eval/**'
```
Expected: 只有 `workflows/dev-module.js`（T4 改的）。**不应**有顶层/其他模块/M2 改动。

- [ ] **Step 4: 真人（用户）最终审准备**

通读 s1–s7 markdown（教学目标清单 + 端到端 why + 填空 why 引导），逐条对照每 notebook 顶部「学完应能讲清」自检能否口头答。准备好给用户的验收摘要：
- 双 env 自包含（两子项目 uv sync + lm_eval[vllm] + 模型）
- 7 notebook 齐 + 教学法齐（清单/摸一摸/why/填空 why）
- workflow acceptance 全绿
- s1/s5 L3 实证（敏感层 + Pareto）
- scaffold 未污染

- [ ] **Step 5: 推分支 + 开 PR（用户审过后）**

```bash
git push -u origin m3-tuning-eval
gh pr create --title "feat(m3): 精度调优与评测（layer fallback，双 env）" --body "..."
```
> merge 由用户决定（用户最终审满意后）。merge 后更新 `.superpowers/sdd/progress.md` 与内存 `course-dev-status.md`（M3 完成 + dev-module.js 双 env 支持）。

---

## Self-Review（写完后自审 spec 覆盖）

**1. Spec 覆盖**：
- §3 模块结构（双 env 子项目 + 共享 models/out）→ Task 1–3 ✓
- §4 env 设计（各子项目自带 env + notebook、不跨 env、_find_module_root 适配、lm_eval[vllm] 坑）→ Task 2/3 + Task 4 ENV_HELP ✓
- §5 7 notebook 设计表（教学目标 + 候选填空 + env + L2/L3）→ Task 5（dev-agent 按 spec §5 + OUTLINE 写）✓
- §6 教学法（套 M2 lesson）→ Global Constraints + Task 5 dev-agent prompt 锚定 ✓
- §7 构建流程（dev-module.js 全流程 + 双 env 执行验证 + 脚手架先建）→ Task 1–5 ✓
- §8 验收标准 6 条 → Task 6 ✓

**2. Placeholder 扫描**：无 TBD/TODO；notebook 内容明确归属 dev-agent（非 placeholder——spec §5 + CONV 是其需求来源），脚手架/workflow 代码均完整给出。

**3. 类型/命名一致性**：envs 配置（Task 5 args 与 Task 4 ENVS 结构 `{dir, smoke}` 一致）；notebook 名（Task 1 README、Task 5、Task 4 glob 均用 `steps/quant/s{1..6}` + `steps/vllm/s7_evaluation`）；SMOKE_CMD/ENV_HELP 命名贯穿 Task 4。

**4. 风险标注**：
- vLLM wheel 很大（Task 3 Step 2 首次 5–15 分钟）。
- lm_eval[vllm] extra 不进 lock，`uv sync` 一致性未实测（OUTLINE 附录 B #5）——Task 4 ENV_HELP 让 reviewer 跑 s7 前幂等重装，缓解。
- T4 改 dev-module.js 需 M2 回归（Task 4 Step 8）防伤单 env 路径。
- T5 workflow 耗时长（dev 写 7 notebook + 双 env 执行验证）。

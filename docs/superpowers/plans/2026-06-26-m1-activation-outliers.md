# M1 激活离群点与现代量化方法（原理层）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `course/m1-activation-outliers/` 下产出**自包含的原理层模块**——精简 env（无 llm-compressor）、模块内下载脚本、README、7 个从零用 PyTorch 实现量化数学的 notebook（s1-s7，对齐 OUTLINE 1.1-1.7），通过 `dev-module.js` 全流程（dev→架构师→QA→学生试课）双重验收，并顺带纠正 OUTLINE 的 `smoothing_strength` 事实错误。

**Architecture:** 与 M2 同一自包含模板：`course/m1-activation-outliers/` = 独立 uv 项目（`pyproject.toml`+`uv.lock`+`.venv`+`scripts/download_model.sh`+`models/`+`README`+`steps/`）。notebook 用「向上发现模块根」解析路径。区别于 M2：M1 是**原理层**——从零用 PyTorch 实现量化数学（hook 激活、手写 INT8/INT4/FP8 量化、SmoothQuant 等价变换、成本模型），**不调 llm-compressor**，故 env 精简（去 llmcompressor/compressed-tensors、加 matplotlib）。7 个 notebook 中 s1-s5 含 GPU（L3 真 0.5B→7B），s6/s7 是纯概念/选型（CPU 成本模型，无 GPU 守卫，spec §5 已定为 CONV 例外）。

**Tech Stack:** uv（独立项目）、Python 3.11、torch cu128（2.11.0+cu128）/ transformers v5（仅加载真 Qwen2.5，不用 llm-compressor）、matplotlib（s1 激活分布图）、ipytest/nbval/nbconvert、`hf` CLI、Workflow 工具 + git worktree（学生隔离）。

**Spec:** [`docs/superpowers/specs/2026-06-26-m1-activation-outliers-design.md`](../specs/2026-06-26-m1-activation-outliers-design.md)

## Global Constraints

- 版本基线（已实测，沿用 M2）：torch 2.11.0+cu128、transformers 5.10.1（v5）、CUDA 12.8 / 驱动 570、Python 3.11。torch **必须**走 cu128 index（PyPI 默认 cu130 在 12.8 驱动上失败）。
- M1 env **不含** `llmcompressor` / `compressed-tensors`（M1 从零实现，YAGNI）；**加** `matplotlib>=3.8`（s1 画 per-channel 激活幅值，M2 无此依赖）。其余与 M2 同（transformers/accelerate/datasets/safetensors/sentencepiece/huggingface-hub/hf-transfer/ipytest/nbval/nbconvert/jupyter）。
- 硬件措辞：本节点是 **H200**（OUTLINE §技术栈口径），notebook/README 用 "H200"。L3 cell 用 `if torch.cuda.is_available():` 守卫；s6/s7 纯概念 CPU-only（无 GPU 代码，无守卫，spec §5 明确为 CONV 例外）。
- notebook-only：每 Step 一个 `.ipynb`，提交前 `nbconvert --clear-output`。严格按 `course/NOTEBOOK_CONVENTIONS.md` cell 顺序。
- 填空哲学（吸取 M2"偏易"教训，spec §6）：从零实现公式/算法（非抄 API 字面量）；每方法 step **≥1 个判断型空**（需推理/选择，非取字面量）；docstring 给方向不给逐字答案。
- 顺带纠正：OUTLINE 把"库默认 `smoothing_strength=0.8`"当事实（错）——实测 0.12.0 `SmoothQuantModifier()` 类默认 = **0.5**（=论文 α）；0.8 是课程显式取值。M2 s3 已修；本计划 T4 同步纠正 OUTLINE。
- 在 `course-development` 分支开发。
- 内容 agent 编辑范围（护栏，已在 dev-module.js prompt 收口）：**只能**创建/修改 `course/m1-activation-outliers/steps/*.ipynb`；严禁改 pyproject/uv.lock/scripts/workflows/README/CONV/顶层。脚手架由本计划 T1-T3 先建。
- 分支：`course-development` 已合并入 `main` 并删除（2026-06-26）；spec(4de61ec)+本 plan(a8e5515) 已在 `main`（领先 origin/main 2 commit，未 push）。M1 实现的分支策略（在 main 继续 / 为 M1 开 feature 分支）在执行前由人定；文档产物（spec/plan）留在 main 即可。

## File Structure

- Create: `course/m1-activation-outliers/pyproject.toml`（M1 精简 env）
- Create: `course/m1-activation-outliers/uv.lock`（`uv lock` 生成，提交）
- Create: `course/m1-activation-outliers/scripts/download_model.sh`（下 7B+0.5B，含 MODEL_CACHE，复用 M2 脚本逻辑）
- Create: `course/m1-activation-outliers/README.md`（自包含 setup + 7 Steps）
- Create: `course/m1-activation-outliers/steps/s1_activation_outliers.ipynb` ~ `s7_timeline_ptq.ipynb`（7 个，由 dev-module.js dev-agent 生成）
- Modify: `OUTLINE.md`（纠正 `smoothing_strength` 事实错误，多处）
- Modify: `workflows/dev-module.js`（加 `args.spec` 透传，让 M1 agent 读 M1 spec）
- `.venv/` `models/` `out/` 在模块内，已 gitignore。

---

### Task 1: M1 自包含 env（pyproject.toml + uv.lock）

**Files:**
- Create: `course/m1-activation-outliers/pyproject.toml`
- Create: `course/m1-activation-outliers/uv.lock`（uv 生成）

**Interfaces:** Produces `course/m1-activation-outliers/.venv`（装好 torch cu128 / transformers v5 / matplotlib，**不含** llmcompressor），供 Task 2 的 `hf` CLI、Task 5 的 dev-agent notebook 执行、学生试课使用。

- [ ] **Step 1: 写 `course/m1-activation-outliers/pyproject.toml`**

内容（= M2 pyproject 去 `llmcompressor`/`compressed-tensors`、加 `matplotlib`、name/description 改 M1；cu128 index/sources 原样）：

```toml
[project]
name = "m1-activation-outliers"
version = "0.1.0"
description = "M1 激活离群点与现代量化方法（原理层）env。从零用 PyTorch 实现量化数学（不调 llm-compressor）。模块自包含——本模块自己的 uv 项目，不依赖顶层 envs/。"
requires-python = ">=3.10,<3.13"
# torch 必须是 CUDA 12.8 (cu128) build 以匹配本节点驱动 (570.x / CUDA 12.8)：
# PyPI 默认 torch 是 cu130，在 12.8 驱动上 CUDA 初始化失败。经 cu128 index 取 torch 2.11.0+cu128。
# 注意：M1 从零实现量化数学，故**不含** llmcompressor/compressed-tensors（那是 M2 工具链）；
#       额外加 matplotlib（s1 画 per-channel 激活幅值分布，M2 无此依赖）。
dependencies = [
    "torch>=2.10,<2.12",       # cu128 build（见 [tool.uv]）
    "transformers>=5.0",       # 加载真 Qwen2.5（仅加载，不调 llm-compressor）
    "accelerate>=1.0",         # device_map 多卡加载 7B
    "datasets>=2.20",          # s1 取样本文本喂模型看激活
    "matplotlib>=3.8",         # s1 画 per-channel 激活幅值分布（M1 特有）
    "safetensors>=0.4",
    "sentencepiece>=0.2",
    "huggingface-hub>=0.26",   # hub API + hf CLI
    "hf-transfer>=0.1",        # 快速下载；脚本设 HF_HUB_ENABLE_HF_TRANSFER=1
    "ipytest>=0.14",           # notebook 内跑 pytest
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

- [ ] **Step 2: 生成 uv.lock + 建 .venv**

Run: `uv lock --directory course/m1-activation-outliers && uv sync --directory course/m1-activation-outliers`
Expected: 解析成功，生成 `course/m1-activation-outliers/uv.lock`，建 `.venv`。无 `unsatisfiable` 错误（因去掉 llmcompressor 后依赖更轻，解析应更顺）。

- [ ] **Step 3: 冒烟验证导入（含 matplotlib）**

Run: `uv run --directory course/m1-activation-outliers python -c "import torch,transformers,matplotlib; print('transformers',transformers.__version__,'torch',torch.__version__)"`
Expected: 打印 `transformers 5.x torch 2.11.0+cu128`，无 ImportError。**反向核对**不含 llmcompressor：`uv run --directory course/m1-activation-outliers python -c "import importlib.util as u; print('llmcompressor present' if u.find_spec('llmcompressor') else 'llmcompressor absent (expected)')"` → 应打印 `llmcompressor absent (expected)`。

- [ ] **Step 4: 提交**

```bash
git add course/m1-activation-outliers/pyproject.toml course/m1-activation-outliers/uv.lock
git commit -m "feat(m1): self-contained module env (pyproject+uv.lock; from-scratch PyTorch, no llmcompressor, +matplotlib)"
```

---

### Task 2: 模块内 download_model.sh（7B+0.5B，含 MODEL_CACHE）+ 实测可跑

**Files:**
- Create: `course/m1-activation-outliers/scripts/download_model.sh`

**Interfaces:** Produces `course/m1-activation-outliers/models/{Qwen2.5-7B-Instruct,Qwen2.5-0.5B-Instruct}`，供 Task 5 的 s1-s5 notebook L3 + 学生试课加载。`MODULE` = `scripts/..`（脚本位置派生，cwd 无关）。逻辑与 M2 脚本一致（直接复用，仅注释改 M1）。

- [ ] **Step 1: 写 `course/m1-activation-outliers/scripts/download_model.sh`**

```bash
#!/usr/bin/env bash
# course/m1-activation-outliers/scripts/download_model.sh
# 下文本模块所需基线模型（Qwen2.5-7B-Instruct + 0.5B）到本模块 ./models/。
# 复用已下好的共享缓存免重复下载（同卷硬链，跨卷拷贝）：
#   MODEL_CACHE=<repo>/models bash scripts/download_model.sh
set -euo pipefail

MODULE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # scripts/.. = 模块根
MODELS_DIR="${MODULE}/models"
mkdir -p "$MODELS_DIR"

REPOS=("Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-0.5B-Instruct")

# ---- HF token（可选；Qwen2.5 非 gated）-----------------------------------------
if [ -n "${HF_TOKEN:-}" ]; then echo "[download_model] HF_TOKEN set (len=${#HF_TOKEN})."; fi
export HF_HUB_ENABLE_HF_TRANSFER=1   # 装了 hf-transfer 时走快路径

# ---- 选 hf CLI：优先本模块 .venv，其次系统 ------------------------------------
HF_CLI=""
if [ -x "${MODULE}/.venv/bin/hf" ]; then HF_CLI="${MODULE}/.venv/bin/hf"
elif command -v hf >/dev/null 2>&1; then HF_CLI="$(command -v hf)"
else echo "[download_model] 未找到 hf CLI。先在本模块内 uv sync。" >&2; exit 1; fi
echo "[download_model] using: $HF_CLI"

CACHE="${MODEL_CACHE:-}"   # 可选共享缓存目录（含 <name>/ 子目录）

for repo in "${REPOS[@]}"; do
  name="${repo#*/}"                 # Qwen2.5-7B-Instruct
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

  # 校验 loader 必需文件
  for f in config.json tokenizer_config.json; do
    [ -f "${dst}/${f}" ] || { echo "[download_model] FAILED: 缺 ${dst}/${f}" >&2; exit 1; }
  done
  echo "  ✓ ${name}: $(du -sh "$dst" 2>/dev/null | cut -f1)"
done

echo "[download_model] Done. 模型在 ${MODELS_DIR}/"
```

赋可执行权限：`chmod +x course/m1-activation-outliers/scripts/download_model.sh`

- [ ] **Step 2: 实测——MODEL_CACHE 复用路径（主路径，免重下）**

Run: `MODEL_CACHE="$PWD/models" bash course/m1-activation-outliers/scripts/download_model.sh`
Expected: 打印「复用缓存 … -> …（硬链）」，`course/m1-activation-outliers/models/Qwen2.5-7B-Instruct/config.json` 与 `Qwen2.5-0.5B-Instruct/config.json` 都存在；无 FAILED。（顶层 `models/` 已有基线作缓存源。）

- [ ] **Step 3: 提交**

```bash
git add course/m1-activation-outliers/scripts/download_model.sh
git commit -m "feat(m1): module-local download_model.sh (7B+0.5B, MODEL_CACHE reuse)"
```

---

### Task 3: M1 README 自包含 setup + 7 Steps

**Files:**
- Create: `course/m1-activation-outliers/README.md`

**Interfaces:** 学员/agent 按 README 即可从零配好本模块（uv sync + 下模型 + 跑 notebook），不碰顶层。7 Steps 列表对齐 spec §5 / OUTLINE 1.1-1.7。

- [ ] **Step 1: 写 `course/m1-activation-outliers/README.md`**

```markdown
# M1 — 激活离群点与现代量化方法（原理层）

对应 [OUTLINE.md](../../../OUTLINE.md) 模块 1（1.1-1.7）。本课是**原理层**：从零用 PyTorch 实现量化数学，建立 SmoothQuant/AWQ/FP8 的思想模型与选型判断力。

> 与 M2 的边界：M1 = **从零实现、理解 why**（可视化、等价变换、成本模型，不调 llm-compressor）；M2 = **llm-compressor 工具链出可部署产物**（how to ship）。同方法名、不同角度。

## Setup（本目录自包含，全部命令在此完成）

1. **建 env**（在本目录下，一次性）：
   ```bash
   cd course/m1-activation-outliers
   uv sync            # 读 uv.lock，建 .venv 并装好依赖（torch cu128 / transformers v5 / matplotlib …）
   ```
2. **拉基线模型**（7B + 0.5B，下到本目录 `./models/`；本模块自包含）：
   ```bash
   bash scripts/download_model.sh
   # 已在别处下过？复用免重下（同卷硬链）：
   # MODEL_CACHE=<repo>/models bash scripts/download_model.sh
   ```

## Steps（每个都是一个自包含 notebook，含填空 + ipytest + tiny 验证 + H200 执行）

- `steps/s1_activation_outliers.ipynb` — 1.1 激活离群点：hook Linear 输入，亲见 emergent outliers（少数 channel 幅值 20×+）
- `steps/s2_llmint8.ipynb` — 1.2 LLM.int8()：混合精度分解（离群维度切 FP16 相加）——SmoothQuant 的对照基准
- `steps/s3_smoothquant.ipynb` — 1.3 SmoothQuant：等价缩放迁移 `Y=(X·diag(s)⁻¹)·(diag(s)·W)`，把激活难度迁到权重 → 纯 INT8 W8A8
- `steps/s4_awq.ipynb` — 1.4 AWQ：显著通道权重缩放 + per-group INT4（weight-only W4A16）
- `steps/s5_fp8.ipynb` — 1.5 FP8：E4M3 vs E5M2，推理用 E4M3（动态范围自带抗离群）
- `steps/s6_selection.ipynb` — 1.6 选型：W4A16（访存墙）/ W8A8（计算墙）/ FP8 决策（**纯 CPU 成本模型，无需 GPU**）
- `steps/s7_timeline_ptq.ipynb` — 1.7 时间线 + 为何 PTQ 为主（**纯 CPU 概念，无需 GPU**）

跑法：在本目录下 `uv run jupyter lab` 打开 notebook；按序填空 → 跑 ipytest 测试 cell（L1）→ 跑 tiny 验证 cell（L2）→ 跑 H200 执行 cell（L3，s1-s5；s6/s7 为 CPU 概念 cell）。
```

- [ ] **Step 2: 核对无顶层残留引用**

Run: `grep -nE 'envs/(quant|deploy)|scripts/setup_env|llmcompressor|llm-compressor' course/m1-activation-outliers/README.md`
Expected: 仅命中边界说明里那句「不调 llm-compressor」（这是刻意的对比说明，非残留引用）；无 `envs/`/`setup_env` 命中。

- [ ] **Step 3: 提交**

```bash
git add course/m1-activation-outliers/README.md
git commit -m "docs(m1): self-contained README setup + 7 steps (principles layer, from-scratch PyTorch)"
```

---

### Task 4: 纠正 OUTLINE 的 smoothing_strength 事实错误（先于 workflow，免 dev-agent 抄错）

**Files:**
- Modify: `OUTLINE.md`（多处）

**Interfaces:** 让 OUTLINE 不再把「库默认 `smoothing_strength=0.8`」当事实——实测 0.12.0 `SmoothQuantModifier()` 类默认 = **0.5**（=论文 α）。这是 Task 5 dev-agent 生成 s3（讲 SmoothQuant）的事实前提，故**必须在跑 workflow 前纠正**。

- [ ] **Step 1: 定位所有命中**

Run: `grep -nE 'smoothing_strength|默认 0\.8|默认 \*\*0\.8' OUTLINE.md`
Expected: 至少命中 line 106、192、200、396、462（见下逐条处理）。

- [ ] **Step 2: 纠正 line 106（1.3 SmoothQuant 版本要点，把"默认 0.8"当库默认）**

Edit：把
```
- ⚠️ 版本要点：llm-compressor 参数名是 **`smoothing_strength`（默认 0.8，即 α=0.8）**，**不是论文的 α**，且默认偏向多迁移到权重——这是最易写错的点
```
改为
```
- ⚠️ 版本要点：llm-compressor 参数名是 **`smoothing_strength`**（实测 0.12.0 的 `SmoothQuantModifier()` 类默认 = **0.5，即论文 α=0.5**）；课程在 s3/M2 刻意显式取 **α=0.8** 以偏向多迁权重——调用时写 `smoothing_strength=0.8`，别误以为 0.8 是库默认（这是最易写错的点）
```

- [ ] **Step 3: 纠正 line 200（M2.5 标注里的"默认 0.8"）**

Edit：把
```
- ⚠️ 导入路径：`SmoothQuantModifier` 在 `llmcompressor.modifiers.transform.smoothquant`（不是 `modifiers.smoothquant`）；`smoothing_strength` 默认 **0.8**（非论文 α=0.5）
```
改为
```
- ⚠️ 导入路径：`SmoothQuantModifier` 在 `llmcompressor.modifiers.transform.smoothquant`（不是 `modifiers.smoothquant`）；`SmoothQuantModifier()` 类默认 `smoothing_strength=0.5`（=论文 α）；课程显式取 0.8 偏向多迁权重
```

- [ ] **Step 4: 纠正 line 396（M3 代码示例的注释，代码值 0.8 保留）**

先核对精确文本（含对齐空格）：`sed -n '393,397p' OUTLINE.md`。可见该行形如
`    SmoothQuantModifier(smoothing_strength=0.8),<对齐空格># 默认 0.8，非论文 α=0.5`（`),` 后留空格与下一行 GPTQModifier 的 `#` 对齐）。
用 Edit 把行尾注释从 `# 默认 0.8，非论文 α=0.5` 改为 `# 显式取 0.8（库默认 0.5=论文 α）`——**保留** `SmoothQuantModifier(smoothing_strength=0.8),` 这段代码值 0.8（课程显式取值，正确）与对齐空格；只替换注释文案。以 `sed` 实际输出为 Edit 的 old_string 基准（勿照抄本计划的空格数）。

- [ ] **Step 5: 纠正 line 462（附录"待验证事项"把"默认 0.8"当默认）**

Edit：把
```
11. **`smoothing_strength` 在 Qwen2.5-7B 上的最优值**：默认 0.8，需 grid search（0.5/0.8/1.0 对比）。
```
改为
```
11. **`smoothing_strength` 在 Qwen2.5-7B 上的最优值**：库默认 0.5（=论文 α），课程取 0.8，需 grid search（0.5/0.8/1.0 对比）。
```

- [ ] **Step 6: 核对 line 192（M2 章节 recipe 代码块内的 `smoothing_strength=0.8`）**

已核对（`sed -n '188,194p'`）：line 192 是 `recipe=[...]` 代码块里的纯代码 `      SmoothQuantModifier(smoothing_strength=0.8),`，**无**「默认」误导注释 → **保留不动**（0.8 是课程刻意取值，正确）。仅当执行环境 OUTLINE 已被改动致此处出现「默认 0.8」注释时，才按 Step 4 同法纠正注释。

- [ ] **Step 7: 全量核对无遗漏的"默认 0.8"误导**

Run: `grep -nE 'smoothing_strength.*默认.{0,4}0\.8|默认.{0,4}0\.8.*smoothing|默认.{0,4}\*\*0\.8\*\*.*α' OUTLINE.md`
Expected: 无输出（所有把 0.8 当库默认的文案已改；显式取 0.8 的代码/说明保留）。

- [ ] **Step 8: 提交**

```bash
git add OUTLINE.md
git commit -m "docs(outline): fix smoothing_strength fact — lib default is 0.5 (paper α), not 0.8 (0.8 is course-chosen)"
```

---

### Task 5: dev-module.js 加 args.spec 透传 + 跑全流程（skipDev=false）生成 7 notebook + 双重验收

**Files:**
- Modify: `workflows/dev-module.js`（加 `args.spec`）
- Create（由 workflow dev-agent）: `course/m1-activation-outliers/steps/s1_activation_outliers.ipynb` ~ `s7_timeline_ptq.ipynb`
- 可能由 workflow 评审/学生修订：同上 steps/*.ipynb

**Interfaces:** M1 notebook 尚不存在 → 跑 dev-module.js **全流程**（skipDev=false）。dev-agent 按 spec §5 + CONV + OUTLINE + **M1 spec** 生成 7 notebook。验收对齐 spec §9。

> 为何先改 dev-module.js：现 `const SPEC`（line 29）硬编码旧总 spec，M1 dev-agent 读不到 M1 spec（从零 PyTorch、matplotlib、s6/s7 CPU 例外、填空哲学等关键设计都在 M1 spec 里）。加 `args.spec` 让各 agent 读正确的 spec。

- [ ] **Step 1: 改 `workflows/dev-module.js` 的 SPEC 为可配置**

把（约 line 29）：
```js
const SPEC = 'docs/superpowers/specs/2026-06-24-course-development-design.md'
```
改为：
```js
// SPEC 可按模块透传（args.spec）；缺省回退旧总 spec（向后兼容 M2）
const SPEC = ARGS.spec || 'docs/superpowers/specs/2026-06-24-course-development-design.md'
```
（`ARGS` 已在 line 15 由 `typeof args === 'string' ? JSON.parse(args) : (args||{})` 解析，`ARGS.spec` 会正确取到。各 agent prompt 里 `${SPEC}` 自动带上 M1 spec 路径，dev/架构师/QA 都会读 M1 spec。）

- [ ] **Step 2: 静态自检 workflow**

Run: `node --check workflows/dev-module.js 2>&1 || echo "(Workflow 脚本非纯 Node；人工核对 ARGS.spec 取值正确、meta 纯字面量)"`
Expected: 无明显语法错。

- [ ] **Step 3: 提交 workflow 改动**

```bash
git add workflows/dev-module.js
git commit -m "feat(workflow): allow args.spec override so per-module spec (e.g. M1) reaches dev/arch/QA agents"
```

- [ ] **Step 4: 跑 dev-module.js 全流程（skipDev=false，后台，完成通知）**

```
Workflow({ scriptPath: "workflows/dev-module.js",
           args: { module: "m1-activation-outliers", skipDev: false,
                   spec: "docs/superpowers/specs/2026-06-26-m1-activation-outliers-design.md" } })
```
用 `/workflows` 看实时进度（开发 → 架构审核 → QA loop → 学生试课 loop → 完整性闸门 → 定稿）。args 现已 `JSON.parse` 修复（probe 实测生效），module/spec/skipDev 会正确解析。

- [ ] **Step 5: 取回 acceptance 核对（spec §9 验收 4）**

workflow 完成后读返回的 `acceptance`，全部要满足：
- `qaResult.verdict === "pass"`
- `studentResult.satisfied === true`
- `scaffoldClean === true`（闸门无违规、冒烟过）
- `blanksPerStepOk === true`（总填空数 ≥ 2 × notebook 数）

另抽查 dev-agent 输出：7 个 notebook 齐全（s1-s7）、每步 ≥2 填空且 ≥1 判断型空（spec §6）、s1-s5 有 GPU 守卫 L3、s6/s7 为 CPU 概念 cell（无守卫，spec §5 例外）。

- [ ] **Step 6: 不通过则迭代**

若未全过：按 `qaResult.findings` / `studentResult.designIssues` 直接改 notebook（或重跑对应阶段），再验证。封顶（QA/学生各 3 轮）仍不过 → 升级人工，带上 findings。**注意 s6/s7 CPU-only 是 spec §5 明确的 CONV 例外**——若 QA 按CONV 严格要求其加 GPU 守卫，引用 spec §5 驳回（它们是纯成本模型/元数据表，无 GPU 代码）。

- [ ] **Step 7: 清输出 + 提交 7 notebook（核对护栏：只动 steps/）**

```bash
for nb in course/m1-activation-outliers/steps/*.ipynb; do
  uv run --directory course/m1-activation-outliers jupyter nbconvert --clear-output --inplace "$nb"
done
git add course/m1-activation-outliers/steps/
git diff --cached --stat   # 核对只动 steps/*.ipynb（护栏：不应有 env/脚本/顶层改动）
git commit -m "feat(m1): 7 principles-layer notebooks via dev-module full flow (dev->arch->QA->student double-acceptance)"
```

---

### Task 6: 最终验收（spec §9 六条逐条核对）

**Files:** 无新增（只核对）；若抽查发现缺陷，回到 Task 5 迭代。

**Interfaces:** 逐条对齐 spec §9「完成的定义」，产出可合并结论。

- [ ] **Step 1: 验收 1——自包含 env 冒烟**

Run: `cd course/m1-activation-outliers && uv sync && uv run python -c "import torch,transformers,matplotlib; print('ok')"`
Expected: `uv sync` 成功，打印 `ok`。

- [ ] **Step 2: 验收 2——下载脚本可跑、模型齐**

Run: `ls course/m1-activation-outliers/models/Qwen2.5-7B-Instruct/config.json course/m1-activation-outliers/models/Qwen2.5-0.5B-Instruct/config.json`
Expected: 两个 config.json 都存在（Task 2 已下）。若学生 worktree 跑过后被清，重跑 `MODEL_CACHE="$PWD/models" bash course/m1-activation-outliers/scripts/download_model.sh`。

- [ ] **Step 3: 验收 3——7 notebook 齐全、cell 顺序、填空覆盖**

Run:
```bash
ls course/m1-activation-outliers/steps/ | grep -cE 's[1-7]_.*\.ipynb'   # 期望 7
grep -rcE 'NotImplementedError|TODO' course/m1-activation-outliers/steps/*.ipynb | awk -F: '{s+=$2} END{print "blanks(total):",s}'   # 期望 ≥ 2×7=14
```
Expected: notebook 数 = 7；总填空 ≥ 14。人工抽查每步 ≥1 个判断型空（spec §6）、cell 顺序合 CONV（标题→导入→路径 cell→讲解→填空→ipytest→L2→L3→产物）。

- [ ] **Step 4: 验收 4——dev-module 全流程 pass（已在 Task 5 Step 5 核对）**

确认 `qaPassed && studentSatisfied && scaffoldClean && blanksPerStepOk` 全 true（承接 Task 5 Step 5 结果）。

- [ ] **Step 5: 验收 5——抽查 s1/s3 的 L3（真 7B）跑通**

注入参考实现（参考 spec §5 候选填空：s1 `per_channel_magnitudes`/`outlier_ratio`、s3 `compute_smooth_scale`/`apply_smooth_transform`），从模块目录启动执行 s1、s3 的 L3 cell：
```bash
cd course/m1-activation-outliers
uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1200 steps/s1_activation_outliers.ipynb steps/s3_smoothquant.ipynb
```
Expected: 无执行错误；s1 产出 `outlier_scan.json`（见 emergent outlier）、s3 forward 等价误差 ~1e-7（先在 0.5B 验，再 7B）。完成后 `nbconvert --clear-output --inplace` 还原无输出态。无 GPU 则记录 skip（H200 节点会真跑）。

- [ ] **Step 6: 验收 6——OUTLINE smoothing_strength 已纠正（已在 Task 4）**

Run: `grep -nE '默认.{0,4}0\.8' OUTLINE.md | grep -i smooth`
Expected: 无把 0.8 当库默认的残留（承接 Task 4 Step 7）。

- [ ] **Step 7: 汇总结论**

全部 6 条通过 → 模块就绪，进入 `superpowers:finishing-a-development-branch` 收尾（merge/PR）。任一未过 → 回 Task 5 迭代对应 notebook，不强行收尾。

---

## Self-Review（写完自查）

**Spec 覆盖**：§3 模块结构→T1+T2+T3（env/脚本/README）；§4 env→T1；§5 七 notebook→T5（dev-agent 生成，T6 抽查）；§6 填空哲学→T5 Step 5 抽查 + dev-agent prompt 读 M1 spec §6；§7 OUTLINE 纠正→T4；§8 构建流程（全 workflow skipDev=false + 先建脚手架）→T1-T3（脚手架）+T5（workflow）；§9 验收六条→T6 Step 1-6 逐条。全覆盖。

**占位符**：无 TBD/TODO；pyproject/download_model/README/OUTLINE Edit/dev-module.js diff 均给了完整内容；T5 workflow 调用给了完整 args。

**类型/命名一致**：`course/m1-activation-outliers/` 路径在 T1-T6 一致；`ARGS.spec`/`SPEC` 在 T5 Step 1 定义、Step 4 调用一致；7 notebook 文件名（`s1_activation_outliers`…`s7_timeline_ptq`）在 T3 README 列表、T5 生成、T6 抽查一致（与 spec §5 表一致）；模型名 `Qwen2.5-7B-Instruct`/`Qwen2.5-0.5B-Instruct` 在 T2 脚本、T3 README、T6 抽查一致。

**与 M2 plan 风格一致**：脚手架任务给完整代码 + 实测；workflow 任务给调用 + acceptance 核对；验收任务逐条核对。

# M4 vLLM 声明式部署与端到端闭环 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `course/m4-deploy-loop/` 下产出 5 个自包含 notebook（s1–s5，对应 OUTLINE 4.1–4.4 + 4.6/4.7），单 env vllm，以声明式部署走通「加载识别（含自己写 quantization_config）→ 多卡 TP → 压测 → 报错排查 → 端到端闭环」，并通过 dev-module.js 全流程（5 阶段）双重验收。

**Architecture:** M4 是**单 env** 模块（模块根 `pyproject.toml`，vllm + ipytest/nbconvert/jupyter；**不装 llmcompressor / lm_eval / huggingface_hub**——部署用 vllm bench/metrics，非评测/非发布）。比 M3 双 env 简单。产物来源**混合**：0.5B 兜底（download，FP16 即可 vllm 部署，逻辑层独立可跑）+ 7B 量化产物跨模块只读引用 `course/m2-*/out`、`course/m3-*/out`（`REPO_COURSE = MODULE_ROOT.parent`）。教学编排**命令构造 + 离线 LLM()**：填空=纯逻辑函数（构造命令/解析 config 选 kernel/诊断报错/决策树），L1 ipytest 无 GPU 验、L2 真实 config 结构验、L3 vllm 真跑（GPU+SKIP_L3）。声明式边界（两层适配 + 三条件 + vLLM 内部 6 步）是 s1 核心，含构造型填空 `build_quantization_config`（学员从需求**自己写出** config，非机械抄）。脚手架（env/scripts/README）由 T1–T2 先建；notebook 由 T3 的 dev-module.js workflow 的 dev-agent 按 spec 写；T4 验收 + merge。

**Tech Stack:** vLLM PyPI 0.x（`>=0.11`，wheel 自带并锁定 CUDA12.8 配对 torch/transformers，**不另装 torch**）、ipytest>=0.14、nbconvert>=7.0、jupyter>=1.0、CUDA 12.8 / 驱动 570、Python 3.11、uv 0.8.x。

## Global Constraints

（来自 spec §2/§3/§4/§5 + OUTLINE 版本基线；每个 task 的需求隐含包含本节）

- **分支**：已从 `main` 开 feature 分支 `m4-deploy-loop`（spec/plan 已提交其上）。不在 main 直接提交实现。
- **单 env**：模块根 `course/m4-deploy-loop/pyproject.toml`（不像 M3 双子项目）。**不装** llmcompressor / lm_eval / huggingface_hub——M4 用 `vllm bench serve` + `/metrics` 压测，不跑下游评测（M3 s7 才用 lm_eval），Hub 发布已砍（s5 仅 markdown 提示不实操）。
- **产物来源混合**：0.5B 兜底（`scripts/download_model.sh` 拉，FP16 即可 vllm 部署，**不依赖任何量化产物**——M4 逻辑层独立可跑）；7B 量化产物跨模块**只读**引用 `REPO_COURSE = MODULE_ROOT.parent` 下的兄弟模块 out/（`course/m2-quant-pipeline/out/qwen7b-{fp8,awq,smoothquant}`、`course/m3-tuning-eval/out/`），带存在性检查 + 友好报错，缺产物降级 0.5B。FP16 基线 = download 的 7B 本身。
- **版本基线（pyproject 下限）**：`vllm>=0.11`（PyPI 0.x，**不是** CalVer `>=25.0`）；Python `>=3.10,<3.13`（锁定 3.11）；**不另装 torch**（vLLM wheel 自带 CUDA 12.8 配对 torch，OUTLINE 坑#2）。
- **vllm 不能 CPU 真跑**（无实用 CPU 后端跑 FP8/AWQ kernel，与 M3 的 L2 能 CPU 跑 llmcompressor 不同）：L1（ipytest 纯逻辑）+ L2（真实 config 结构验证）CPU 可跑、workflow 验；**L3（vllm 真跑）全 GPU 守卫** `if torch.cuda.is_available() and not os.environ.get('SKIP_L3'):`——workflow 执行验证设 `SKIP_L3=1` 跳过（7B serve 分钟级 + 多卡，太重），学员/真人不设时 L3 实证。
- **_find_module_root 用 CONV 原版**（`scripts/` + `pyproject.toml` 判据）——M4 单 env 在模块根有 pyproject.toml，**不像 M3 双 env 要改 `scripts/+steps/` 判据**。M4 setup cell 额外加 `REPO_COURSE = MODULE_ROOT.parent` 解析跨模块路径。
- **提交规则**：notebook commit 前 `jupyter nbconvert --clear-output --inplace steps/*.ipynb` + 删 cell `metadata.execution`（与 M1/M2/M3 一致）；改 pyproject 后必须重生成 `uv.lock` 并一并 commit。
- **编辑范围**：T1–T2 建 `course/m4-deploy-loop/` 下脚手架；T3 的 dev-agent（dev-module.js workflow 内）只改 `course/m4-deploy-loop/steps/*.ipynb`，严禁改 pyproject/uv.lock/scripts/README/workflows/CONV。
- **教学法（套 M2/M3 验证过的 lesson）**：每 notebook 顶部「## 学完应能讲清」清单（3–5 问）+ 摸一摸 cell + 端到端 why + 填空 docstring「为什么这么设计」段（**给语义/约束，不给逐字答案**）。s1 的 `build_quantization_config` 是构造型核心——学员从需求**自己写出** config_groups/ignore 结构，非抄 JSON。
- **不外发**：Hub 已砍；notebook 不自动起对外服务（vllm serve 教学默认 bind localhost）、不自动上传。跨模块只读不改兄弟模块。
- **dev-module.js 单 env 向后兼容**：调用 `Workflow({scriptPath:'workflows/dev-module.js', args:{module:'m4-deploy-loop', skipDev:false, spec:'<path>'}})`——不带 `envs`（缺失→默认单根 env，行为同 M1/M2；M3 加的双 env 分支不触发）。reviewer/dev-fix 执行验证已强制 foreground nbconvert + `SKIP_L3=1` + 禁 background/Monitor（M3 af3d9be 已修，M4 复用）。

---

## File Structure

```
course/m4-deploy-loop/
  README.md                          # T1：双前置（本 env sync；跨模块 L3 需先跑 M2/M3）+ Steps + 跑法
  .gitignore                         # T1：忽略 models/ out/ .venv/ __pycache__/
  scripts/
    download_model.sh                # T1：下 7B+0.5B 到 ./models/（hf CLI 优先 .venv/bin/hf；MODEL_CACHE 复用）
  pyproject.toml                     # T2：单 env（vllm + ipytest + nbconvert + jupyter；不装 llmcompressor/lm_eval/huggingface_hub）
  uv.lock                            # T2 uv sync 生成，commit
  .venv/                             # gitignore
  models/                            # gitignore；download 下此（7B + 0.5B）
  out/                               # gitignore；M4 压测报告/产物
  steps/
    s1_load_quantized.ipynb          # T3 dev-agent 写（3 填空：detect + build_quantization_config + pick）
    s2_multigpu_deploy.ipynb         # T3（2 填空：build_serve_cmd + recommend_tp）
    s3_benchmark.ipynb               # T3（2 填空：build_bench_cmd + parse_metrics）
    s4_error_cheatsheet.ipynb        # T3（2 填空：diagnose_error + recommend_fix）
    s5_e2e_loop.ipynb                # T3（2 填空：build_decision_tree + build_delivery_bundle + 4.7 前瞻 markdown）
```

每文件职责：`scripts/download_model.sh` 模型唯一入口；`pyproject.toml` env 唯一事实来源（`uv.lock` 复现性来源）；`workflows/dev-module.js` 全模块通用开发 workflow（M4 单 env 复用，不改）。

---

### Task 1: M4 模块根脚手架（README + .gitignore + download_model.sh）

**Files:**
- Create: `course/m4-deploy-loop/README.md`
- Create: `course/m4-deploy-loop/.gitignore`
- Create: `course/m4-deploy-loop/scripts/download_model.sh`（+ `chmod +x`）

**Interfaces:**
- Produces: `download_model.sh`（hf CLI 优先 `.venv/bin/hf`）、共享 `models/`（被 T2 冒烟后下模型 + T3 notebook 读）、`out/`（被 T3 notebook 写压测报告）、README 双前置说明（被 T3 dev-agent 读，理解跨模块依赖）。

- [ ] **Step 1: 确认在 feature 分支**

```bash
git branch --show-current   # 应为 m4-deploy-loop
```
Expected: 打印 `m4-deploy-loop`（已在 spec/plan 提交时开好；若不是，`git checkout m4-deploy-loop`）。

- [ ] **Step 2: 建目录结构**

```bash
mkdir -p course/m4-deploy-loop/scripts course/m4-deploy-loop/steps course/m4-deploy-loop/models course/m4-deploy-loop/out
```
Expected: `course/m4-deploy-loop/{scripts,steps,models,out}/` 存在。

- [ ] **Step 3: 写 `.gitignore`**

Create `course/m4-deploy-loop/.gitignore`:
```
# Python / uv
.venv/
__pycache__/
*.pyc

# Downloaded models & outputs (target of scripts/download_model.sh / notebook writes)
models/
out/
```

- [ ] **Step 4: 写 `scripts/download_model.sh`**（M2 单 env 模板 + M4 适配）

Create `course/m4-deploy-loop/scripts/download_model.sh`:
```bash
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
```

```bash
chmod +x course/m4-deploy-loop/scripts/download_model.sh
```

- [ ] **Step 5: 写 `README.md`**（双前置 + Steps + 跑法）

Create `course/m4-deploy-loop/README.md`:
```markdown
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
```

- [ ] **Step 6: Commit**

```bash
git add course/m4-deploy-loop/README.md course/m4-deploy-loop/.gitignore course/m4-deploy-loop/scripts/download_model.sh
git commit -m "feat(m4): module scaffold — README (dual-prereq) + .gitignore + download_model.sh

Co-Authored-By: Claude <noreply@anthropic.com>"
```
Expected: 3 files committed。

---

### Task 2: 单 env（pyproject + uv.lock + 冒烟）

**Files:**
- Create: `course/m4-deploy-loop/pyproject.toml`
- Create: `course/m4-deploy-loop/uv.lock`（`uv sync` 生成）

**Interfaces:**
- Consumes: T1 的 `scripts/download_model.sh`（`.venv/bin/hf` 来源——env sync 后 download 才有 hf CLI）。
- Produces: `.venv/`（gitignore；T3 dev-agent/reviewer 在此跑 nbconvert）、`pyproject.toml` + `uv.lock`（env 唯一事实来源 + 复现性来源）。

- [ ] **Step 1: 写 `pyproject.toml`**（单 env，干净）

Create `course/m4-deploy-loop/pyproject.toml`:
```toml
[project]
name = "m4-deploy-loop"
version = "0.1.0"
description = "M4 vLLM 声明式部署与端到端闭环（单 env：vllm + ipytest + nbconvert + jupyter）。vLLM wheel 自带并锁定 CUDA12.8 配对的 torch/transformers，不另装。不装 lm_eval（部署用 vllm bench/metrics，非评测）、不装 huggingface_hub（Hub 发布已砍）。"
requires-python = ">=3.10,<3.13"
# vLLM wheel 自带 torch/transformers（CUDA 12.8 配对）——不要另装 torch（OUTLINE 坑#2，会覆盖 vllm build 破坏 FP8/INT8 kernel）。
# 不装 lm_eval（M4 不评测）、不装 huggingface_hub（Hub 砍）——env 干净，无 extra 旁路坑。
dependencies = [
    "vllm>=0.11",            # PyPI 0.x（非 CalVer）；wheel 自带 torch/transformers
    "ipytest>=0.14",         # notebook 内 pytest（L1）
    "nbconvert>=7.0",        # --execute / --clear-output
    "jupyter>=1.0",          # 学员交互跑
]
```

- [ ] **Step 2: `uv sync` 建 env**（vLLM wheel 大，首次需几分钟）

```bash
cd course/m4-deploy-loop && uv sync
```
Expected: `.venv/` 建好，`uv.lock` 生成，无报错。若 vllm 解析失败（写了 CalVer `>=25.0` 之类）→ 检查 pyproject 是 `>=0.11`。

- [ ] **Step 3: 冒烟**（用 `.venv/bin/python`，不触发 sync）

```bash
./.venv/bin/python -c "import vllm, ipytest, nbconvert; print('vllm', vllm.__version__); print('env ok')"
```
Expected: 打印 `vllm <0.x版本>` + `env ok`。若 `No module named 'vllm'` → Step 2 sync 未完成，重跑。

- [ ] **Step 4: 下模型**（env 有了 hf CLI 才能下）

```bash
bash scripts/download_model.sh
```
Expected: `models/Qwen2.5-7B-Instruct` + `models/Qwen2.5-0.5B-Instruct` 存在（或复用 MODEL_CACHE 硬链）。

- [ ] **Step 5: Commit**（pyproject + uv.lock 一并）

```bash
cd course/m4-deploy-loop
git add pyproject.toml uv.lock
git commit -m "feat(m4): single vllm env (pyproject + uv.lock) — no llmcompressor/lm_eval/huggingface_hub

Co-Authored-By: Claude <noreply@anthropic.com>"
```
Expected: 2 files committed（`.venv/`/`models/` 被 gitignore 不进）。

---

### Task 3: dev-module.js 全流程产出 5 notebook + controller 独立复核

**Files:**
- Create: `course/m4-deploy-loop/steps/s1_load_quantized.ipynb`（3 填空）
- Create: `course/m4-deploy-loop/steps/s2_multigpu_deploy.ipynb`（2 填空）
- Create: `course/m4-deploy-loop/steps/s3_benchmark.ipynb`（2 填空）
- Create: `course/m4-deploy-loop/steps/s4_error_cheatsheet.ipynb`（2 填空）
- Create: `course/m4-deploy-loop/steps/s5_e2e_loop.ipynb`（2 填空 + 4.7 前瞻 markdown）

**Interfaces:**
- Consumes: T1 README（双前置/跨模块说明）、T2 `.venv`（nbconvert 执行）、spec（dev-agent 读 `docs/superpowers/specs/2026-06-29-m4-deploy-loop-design.md` 拿教学法 + 填空设计 + 跨模块路径 + 声明式边界）、M2/M3 `out/`（L3 跨模块只读 7B 量化产物）。
- Produces: 5 notebook（套教学法 + L1/L2/L3 + SKIP_L3 双守卫 + 跨模块友好报错）+ dev-module.js acceptance（reviewer pass[含执行验证] + learner satisfied + gate smokeOk）。

- [ ] **Step 1: 跑 dev-module.js workflow（单 env，skipDev=false）**

controller 执行（Workflow tool）：
```javascript
Workflow({
  scriptPath: "workflows/dev-module.js",
  args: {
    module: "m4-deploy-loop",
    skipDev: false,
    spec: "docs/superpowers/specs/2026-06-29-m4-deploy-loop-design.md"
  }
})
```
说明：不带 `envs`（单 env，向后兼容 M1/M2）。workflow 内部：① 开发专家按 spec 写 5 notebook（套教学法 + build_quantization_config 构造型填空 + 跨模块 REPO_COURSE 路径 + L3 双守卫）→ ② reviewer 审内容（锚「学完应能讲清」）+ 执行验证（注入参考实现 foreground nbconvert 跑 L1+L2，SKIP_L3=1）loop≤3 → ③ 学员代理理解探测 loop≤3 → ④ 完整性闸门（单 env `uv sync` 冒烟）→ ⑤ 定稿 acceptance。
Expected: workflow 完成，`acceptance = {reviewerPassed:true, execVerified:true, learnerSatisfied:true, scaffoldClean:true}`。若 acceptance 非全绿 → 看 workflow log 找未过项（reviewer findings / learner designIssues / gate violations），针对性让 dev-agent 修后重跑。

- [ ] **Step 2: controller 独立复核（M3 教训：reviewer/learner 抓不到的盲点）**

```bash
cd course/m4-deploy-loop
# (a) 答案泄露：参考实现不得内联进 notebook（只进 reviewer 注入）
grep -rnE "def _.*_ref|= _.*_ref|参考实现|标准答案" steps/ && echo "[FAIL] 答案泄露" || echo "[OK] 无泄露"
# (b) L3 双守卫：每 notebook 的 vllm 真跑 cell 必须有 cuda + SKIP_L3 双守卫
grep -rL "torch.cuda.is_available() and not os.environ.get('SKIP_L3')" steps/*.ipynb && echo "[WARN] 以下 notebook 缺 L3 双守卫" || echo "[OK] L3 守卫齐全"
# (c) 跨模块路径友好报错：setup cell 应有 REPO_COURSE + 存在性检查
grep -rl "REPO_COURSE" steps/*.ipynb | wc -l   # 应 ≥1（s1 setup cell）
# (d) 清输出：提交前无 execution_count / outputs
grep -rl '"execution_count":' steps/*.ipynb && echo "[WARN] 未清输出" || echo "[OK] 已清输出"
```
Expected: 全部 `[OK]`。任何 `[FAIL]`/`[WARN]` → 让 dev-agent 修（泄露=移除参考实现 cell；缺守卫=补 `if torch.cuda.is_available() and not os.environ.get('SKIP_L3'):`）。

- [ ] **Step 3: dev-agent commit notebook**（workflow 内 dev-agent 已写但未必 commit；controller 确认/补 commit）

```bash
cd course/m4-deploy-loop
# 清输出（确保干净）
for f in steps/*.ipynb; do uv run jupyter nbconvert --clear-output --inplace "$f"; done
git add steps/*.ipynb
git status --short   # 确认 5 个 notebook + 无 steps/ 外改动
```
然后 dev-agent（或 controller 代）commit：
```bash
git commit -m "feat(m4): 5 notebooks via dev-module workflow (acceptance green)

s1 load/id + quantization_config construct + declarative boundary (3 blanks),
s2 multigpu TP (2), s3 benchmark (2), s4 error cheatsheet (2),
s5 e2e loop finale + QAT/NVFP4 前瞻 (2). 11 blanks total.
L1 ipytest + L2 config-structure verified (CPU); L3 vllm-run GPU+SKIP_L3 guarded.

Co-Authored-By: Claude <noreply@anthropic.com>"
```
Expected: 5 notebook committed，`git status` 无 steps/ 外被跟踪改动。

---

### Task 4: 验收 + merge 准备

**Files:** 无新建（验收现有 5 notebook + 脚手架）。

**Interfaces:**
- Consumes: T1–T3 全部产出。
- Produces: 确认 acceptance 全绿 + 提交链完整 + merge 准备就绪（不自动 merge，等用户）。

- [ ] **Step 1: 复核 acceptance 全绿**

```bash
# T3 Step 1 workflow 返回的 acceptance 四项必须全 true
# T3 Step 2 controller 独立复核四项必须全 [OK]
```
Expected: reviewerPassed + execVerified + learnerSatisfied + scaffoldClean 全 true；泄露/守卫/清输出全 OK。任一未达 → 回 T3 修。

- [ ] **Step 2: 确认提交链完整**

```bash
git log --oneline main..HEAD
```
Expected: 看到 M4 提交链——`feat(m4): 5 notebooks` → `feat(m4): single vllm env` → `feat(m4): module scaffold` → `docs(spec): deepen s1` → `docs(spec): m4-deploy-loop design`（顺序可能含 fixup/reviewer 修复提交）。

- [ ] **Step 3: 整体 sanity（notebook 可被 nbconvert 解析）**

```bash
cd course/m4-deploy-loop
for f in steps/*.ipynb; do ./.venv/bin/jupyter nbconvert --to notebook --stdout "$f" >/dev/null 2>&1 && echo "[OK] $f" || echo "[FAIL] $f"; done
```
Expected: 5 notebook 全 `[OK]`（JSON 结构合法、可被 nbconvert 读）。

- [ ] **Step 4: merge 准备（不自动 merge）**

向用户报告 acceptance 全绿 + 提交链，等用户确认 merge 策略（直接 merge main / 开 PR）。**不自动 merge**（用户 M3 时选"直接 merge 回 main"，但 M4 等用户明确指示）。

---

## Self-Review（写完后自审 spec 覆盖）

**1. Spec coverage**（逐 spec 节对照 task）：
- §1 目标/范围/砍 4.5 → T3 写 5 notebook（无 Hub notebook）+ README Steps 列 5 个 ✅
- §2 单 env（不装 lm_eval/huggingface_hub）→ T2 pyproject 4 依赖 ✅
- §3 产物混合（0.5B + 跨模块 REPO_COURSE）→ T1 download 下 0.5B+7B + README 前置 2 + T3 dev-agent 按 spec 写跨模块路径 + T3 Step 2(c) grep 验 ✅
- §4 教学编排（命令构造+离线 LLM()；L1/L2/L3 三层；SKIP_L3）→ T3 dev-agent 按 spec §4 写 + Step 2(b) grep 验 L3 双守卫 ✅
- §5 声明式边界（两层适配+三条件+6 步+字段映射）→ T3 dev-agent 按 spec §5 写进 s1 讲解 ✅
- §6 五 notebook 填空设计（11 填空）→ T3 文件清单标注填空数 + commit message 记 11 ✅
- §7 不外发 → Global Constraints"不外发" + 无 Hub notebook ✅
- §8 验收 → T3 workflow acceptance + Step 2 独立复核 + T4 复核 ✅
- §9 风险 → Global Constraints 逐条对应缓解 ✅
- §10 OUTLINE 对应 + 砍 4.5 → README Steps + T3 notebook 清单 ✅

**2. Placeholder scan**：无 TBD/TODO；T1/T2 给完整文件内容（README/download_model.sh/pyproject）；T3 给完整 Workflow 调用 + grep 复核命令 + commit message。✅

**3. Type consistency**：填空函数名跨 spec↔plan 一致——`detect_quant_scheme`/`build_quantization_config`/`pick_vllm_flag_and_kernel`(s1) / `build_serve_cmd`/`recommend_tp`(s2) / `build_bench_cmd`/`parse_metrics`(s3) / `diagnose_error`/`recommend_fix`(s4) / `build_decision_tree`/`build_delivery_bundle`(s5)；SKIP_L3 守卫表达式 `torch.cuda.is_available() and not os.environ.get('SKIP_L3')` 跨 spec↔plan↔dev-module.js 一致。✅

无遗漏，plan 可执行。

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-29-m4-deploy-loop.md`. Two execution options:

**1. Subagent-Driven (recommended)** — 我（controller）dispatch fresh implementer subagent 逐 task 执行（T1→T2→T3→T4），每 task 后 task review，最后全分支 review。与 M3 一致（用户上次选此）。

**2. Inline Execution** — 在本会话内逐 task 执行，批量 + checkpoint。

注：T1/T2 是机械脚手架（单文件/命令），适合 cheap model；T3 是 dev-module.js workflow 调用（controller 直接跑 Workflow tool，不 dispatch subagent——workflow 内部自有 dev-agent/reviewer/learner）；T4 是 controller 验收。

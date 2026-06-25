# 每节课自包含重构 + 双重验收 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 M2 重构成**自包含模块**（env + 模型 + 脚本 + README 全在 `course/m2-quant-pipeline/`，顶层 `envs/`+`scripts/` 退役），实测下载脚本可直接跑通，升级 `dev-module.js`（护栏 + skipDev + 学生全流程），再跑**双重验收**（架构师→QA→学生试课）。

**Architecture:** 每节课 = 一个独立 uv 项目目录（`pyproject.toml`+`uv.lock`+`.venv`+`scripts/`+`models/`+`README`+`steps/`）。notebook 用「向上发现模块根」解析路径，对 cwd/git 鲁棒。`dev-module.js` 加护栏（内容 agent 只许改 `steps/*.ipynb`，定稿前完整性闸门核对+还原+冒烟）、`skipDev`（M2 已开发，跳开发期）、学生试课跑完整流程（worktree 内 uv sync + 复用基线模型 + L1/L2/L3 真 7B）。

**Tech Stack:** uv（每模块独立项目）、Python 3.11、torch cu128 / llmcompressor 0.12 / transformers v5、`hf` CLI（huggingface-hub）、ipytest/nbval/nbconvert、Workflow 工具 + git worktree（学生隔离）。

**Spec:** [`docs/superpowers/specs/2026-06-25-per-module-self-contained-design.md`](../specs/2026-06-25-per-module-self-contained-design.md)

## Global Constraints

- 版本基线（已实测，2026-06-25）：torch 2.11.0+cu128、transformers 5.10.1（v5，llmcompressor 0.12 必须）、llmcompressor 0.12.0、compressed-tensors 0.17.1、CUDA 12.8 / 驱动 570、Python 3.11。torch 必须走 cu128 index（PyPI 默认 cu130 在 12.8 驱动上失败）。
- vLLM **不在** M2 env（它要 transformers<5，与 llmcompressor v5 冲突）——M2 pyproject 不含 vllm/lm-eval。
- M2 pyproject 依赖 = `envs/quant/pyproject.toml`(HEAD) 去掉 `lm-eval`，name 改 `m2-quant-pipeline`。
- notebook-only：每 Step 一个 `.ipynb`，提交前 `nbconvert --clear-output`。
- H200 cell 用 `if torch.cuda.is_available():` 守卫。
- 顶层 `models/`（已下好 7B+0.5B）**保留**作 `MODEL_CACHE` 缓存源；只删 `envs/`+`scripts/`。
- 在 `course-development` 分支开发。

## File Structure

- Create: `course/m2-quant-pipeline/pyproject.toml`（M2 自包含 env）
- Create: `course/m2-quant-pipeline/uv.lock`（`uv lock` 生成，提交）
- Create: `course/m2-quant-pipeline/scripts/download_model.sh`（下 7B+0.5B，含 MODEL_CACHE）
- Modify: `course/m2-quant-pipeline/README.md`（自包含 setup）
- Modify: `course/m2-quant-pipeline/steps/{s1,s2,s3}*.ipynb`（cell 2 路径 → module-local）
- Modify: `course/NOTEBOOK_CONVENTIONS.md`（路径/跑法 → module-local）
- Modify: `OUTLINE.md`（env 架构章节 → 每模块自包含）
- Modify: `workflows/dev-module.js`（护栏 + skipDev + 自包含感知 + 学生全流程）
- Delete: `envs/`（quant+deploy）、`scripts/`（setup_env.sh+download_model.sh）

---

### Task 1: M2 自包含 env（pyproject.toml + uv.lock）

**Files:**
- Create: `course/m2-quant-pipeline/pyproject.toml`
- Create: `course/m2-quant-pipeline/uv.lock`（uv 生成）

**Interfaces:** Produces `course/m2-quant-pipeline/.venv`（装好 torch cu128/llmcompressor/transformers v5），供 Task 2 的 `hf` CLI、Task 4 的 notebook 执行、Task 8 的学生试课使用。

- [ ] **Step 1: 写 `course/m2-quant-pipeline/pyproject.toml`**

内容（= `envs/quant/pyproject.toml` 去 `lm-eval`、name 改、description 改；cu128 index/sources 原样保留）：

```toml
[project]
name = "m2-quant-pipeline"
version = "0.1.0"
description = "M2 量化流水线 env（llm-compressor oneshot: FP8/AWQ/SmoothQuant）。模块自包含——本模块自己的 uv 项目，不依赖顶层 envs/。"
requires-python = ">=3.10,<3.13"
# torch 必须是 CUDA 12.8 (cu128) build 以匹配本节点驱动 (570.x / CUDA 12.8)：
# PyPI 默认 torch 是 cu130，在 12.8 驱动上 CUDA 初始化失败。经 cu128 index 取 torch 2.11.0+cu128。
# 依赖从 envs/quant 迁来，去掉 lm-eval（那是 M3 评测用的，留给 M3 模块 env）。
dependencies = [
    "torch>=2.10,<2.12",       # cu128 build（见 [tool.uv]）；llmcompressor 0.12 要 torch [2.10,2.12]
    "llmcompressor>=0.9.0",    # 0.12 要 transformers v5
    "compressed-tensors>=0.10",
    "accelerate>=1.0",         # device_map / multi-gpu load during oneshot
    "datasets>=2.20",
    "transformers>=5.0",       # llmcompressor 0.12 requires transformers v5
    "safetensors>=0.4",
    "sentencepiece>=0.2",
    "huggingface-hub>=0.26",   # hub API + hf CLI；[hf-transfer] extra 在 hub 1.x 已移除，单独装下面
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

Run: `uv lock --directory course/m2-quant-pipeline && uv sync --directory course/m2-quant-pipeline`
Expected: 解析成功，生成 `course/m2-quant-pipeline/uv.lock`，建 `.venv`。无 `unsatisfiable` 错误。

- [ ] **Step 3: 冒烟验证导入**

Run: `uv run --directory course/m2-quant-pipeline python -c "import llmcompressor,transformers,compressed_tensors; print('transformers',transformers.__version__,'torch',__import__('torch').__version__)"`
Expected: 打印 `transformers 5.x torch 2.11.0+cu128`，无 ImportError。

- [ ] **Step 4: 提交**

```bash
git add course/m2-quant-pipeline/pyproject.toml course/m2-quant-pipeline/uv.lock
git commit -m "feat(m2): self-contained module env (pyproject+uv.lock, migrated from envs/quant minus lm-eval)"
```

---

### Task 2: 模块内 download_model.sh（7B+0.5B，含 MODEL_CACHE）+ 实测可跑

**Files:**
- Create: `course/m2-quant-pipeline/scripts/download_model.sh`

**Interfaces:** Produces `course/m2-quant-pipeline/models/{Qwen2.5-7B-Instruct,Qwen2.5-0.5B-Instruct}`，供 Task 4 notebook L3 + Task 8 学生试课加载。`MODULE` = `scripts/..`（脚本位置派生，cwd 无关）。

- [ ] **Step 1: 写 `course/m2-quant-pipeline/scripts/download_model.sh`**

```bash
#!/usr/bin/env bash
# course/m2-quant-pipeline/scripts/download_model.sh
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

赋可执行权限：`chmod +x course/m2-quant-pipeline/scripts/download_model.sh`

- [ ] **Step 2: 实测——MODEL_CACHE 复用路径（主路径，免重下）**

Run: `MODEL_CACHE="$PWD/models" bash course/m2-quant-pipeline/scripts/download_model.sh`
Expected: 打印「复用缓存 … -> …（硬链）」，`course/m2-quant-pipeline/models/Qwen2.5-7B-Instruct/config.json` 与 `Qwen2.5-0.5B-Instruct/config.json` 都存在；无 FAILED。

- [ ] **Step 3: 实测——真实下载路径（用 0.5B，体积小）**

清掉 0.5B 的复制品，不带 cache 重下（验 `hf download` 真路径）：
```bash
rm -rf course/m2-quant-pipeline/models/Qwen2.5-0.5B-Instruct
bash course/m2-quant-pipeline/scripts/download_model.sh
```
Expected: 0.5B 走「下载 … ->」，完成后 `config.json` 存在；7B 因已存在跳过。无 FAILED。

- [ ] **Step 4: 提交**

```bash
git add course/m2-quant-pipeline/scripts/download_model.sh
git commit -m "feat(m2): module-local download_model.sh (7B+0.5B, MODEL_CACHE reuse)"
```

---

### Task 3: M2 README 自包含 setup

**Files:**
- Modify: `course/m2-quant-pipeline/README.md`（Setup 段 + 跑法）

**Interfaces:** 学员/agent 按 README 即可从零配好本模块（uv sync + 下模型 + 跑 notebook），不碰顶层。

- [ ] **Step 1: 重写 README 的 Setup 段为自包含**

把现有「## Setup 前置」整段替换为：

```markdown
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
```

把末尾「跑法」行替换为：

```markdown
跑法：在本目录下 `uv run jupyter lab` 打开 notebook；按序填空 → 跑 ipytest 测试 cell → 跑 tiny 验证 cell → 跑 H200 执行 cell。
```

- [ ] **Step 2: 核对无顶层残留引用**

Run: `grep -nE 'envs/(quant|deploy)|scripts/setup_env|<repo>/models|顶层的? ?envs' course/m2-quant-pipeline/README.md`
Expected: 无输出（无残留顶层引用）。

- [ ] **Step 3: 提交**

```bash
git add course/m2-quant-pipeline/README.md
git commit -m "docs(m2): self-contained README setup (uv sync + download + run, module-local)"
```

---

### Task 4: notebook 路径 cell 改 module-local（3 个）+ 路径逻辑验证

**Files:**
- Modify: `course/m2-quant-pipeline/steps/s1_fp8_quant.ipynb`（cell 2）
- Modify: `course/m2-quant-pipeline/steps/s2_awq_quant.ipynb`（cell 2）
- Modify: `course/m2-quant-pipeline/steps/s3_smoothquant.ipynb`（cell 2）

**Interfaces:** notebook 用 `MODULE_ROOT`/`MODEL_DIR`/`TINY_MODEL_DIR`/`OUT_ROOT`（module-local），后续 L2/L3/artifact cell 不变（它们引用这些变量名）。`MODEL_DIR` 指向 `<module>/models/Qwen2.5-7B-Instruct`，与 Task 2 的下载脚本一致。

- [ ] **Step 1: 替换 3 个 notebook 的 cell 2**

3 个 notebook 的 cell 2 当前内容相同（`REPO_ROOT = git rev-parse…`）。全部替换为：

```python
# 解析模块根（cwd 无关、git 无关）：从 cwd 向上找同时含 steps/ 和 pyproject.toml 的目录。
# notebook 通过 `cd course/m2-quant-pipeline && uv run jupyter lab` 启动，但 jupyter 的
# cwd 是所在 shell 的 cwd，所以这里自己解析模块根，绝不依赖裸相对路径或仓库根。
import pathlib

def _find_module_root(start):
    p = pathlib.Path(start).resolve()
    for cand in [p, *p.parents]:
        if (cand / "steps").is_dir() and (cand / "pyproject.toml").exists():
            return cand
    raise RuntimeError("找不到模块根（含 steps/ + pyproject.toml 的目录）；请在模块目录内启动 jupyter")

MODULE_ROOT = _find_module_root(pathlib.Path.cwd())
MODEL_DIR      = MODULE_ROOT / "models" / "Qwen2.5-7B-Instruct"      # 与 scripts/download_model.sh 一致
TINY_MODEL_DIR = MODULE_ROOT / "models" / "Qwen2.5-0.5B-Instruct"     # L3 先在 0.5B 上验，再上 7B
OUT_ROOT       = MODULE_ROOT / "out"                                 # 已 gitignore
OUT_ROOT.mkdir(parents=True, exist_ok=True)
print("MODULE_ROOT:", MODULE_ROOT)
print("GPU OK" if __import__("torch").cuda.is_available() else "无 GPU（仅 L1/L2 可跑）")
```

（编辑方式：用 nbformat 或 jupyter UI 改 cell 2 的 source；保持 cell 类型为 code、不改其它 cell。）

- [ ] **Step 2: 验证路径逻辑（在模块 env 内，从模块目录启动）**

Run:
```bash
cd course/m2-quant-pipeline
uv run python -c "
import pathlib
def _find_module_root(start):
    p = pathlib.Path(start).resolve()
    for cand in [p, *p.parents]:
        if (cand/'steps').is_dir() and (cand/'pyproject.toml').exists(): return cand
    raise RuntimeError('not found')
m = _find_module_root(pathlib.Path.cwd())
assert m.name == 'm2-quant-pipeline', m
assert (m/'models'/'Qwen2.5-7B-Instruct').is_dir(), 'models 未下载？先 bash scripts/download_model.sh'
print('module root OK:', m)
"
```
Expected: 打印 `module root OK: .../course/m2-quant-pipeline`，无 AssertionError。（完整 L1/L2/L3 执行由 Task 8 的双重验收覆盖。）

- [ ] **Step 3: 清输出 + 提交**

```bash
for nb in course/m2-quant-pipeline/steps/*.ipynb; do
  uv run --directory course/m2-quant-pipeline jupyter nbconvert --clear-output --inplace "$nb"
done
git add course/m2-quant-pipeline/steps/
git commit -m "refactor(m2): notebook path cells -> module-local (find module root, drop repo-root dep)"
```

---

### Task 5: 更新 NOTEBOOK_CONVENTIONS.md + OUTLINE.md

**Files:**
- Modify: `course/NOTEBOOK_CONVENTIONS.md`
- Modify: `OUTLINE.md`

**Interfaces:** 约定文档与大纲描述「每模块自包含」，与 Task 1–4 的现实一致；dev-module.js 的 agent 读它们（Task 7）。

- [ ] **Step 1: NOTEBOOK_CONVENTIONS.md —— cell 2 路径说明 + 跑法 + L3 示例改 module-local**

把现有「cwd 无关路径解析…`uv run --directory envs/quant jupyter lab`…必须自行解析仓库根」那段（约第 7 行）替换为：

```markdown
   - **cwd 无关路径解析**：notebook 通过 `cd course/<module> && uv run jupyter lab` 启动，但 jupyter 的 cwd 是*所在 shell 的* cwd，所以路径绝不依赖裸相对路径。notebook 自己**向上发现模块根**（含 `steps/` + `pyproject.toml` 的目录），从模块根派生 `models/`、`out/`：
     ```python
     def _find_module_root(start):
         p = pathlib.Path(start).resolve()
         for cand in [p, *p.parents]:
             if (cand / "steps").is_dir() and (cand / "pyproject.toml").exists():
                 return cand
         raise RuntimeError("在模块目录内启动 jupyter")
     MODULE_ROOT = _find_module_root(pathlib.Path.cwd())
     ```
```

把 H200 执行 cell 示例里硬编码的 `/shared/models/...` 改为 `MODEL_DIR`（module-local 变量）；「跑法」改为 `cd course/<module> && uv run jupyter lab`。

- [ ] **Step 2: OUTLINE.md —— env 架构章节改为每模块自包含**

针对 grep 命中的行，按下面改（保持版本基线表不变）：
- 第 52 行（uv 行）：把「管理两个隔离 env（`envs/quant`、`envs/deploy`）」改为「每模块一个独立 uv 项目目录（`course/<module>/`，各自 `pyproject.toml`+`uv.lock`+`.venv`）」。
- 第 59、81、142 行（路径 A 两 env 架构）：把「路径 A：两个独立 uv 项目目录 `envs/quant`+`envs/deploy`」整段改为「**每模块自包含 uv 项目**：`course/<module>/pyproject.toml`+`uv.lock`+`.venv`+`scripts/`+`models/`+`README`，顶层不再有 `envs/`/`scripts/`。M2/M1 用 quant env（llmcompressor），M3/M4 各自模块内建 vLLM env。」
- 第 72、76、423–430 行（uv 命令示例）：把 `uv sync --directory envs/quant` 等改为 `cd course/<module> && uv sync`；删去 `setup_env.sh` 相关。
- 第 147、152、210 行（动手 setup_env/download_model）：改为「在模块目录内 `uv sync` + `bash scripts/download_model.sh`（下到模块 `./models/`，可 `MODEL_CACHE` 复用）」。
- 第 364、477 行：`uv run --directory envs/deploy` → `cd course/m4-deploy-loop && uv run`；`envs/quant/pyproject.toml 的 transformers>=4.45→>=5.0` 改为「各模块 pyproject 按 llmcompressor 0.12 要求 `transformers>=5.0`」。

- [ ] **Step 3: 核对无悬空顶层 envs/scripts 引用（除历史 docs/）**

Run: `grep -rnE 'envs/(quant|deploy)|scripts/setup_env' OUTLINE.md course/NOTEBOOK_CONVENTIONS.md`
Expected: 无输出。

- [ ] **Step 4: 提交**

```bash
git add course/NOTEBOOK_CONVENTIONS.md OUTLINE.md
git commit -m "docs: conventions + OUTLINE -> per-module self-contained env architecture"
```

---

### Task 6: 删除顶层 envs/ + scripts/，核对无悬空引用

**Files:**
- Delete: `envs/`（quant+deploy）、`scripts/`（setup_env.sh+download_model.sh）

**Interfaces:** 仓库顶层不再有 env/scripts；所有引用已在 Task 3–5 改为 module-local。

- [ ] **Step 1: 全仓核对引用（排除历史 docs/ 与 .venv/models/out）**

Run: `grep -rnE 'envs/(quant|deploy)|scripts/(setup_env|download_model)' --include='*.md' --include='*.sh' --include='*.js' --include='*.ipynb' . | grep -vE '/(\.venv|models|out)/|^\./docs/superpowers/(plans|specs)/2026-06-24|^\./docs/REVIEW'`
Expected: 无输出（仅历史 2026-06-24 plan/spec 与 REVIEW.md 残留，属存档，不动）。

- [ ] **Step 2: 删除**

```bash
git rm -r envs scripts
```

- [ ] **Step 3: 冒烟——模块 env 仍可用（不依赖顶层）**

Run: `uv run --directory course/m2-quant-pipeline python -c "import llmcompressor,transformers; print('ok')"`
Expected: 打印 `ok`。

- [ ] **Step 4: 提交**

```bash
git commit -m "chore: remove top-level envs/ + scripts/ (migrated to per-module self-contained)"
```

---

### Task 7: dev-module.js 改造（护栏 + skipDev + 自包含感知 + 学生全流程）

**Files:**
- Modify: `workflows/dev-module.js`

**Interfaces:** 升级后的 workflow 供 Task 8 以 `{module:"m2-quant-pipeline", skipDev:true}` 调用，跑架构师→QA→学生双重验收，并自带护栏防 agent 改坏 env/脚本。

- [ ] **Step 1: 各内容 agent prompt 收口编辑范围 + 改 envs/quant 引用**

在【开发】【架构审核】【QA】【学生】每个 agent 的 prompt 里：
- 把「在 envs/quant 跑 L1…」「uv sync(envs/quant)」等改为「在模块目录内 `cd course/<module> && uv sync`，notebook 用 module-local 路径（`MODULE_ROOT`/`MODEL_DIR`），跑法 `cd course/<module> && uv run jupyter lab`」。
- 追加收口句：`编辑范围：只能创建/修改 course/<module>/steps/*.ipynb。严禁改 pyproject.toml、uv.lock、scripts/、workflows/、README.md、NOTEBOOK_CONVENTIONS.md、顶层任何文件。遇到 env/依赖/CLI 问题写进 findings 报告，不要自己改。`

- [ ] **Step 2: 加 skipDev（跳过开发期，M2 用）**

把现有开发段：
```js
phase('开发')
const devReport = await agent(`你是【课程开发专家】...`, { label: '开发专家', phase: '开发', schema: DEV_REPORT })
```
改为：
```js
phase('开发')
let devReport
if (args.skipDev) {
  // notebook 已存在且 L1/L2 已验过；只做清单，跳过开发期
  devReport = await agent(`列出模块 ${MODULE}（${MODULE_PATH}/steps/）现有 notebook 清单。
对每个 .ipynb：路径 + 填空数（含 NotImplementedError/TODO 的 code cell 数）。
按 schema 报 notebooks + blanksTotal（execResults 给空数组，notes 写 "skipDev"）。`,
    { label: '清单(skipDev)', phase: '开发', schema: DEV_REPORT })
  log(`skipDev：跳过开发，清单 ${devReport.notebooks.length} 个 notebook`)
} else {
  devReport = await agent(`你是【课程开发专家】...（原开发 prompt，含收口句）...`, { label: '开发专家', phase: '开发', schema: DEV_REPORT })
}
```

- [ ] **Step 3: 加完整性闸门 schema + 定稿前闸门 phase**

在 schema 区追加：
```js
const GATE = {
  type: 'object', additionalProperties: false,
  properties: {
    violations: { type: 'array', items: { type: 'string' } },   // steps/ 之外被改的文件
    reverted: { type: 'array', items: { type: 'string' } },      // 已还原的文件
    smokeOk: { type: 'boolean' },                                // uv sync + import 冒烟
  },
  required: ['violations', 'reverted', 'smokeOk'],
}
```
在【Student 试课】loop 之后、【定稿】之前插入：
```js
phase('完整性闸门')
const gate = await agent(`核对模块 ${MODULE} 开发未污染脚手架。
1. 跑 \`git status --porcelain\` 列所有改动。
2. \`course/${MODULE}/steps/\` 之外的改动都是违规（尤其 pyproject.toml/uv.lock/scripts/workflows/README/顶层）。
3. 对每个违规文件 \`git checkout HEAD -- <file>\` 还原；steps/ 下任何文件一律不动。
4. 冒烟：\`cd course/${MODULE} && uv sync && uv run python -c "import llmcompressor,transformers;print('ok')"\`。
按 schema 报 violations/reverted/smokeOk。`, { label: '完整性闸门', phase: '完整性闸门', schema: GATE })
log(`完整性闸门: violations=${gate.violations.length}, smokeOk=${gate.smokeOk}`)
```

- [ ] **Step 4: 学生试课 prompt 改为完整流程 + 复用基线模型**

把学生 agent prompt 中「uv sync(envs/quant) → 按 README 拉模型 → 填空 → ipytest → tiny → H200」段改为：
```
在隔离 worktree 内跑完整流程：
  cd ${MODULE_PATH}
  uv sync
  # 复用主仓已下基线模型，免重下（worktree 是独立工作树）：
  MAIN="$(git rev-parse --git-common-dir)/.." && MODEL_CACHE="$MAIN/models" bash scripts/download_model.sh
  # （若该路径无模型则脚本会正常下载）
  uv run jupyter lab  # 或逐个 nbconvert --execute
然后自己填空(实现 logic 函数) → 跑 L1(ipytest) → L2(tiny) → L3(真 Qwen2.5-0.5B 再 7B H200)。记录能否填出、跑通/报错。
```
（其余 codeRun/designIssues/satisfied/report 要求不变。）

- [ ] **Step 5: acceptance 加 scaffoldClean**

把定稿的 acceptance 对象加一项：
```js
  scaffoldClean: gate && gate.violations.length === 0 && gate.smokeOk,
```

- [ ] **Step 6: 静态自检**

Run: `node --check workflows/dev-module.js 2>&1 || echo "(Workflow 脚本非纯 Node；人工核对 meta 纯字面量、无 Date.now/Math.random)"`
Expected: 无明显语法错；人工确认 meta 纯字面量。

- [ ] **Step 7: 提交**

```bash
git add workflows/dev-module.js
git commit -m "feat(workflow): guardrail (integrity gate) + skipDev + self-contained awareness + student full-flow"
```

---

### Task 8: 双重验收——跑 dev-module.js（skipDev）

**Files:**
- 由 workflow 评审/学生改：`course/m2-quant-pipeline/steps/*.ipynb`（若有合理修订）

**Interfaces:** 调用 Task 7 的 workflow，`{module:"m2-quant-pipeline", skipDev:true}`。验收对齐 spec §7.6。

- [ ] **Step 1: 跑 workflow（后台，完成通知）**

```
Workflow({ scriptPath: "workflows/dev-module.js", args: { module: "m2-quant-pipeline", skipDev: true } })
```
用 `/workflows` 看实时进度（架构审核 → QA loop → 学生试课 loop → 完整性闸门 → 定稿）。

- [ ] **Step 2: 取回 acceptance 核对**

workflow 完成后读返回的 `acceptance`，全部要满足：
- `qaResult.verdict === "pass"`
- `studentResult.satisfied === true`
- `scaffoldClean === true`（闸门无违规、冒烟过）

- [ ] **Step 3: 不通过则迭代**

若未全过：按 `qaResult.findings` / `studentResult` 直接改 notebook（或重跑对应阶段），再验证。封顶（QA/学生各 3 轮）仍不过 → 升级人工，带上 findings。

- [ ] **Step 4: 清输出 + 提交评审/学生修订**

```bash
for nb in course/m2-quant-pipeline/steps/*.ipynb; do
  uv run --directory course/m2-quant-pipeline jupyter nbconvert --clear-output --inplace "$nb"
done
git add course/m2-quant-pipeline/
git diff --cached --stat   # 核对只动 steps/*.ipynb（护栏：不应有 env/脚本改动）
git commit -m "feat(m2): double-acceptance pass (architect+QA+student trial) on self-contained module"
```

---

## Self-Review（写完自查）

**Spec 覆盖**：§4.1→Task1；§4.2→Task2；§4.3→Task3；§4.4→Task4；§4.5/4.6→Task5；§3 删除+§5 引用→Task6；§4.7→Task7；§7.6 双重验收→Task8。验收 1→T1S3，2→T2S2/S3，3→T3S2，4→T4S2+T8，5→T5S3+T6S1，6→T8S2。全覆盖。

**占位符**：无 TBD/TODO；脚本/pyproject/notebook cell 均给了完整代码。

**类型一致**：`MODULE_ROOT/MODEL_DIR/TINY_MODEL_DIR/OUT_ROOT` 在 Task4 定义、Task8 学生 prompt 引用一致；`gate.violations/reverted/smokeOk`、`acceptance.scaffoldClean` 在 Task7 定义/引用一致；`args.skipDev` 定义(Task7S2)与调用(Task8S1)一致。

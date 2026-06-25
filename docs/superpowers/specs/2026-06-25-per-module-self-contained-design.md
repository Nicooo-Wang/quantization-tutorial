# 每节课自包含重构 + 双重验收 设计

> 日期：2026-06-25　分支：`course-development`　关联：`docs/superpowers/specs/2026-06-24-course-development-design.md`（旧架构 spec，历史存档）

## 1. 背景与动机

M2 已在旧架构下开发完成（3 个 notebook 已提交 `ff9a6b4`，L1/L2 人工复验通过）。旧架构是**集中式**：顶层 `envs/quant`（量化 env）+ `envs/deploy`（vLLM env）+ 顶层 `scripts/setup_env.sh`、`scripts/download_model.sh`，模型下到顶层 `models/`，notebook 用 `uv run --directory envs/quant`。

问题：每节课的 env、模型、脚本、setup 全堆在仓库顶层，**不自包含**——学员打开一节课，要回仓库根去配 env、找脚本、理解全局布局。且 dev-module workflow 的内容 agent 上一轮把顶层 env/脚本"修"坏了（重加 vllm 致 `uv` 解析失败）。

新方向：**每节课自包含**——`course/<module>/` 下放该模块自己的 `pyproject.toml`+`uv.lock`+`scripts/`+`models/`+`README`，顶层 `envs/`/`scripts/` 退役。并在自包含结构上跑**双重验收**（架构师审核 → QA 评审 → 学生试课）。

**本次范围：M2 作为试点**。M1/M3/M4 之后按同一模板做（各自的 `dev-module.js` 跑）。

## 2. 目标 / 非目标

**目标**
- M2 目录自包含：env + 模型 + 脚本 + README 全在 `course/m2-quant-pipeline/`。
- `bash scripts/download_model.sh` 直接执行能下到本模块 `models/`（7B+0.5B），并支持 `MODEL_CACHE` 复用已下模型，避免重复下载。
- README 把 uv 初始化/同步/下模型/跑 notebook 的命令写齐，不依赖顶层。
- notebook 路径鲁棒（module-local，不依赖 cwd/git 根）。
- `dev-module.js` 升级：自包含感知 + 护栏（防 agent 改坏 env/脚本）+ `skipDev`（M2 跳过开发期）+ 学生试课跑**完整流程（含 L3 真 7B）**。
- 跑双重验收：架构师审核 → QA loop（到 pass）→ 学生试课 loop（到满意）。

**非目标**
- 不动 M2 三个 notebook 的**教学内容/填空设计/教学法**（只改 env/path 相关 cell）。
- 不开发 M1/M3/M4 内容。
- 不重建 deploy env（删 `envs/deploy`；M4 建模块时按自包含模板重建，内容已存档见 §8）。
- 不改写历史 plan/spec（`docs/superpowers/plans/2026-06-24-*`、旧 spec、`docs/REVIEW.md` 保持原样作存档）。

## 3. 目标目录结构（M2）

```
course/m2-quant-pipeline/
  README.md                  # 自包含 setup：uv sync + 下模型 + 跑法（命令写齐）
  pyproject.toml             # 从 envs/quant/pyproject.toml 迁来（路径无关化）
  uv.lock                    # 提交（在模块目录内 uv lock 重生成）
  scripts/
    download_model.sh        # 默认下到 ./models/；MODEL_CACHE=<dir> 复用共享缓存
  steps/
    s1_fp8_quant.ipynb
    s2_awq_quant.ipynb
    s3_smoothquant.ipynb
  # 运行时产物（已被 .gitignore 的 .venv/ models/ out/ 覆盖，本模块内）
```

删除：顶层 `envs/`（quant + deploy）、顶层 `scripts/`（setup_env.sh + download_model.sh）。
保留：顶层 `models/`（已 gitignore，里面已下好 7B+0.5B）作为 `MODEL_CACHE` 的共享缓存源——M2 首次跑可 `MODEL_CACHE=<repo>/models` 复用，免重下。

`.gitignore` 现有非锚定规则 `.venv/`、`models/`、`out/` 已自动覆盖模块内同名目录，无需改。

## 4. 组件设计

### 4.1 `pyproject.toml` 迁移
- 在 `course/m2-quant-pipeline/pyproject.toml` 写**M2 所需依赖子集**（自包含 = 每模块只装自己要的）：torch（cu128）/ `llmcompressor>=0.9` / `compressed-tensors` / `transformers>=5.0` / `accelerate` / `datasets` / `safetensors` / `sentencepiece` / `huggingface-hub[hf-transfer]` / `ipytest` / `nbval` / `nbconvert` / `jupyter`。**去掉 `lm-eval`**（那是 M3 评测用的，留给 M3 模块 env）。版本下界沿用已验证的 `envs/quant`。
- `name = "m2-quant-pipeline"`；`requires-python>=3.10,<3.13` 保留；`[tool.uv]` cu128 index 配置保留（torch 必须走 cu128 匹配 12.8 驱动）。
- 在模块目录内 `uv lock` 重生成 `uv.lock` 并 `uv sync` 建 `.venv`。
- 冒烟：`uv run --directory course/m2-quant-pipeline python -c "import llmcompressor,transformers;print('ok')"`。

### 4.2 `scripts/download_model.sh`（模块内）
- `MODULE` = 脚本所在目录的父目录（`scripts/..` = 模块根），不依赖 cwd。
- 默认 `MODEL_DIR=$MODULE/models/<repo-basename>`。
- **下载 7B + 0.5B 两个**（notebook 都要用：0.5B 先验、7B 出产物）——循环两个 repo id。
- `MODEL_CACHE=<dir>`（环境变量，可选）：下载/复用到缓存目录，再硬链进 `$MODULE/models/`；同卷硬链，跨卷回退拷贝（软链不用于模型文件，避免 vLLM/transformers 解析问题）。
- 用 `hf` CLI（`huggingface-cli` 在 hub 1.x 已移除，`hf` 前向兼容）。
- 下载后校验每个 `config.json` 存在且可读。
- **验收（实测）**：① `MODEL_CACHE=<repo>/models bash scripts/download_model.sh`（复用已下好的顶层模型）跑通；② 不带 cache 跑 0.5B 子路径验真实下载（7B 太大，0.5B 小可真下）。

### 4.3 `README.md`（自包含 setup）
Setup 段重写为完全在本目录内：
```bash
cd course/m2-quant-pipeline
uv sync                            # 用 uv.lock 建 .venv 并装依赖
bash scripts/download_model.sh     # 下 7B+0.5B 到 ./models/（可选 MODEL_CACHE=<dir> 复用）
uv run jupyter lab                 # 打开 steps/*.ipynb，按序填空→ipytest→tiny→H200
```
校准数据准备仍在各 notebook 首 cell（已是）。删去旧的对顶层 `scripts/setup_env.sh`、`envs/quant`、`<repo>/models` 的引用。

### 4.4 notebook 路径改造（module-local，3 个 notebook 的 cell 2）
现状：`REPO_ROOT = git rev-parse --show-toplevel`；`MODEL_DIR = REPO_ROOT/"models"/...`。
改为**向上发现模块根**（对 cwd 鲁棒，不依赖 git）：
```python
def _find_module_root(start):
    p = start.resolve()
    for cand in [p, *p.parents]:
        if (cand / "steps").is_dir() and (cand / "pyproject.toml").exists():
            return cand
    raise RuntimeError("找不到模块根（含 steps/ + pyproject.toml 的目录）")
MODULE_ROOT = _find_module_root(pathlib.Path.cwd())
MODEL_DIR      = MODULE_ROOT / "models" / "Qwen2.5-7B-Instruct"
TINY_MODEL_DIR = MODULE_ROOT / "models" / "Qwen2.5-0.5B-Instruct"
OUT_ROOT       = MODULE_ROOT / "out"            # 已 gitignore
OUT_ROOT.mkdir(parents=True, exist_ok=True)
```
cell 1 的 `%%capture` 导入不变。改完 3 个 notebook 需重跑 L1/L2/L3 复验（双重验收覆盖）。

### 4.5 `NOTEBOOK_CONVENTIONS.md` 更新
- cell 2 说明改为"向上发现模块根（含 `steps/`+`pyproject.toml`），路径 module-local"。
- L3 H200 示例与产物检查示例改为 module-local 路径。
- "跑法"改为 `cd course/<module> && uv run jupyter lab`。

### 4.6 `OUTLINE.md` 更新（架构章节）
把描述"两 env 架构（路径 A：`envs/quant`+`envs/deploy`）"的章节（~§技术栈、§环境隔离、§动手、§命令示例）改为"**每模块自包含 uv 项目目录**（`course/<module>/pyproject.toml`+`uv.lock`+`.venv`+`scripts/`），顶层不再有 envs/scripts"。保留版本基线表（vLLM 0.x / llmcompressor 0.12 / transformers v5 / torch cu128 / CUDA 12.8）。M3/M4 的 deploy env 表述改为"各自模块内建 vLLM env"。

### 4.7 `workflows/dev-module.js` 改造
1. **自包含感知**：各 agent prompt 把"模块目录 = 自包含（pyproject+uv.lock+scripts+README+steps），notebook 用 module-local 路径，跑法 `cd <module> && uv sync && bash scripts/download_model.sh && uv run jupyter lab`"写进去。CONV 引用不变（已是 module 通用）。
2. **护栏（防 agent 改坏 env/脚本）**：
   - 每个 agent prompt 收口："只能创建/修改 `course/<module>/steps/*.ipynb`；**严禁**改 `pyproject.toml`、`uv.lock`、`scripts/`、`workflows/`、`README.md`、`NOTEBOOK_CONVENTIONS.md`、顶层任何文件。遇到 env/依赖/CLI 问题写进 findings 报告，不要自己改。"
   - 定稿前加**完整性闸门 agent**：`git status --porcelain` 核对；`course/<module>/steps/` 之外的改动视为违规并 `git checkout HEAD -- <file>` 还原（steps/ 下任何文件不动）；冒烟 `uv run --directory course/<module> python -c "import llmcompressor,transformers;print('ok')"`。按 schema 报 `{violations, reverted, smokeOk}`，计入 `acceptance.scaffoldClean`。
3. **`skipDev` 选项**：`args.skipDev===true` 时跳过【开发】阶段（notebook 已存在且已验过 L1/L2），直接进【架构审核】。M2 本次用。
4. **学生试课全流程**：学生 agent 在 worktree 内跑完整流程——`cd <module> && uv sync` → `MODEL_CACHE=<repo>/models bash scripts/download_model.sh`（复用基线，免每轮重下）→ 自己填空 → 跑 **L1（ipytest）+ L2（tiny）+ L3（真 7B H200）** → 报 `codeRun`（filledBlanksOk/notebooksRan/errors）+ `designIssues` + `satisfied`；不满意由架构师+评审改→再 loop（封顶 3 轮）。

## 5. 数据流（学员/agent 跑一节课）

```
cd course/m2-quant-pipeline
  → uv sync               （读 uv.lock → 建 .venv → 装依赖）
  → bash scripts/download_model.sh   （hf 下载 7B+0.5B → ./models/；MODEL_CACHE 可复用）
  → uv run jupyter lab    （notebook cell: 向上发现模块根 → 读 ./models/ → 填空 → L1/L2/L3 → 产物写 ./out/）
```

每节课目录是独立 uv 项目，互不影响；M1/M3/M4 各自重复此流。

## 6. 错误处理 / 鲁棒性

- **模块根发现失败**：`_find_module_root` 找不到含 `steps/`+`pyproject.toml` 的目录 → 抛清晰错误（提示在模块目录内启动）。
- **MODEL_CACHE 跨卷**：硬链失败（EXDEV）→ 回退拷贝；记录日志。
- **下载校验**：每个模型下载后检查 `config.json`；缺失则报错退出。
- **护栏闸门**：发现违规改动 → 还原 + 冒烟；冒烟失败 → `acceptance.scaffoldClean=false`，定稿标红，升级人工。
- **L3 无 GPU**：cell 仍 `if torch.cuda.is_available()` 守卫；无 GPU 记录 skip（H200 节点会真跑）。

## 7. 验收标准（"完成"的定义）

**执行顺序**：先由计划执行者完成 §4.1–4.6 的重构 + §5 引用更新，满足验收 1–5；再跑 `dev-module.js`（`skipDev:true`）满足验收 6（双重验收）。

1. `cd course/m2-quant-pipeline && uv sync` 成功；`import llmcompressor,transformers` 冒烟过。
2. `bash scripts/download_model.sh`（MODEL_CACHE 复用）跑通，`./models/` 有 7B+0.5B 的 `config.json`；0.5B 真实下载路径也验过。
3. README 的 setup 命令逐条可跑，无对顶层 `envs/`/`scripts/`/`<repo>/models` 的残留引用。
4. 3 个 notebook 路径 cell 改为 module-local 后，注入参考实现跑 L1（ipytest 全过）+ L2（tiny 真跑）+ L3（真 7B H200，产出 compressed-tensors 产物）全过。
5. 顶层 `envs/`、`scripts/` 已删；`OUTLINE.md`/`NOTEBOOK_CONVENTIONS.md`/`workflows/dev-module.js` 引用已更新，无悬空引用。
6. 双重验收：`dev-module.js`（skipDev）跑完，`qaResult.verdict==="pass"` 且 `studentResult.satisfied===true` 且 `acceptance.scaffoldClean===true`。

## 8. 存档：deploy env（M4 用）

`envs/deploy/pyproject.toml`（删除前内容，M4 建模块时在 `course/m4-deploy-loop/pyproject.toml` 重建）：
```toml
[project]
name = "deploy-env"
version = "0.1.0"
requires-python = ">=3.10,<3.13"
dependencies = [
    "vllm>=0.11.0",                       # CUDA-12.8 wheel 自带匹配的 torch/transformers
    "huggingface-hub[hf-transfer]>=0.26",
    "compressed-tensors>=0.10",
]
[tool.uv]
[tool.uv.pip]
```
M3（评测）若需 vLLM eval backend，在其模块 env 内额外 `uv pip install --python <venv>/bin/python 'lm_eval[vllm]'`（OUTLINE §uv 陷阱已记：`uv pip` 不认项目 venv，必须显式 `--python`）。

## 9. 取舍记录

| 决策 | 选择 | 理由 | 备选（未选） |
|---|---|---|---|
| pyproject 来源 | 迁移 envs/quant | 保留已验证 deps，免重验 | 重写（浪费、风险） |
| M2 workflow 范围 | `skipDev` 跳开发 | notebook 已存在且 L1/L2 已验 | 全量重跑（重新生成已验证内容、agent 改坏风险） |
| 学生试课模型 | `MODEL_CACHE` 复用 | 免每 worktree/每轮重下 15GB | 每 worktree 重下（慢、费盘） |
| 模型复用机制 | 硬链（跨卷拷贝） | 同卷零拷贝、路径仍是模块内 | 软链（解析风险） |

## 10. 后续（本设计之后）

- 本设计批准 → 写实施计划（writing-plans）→ 执行。
- 执行完 M2 自包含重构 + 双重验收通过 → M1/M3/M4 各自按本模板（`dev-module.js` 全流程或 skipDev）开发。

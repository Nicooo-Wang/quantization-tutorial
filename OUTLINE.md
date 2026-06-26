# 大语言模型量化实战：H200×8 工业级端到端流水线（SmoothQuant + AWQ + FP8）

> 本大纲由 4 人 Agent 团队协作产出，并经量化技术专家 + 代码导师对 **2025–2026 当前版本库 API（vLLM / llm-compressor / AutoAWQ / lm-eval）Web 查证**后定稿。版本敏感的命令与 flag 均已核对官方文档，关键修正见 [附录 A](#附录-a版本敏感-api-速查) 与 [`docs/REVIEW.md`](docs/REVIEW.md)。
>
> **课程主线**：`M2 量化 → M3 精度调优(layer fallback) → M3 评测 → M4 vLLM 声明式部署`，三种方法（SmoothQuant / AWQ / FP8）贯穿，端到端闭环以 SmoothQuant 为范例走通。
>
> **本轮修订关键升级**：Python 环境全面迁移到 **uv** 管理（**每模块一个独立 uv 项目目录** `course/<module>/`）；M4 明确声明式部署、无需改模型代码；M4.6 去冗余收敛。

---

## 课程定位

一门**面向工业落地**的 LLM 量化实战课。在 **NVIDIA H200×8** 上，用 **SmoothQuant（W8A8）+ AWQ（W4A16）+ FP8** 三种方法，完整跑通「端到端量化 → 精度调优 → 多维评测 → vLLM 多卡部署」的工业闭环，基线模型 **Qwen2.5-7B**。

## 目标学员

算法工程师 / 推理工程师 / AI 应用开发者，**已掌握**对称/非对称量化、per-token / per-channel / per-group 粒度等量化基础（本课跳过基础数学，直接从激活离群点起步），希望在 H200 这类数据中心卡上把量化做到生产可部署。

## 先修知识

- 熟练 Python / PyTorch（能读懂 `nn.Module`、`forward`、`torch.cuda` 调用）
- 了解 Decoder-only LLM 结构（Attention / FFN / RMSNorm / embedding / lm_head）
- 会用 HuggingFace Transformers 加载与生成（`AutoModelForCausalLM`、`from_pretrained`、`generate`）
- **已掌握**量化基础：对称/非对称、per-tensor/channel/group、scale 与 zero_point 的几何含义
- 了解 Linux 命令行与 Docker 基本操作；会读 TOML 配置文件（`pyproject.toml`）

## 硬件前提：H200×8

| 精度 | H200（Hopper） | 说明 |
|------|--------------|------|
| **FP8**（E4M3/E5M2） | ✅ 原生 Tensor Core，约 4 PFLOPS FP8 | **本课核心红利**，生产首选 |
| FP16/BF16 | ✅ | 基线与回退精度 |
| **NVFP4 / FP4** | ❌ Blackwell（B100/B200）独有 | Hopper 无 FP4 Tensor Core，零 FP4 吞吐；仅作一句话前瞻 |
| INT8/INT4 | ✅（软件层） | W8A8 / W4A16 路径 |

- **H200 = 141GB HBM3e（4.8TB/s）**；H100 = 80GB HBM3（3.35TB/s）。同为 Hopper 架构，FP8/INT8 Tensor Core 能力**完全一致**，但 HBM3e 带宽显著更高（4.8TB/s vs 3.35TB/s）——对 W4A16 访存墙方案和大 KV-Cache 场景有利，这是 H200 相对 H100 的关键差异，**不可简单说「仅显存更大」**。
- 8 卡单节点：纯张量并行 `--tensor-parallel-size 8`，单节点不需要流水线并行（PP）。

## 技术栈与版本基线（重要，版本错配是头号坑）

| 组件 | 版本基线（2025–2026） | 备注 |
|------|---------------------|------|
| CUDA | **12.8**（vLLM wheel 已 rebase 到 12.8 基线） | **driver ≥ 570.x**（570.26 Linux）；不要写成 535（那是 12.1/12.2 的 driver 下限）；用 12.4/12.1 镜像跑最新 vLLM 会 `undefined symbol` |
| vLLM | **PyPI 0.x（当前 0.23.x）** | PyPI 包仍是 0.x（实测 2026-06 pypi.org：0.23.0）。「CalVer 25.x（如 25.09/25.10/25.11）」指的是 NGC/容器镜像 tag 命名，**不是** `pip install vllm` 的版本号 —— `pyproject.toml` 必须用 0.x 下限（`>=0.11.0`），写 `>=25.0` 会解析失败 |
| llm-compressor (`llmcompressor`) | **0.12.x** | `>=0.7` 支持 mixed-precision config_groups；`>=0.9` 覆盖现代 `QuantizationModifier` API；**0.12.x 要求 transformers v5**。常需配合当前 vLLM stable |
| compressed-tensors | **0.17.x** | llm-compressor 三方法（FP8/AWQ/SmoothQuant）的统一产物格式；vLLM `auto` 识别的依据。`pyproject.toml` 下限 `>=0.10` |
| AutoAWQ | **已废弃** | vLLM 官方已废弃 AutoAWQ，AWQ 能力收编进 llm-compressor（输出 compressed-tensors）。仅作遗留路径对照，不作为主路径 |
| lm-eval (`lm_eval`) | **0.4.12** | 核心包**不再内置 model backend** —— 评测 vLLM 必须额外 `cd course/<module> && uv pip install --python ./.venv/bin/python 'lm_eval[vllm]'`，否则 `--model vllm` / `local-completions` 后端不可用。中文任务 `ceval-valid`/`cmmlu` 以 `lm_eval --tasks list` 实测为准 |
| transformers | **5.12.x** | vLLM wheel 自带匹配 transformers，不要另装冲突版本；llmcompressor 0.12 要求 v5 |
| accelerate | **1.14.x** | oneshot 期间 `device_map` / 多卡加载需要 |
| huggingface-hub | **1.20.x** | 用 `huggingface-hub[hf-transfer]` extra 拉 `hf_transfer` 加速；纯 `huggingface-hub` 不含 hf-transfer |
| uv | **0.8.x** | Python 环境与依赖管理器，替代 pip。课程**每模块一个独立 uv 项目目录**（`course/<module>/`，各自 `pyproject.toml` + `uv.lock` + `.venv`），用 `cd course/<module> && uv sync` 复现 |
| Python | **3.11**（所有模块统一基线） | llm-compressor 要求 `>=3.10`；课程锁定 3.11，各模块必须用同一 Python 次版本，否则 vLLM/torch ABI 可能不一致影响产物一致性 |

> **工程建议**：每节课 = 一个独立 uv 项目目录 `course/<module>/`（量化模块用 llmcompressor env，部署模块 M3/M4 各自模块内建 vLLM env），用模块内的 `uv.lock` 固定一套验证过的版本组合。开课前在各模块目录 `uv sync` + commit `uv.lock`。

## uv 工作流（每模块自包含 uv 项目）

采用**每模块自包含 uv 项目**：`course/<module>/pyproject.toml` + `uv.lock` + `.venv` + `scripts/` + `models/` + `README`，顶层不再有 `envs/`/`scripts/`。M2/M1 用 quant env（llmcompressor），M3/M4 各自模块内建 vLLM env。

**理由**：vLLM wheel 自带并锁定 transformers/torch，而 llmcompressor/lm-eval 需要不同的 transformers —— 把它们放一个 env 会让 uv/pip 互相争抢版本；按模块拆成独立 uv 项目后，各自 `uv.lock` 自然隔离，且学员在一个模块内即可闭环（env + 脚本 + 模型 + notebook），不必跨目录。

**工作流三件套职责**（均在模块目录内执行）：
- `cd course/<module>`：进入对应模块目录（每个模块就是一个 uv 项目根）。
- `uv sync`：读该目录 `pyproject.toml` + `uv.lock`，把依赖装入该目录 `.venv`。
- `uv run <cmd>`：在模块 `.venv` 内执行（如 `uv run jupyter lab`、`uv run python recipes/xxx.py`）；env 已 sync 时是 no-op。

**复现性**：来自模块目录内的 `uv.lock`（跨平台精确版本图），不是 `pyproject.toml`。每个模块的 `uv.lock` 都要 commit；新 H200 节点 `cd course/<module> && uv sync` 即可复现。激活方式 `source course/<module>/.venv/bin/activate`。

**关键坑（必须避免）**：

1. 不要在仓库根跑 `uv sync` —— 根目录无 `pyproject.toml`（顶层不再有 envs/scripts），会报 `no project`。必须先 `cd course/<module>` 进入模块目录。
2. 不要另装 torch：vLLM wheel 自带 CUDA 12.8 配对的 torch，单独装 torch 会覆盖该 build，破坏 FP8/INT8 kernel。
3. 环境检查不要用 `uv run`（每次会触发 sync 检查，vLLM env 可能触发数分钟重装）—— 用 `.venv/bin/python` 直接调。
4. **lm-eval 核心包不含 model backend，quant 模块需额外装 `lm_eval[vllm]` extra。但真正风险是 `uv pip` 只认"激活的 env / `--python <path>`"，**不认项目 venv** —— 直接 `uv pip install 'lm_eval[vllm]'` 在未激活时不知道装哪。**正确写法是显式传 `--python` 指向目标 venv 的 python**：
   ```bash
   cd course/<module>
   uv pip install --python ./.venv/bin/python 'lm_eval[vllm]'
   ```

**教学锚点**：每个模块的 `pyproject.toml` 是该模块 env 的唯一事实来源；`scripts/download_model.sh` 是一键拉模型入口（下到模块 `./models/`）。不采用单根 `pyproject.toml` 多 venv（需 `UV_PROJECT_ENVIRONMENT` 环境变量，语义不清晰）。如未来改用 uv workspace（`[tool.uv.workspace] members=course/*`）可共享根 `uv.lock` 同时各自独立 venv，是更集中的演进方向，本轮先用每模块自包含保证清晰。

---

## 课程结构（4 模块 / 共 28 课时）

### 模块 1：激活离群点与现代量化方法（原理层，跳过基础）

**模块目标**：从「激活离群点」这一 LLM 量化头号根因起步，建立 SmoothQuant / AWQ / FP8 三种方法的思想模型与选型判断力。学员已懂基础量化数学，本模块只讲**为什么 LLM 量化难**与**每种方法在优化什么**。

**课时安排**：

#### 1.1 激活离群点：LLM 量化的头号根因（~50 分钟）
- 📌 要点：Dettmers 发现 transformer ≥6.7B 会出现 **emergent features**——少数 channel 激活幅值比其它大 20× 以上，且固定在特定维度、跨 token 持续出现；这 0.1% 的大幅值通道会把其余 99.9% 的有效量化范围"挤压"掉，导致精度坍塌
- 📌 与 CNN 的本质区别：LLM 离群点是**结构性、跨 token 持续**的；CNN 激活分布相对均匀，per-channel 即可覆盖 → LLM 量化方法无法照搬 CNN 经验
- 🛠 动手：hook Qwen2.5-7B 各 Linear 层输入，画 per-channel 激活幅值分布，**亲眼看见 emergent outliers**（实测证伪"7B 还没到出现离群点的规模"）

#### 1.2 LLM.int8()：混合精度分解（基准）（~40 分钟）
- 📌 要点：vector-wise absmax INT8 + mixed-precision decomposition——99.9% 常规维度做 INT8，离群维度切出用 FP16，结果相加
- 📌 关键认知：它是 **W8A8 + FP16 混合**（不是纯 W8A8），激活侧是 vector-wise（非 per-token dynamic），这与 SmoothQuant 输出的纯 W8A8 本质不同；混合精度路径切换带来推理开销
- 📌 定位：作为后续 SmoothQuant 的对照基准（SmoothQuant 直接产出纯 W8A8，更适合 vLLM 部署）

#### 1.3 SmoothQuant：把离群点迁移到权重（W8A8）（~55 分钟）
- 📌 核心思想：通过**数学等价的 per-channel 缩放**，把激活离群点幅值"迁移"到权重上，使两边都易量化 → 变换后可走**纯 INT8 W8A8 GEMM**，无需运行时混合精度
- 📌 公式：`Y = (X·diag(s)⁻¹)·(diag(s)·W)`，其中 `s_j = max(|X_j|)^α / max(|W_j|)^(1-α)`，α∈[0,1] 是迁移强度；α≈0.5 论文最优
- ⚠️ 版本要点：llm-compressor 参数名是 **`smoothing_strength`**（实测 0.12.0 的 `SmoothQuantModifier()` 类默认 = **0.5，即论文 α=0.5**）；课程在 s3/M2 刻意显式取 **α=0.8** 以偏向多迁权重——调用时写 `smoothing_strength=0.8`，别误以为 0.8 是库默认（这是最易写错的点）
- 📌 教学深度：讲透"等价变换不改变数学结果但改变量化难度"这一贯穿全课的公理

#### 1.4 AWQ：激活感知权重显著性 + per-group（W4A16）（~50 分钟）
- 📌 核心思想：不依赖反向传播，而是"激活感知"地给**显著通道**的权重乘缩放因子 s（大权重更抗量化截断），s 的倒数吸收进前一层 RMSNorm/Linear（保证等价）；量化是 per-group INT4，group_size 典型 128
- 📌 关键认知：AWQ 是 **weight-only**（激活全程 FP16）→ 故称 W4A16，是**访存墙**方案而非计算墙方案
- ⚠️ 版本要点：AutoAWQ 默认 `zero_point=True`（非对称）、`q_group_size=128`、`w_bit=4`、`version=GEMM`；**Marlin 内核要求 `zero_point=False`**

#### 1.5 FP8：H200 原生红利（~45 分钟）
- 📌 两种格式：**E4M3**（动态范围小、精度高，适合前向的权重+激活） vs **E5M2**（动态范围大、精度低，适合梯度/反向传播）
- 📌 推理侧结论：**权重和激活都用 E4M3**（推理无梯度，不需 E5M2 的大动态范围）；浮点自带动态范围 → 天然抗离群点，**无需 SmoothQuant** 即可保持精度
- 📌 H200：原生 FP8 Tensor Core，性能 ≈ INT8 W8A8 但精度更好、更简单
- ⚠️ 易错："权重 E4M3、激活 E5M2"是**训练场景**配置，套到推理上是错的

#### 1.6 三范式选型总纲（~40 分钟）
- 📌 **W4A16（AWQ）**：weight-only，4-bit 权重 + FP16 激活，瓶颈是**访存**，适合 batch=1 / 小 batch / 显存极度受限
- 📌 **W8A8 INT8（SmoothQuant）**：权重激活都 8-bit，可用 INT8 Tensor Core GEMM，瓶颈是**计算**，适合大 batch / 吞吐优先
- 📌 **FP8（E4M3 W8A8）**：W8A8 的浮点版，Hopper 上性能≈INT8 但精度更好、无需 SmoothQuant
- ⚠️ **关键行业事实**：2025–2026 共识是 **Hopper GPU 上 FP8-Dynamic 几乎全面替代 INT8-W8A8**（更简单、精度更好、无需 SmoothQuant）。**H200 生产默认是 FP8**；INT8 W8A8 主要用于非 Hopper（Ampere/Ada），本课保留它作为"显式离群点处理 + layer fallback 教学"的最佳载体

#### 1.7 量化方法时间线：为何 PTQ 为主（~30 分钟）
- 📌 2022 LLM.int8 → 2022 GPTQ → 2023 AWQ/SmoothQuant → 2024 AutoRound/HQQ → 2025 NVFP4（Blackwell）
- 📌 PTQ 为 LLM 时代绝对主流（QAT 对千亿模型训练成本不可接受）；QAT 仅作蒸馏/继续训练场景（本课仅概念，见 M4.7）

**🔧 模块动手实验**：在 Qwen2.5-7B 上实测激活幅值分布，定位 emergent outlier 通道，用一段纯 PyTorch 复现 SmoothQuant 的等价变换（`X·diag(s)⁻¹` 与 `diag(s)·W`），验证变换前后 forward 输出数值一致、但激活幅值分布被"压平"。

---

### 模块 2：H200×8 环境与端到端量化流水线（动手）

**模块目标**：用 uv 管理搭好 H200×8 可复现的模块自包含环境（M2/M1 用 quant env、M3/M4 各自模块内建 vLLM env），用 **llm-compressor 统一工具链**产出三种可部署量化模型（FP8 / AWQ / SmoothQuant），讲清方法间的工程差异。这是闭环的起点。

**课时安排**：

#### 2.1 环境搭建：每模块自包含 uv 工作流（~50 分钟）
- 📌 **为什么用 uv 而非 pip**：vLLM wheel 自带并锁定 transformers/torch，而 llmcompressor/lm-eval 需不同 transformers —— 一个 env 会让版本互锁；uv 用各自 `pyproject.toml` + `uv.lock` 自然隔离，且 `uv.lock` 是跨平台精确锁（`pip freeze` 只是扁平列表、传递依赖会漂移）
- 📌 **每模块自包含 uv 项目**：`course/<module>/pyproject.toml` + `uv.lock` + `.venv` + `scripts/` + `models/` + `README`，顶层不再有 `envs/`/`scripts/`；M2/M1 用 quant env（llmcompressor），M3/M4 各自模块内建 vLLM env
- 📌 uv 三件套职责：`cd course/<module>` 进入项目目录、`uv sync` 读 `pyproject`+`uv.lock` 装依赖到模块 `.venv`、`uv.lock` 跨平台锁精确版本图（复现性来源，每个模块 `uv.lock` 都 commit）
- 📌 模块内一键建 env：`cd course/m2-quant-pipeline && uv sync`；模块 `pyproject.toml` 是该 env 唯一事实来源
- 📌 不要另装 torch（vLLM wheel 自带 CUDA 12.8 配对 torch）；lm-eval 核心包不含 model backend，需额外 `cd course/<module> && uv pip install --python ./.venv/bin/python 'lm_eval[vllm]'`
- ⚠️ 关键坑：`uv pip` **只认激活的 env / `--python <path>`，不认项目 venv**。给 quant 模块装 `lm_eval[vllm]` extra 必须先 `cd course/<module>` 再显式 `--python ./.venv/bin/python`（未激活时直接 `uv pip install` 不知道装哪）
- 🛠 动手：在模块目录内 `cd course/m2-quant-pipeline && uv sync` 建 env；`course/m2-quant-pipeline/pyproject.toml` 是 env 唯一事实来源；输出 `nvidia-smi` / `nvcc` / 各库版本核对（vllm 0.23.x、llmcompressor 0.12.x、compressed-tensors 0.17.x、torch CUDA 12.8）

#### 2.2 统一基线 Qwen2.5-7B-Instruct + 中英文校准/评测集（~40 分钟）
- 📌 基线模型：全课统一 **Qwen2.5-7B-Instruct**（与 `download_model.sh` 默认 `MODEL_REPO=Qwen/Qwen2.5-7B-Instruct` 对齐，保证四维对比可比性；课程选 Instruct 因为也要做指令遵循/工具调用评测）
- 📌 ⚠️ **base vs Instruct 差异**：base 模型未做指令微调，在 GSM8K 5-shot / MMLU 5-shot 等指令遵循任务上分数显著低于 Instruct；课程统一用 Instruct 以保证评测可比。如需对照 base，`MODEL_REPO=Qwen/Qwen2.5-7B` 覆盖
- 📌 在模块目录内用 `scripts/download_model.sh` 把模型下到模块 `./models/`（本模块自包含；可在别处下过后用 `MODEL_CACHE=<repo>/models bash scripts/download_model.sh` 复用免重下）；`huggingface-cli download <repo-id> --local-dir <dir>`
- 📌 HF_TOKEN 说明：Qwen2.5-7B-Instruct 非 gated，匿名可下，但设 `HF_TOKEN` 避免匿名限流（脚本自动认 `HF_TOKEN`，无需 `--token` 明文）；`HF_HUB_ENABLE_HF_TRANSFER=1` 加速但必须先装 `hf_transfer` 包（否则报下载失败/被误诊为 unrecognized model，transformers issue #37477）
- 📌 下载后校验 `config.json` `architectures=Qwen2ForCausalLM`，确认声明式部署前提（无需 `trust_remote_code`）
- 📌 校准集：`wikitext2-raw-v1` train split，`shuffle(seed=42).select(range(512))`，每条 2048 tokens（INT8 W8A8 至少 512 样本；FP8 无需校准；AWQ 极省样本 128–256 即可）
- 📌 评测集：中文 `ceval-valid` / `cmmlu` + 英文 `mmlu` / `gsm8k`
- ⚠️ 复现性：`set_seed(42)`；vLLM 推理复现需 `SamplingParams(temperature=0)` + `--enforce-eager`；**lm_eval 必须加 `add_bos_token=True`**（量化模型常见坑，漏加会让 PPL 虚低/虚高）

#### 2.3 FP8 量化全流程（llm-compressor，最简单，H200 首选）（~45 分钟）
- 📌 当前版本确切 API（已查证，**不是**已废弃的 `Float8Modifier`）：
  ```python
  from llmcompressor import oneshot
  from llmcompressor.modifiers.quantization import QuantizationModifier
  recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
  oneshot(model=model, recipe=recipe)   # 注意：FP8 的 oneshot 不传 dataset
  model.save_pretrained(SAVE_DIR)
  ```
- 📌 `scheme="FP8_DYNAMIC"` = 权重 per-channel static E4M3 + 激活 dynamic per-token E4M3；**无需校准数据**（动态激活量化），**无需 SmoothQuant**
- 📌 本节只讲精确 `ignore`（如 `ignore=["lm_head"]`）；**正则 `ignore` 写法（`re:` 前缀）统一放 M3.3 讲**
- ⚠️ 易错：旧博客的 `from llmcompressor.modifiers.quantization import Float8Modifier` 新版会 ImportError

#### 2.4 AWQ W4A16 全流程（llm-compressor 统一路径；AutoAWQ 遗留对照）（~55 分钟）
- 📌 **推荐路径：llm-compressor 的 AWQ modifier**（输出 compressed-tensors，与 SmoothQuant/FP8 工具链统一）——因 AutoAWQ 已被 vLLM 官方废弃
- 📌 **遗留路径对照（AutoAWQ）**：
  ```python
  from awq import AutoAWQForCausalLM
  quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
  model = AutoAWQForCausalLM.from_pretrained(model_path)
  model.quantize(tokenizer, quant_config=quant_config, calib_data=data)  # data 128–256 条
  model.save_quantized(quant_path, safetensors=True, shard_size="4GB")
  ```
- ⚠️ 关键修正：AutoAWQ 的 `quant_config` **没有 `exclude_modules` 字段**（常见误写）；AWQ 排除层走 HF Transformers/vLLM 的 `AwqConfig.modules_to_not_convert`（**加载侧**生效，默认 `['lm_head']`），量化期 AutoAWQ 硬编码跳过 lm_head。课程讲"AWQ 排除层"必须区分**量化期 vs 加载期**
- 📌 *（脚注：Falcon 模型只兼容 group_size=64；要走 Marlin 内核必须 `zero_point=False` + `version="Marlin"`。Qwen2.5 不受此限，故作为脚注而非主流程。）*

#### 2.5 SmoothQuant W8A8 全流程（llm-compressor 两段式）（~55 分钟）
- 📌 当前版本确切 API（**两段式，不是单 modifier**）：
  ```python
  from llmcompressor import oneshot
  from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier
  from llmcompressor.modifiers.gptq import GPTQModifier
  recipe = [
      SmoothQuantModifier(smoothing_strength=0.8),
      GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
  ]
  oneshot(model=model, dataset=ds, recipe=recipe,
          max_seq_length=2048, num_calibration_samples=512)
  model.save_pretrained(SAVE_DIR, save_compressed=True)
  ```
- 📌 `scheme="W8A8"` = 权重 per-channel 对称 INT8 + 激活 **dynamic per-token** 对称 INT8；**必须先 SmoothQuant 平滑**否则 INT8 激活量化精度崩
- ⚠️ 导入路径：`SmoothQuantModifier` 在 `llmcompressor.modifiers.transform.smoothquant`（不是 `modifiers.smoothquant`）；`SmoothQuantModifier()` 类默认 `smoothing_strength=0.5`（=论文 α）；课程显式取 0.8 偏向多迁权重
- ⚠️ `ignore=["lm_head"]` 必须是**列表**（写成字符串会报错）
- 📌 版本说明：以课程锁定的 llmcompressor 版本号为准（0.12.x）；如未来 0.9.0+ 引入 attention quantization/MXFP4/AutoRound，导入路径与 modifier 组合需重新核对

#### 2.6 三方法产物与 compressed-tensors 格式（~35 分钟）
- 📌 llm-compressor 输出统一为 **compressed-tensors** 格式（`config.json` 里 `quantization_config.config_groups`）；AutoAWQ 输出 `quant_method="awq"` 旧格式
- 📌 `quantization_config` 字段是 vLLM `auto` 识别的依据——少了它 vLLM 会当 FP16 加载（HF Hub 发布必带，见 M4.5）
- 📌 `save_pretrained` 会把 tokenizer（含 chat template）与 `generation_config` 一并保存到产物目录，部署侧原样读取，无需写模板代码
- 🛠 动手：产出三个可部署量化模型（`Qwen2.5-7B-FP8` / `-AWQ-W4A16` / `-SmoothQuant-W8A8`），打印各自 `config.json` `quantization_config` 与显存占用对比

**🔧 模块动手实验**：「环境体检 + 三方法量化初体验」——在模块目录内 `cd course/m2-quant-pipeline && uv sync` 建 env，跑 `bash scripts/download_model.sh` 下模型，对 Qwen2.5-7B-Instruct 分别产出 FP8 / AWQ / SmoothQuant 三个量化模型，提交显存对比表（FP16 / FP8 / W4A16 / W8A8）与环境检查 cell 输出。

---

### 模块 3：精度调优与评测（layer fallback 核心，SmoothQuant 为范例）

**模块目标**：本课灵魂模块。以 **SmoothQuant W8A8** 为范例，掌握工业级**敏感层识别 → 混合精度回退 → Pareto 调优 → 多维评测**全流程。（同一流程适用于 AWQ / FP8。）最终产出"最少回退 → 最大精度"的调优结果与四维 Pareto 对比表。

**课时安排**：

#### 3.1 敏感层识别：profiling + 逐层量化测 PPL（~55 分钟）
- 📌 工业标准两步法：(1) **激活异常 profiling**——hook 每个 Linear 输入，统计 per-channel 幅值分布，找方差极大/离群点密集的层；(2) **逐层单独量化测 PPL/lm_eval**——保持其它层 FP16，每次只量化目标层，PPL 跳升者即敏感层
- 📌 加速代理：KL-divergence（base vs quant 模型 forward 输出分布）作为比 PPL 更快的代理指标
- ⚠️ 易错：只看激活幅值不够——有些层激活正常但对量化误差累积敏感，**必须用 PPL/lm_eval 做最终判据**；且敏感度与 scheme 相关（FP8 vs INT4 敏感层不同），不能跨 scheme 复用结论

#### 3.2 "不该量化什么"：工业经验法则（~40 分钟）
- 📌 经验共识：(1) **lm_head 几乎总是回退**（所有官方 example 都 `ignore=["lm_head"]`）；(2) **embedding** 不量化；(3) **MoE router/gate** 回退（量化后路由逻辑崩）；(4) 部分 **RMSNorm/LayerNorm** 不量化（参数少、量化收益小风险大）；(5) **首末几层 transformer block** 有时敏感（需 profile 确认，非铁律）；(6) 某些模型的 **down_proj** 对低比特敏感
- ⚠️ 易错：把"首末层回退"当铁律——Qwen2.5 实测首末层未必最敏感，**必须 profile**；旋转位置编码下 v_proj/k_proj 可能更敏感

#### 3.3 回退机制：llm-compressor ignore 语法（含正则统一讲解）（~45 分钟）
- 📌 `ignore` 接受**字符串列表**，三种写法：(1) 精确匹配 `ignore=["lm_head"]`（M2.3 / M2.5 用到的就是这种）；(2) 正则匹配加 `re:` 前缀 `ignore=["re:.*lm_head", "re:.*mlp.gate$"]`；(3) 混合
- 📌 **正则 ignore 统一在本节讲透**（M2.3 只讲精确 ignore，避免分散）——多个 modifier 共享同一正则语义
- 📌 `targets="Linear"` 是**类名匹配**（所有 nn.Linear）；想按层名用 `ignore` 的正则
- 📌 所有 `QuantizationModifier` / `GPTQModifier` / `SmoothQuantModifier` 都接受 `targets` + `ignore` 这对参数
- ⚠️ 易错：正则**不加 `re:` 前缀不生效**（被当字面字符串）；`ignore` 元素是模块名模式不是层索引

#### 3.4 细粒度混合精度回退：mixed-precision config_groups（课程高阶核心）（~55 分钟）
- 📌 llm-compressor 0.7.0+ 支持 **mixed-precision**：在一次 `oneshot` 里用**多个 QuantizationModifier 实例**（不同 `targets` + `scheme`），自动合并成 compressed-tensors 的 `config_groups`——给敏感层配更高比特、其余层用低比特
- 📌 示例结构：`group_0`（低比特 targets 多数 attention/MLP 层）+ `group_1`（高比特 targets 敏感的 down_proj）
- 📌 **vLLM 0.10.1+ 原生可跑 mixed-precision 模型**——这是"最少回退 → 最大精度"Pareto 优化的工程基础，**优于粗粒度 FP16 全回退**；`ignore` 支持 `re:` 正则前缀
- ⚠️ 易错：mixed-precision **不是手动改 config.json**——应通过多个 modifier 让 llm-compressor 生成正确结构；手动改易漏 `observer`/`observer_kwargs` 字段导致 vLLM 加载失败

#### 3.5 调优流程：逐步回退 → PPL 恢复曲线 → Pareto 拐点（~50 分钟）
- 📌 标准流程：(1) 全量化基线测 PPL/lm_eval（如 gsm8k 5-shot）；(2) 逐层敏感度排序；(3) 按敏感度从高到低**逐步加入 ignore**，每加一层**重新量化并测 PPL**，记录（回退层数, PPL）对；(4) 画 PPL 恢复曲线找拐点（PPL 快速恢复到接近 FP16 的最少回退层数）= Pareto 最优点
- 📌 评估命令：`lm_eval --model vllm --model_args pretrained=MODEL,add_bos_token=True --tasks gsm8k --num_fewshot 5 --limit 250`
- ⚠️ 易错：每次回退**都要重新量化**（GPTQ 的 Hessian 会因 ignore 改变而变，不能只改 config 不重量化）；`limit=250` 太小方差大，最终评估要全量

#### 3.6 group_size 调参（~30 分钟）
- 📌 `group_size=128` 经验最优：太小（32）→ scale 开销占比上升、压缩比下降；太大（256+）→ 组内离群点影响扩大；128 在常见 hidden_size（4096/8192 整数因子）下平衡压缩比与精度
- 📌 调参方向：显存极度紧张可降到 64；精度优先可试 64（每组单独 scale 更精细）；**group_size 必须整除 hidden_size** 否则 vLLM 加载报错
- ⚠️ 易错：group_size 不是越小越好——32 以下压缩比反而变差（scale pack 开销）

#### 3.7 评测方法论：性能 + 质量（~55 分钟）
- 📌 **性能测量**：`vllm bench serve`（测 online serving 吞吐 + TTFT，输出 mean/median/P99） / `vllm bench throughput`（离线吞吐） / `vllm bench latency`（端到端延迟）
- ⚠️ 方法学坑：V1 引擎下用 `llm.generate()` 拿 per-request TTFT **不可靠**，**必须用 server 端 `/metrics` + `vllm bench serve`**
- 📌 **显存拆分（权重 vs KV-Cache）**：推荐读 vLLM `/metrics`（Prometheus）的 `vllm:gpu_cache_usage_perc` / `num_gpu_blocks`；**差值法**（空载记权重基线、长上下文生成记峰值、差值≈KV-Cache）有 nvidia-smi 偏差（显示进程已映射显存，会**高估**实际 KV 占用），课程要讲清这个偏差。**抓取命令示例**：
  ```bash
  curl -s http://localhost:8000/metrics | grep -E 'gpu_cache_usage_perc|num_gpu_blocks'
  ```
- 📌 **质量评估**：`lm_eval` 跑 `ceval-valid,cmmlu,mmlu,gsm8k`，量化前后用**相同 seed/prompt/num_fewshot**；lm-eval 核心包不含 model backend，需 `lm_eval[vllm]` extra
- 📌 在线评测模板（harness 走 vLLM OpenAI 端点）：
  ```bash
  lm_eval --model local-completions \
      --model_args model=<name>,base_url=http://localhost:8000/v1/completions,num_concurrent=64,max_length=4096 \
      --tasks ceval-valid,cmmlu,mmlu,gsm8k --batch_size 64 --output_path ./results/<run>
  ```

#### 3.8 四维 Pareto 对比表（~35 分钟）
- 📌 对 FP16 / W8A8(SmoothQuant) / W4A16(AWQ) / FP8 四种配置，统一生成「质量(PPL+下游) | 显存 | 吞吐 | TTFT」四维对比表与 Pareto 前沿图
- 📌 重点观察：代码/数学任务对量化更敏感（PPL 几乎不变但 GSM8K 可能掉）；W4A16 在算力受限卡提速、在 H200 上 dequant 开销可能拖慢

**🔧 模块动手实验**：「SmoothQuant 逐层敏感度 → mixed-precision 回退调优 → PPL 恢复曲线 → 四维评测」——对 M2 产出的 SmoothQuant 模型：(1) 跑逐层敏感度排序；(2) 用多个 QuantizationModifier 做 mixed-precision 回退；(3) 画 PPL 恢复曲线找 Pareto 拐点；(4) 用统一 benchmark 生成四维对比表。注明同一流程可套用到 AWQ/FP8。

---

### 模块 4：vLLM 声明式部署与端到端闭环项目

**模块目标**：把调优后的模型部署到 H200×8 的 vLLM，完成端到端闭环 mini 项目，并补齐部署侧的报错排查与产物发布。引擎**只用 vLLM**。

> **声明式部署（开篇明确）**：部署是声明式的、**无需改模型代码**——Qwen2.5-7B 的 `architectures` 字段是 `Qwen2ForCausalLM`，已原生集成进 transformers 与 vLLM，纯文本 Qwen2.5-7B 无需 `--trust-remote-code` 加载模型与 tokenizer。vLLM `--quantization` 默认 `auto`，读 `config.json` 的 `quantization_config` 自动识别 compressed-tensors 格式（FP8/AWQ/SmoothQuant 三产物统一）并选 kernel；`tokenizer_config.json` 的 chat template 由 vLLM 自动加载，调用 `/v1/chat/completions` 自动套用，无需 `--chat-template`；`generation_config.json` 的 `temperature`/`top_p` 自动作为默认采样参数。
>
> 本模块学员写的代码只有三类且都不是模型适配：① 量化 recipe（M2 产出）② `vllm serve` 命令（本模块）③ 客户端调用（OpenAI SDK）。最小命令就是 `vllm serve <path> --tensor-parallel-size 8`。
>
> **边界**：仅以下情况才需自定义代码或 flag——(a) Qwen2.5-VL 等多模态变体（架构 `Qwen2_5_VLForConditionalGeneration`，可能需 `trust-remote-code` 或特定 vLLM 版本）；(b) 非标准/自定义 quantization scheme（不在 compressed-tensors 内置 scheme 列表）；(c) vLLM 未原生支持的冷门架构（需走 Transformers fallback，有性能损失）。本课程用 Qwen2.5-7B 纯文本 + 三种标准 compressed-tensors scheme，全部落在声明式范围内。

**课时安排**：

#### 4.1 vLLM 加载量化模型：auto 自动识别 + 三方法 flag/kernel 路径（~50 分钟）
- 📌 vLLM `--quantization` **默认 `auto`**：读模型 `config.json` 的 `quantization_config`，多数情况**不需要显式传 flag**
- 📌 **方法 → flag → kernel 速查表（已 Web 核对当前 vLLM 版本）**：

  | 方法 | vllmFlag | kernel（H200） | 备注 |
  |------|----------|---------------|------|
  | **SmoothQuant INT8 W8A8**（compressed-tensors） | 不传（`auto` 识别）；或 `--quantization compressed-tensors` | INT8 Marlin / CUTLASS `scaled_mm` | 校准时务必用 SmoothQuant 把激活 scale 吸收进权重 |
  | **AWQ（推荐，llm-compressor compressed-tensors）** | 不传（`auto` 识别） | Machete / Marlin（W4A16 mixed-input） | 与 SmoothQuant/FP8 工具链统一 |
  | **AWQ（遗留 AutoAWQ 格式）** | `--quantization auto_awq`（带下划线，**不是** `awq`） | awq_marlin（自动） | AutoAWQ 量化必须 `zero_point=False` 才能走 Marlin；库已废弃 |
  | **FP8 W8A8** | 不传（`auto` 识别）；或 `--quantization fp8` | CUTLASS `torch._scaled_mm`（Hopper 原生 FP8） | 静态 scale 需校准时拿到；H200 上不会 fallback 到 FP8-Marlin |

- ⚠️ 易错：旧 AutoAWQ 格式 flag 是 `auto_awq`（带下划线）；写成 `awq` 会报错；SmoothQuant 实际走 INT8-Marlin 还是 scaled_mm 建议课堂上 `VLLM_LOGGING_LEVEL=DEBUG` 打印一次确认

#### 4.2 H200×8 多卡部署（~50 分钟）
- 📌 单节点 8 卡 → **纯张量并行 `--tensor-parallel-size 8`**，单节点不混 PP（PP 只在跨节点才有意义）
- 📌 命令模板：
  ```bash
  vllm serve <quantized-model-path> \
      --tensor-parallel-size 8 \
      --gpu-memory-utilization 0.90 \
      --max-model-len 32768 \
      --enable-prefix-caching \
      --disable-custom-all-reduce
  ```
- 📌 continuous batching / PagedAttention：vLLM V1 引擎默认开启，无需 flag
- ⚠️ 高频坑：**NCCL hang at startup**（TP>1）→ 查 `nvidia-nccl-cu12` 是否匹配 CUDA 12.8，调试 `NCCL_P2P_DISABLE=0 NCCL_IB_DISABLE=1` / `NCCL_DEBUG=INFO`；all-reduce 报错加 `--disable-custom-all-reduce`；fork 问题设 `VLLM_WORKER_MULTIPROC_METHOD=spawn`
- 📌 **TP=2 + 数据并行（提吞吐）**：Qwen2.5-7B 量化后单卡即可放，TP=8 更多是教学演示；真实提吞吐可在单节点起多个 vLLM 实例（每实例 `--tensor-parallel-size 2` 占 2 卡），前面挂负载均衡分发流量，单实例内 TP=2 降通信开销、多实例并行提总吞吐。> *（数据并行涉及多实例编排 + LB 配置，属运维范畴，本课不展开实操，仅点明方向。）*

#### 4.3 部署性能压测（~40 分钟）
- 📌 `vllm bench serve --backend vllm --base-url http://localhost:8000 --model <name> --dataset-name sharegpt --num-prompts 1000` → 吞吐 + TTFT(mean/median/P99)
- 📌 读 `/metrics` 看 KV-Cache 占用与 residency（示例：`curl -s http://localhost:8000/metrics | grep -E 'gpu_cache_usage_perc|num_gpu_blocks'`）
- ⚠️ 评测坑：V1 引擎下 `llm.generate()` 测 TTFT **不可靠** → 用 `/metrics` + `vllm bench serve`；显存差值法（nvidia-smi）会**高估** KV-Cache
- 🛠 动手：对三种量化模型 + FP16 基线在 H200×8 上压测，生成吞吐/TTFT/显存对比

#### 4.4 常见报错 cheatsheet（~45 分钟）
- 📌 单一权威排查表：

  | 症状 | 根因 | 排查/修复 |
  |------|------|----------|
  | `No compatible kernel found` / 加载即 OOM 但权重很小 | compressed-tensors 版本与 vLLM 不匹配 → fallback 错误路径 | `uv pip install -U compressed-tensors vllm llmcompressor`；对照 issue #2268 |
  | AWQ 模型 `--quantization awq` 报错 | flag 名错（应是 `auto_awq` 或不传） | 改 `auto_awq` 或删 flag |
  | AWQ Marlin 报 `zero_point` 错 | AutoAWQ 量化用了 `zero_point=True` | 重新量化 `zero_point=False`（issue #6985） |
  | TP 启动 hang / rank timeout | NCCL 不匹配 / P2P | `pip show nvidia-nccl-cu12`；`--disable-custom-all-reduce`；`NCCL_DEBUG=INFO` |
  | `CUDA error: no kernel image` | wheel CUDA 基线(12.8) ≠ 实际 CUDA | 用 CUDA 12.8 镜像或源码 build |
  | OOM at profile run | `--gpu-memory-utilization` 太高 / KV 太大 | 降到 0.85 或降 `--max-model-len`（量化模型 OOM 多为 KV） |
  | `ValueError: task 'ceval-valid' not found` | lm-eval 版本太旧 / `lm_eval[vllm]` extra 未装 | `cd course/<module> && uv pip install --python ./.venv/bin/python -U 'lm_eval[vllm]'`，`lm_eval --tasks list` 核对 |
  | llm-compressor `oneshot` 报 transformers 版本不符 | transformers 被 vLLM 锁版本 | 量化与部署分模块（本课已用每模块自包含 uv 项目隔离） |

#### 4.5 发布量化模型到 HuggingFace Hub（~35 分钟）
- 📌 用 `huggingface_hub` 的 `upload_folder`（比 git push 大文件更稳、可断点续传）：
  ```python
  from huggingface_hub import HfApi
  api = HfApi(token="hf_xxx")
  api.create_repo("your-org/Qwen2.5-7B-SmoothQuant-W8A8", exist_ok=True)
  api.upload_folder(folder_path="./quantized_model",
                    repo_id="your-org/Qwen2.5-7B-SmoothQuant-W8A8", repo_type="model")
  ```
- ⚠️ **必须带 `config.json` 的 `quantization_config`**（compressed-tensors 自动写入）——这是 vLLM `auto` 识别的依据，少了会被当 FP16 加载
- 📌 model card 硬约定：写明基线模型、量化方法、校准集、lm-eval 评分、vLLM 版本（vllm **PyPI 版本号**，如 0.23.x）、推荐启动命令

#### 4.6 端到端闭环 mini 项目（~90 分钟，综合）
- 📌 **本节不重新教部署/压测**——部署与压测步骤引用 4.1（加载 flag/kernel 速查）–4.2（TP=8 多卡）–4.3（`vllm bench serve` / metrics）；不再重复列 `vllm serve` 命令、不再重教 `--tensor-parallel-size`、不再重教 bench 命令
- 📌 **聚焦①整条链路串接**：量化产物（`config.json` `quantization_config` + 量化权重）→ 调优（`ignore`/mixed-precision 回退后的最终产物）→ 评测（四维 Pareto 精度数）→ 部署（引用 4.1-4.2 的 serve 命令）→ 压测（引用 4.3 的 `vllm bench serve` / metrics）；强调链路唯一交接物是 compressed-tensors 产物目录
- 📌 **聚焦②选型决策树**：把 M1.6 三范式选型（W4A16 访存墙/W8A8 计算墙/FP8）与 M3.8 四维 Pareto 数据结合，产出按 H200×8 + 目标吞吐/显存/精度约束选哪条路径的决策树
- 📌 **聚焦③可复现交付物四件套**：量化 recipe（`.py`/yaml）+ 各模块 `uv.lock` + 产物 HF Hub 路径 + serve 命令 + 压测报告，打包成可复现交付物（固定 seed/prompt/batch 的 notebook + README + 选型决策树一页纸）
- 📌 评分维度：可复现性（固定 seed/prompt/batch）、指标完整性（四维）、调优是否找到 Pareto 拐点、选型建议是否场景化

#### 4.7 前瞻：QAT 概念 + NVFP4（~25 分钟）
- 📌 **QAT**（仅概念）：前向插入 fake-quant + 直通估计器 STE，理论精度上限高于 PTQ，但 LLM 上需万亿 token 重训、成本极高，仅蒸馏/继续训练场景划算——**本课不动手**
- 📌 **NVFP4**（一句话）：Blackwell 第五代 Tensor Core 独有，H200 不支持；作未来硬件升级时的前瞻

**🔧 模块动手实验**：「端到端闭环项目交付（去冗余版）」——不重新部署/压测，引用 4.1-4.3 既有命令与产物，整合 M1–M4 全部能力，产出：① 整条链路串接演练（以 compressed-tensors 产物为唯一交接物）② 选型决策树（M1.6 三范式 + M3.8 四维 Pareto）③ 可复现交付物四件套（recipe + uv.lock + Hub 路径 + serve 命令 + 压测报告）。

---

## 评估与练习（分层）

1. **跑通层**：在 Qwen2.5-7B-Instruct 上用 llm-compressor 产出 FP8 模型并用 vLLM `--tensor-parallel-size 8` 加载 generate，提交显存对比 + 环境检查 cell（在部署模块目录内 `cd course/m4-deploy-loop && uv run` 跑）。
2. **对比层**：对同一 Qwen2.5-7B-Instruct 分别产出 SmoothQuant/AWQ/FP8 三模型（固定校准集与 seed），提交「方法 | 比特 | group_size | PPL | 显存 | tokens/s」对比表，解释差异。
3. **调优层**：对 SmoothQuant 模型做逐层敏感度分析 + mixed-precision 回退，提交 PPL 恢复曲线与 Pareto 拐点结论。
4. **闭环交付（Mini 项目）**：M4.6 的完整闭环交付（选型决策树 + 可复现交付物四件套）。
5. **反例诊断题**：给一份 PPL 异常的量化结果（量化了 lm_head？随机噪声校准？对称 INT4 激活？add_bos_token 漏加？），诊断原因并给修复方案。
6. **测量规范题**：给一段有 bug 的压测脚本（缺 warmup / 把首 token 延迟当吞吐 / 没区分权重与 KV-Cache 显存 / 用 `llm.generate()` 测 TTFT），找出并修复所有测量错误。
7. **格式-kernel 配套题**：判断下列组合是否可行并说明 kernel 路径——(a) 遗留 AutoAWQ 模型 `--quantization awq`；(b) AutoAWQ 量化 `zero_point=True` 后走 Marlin；(c) compressed-tensors SmoothQuant 模型不传 `--quantization` flag。
8. **uv 工作流题**：给一份错误的模块 env 搭建命令（在仓库根直接 `uv sync` 报 `no project`；以及未 `cd` 进模块目录就 `uv pip install 'lm_eval[vllm]'` 不知道装哪），指出为什么失败（根目录无 `pyproject.toml`；`uv pip` 未激活时只认 `--python <path>`、不认项目 venv）并改写为正确的 `cd course/<module> && uv sync` / `cd course/<module> && uv pip install --python ./.venv/bin/python 'lm_eval[vllm]'` 写法；说明 `uv.lock` 与 `pyproject.toml` 在复现性上的职责区别。
9. **声明式部署题**：给定一段 vLLM serve 命令（误加 `--trust-remote-code` / `--chat-template` / `--quantization fp8`），判断哪些 flag 对 Qwen2.5-7B compressed-tensors 模型是多余的或错误的，并说明自动加载机制（`quantization_config` / `chat_template` / `generation_config`）。
10. **选型论述题**：给定「H200×8 + 32k 上下文 + 可接受 PPL 升幅<3%」场景，论述应选 SmoothQuant/AWQ/FP8 中哪种，并用本课数据支撑（须同时考虑 KV-Cache 显存与长上下文质量劣化）。

---

## 附录 A：版本敏感 API 速查

```python
# ============ 版本敏感 API 速查（2026-06 实测核实）============
# 版本基线：vllm PyPI 0.23.x（0.x，非 CalVer）、llmcompressor 0.12.x、
#           compressed-tensors 0.17.x、transformers 5.12.x、lm-eval 0.4.12
#           CUDA 12.8（driver ≥ 570.x）、uv 0.8.x、Python 3.11

# === FP8 W8A8（无需校准数据，H200 首选）===
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
oneshot(model=model, recipe=recipe)                       # 不传 dataset
model.save_pretrained(SAVE_DIR)

# === SmoothQuant W8A8 INT8（两段式，需校准）===
from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier  # 注意路径
from llmcompressor.modifiers.gptq import GPTQModifier
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),          # 显式取 0.8（库默认 0.5=论文 α）
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),  # ignore 必须是列表
]
oneshot(model=model, dataset=ds, recipe=recipe,
        max_seq_length=2048, num_calibration_samples=512)
model.save_pretrained(SAVE_DIR, save_compressed=True)

# === AWQ W4A16（推荐 llm-compressor 统一路径；下方为遗留 AutoAWQ 对照）===
# 主路径：llm-compressor AWQ modifier，输出 compressed-tensors（加载侧 --quantization auto）
# 遗留对照（AutoAWQ，已废弃，仅历史）：
from awq import AutoAWQForCausalLM
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
# 注意：quant_config 没有 exclude_modules 字段；排除层走加载侧 AwqConfig.modules_to_not_convert
model = AutoAWQForCausalLM.from_pretrained(model_path)
model.quantize(tokenizer, quant_config=quant_config, calib_data=data)   # data 128–256 条
model.save_quantized(quant_path, safetensors=True, shard_size="4GB")
# 遗留产物加载：--quantization auto_awq（带下划线）+ awq_marlin（须 zero_point=False）

# === mixed-precision 回退（细粒度，0.7.0+）===
# 用多个 QuantizationModifier（不同 targets + scheme）在一次 oneshot 中组合
# → llm-compressor 自动合并成 compressed-tensors 的 config_groups
# → vLLM 0.10.1+ 原生可跑（非手动改 config.json，否则易漏 observer 字段致加载失败）
# ignore 支持正则前缀：ignore=["re:.*lm_head", "re:.*down_proj"]
```

```bash
# === uv 每模块自包含工作流（替代 pip）===
cd course/m2-quant-pipeline                       # 进入模块目录（每个模块就是一个 uv 项目）
uv sync                                           # 读 pyproject.toml + uv.lock，装依赖到模块 ./.venv
# lm-eval 核心不含 model backend —— 注意：uv pip 只认激活 env / --python，不认项目 venv！
# 未激活时必须显式 --python 指向目标 venv，否则不知道装哪。
uv pip install --python ./.venv/bin/python 'lm_eval[vllm]'
# M3/M4 部署模块同理（cd course/<module> && uv sync）
# 运行：cd course/<module> && uv run jupyter lab  （或 uv run python recipes/xxx.py）
# 复现：commit 各模块 uv.lock，新节点 cd course/<module> && uv sync 即可

# === vLLM H200×8 声明式部署（Qwen2.5-7B 无需 --trust-remote-code）===
vllm serve <model> --tensor-parallel-size 8 --gpu-memory-utilization 0.90 \
    --max-model-len 32768 --enable-prefix-caching --disable-custom-all-reduce
# --quantization 默认 auto，读 config.json quantization_config 自动识别；不要画蛇添足加 flag

# === 评测（lm_eval[vllm] extra 必装；add_bos_token=True 必加）===
lm_eval --model local-completions \
    --model_args model=<name>,base_url=http://localhost:8000/v1/completions,num_concurrent=64 \
    --tasks ceval-valid,cmmlu,mmlu,gsm8k --batch_size 64 --output_path ./results/<run>

# === 压测（不要用 llm.generate() 测 TTFT）===
vllm bench serve --backend vllm --base-url http://localhost:8000 \
    --model <name> --dataset-name sharegpt --num-prompts 1000

# === 读 vLLM /metrics（权重 vs KV-Cache 显存拆分）===
curl -s http://localhost:8000/metrics | grep -E 'gpu_cache_usage_perc|num_gpu_blocks'
```

## 附录 B：需在开课前实测确认的点（专家标注的 open questions）

1. **uv 结构选型**：每模块用完全独立 uv 项目（各自 `uv.lock`，本课方案）还是 workspace（共享根 `uv.lock`，`[tool.uv.workspace] members=course/*`）？workspace 复现性更集中，但 GitHub Issue #8722 指出 workspace member 的 per-member dependency-groups 同步仍有局限。需课程方定夺结构偏好（本轮先用每模块自包含）。
2. **deploy 模块的 vLLM 版本锁定**：是否需要锁到具体 PyPI 版本（0.23.x 还是更早）？不同小版本对 compressed-tensors scheme 的 kernel 支持有差异（如 0.10.1+ 才原生跑 mixed-precision config_groups），锁定哪个版本影响各部署模块 `uv.lock` 内容。
3. **完整 uv sync 实测**：实际 `cd course/<module> && uv sync`（含 vLLM CUDA12.8 wheel + llmcompressor 全量传递依赖）在各模块目录的总安装时长与磁盘占用未实测——本轮只验证了 `pyproject.toml` 语法层在 uv 0.8.x 解析通过，**完整 sync + `lm_eval[vllm]` 装入目标 venv 待目标 H200 节点实测**（vLLM wheel 很大）。
4. **`lm_eval[vllm]` extra 与 vllm wheel transformers 冲突**：extra 装入 quant env 后是否会与 vllm wheel 自带 transformers 冲突——需目标节点 `uv sync` 后实测 `lm_eval --model vllm` 能否正常起。
5. **`uv pip install --python ... 'lm_eval[vllm]'` 与后续 `uv sync` 的一致性**：这种方式装的 extra 是否会被后续 `uv sync` 视为环境偏离 `uv.lock` 而触发重装/卸载 extra，需实测确认 uv 对 `uv pip install` 与 `uv sync` 的一致性处理。
6. **hf-transfer 容器兼容**：`download_model.sh` 的 hf-transfer 在 H200 节点（可能是容器内）是否生效——容器环境偶有 hf-transfer 与 glibc 兼容问题，需实测下载一次大文件确认加速生效。
7. **共享存储路径**：课程 H200×8 节点的共享存储路径约定（`/shared/models` 还是仓库内 `models/`）需与实际集群对齐，`download_model.sh` 的默认路径占位需替换。
8. **SmoothQuant 在 llmcompressor 0.9.0+ 的 API 稳定性**：两段式在 0.9.0+（2026-01 引入 attention quantization/MXFP4/AutoRound）下导入路径与 modifier 组合是否有变化？本轮未深挖 0.9.0+ 对 `SmoothQuantModifier` 的影响，建议 M2.5 标注以课程锁定的 llmcompressor 版本号（0.12.x）为准。
9. **Qwen2.5-7B 三方法实测精度**：官方文档给的是 Llama 3 的 gsm8k 数字，Qwen2.5-7B 需课程实测。
10. **Qwen2.5-7B 逐层敏感度结论**：社区无公开报告，需实测，不能套用 Llama 3 结论（lm_head 之外哪些层最敏感）。
11. **`smoothing_strength` 在 Qwen2.5-7B 上的最优值**：库默认 0.5（=论文 α），课程取 0.8，需 grid search（0.5/0.8/1.0 对比）。
12. **SmoothQuant W8A8 在 H200 上 vLLM 实际 kernel 名**（INT8-Marlin vs CUTLASS scaled_mm）：课堂上 `VLLM_LOGGING_LEVEL=DEBUG` 抓一次确认。
13. **CUDA 12.8 driver 下限**：570.x 基于 NVIDIA 通用 release notes——具体 H200 SXM/PCIe 卡型与所用驱动分支（datacenter vs enterprise）是否一致，建议开课前在目标节点 `nvidia-smi` 实测确认。
14. **M4 声明式部署「Qwen2.5 不需 trust_remote_code」的前置条件**：基于 `Qwen2ForCausalLM` 原生——若课程将来改用 Qwen2.5-MoE 或 Qwen3 等新架构，需重新核实其 `architectures` 字段是否在 vLLM 原生列表内。

---

## 修订说明（本轮评审定稿 revisionNotes）

本轮在 v3 定稿基础上，按严格评审要求落实 C1（CRITICAL）/M1/m1/m3 四项**必须改动**，并采纳覆盖澄清建议（M2.2 / M3.7 / M2.4 / M4.2 / m4）。保留 v3 已查证的全部版本纠正与结构。

**【必须改动，已落实】**：

- **C1（CRITICAL）uv pip 环境 bug**：所有 `uv pip install 'lm_eval[vllm]'` 一律改为 `cd course/<module> && uv pip install --python ./.venv/bin/python 'lm_eval[vllm]'`。`uv pip` 未激活时只认 `--python <path>`、不绑定目标解释器（实测 `uv pip` 只认激活 env / `--python <path>`，不认项目 venv）。涉及各模块 `scripts/download_model.sh` 后的 extra 装配、附录 A 速查、`uvEnvApproach` 坑#4、`assessmentIdeas` 的 uv 题与 cheatsheet 题干；坑#4 措辞改为"真正风险是 uv pip 只认激活 env / `--python`，不认项目 venv"。
- **M1**：各模块 `pyproject.toml` 的 `transformers>=4.45` → `>=5.0`（llmcompressor 0.12 要求 transformers v5）。
- **m1**：`scripts.purpose` 与 `revisionNotes` 中"4 脚本已实测可用"降级为"**语法层 uv 0.8.x 解析通过；完整 sync + `lm_eval[vllm]` 装入目标 venv 待目标 H200 节点实测**"，与附录 B 口径统一。
- **m3**：两个 `pyproject.toml` 的 `compressed-tensors>=0.9.0` → `>=0.10`。

**【覆盖澄清建议，一并落实】**：

- **(a) M2.2 统一基线**：明确为 **Qwen2.5-7B-Instruct**（与 `download_model.sh` 默认 `MODEL_REPO` 对齐），并点出 base vs Instruct 在指令遵循任务上的评测差异。
- **(b) M3.7 `/metrics` 抓取示例**：补 `curl -s http://localhost:8000/metrics | grep -E 'gpu_cache_usage_perc|num_gpu_blocks'`（M4.3 与附录 A 同步补）。
- **(c) M2.4 Falcon group_size=64**：降为脚注（Qwen2.5 不受此限，非主流程）。
- **(d) M4.2 TP=2+数据并行**：给最小说明（单节点起多实例 `--tensor-parallel-size 2` + 负载均衡），并标注编排属运维范畴、超纲不实操。
- **(e) m4 ignore 正则归位**：M2.3 只讲精确 `ignore=["lm_head"]`，正则 `re:` 前缀统一放 M3.3 讲透。

**【保留不变】**：v3 的版本纠正（vLLM 0.x 非 CalVer、CUDA 12.8/driver 570、H200 141GB HBM3e 4.8TB/s、lm_eval[vllm] extra、各包版本号）、4 模块结构、28 课时、M1/M3 全部内容、已查证事实（AutoAWQ 废弃、FP8/SmoothQuant/W8A8 API、mixed-precision、评测坑、声明式部署本质）全部保留。

---

*本文件由 Agent 团队生成并经版本查证定稿。技术栈与库 API 会随版本演进，开课前请按附录 B 实测确认。裁剪决策与关键修正见 [`docs/REVIEW.md`](docs/REVIEW.md)。*

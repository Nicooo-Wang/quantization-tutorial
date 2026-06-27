# M2 教学法重设计（AWQ 试点）设计

> 日期：2026-06-27　分支：`main`（开发在 feature 分支）　关联：OUTLINE §模块 2、`docs/superpowers/specs/2026-06-25-per-module-self-contained-design.md`（M2 自包含架构）、`course/NOTEBOOK_CONVENTIONS.md`

## 1. 背景与目标

用户审核 M2 后反馈：三个 lab（FP8/AWQ/SmoothQuant）是「机械化填空」——能填 recipe 字段、跑 `oneshot` 通，但**不理解整个通路**：`oneshot(model, dataset, recipe)` 内部到底干了什么、`dataset`/`recipe`/`modifier`/`scheme`/`quantization_config` 这些组件是什么/为何需要/什么格式、三种方法是什么关系、每步为何需要。学完一个 lab 没有真正掌握。

**根因**：M2 是早期开发的（`ff9a6b4`），后来做了自包含重构 + 双重验收，但那次**明确不动教学内容**，所以教学法缺口留存。当前 lab 讲清了各方法的**算法原理**（如 AWQ「保护显著通道」），却没讲 **llm-compressor 工具链的端到端通路**与**三方法关系**。

**本设计目标**：把 M2 重设计成「端到端指导」——讲透通路 + 组件 + 关系 + 每步 why，学完真正掌握。**AWQ 试点**（用户最关心），跑通满意后套用到 FP8/SmoothQuant。

## 2. 目标 / 非目标

**目标**
- 新增 `s0_pipeline_overview.ipynb`：三 lab 共享的「通路总览」（oneshot pipeline + 五大组件 + 三方法关系图）。
- 重写 `s2_awq_quant.ipynb`：端到端讲解 + 填空带 why 引导。
- 学完 AWQ lab，学员能讲清：oneshot 通路数据流、`dataset`/`recipe`/`modifier`/`scheme`/`quantization_config` 各是什么与为何需要、AWQ 在通路哪步（两段式 recipe）、为何两段、W4A16 产物怎么读。

**非目标**
- 不动 `s1_fp8_quant.ipynb` / `s3_smoothquant.ipynb`（AWQ 试点满意后套用）。
- 不动 M2 env（pyproject/uv.lock）、scripts/、README——只改教学法层（markdown + 填空引导）。
- 不动 AWQ lab 的**代码逻辑**（函数实现、L2 tiny、L3 真 7B、产物检查代码）——重设计是教学法层，不是推倒代码。
- 不动 M1/M3/M4、OUTLINE（M2 内容大纲不变，只是教学法重构）。

## 3. 交付物 + 结构

```
course/m2-quant-pipeline/steps/
  s0_pipeline_overview.ipynb   # 新增：三 lab 共享通路总览
  s1_fp8_quant.ipynb           # 不动
  s2_awq_quant.ipynb           # 重写：端到端讲解 + 填空带 why（代码逻辑保留）
  s3_smoothquant.ipynb         # 不动
```

跑法不变（自包含 uv 项目：`cd course/m2-quant-pipeline && uv sync && bash scripts/download_model.sh && uv run jupyter lab`）。

## 4. s0 通路总览设计（新增，markdown 为主 + 最小代码演示）

讲三件事，markdown 为主，配一段最小真 oneshot 代码演示通路（不填空）：

### 4.1 oneshot 通路数据流
`oneshot(model, dataset, recipe)` 内部：**加载模型 → 用校准数据跑前向收集激活统计 → 按 recipe 里 modifier 的顺序逐个应用（每个 modifier 是一步量化变换）→ 保存量化模型（compressed-tensors）**。配一段最小真代码演示（tiny Qwen2，用通用 W8A8 scheme 走通全通路，CPU 可跑、避开 FP8 的 GPU 要求），展示「输入 model+dataset+recipe → 产物落盘」的完整数据流。

### 4.2 五大组件（每个讲：是什么 / 为什么需要 / 什么格式）
- **`dataset`**：校准数据。格式 = HuggingFace `Dataset`（`load_dataset(...)` 或 `Dataset.from_list(...)`），含 `text` 列，oneshot 内部用 tokenizer 处理。**为何需要**：量化算法要看真实激活分布（AWQ 挑显著通道、SmoothQuant 找离群点）才能决定 scale。
- **`recipe`**：`[modifier1, modifier2, ...]`，一个**有序列表**。**为何是序列**：量化分步（如 AWQ 先用 AWQModifier 搜 scale、再用 QuantizationModifier 压 INT4），顺序即流水线。
- **`modifier`**：一个量化步骤的封装（带 `scheme`/`targets`/`ignore` 等参数），如 `QuantizationModifier` / `AWQModifier` / `GPTQModifier` / `SmoothQuantModifier`。
- **`scheme`**：量化方案字符串（`W4A16_ASYM` / `W8A8` / `FP8_DYNAMIC`），决定位宽 + 对称性 + 粒度（per-channel/group）。
- **`quantization_config`**：产物 `config.json` 里的量化元数据（compressed-tensors 格式），记录「怎么量化的」，vLLM 据此加载。

### 4.3 三方法关系图
一张表讲清三者在「量化什么 / 数值格式 / 是否需补偿算法」上的关系，加上各自在 oneshot 里的 modifier 组合：

| 方法 | 权重 | 激活 | 数值格式 | 需补偿算法？ | recipe（modifier 组合） |
|---|---|---|---|---|---|
| AWQ | INT4 | FP16 | 整数 | 是（搜 scale 保护显著通道）| `AWQModifier` + `QuantizationModifier(W4A16_ASYM)` |
| SmoothQuant | INT8 | INT8 | 整数 | 是（离群点迁到权重）| `SmoothQuantModifier` + `GPTQModifier(W8A8)` |
| FP8 | FP8 | FP8 | 浮点 | **否**（浮点大动态范围，直接 cast）| `QuantizationModifier(FP8_DYNAMIC)` |

**关键洞察**：M1 讲过——INT 量化的死穴是「激活离群点让整数网格崩」，所以 AWQ/SmoothQuant 要加算法补偿；FP8 是浮点（有指数位、动态范围远大于定点），离群点不会让它崩，**直接 cast，不需补偿**——所以 FP8 在 H200 上最简单、只需一个 modifier。

### 4.4 s0 的 CONV 定位
s0 是**概念导览**（同 M1 s6/s7 的 CONV 例外）：以 markdown 讲解为主，不含量化算法填空；可有「理解检查」问答（如「为何 FP8 只需一个 modifier、而 AWQ 要两个？」），但不跑完整 L1/L2/L3 量化验证。最小 oneshot 演示代码可执行（tiny + CPU）以佐证通路，但非量化产物 lab。

## 5. s2 AWQ lab 重写结构

### 5.1 开头：回顾 s0，定位 AWQ
一句话回扣通路总览：AWQ 在 oneshot 通路里 = **两段式 recipe**（先用 AWQModifier 搜显著通道 scale、再用 QuantizationModifier 压 INT4），激活保持 FP16（W4A16，weight-only）。点明「本 lab 目标：端到端跑通 + 理解每步为何」。

### 5.2 端到端讲解（markdown 大幅加强，每步讲 why）
按通路顺序，每步先讲「是什么 / 为何需要 / 怎么做」：
- **dataset 步**：为何 AWQ 要校准数据（要看激活幅值挑显著通道）、wikitext-2 格式与命名空间坑（`Salesforce/wikitext`）、为何 128–256 样本就够（AWQ 极省样本）。
- **recipe 步**：为何 AWQ 是**两段**——第一段 AWQModifier 在校准数据上搜 per-channel scale（保护显著通道）；第二段 QuantizationModifier 把权重压 INT4（第一段算的 scale 在此生效）。讲清 `targets`（量化谁）/ `scheme`（W4A16_ASYM 含义：4-bit 非对称 group-wise）/ `ignore`（为何排除 lm_head）各参数为何需要。
- **oneshot 步**：调 `oneshot(model, dataset, recipe)` 时内部发生什么（回扣 s0 通路：前向收集统计 → 两段 modifier 顺序应用）、产物落哪。
- **产物步**：怎么读 `quantization_config`，W4A16 的证据字段（`num_bits=4` / `symmetric=False` / `group_size=128` / `input_activations=None`）。

### 5.3 填空保留 + 带 why（实现逻辑不变）
保留三个填空（实现逻辑不变、已验证能跑），但**每个填空前**加「这个组件在通路哪步、参数为何需要」的引导段：
- `build_awq_recipe(ignore, scheme)` —— 引导：第一段为何不带参数、第二段为何要 targets/scheme/ignore 三参数、scheme 为何用入参（别写死）、ignore 为何转 list。
- `build_calibration_dataset(tokenizer, n_samples, seq_len, raw_texts)` —— 引导：返回 `Dataset` 含 text 列（oneshot 内部 tokenize）、raw_texts 分支为何免网络、wikitext 命名空间为何必要。
- `awq_config_summary(qc)` —— 引导：每个字段（num_bits/symmetric/group_size/input_activations）对应 W4A16 的哪个语义。

### 5.4 L2/L3/产物检查：代码保留 + 加验证说明
L2（tiny）、L3（真 0.5B→7B）、产物检查的**代码框架保留**（已验证能跑），但每个加 markdown 注释讲「这步在验证什么」（如 L2 验「tiny 上两段式跑通 + 产物是 W4A16」、L3 验「真模型显存降到 ~1/4」、产物检查验「读 quantization_config 确认 W4A16」）。

## 6. 填空哲学

- **讲解 + 保留代码填空（带 why）**：保留动手写代码（不退化成纯阅读），但每个填空前讲透「组件在通路哪步、参数为何需要」，让填空从「拼参数」升级为「理解后实现」。
- 填空**实现逻辑不变**（已 dev/arch 验证能跑）——重设计是教学法层（引导 markdown），不动函数实现。
- AWQ 的算法原理（保护显著通道）讲解保留并强化（当前 lab 已有，质量 OK）。

## 7. 构建与验收

### 7.1 构建
走 `dev-module.js` 全流程（`args.spec` = 本 spec，`skipDev: false`），dev-agent 按 spec 范围：**新增 s0 + 重写 s2 教学法层，s1/s3 保留不动**。dev-module.js 的 student worktree 修复（base=head + 每 student 前 commit，2026-06-27 `e792da4`）已生效，student 试课能拿到 notebook。

> 注：dev-agent 默认 prompt 是「为每个 Step 创建 .ipynb」，与本任务「重写 s2 + 新增 s0、不动 s1/s3」略有出入。需在调用时让 dev-agent 读本 spec 明确范围（或在 plan 阶段微调 dev prompt 支持「部分 step 重写」）——plan 阶段定。

### 7.2 验收（「完成」）
1. s0 讲清 oneshot 通路 + 五大组件 + 三方法关系；最小 oneshot 演示（tiny + CPU）能跑。
2. s2 AWQ lab：端到端讲解（dataset/recipe/oneshot/产物 各步 why）、三个填空带 why 引导、L1/L2/L3 跑通（注入参考实现）。
3. 学员视角（student 试课）能讲清：oneshot 通路、组件作用、AWQ 两段 recipe 为何、W4A16 产物怎么读——不再「机械化」。
4. s1/s3 未被改动（护栏：dev 只动 s0/s2）。
5. M2 env/scripts/README 未被改动（完整性闸门）。

### 7.3 真学员视角（补 AI student 盲区）
AI student 代理读代码就懂通路、无法模拟真学员「填了但不理解」的体验，靠它验收「学得懂」不可靠。故本重设计**验收时 controller（主控）逐 lab 充当真学员视角自检**：通读后尝试回答每个 lab 的「学完应能讲清」清单（§4 s0 三件事、§5 AWQ 四步 why），有答不出/不通透处即教学法缺口、返工。这是「先重做 M2、workflow 改进之后单独做」的执行约定——workflow 的「教学目标锚点 + 理解探测」改进作为独立项目，不阻塞本重设计。

## 8. 套用计划（AWQ 试点之后）

AWQ 试点 + 用户审核满意后，s0 已就位（三 lab 共享），FP8（s1）/SmoothQuant（s3）按同模板重写：各自开头回顾 s0 + 端到端讲解（该方法的 specifics）+ 填空带 why。s0 无需重做。

## 9. 取舍记录

| 决策 | 选择 | 理由 | 备选（未选） |
|---|---|---|---|
| 范围 | AWQ 试点先 | 用户最关心；跑通一个验证教学法是否生效，避免一次改三个难评估 | 三个一起重做（改动大、审核重、方向错返工大） |
| lab 形态 | 讲解 + 保留填空带 why | 保留动手 + 加强理解，平衡 | 纯讲解无填空（太被动）/ 讲解为主少量填空（动手不足） |
| 通路总览位置 | 独立 s0，三 lab 共享 | 通路+组件+关系是三 lab 共同缺口，独立 s0 清晰、不重复 | 放 AWQ 开头（套用时再提取）/ 放 README |
| 代码 | 保留 AWQ 逻辑，只改教学法 | 代码已 dev/arch 验证能跑，YAGNI | 推倒重写（浪费 + 风险） |

## 10. 后续

本设计批准 → writing-plans 生实施计划（s0 新增 + s2 重写的具体 cell 内容/填空引导，TDD 式或 dev-module 全流程）→ 执行（feature 分支）→ AWQ 试点验收 → 用户审核 → 套用 FP8/SmoothQuant。

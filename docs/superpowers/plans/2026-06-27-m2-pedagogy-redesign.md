# M2 教学法重设计（AWQ 试点）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 `s0_pipeline_overview.ipynb`（三 lab 共享的 oneshot 通路 + 五组件 + 三方法关系总览）+ 重写 `s2_awq_quant.ipynb` 的教学法层（端到端讲解 + 填空带 why），让学员学完真正掌握通路/组件/关系，不再机械化填空。

**Architecture:** 教学法重写，不是新代码开发——s2 的函数实现/L2/L3/产物检查代码全部保留不变，只改 markdown 讲解 + 填空 docstring 的 why 引导。新增 s0（markdown 为主 + 一段最小 oneshot 演示）。不走 dev-module 全流程（那是新代码开发用的）；用 implementer 写 + reviewer 审 + controller 充当真学员自检（spec §7.3，补 AI student 盲区）。

**Tech Stack:** notebook（.ipynb，nbformat/jupyter 编辑）、M2 env（torch cu128/llmcompressor/transformers v5/datasets，已就位）、ipytest（s2 L1）、CONV cell 约定。

**Spec:** [`docs/superpowers/specs/2026-06-27-m2-pedagogy-redesign-design.md`](../specs/2026-06-27-m2-pedagogy-redesign-design.md)

## Global Constraints

- 范围：**只动 s0（新增）+ s2（重写教学法层）**；s1_fp8 / s3_smoothquant / M2 env(pyproject/uv.lock) / scripts / README / OUTLINE **不动**。
- **s2 代码逻辑不变**：`build_awq_recipe` / `build_calibration_dataset` / `awq_config_summary` 的实现、`run_awq_quantize` 脚手架、L2 tiny 代码、L3 真 7B 代码、产物检查代码——**全部保留**；只改/加 markdown 讲解 + 三个填空 docstring 的 why 引导段。
- notebook 提交前 `nbconvert --clear-output`（不带输出）。
- s0 是**概念导览**（CONV 例外，同 M1 s6/s7）：markdown 为主，不含量化算法填空；可有「理解检查」问答；最小 oneshot 演示代码可执行（tiny + CPU）。
- 真学员自检（spec §7.3）：验收时 controller 通读 s0+s2，回答「学完应能讲清」清单（spec §4 三件事、§5 AWQ 四步 why），不通透处返工。
- 在 feature 分支开发（从 main `1f91441` 开），完成后 merge main。

## File Structure

- Create: `course/m2-quant-pipeline/steps/s0_pipeline_overview.ipynb`（通路总览，markdown 为主 + 最小 oneshot 演示）
- Modify: `course/m2-quant-pipeline/steps/s2_awq_quant.ipynb`（开头回顾 s0 + 四步 why markdown + 三个填空 why 引导 + L2/L3/产物验证说明；**代码逻辑不变**）

---

### Task 1: 新增 s0 通路总览 notebook

**Files:**
- Create: `course/m2-quant-pipeline/steps/s0_pipeline_overview.ipynb`

**Interfaces:** s0 是三 lab（s1/s2/s3）共享的通路总览。s2（Task 2）开头会「回顾 s0」。s0 不被其它 notebook import（纯讲解 + 独立演示）。

**说明：** s0 的内容设计见 spec §4。下面给出每个 cell 的具体内容（markdown 文本 + 演示代码），implementer 据此用 nbformat 或 jupyter 建 notebook。cell 顺序按 CONV（标题→导入→路径→讲解→演示）。

- [ ] **Step 1: 建 notebook 骨架（7 个 cell，按 CONV 顺序）**

用 nbformat 创建 `course/m2-quant-pipeline/steps/s0_pipeline_overview.ipynb`，7 个 cell（类型 + 内容如下）。cell 2（路径）与 s1/s2/s3 完全一致（module-local）。

**cell 0（markdown，标题）：**
```markdown
# Step 0: llm-compressor 量化通路总览（三 lab 共享）

**目标**：在动手 FP8/AWQ/SmoothQuant 之前，先建立**整条量化通路**的地图——`oneshot(model, dataset, recipe)` 内部到底干了什么、`dataset`/`recipe`/`modifier`/`scheme`/`quantization_config` 这五个组件各自是什么、三种方法是什么关系。s1/s2/s3 都走同一条通路，区别只在 recipe 里塞了什么。

**对应 OUTLINE**：模块 2 导览（本 step 是教学法前置，非 OUTLINE 单独课时）。

> 本 step 是**概念导览**（无量化填空、无 L1/L2/L3 量化验证），以 markdown 讲解为主，配一段最小 oneshot 演示佐证通路。
```

**cell 1（code，%%capture 导入）：**
```python
%%capture
import pathlib, json
import torch
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
```

**cell 2（code，路径——与 s1/s2/s3 完全一致）：**
```python
def _find_module_root(start):
    p = pathlib.Path(start).resolve()
    for cand in [p, *p.parents]:
        if (cand / "steps").is_dir() and (cand / "pyproject.toml").exists():
            return cand
    raise RuntimeError("找不到模块根（含 steps/ + pyproject.toml 的目录）；请在模块目录内启动 jupyter")

MODULE_ROOT = _find_module_root(pathlib.Path.cwd())
OUT_ROOT    = MODULE_ROOT / "out"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
print("MODULE_ROOT:", MODULE_ROOT)
```

**cell 3（markdown，oneshot 通路数据流）：**
```markdown
## 1. oneshot 通路：一次调用里发生了什么

三个 lab 的入口都是同一个函数：

```python
oneshot(model, dataset, recipe)
```

它内部是一条流水线（**记住这张数据流图，三个 lab 都套它**）：

```
model(FP 原始) + dataset(校准文本) + recipe([modifier, ...])
        │
        ▼  oneshot(...)
   ① 用 dataset 跑前向 → 收集每层激活统计（给 modifier 决策用）
   ② 按 recipe 里 modifier 的【顺序】逐个应用：
        modifier 1  →  对模型做一步量化变换（如搜 scale）
        modifier 2  →  再做一步（如真正压 INT4）
        ...
   ③ 保存 → compressed-tensors 格式（config.json 里写 quantization_config）
        │
        ▼
  量化模型（可被 vLLM 加载）
```

**关键**：`recipe` 是一个**有序列表**——量化是分步的，顺序即流水线。比如 AWQ 是「先搜 scale、后压 INT4」两步，所以 recipe 有两个 modifier，顺序不能反。
```

**cell 4（markdown，五大组件）：**
```markdown
## 2. 五大组件：是什么 / 为什么需要 / 什么格式

| 组件 | 是什么 | 为什么需要 | 什么格式 |
|---|---|---|---|
| **dataset** | 校准数据 | 量化算法要看**真实激活分布**（AWQ 挑显著通道、SmoothQuant 找离群点）才能定 scale | HuggingFace `Dataset`，含 `text` 列；oneshot 内部用 tokenizer 处理，你只给原始文本 |
| **recipe** | 量化步骤的有序清单 | 量化**分步**（先搜 scale 后量化），顺序即流水线 | `[modifier1, modifier2, ...]`（list） |
| **modifier** | 一个量化步骤的封装 | 每步做一件量化变换（搜 scale / 压位宽 / 平滑…），带参数控制 | `QuantizationModifier(...)` / `AWQModifier()` / `GPTQModifier(...)` 等类的实例 |
| **scheme** | 量化方案 | 决定**位宽 + 对称性 + 粒度**（per-channel/group） | 字符串：`W4A16_ASYM` / `W8A8` / `FP8_DYNAMIC` |
| **quantization_config** | 产物的量化元数据 | vLLM 据此**加载**量化模型；也是你验证「量化对没对」的依据 | 产物 `config.json` 里的一段（compressed-tensors 格式） |

> 一句话：你给 oneshot 一个 **model + dataset + recipe**（recipe 是 modifier 列表，每个 modifier 带 scheme），它吐出一个**带 quantization_config 的量化模型**。
```

**cell 5（markdown，三方法关系）：**
```markdown
## 3. 三种方法什么关系

三种方法都套上面那条通路，区别**只在 recipe 里塞了哪些 modifier + 用什么 scheme**。先看一张总表：

| 方法 | 权重 | 激活 | 数值格式 | 需补偿算法？ | recipe（modifier 组合） |
|---|---|---|---|---|---|
| **AWQ** (s2) | INT4 | FP16 | 整数 | 是（搜 scale 保护显著通道）| `AWQModifier()` + `GPTQModifier(scheme="W4A16_ASYM")` |
| **SmoothQuant** (s3) | INT8 | INT8 | 整数 | 是（离群点迁到权重）| `SmoothQuantModifier()` + `GPTQModifier(scheme="W8A8")` |
| **FP8** (s1) | FP8 | FP8 | 浮点 | **否**（浮点大动态范围，直接 cast）| `QuantizationModifier(scheme="FP8_DYNAMIC")` |

**最该记住的洞察**（M1 讲过原理）：
- **INT 量化（AWQ/SmoothQuant）的死穴是「激活离群点让整数网格崩」**，所以它俩都得加一个 modifier 来补偿——AWQ 用 `AWQModifier` 搜 scale 保护显著通道，SmoothQuant 用 `SmoothQuantModifier` 把离群点迁到权重。这就是它俩 recipe 有**两个** modifier 的原因（先补偿、后压位宽）。
- **FP8 是浮点**（有指数位、动态范围远大于定点 INT），离群点不会让它崩，**直接 cast 就行、不需要补偿**——所以 FP8 的 recipe 只有**一个** modifier。

> 所以：**INT 要两步（补偿 + 量化），FP8 只要一步（直接 cast）**。这是三种方法在通路上最本质的区别。
```

**cell 6（code，最小 oneshot 演示——tiny + CPU，走通全通路）：**
```python
# 最小演示：用一个 tiny 内存模型 + 通用 W8A8 scheme，把上面那条通路真跑一遍（CPU 可跑，不依赖 FP8 硬件）。
# 目的不是产出可用模型，而是让你看到 "model + dataset + recipe → oneshot → 带 quantization_config 的产物" 这条数据流。
from transformers import Qwen2Config, Qwen2ForCausalLM
from datasets import Dataset

tiny = Qwen2ForCausalLM(Qwen2Config(
    num_hidden_layers=2, hidden_size=64, intermediate_size=128,
    num_attention_heads=4, num_key_value_heads=2, vocab_size=320)).eval()

calib = Dataset.from_dict({"text": [" ".join("0 1 2 3 4")] * 4})   # 占位校准文本

recipe = [QuantizationModifier(scheme="W8A8", targets="Linear", ignore=["lm_head"])]  # 一个 modifier

oneshot(model=tiny, dataset=calib, recipe=recipe, max_seq_length=32, num_calibration_samples=4)
demo_out = OUT_ROOT / "s0-demo-w8a8"
demo_out.mkdir(parents=True, exist_ok=True)
tiny.save_pretrained(demo_out)

qc = json.loads((demo_out / "config.json").read_text())["quantization_config"]
print("产物 quantization_config 的 quant_method:", qc.get("quant_method"))
print("通路跑通：model + dataset + recipe → oneshot → 带 quantization_config 的量化模型 ✓")
```

**cell 7（markdown，理解检查）：**
```markdown
## 4. 理解检查（继续前自答）

读完上面，你应该能答出：
1. `oneshot(model, dataset, recipe)` 内部三步是什么？（前向收集统计 → 按 recipe 顺序应用 modifier → 保存）
2. 为什么 `recipe` 是个**列表**、且顺序重要？（量化分步，顺序即流水线）
3. 为什么 AWQ/SmoothQuant 的 recipe 有**两个** modifier，FP8 只有**一个**？（INT 要先补偿离群点再量化；FP8 浮点不需补偿、直接 cast）
4. 校准 `dataset` 为什么不能省？（量化算法要看真实激活分布定 scale）

答得出来再进 s1/s2/s3——它们只是在这条通路上换不同的 recipe。
```

- [ ] **Step 2: 验证 s0 最小演示能跑（CPU）**

Run: `cd course/m2-quant-pipeline && uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 steps/s0_pipeline_overview.ipynb`
Expected: 无错；最后 cell 打印「通路跑通…✓」+ quant_method（compressed-tensors）。tiny 模型 CPU 可跑（W8A8 不需 FP8 硬件）。

- [ ] **Step 3: 内容核对（三方法表 + 五组件准确）**

人工 grep 核对关键内容在位：`grep -c "AWQModifier" steps/s0_pipeline_overview.ipynb`（应 ≥1）、`grep -c "FP8_DYNAMIC" steps/s0_pipeline_overview.ipynb`（≥1）、`grep -c "quantization_config" steps/s0_pipeline_overview.ipynb`（≥2）。确认三方法表的 modifier 组合与 scheme 与 spec §4.3 一致。

- [ ] **Step 4: 清输出 + 提交**

```bash
uv run --directory course/m2-quant-pipeline jupyter nbconvert --clear-output --inplace course/m2-quant-pipeline/steps/s0_pipeline_overview.ipynb
git add course/m2-quant-pipeline/steps/s0_pipeline_overview.ipynb
git commit -m "feat(m2): add s0 pipeline overview (oneshot pipeline + 5 components + 3-method relation)"
```

---

### Task 2: 重写 s2 AWQ lab 教学法层（代码逻辑不变）

**Files:**
- Modify: `course/m2-quant-pipeline/steps/s2_awq_quant.ipynb`（markdown + 三个填空 docstring；**代码逻辑不动**）

**Interfaces:** s2 现有代码（`build_awq_recipe` / `build_calibration_dataset` / `awq_config_summary` / `run_awq_quantize` / L2 / L3 / 产物检查）全部保留；本任务只**加/改 markdown 讲解** + **给三个填空的 docstring 加 why 引导段**。L1 ipytest 不变（函数签名/实现不变）。

**说明：** 现有 s2 的 AWQ 算法原理讲解（「保护显著通道」）质量 OK、保留。本任务补的是**工具链通路 + 每步 why**（spec §5）。下面给出每个新增/修改点的具体内容。

- [ ] **Step 1: 标题 cell 加「回顾 s0」段**

在 s2 标题 cell（cell 0）末尾追加（不改现有标题/目标）：
```markdown

> **先看 s0**：本 lab 走的就是 s0 讲的那条 `oneshot(model, dataset, recipe)` 通路。AWQ 在通路里的位置 = **两段式 recipe**：第一段 `AWQModifier` 搜显著通道 scale，第二段 `GPTQModifier(W4A16_ASYM)` 压 INT4；激活保持 FP16（weight-only）。本 lab 目标：**端到端跑通 + 理解每步为何这么干**。
```

- [ ] **Step 2: 在「原理」cell 后、填空说明前，插入「端到端四步 why」markdown cell**

新插入一个 markdown cell（在现有 AWQ 原理 cell 之后、「本步填空」cell 之前），内容：
```markdown
## 端到端：AWQ 在通路上每步为什么这么干

套用 s0 的通路（`model + dataset + recipe → oneshot → 产物`），AWQ 每步的 why：

**① dataset 步——为什么 AWQ 要校准数据？**
AWQ 要挑「显著通道」（大激活对应的权重通道），**必须看真实激活幅值**才能挑——所以需要校准数据跑前向。用 `wikitext-2`（`Salesforce/wikitext` 命名空间），AWQ 极省样本，128–256 条就够（不像 GPTQ 要 512+）。

**② recipe 步——为什么是两段？**
- 第一段 `AWQModifier()`：在校准数据上前向，为每个 Linear **搜 per-channel scale**（把显著权重通道放大，使其在 INT4 网格下相对误差变小）。这一段**不带参数**（网格搜索用默认候选值）。
- 第二段 `GPTQModifier(scheme="W4A16_ASYM", targets="Linear", ignore=["lm_head"])`：把权重**真正压成 4-bit**，第一段算的 scale 在这里生效。三个参数各有 why：
  - `targets="Linear"`：只量化 Linear 层（embedding/norm 等保持 FP）。
  - `scheme="W4A16_ASYM"`：4-bit 非对称（带 zero-point，适合权重分布）、group-wise（每 128 个权重共享一组 scale/zp）；激活不量化（A16）。
  - `ignore=["lm_head"]`：lm_head 对输出 logits 最敏感、量化易掉点，跳过。

**③ oneshot 步——调用时内部发生什么？**
回扣 s0 通路：`oneshot(model, dataset, recipe)` → 用校准数据前向收集激活 → 按 recipe 顺序：先 AWQModifier 搜 scale、再 GPTQModifier 压 INT4 → 保存。你填的 recipe 决定了它怎么量化。

**④ 产物步——怎么验证量化对了？**
读产物 `config.json` 的 `quantization_config`，确认 W4A16 证据：`weights.num_bits=4` / `weights.symmetric=False` / `weights.group_size=128` / `input_activations=None`（激活没量化）。这就是下面 `awq_config_summary` 要抽的字段。
```

- [ ] **Step 3: 三个填空的 docstring 各加「why 引导」段（实现提示保留，不动函数签名/实现）**

给三个填空函数的 docstring **开头**加一段 why 引导（现有 docstring 的实现提示/易错点保留）。具体追加内容：

`build_awq_recipe` docstring 开头加：
```markdown
**为什么这么设计（填前先想）**：这个函数产出 s0 通路里的 `recipe`——一个 modifier 有序列表。AWQ 要两段：第一段搜 scale（不带参数，回顾端到端第②步）、第二段压 INT4（要 targets/scheme/ignore，各参数 why 见端到端第②步）。scheme 用入参（别写死，便于复用）；ignore 入参是 tuple 要转 list（GPTQModifier 要求 list）。
```

`build_calibration_dataset` docstring 开头加：
```markdown
**为什么这么设计（填前先想）**：这个函数产出 s0 通路里的 `dataset`——oneshot 要校准数据看激活分布（端到端第①步）。返回 HF `Dataset`、含 `text` 列（oneshot 内部 tokenize，你只给原始文本）。`raw_texts` 非 None 时直接包装（免网络、用于 tiny/测试）；为 None 时从 `wikitext-2` 加载（命名空间 `Salesforce/wikitext`，老裸 `wikitext` 会 HfUriError）。
```

`awq_config_summary` docstring 开头加：
```markdown
**为什么这么设计（填前先想）**：这个函数读 s0 通路里的产物 `quantization_config`（端到端第④步），抽出证明「这是 W4A16」的字段：num_bits=4 / symmetric=False（非对称）/ group_size=128 / input_activations=None（激活没量化）。
```

（注：只加 docstring 引导段，函数签名与实现逻辑不动；L1 ipytest 不变。）

- [ ] **Step 4: L2 / L3 / 产物检查 cell 前各加「验证什么」markdown**

在 L2 cell 前、L3 cell 前、产物检查 cell 前，各加一个简短 markdown（不改这些 cell 的代码）：

L2 前：
```markdown
### L2 在验证什么
tiny 模型上真跑 AWQ 两段式 → 确认通路跑通 + 产物确实是 W4A16（num_bits=4、激活没量化）。这是「通路 + 产物」的缩微验证。
```

L3 前：
```markdown
### L3 在验证什么
真 Qwen2.5（先 0.5B 再 7B）上跑 AWQ → 确认在真实规模也跑通，且权重磁盘占用降到 ~1/4（W4A16 = 4-bit 权重的直接收益）。
```

产物检查前：
```markdown
### 产物检查在验证什么
读 7B AWQ 产物的 `quantization_config`，用 `awq_config_summary` 抽字段 → 确认 W4A16（4-bit/非对称/group=128/激活不量化），并与 FP16（原始）对比显存。这是端到端第④步的落地。
```

- [ ] **Step 5: 验证 L1/L2/L3 仍跑通（注入参考实现，确认代码逻辑没被误改）**

Run:
```bash
cd course/m2-quant-pipeline
uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=1200 steps/s2_awq_quant.ipynb
```
（注入参考实现后执行——参考实现可从现有 s2 的测试语义反推，或临时填入正确实现。）
Expected: 无错；L1 ipytest 全过、L2 tiny 跑通、L3（有 GPU 时）真 7B 跑通产物 W4A16。**关键是确认本任务的 markdown/docstring 改动没破坏代码**（函数实现未动，L1/L2/L3 应与改前一致）。

- [ ] **Step 6: 清输出 + 提交**

```bash
uv run --directory course/m2-quant-pipeline jupyter nbconvert --clear-output --inplace course/m2-quant-pipeline/steps/s2_awq_quant.ipynb
git add course/m2-quant-pipeline/steps/s2_awq_quant.ipynb
git commit -m "feat(m2): rewrite s2 AWQ lab pedagogy (end-to-end why + fill-in why guidance); code logic unchanged"
```

---

### Task 3: 真学员自检 + 验收

**Files:** 无改动（只核对）；若自检发现缺口，回 Task 1/2 返工。

**Interfaces:** 对齐 spec §7.2 验收 + §7.3 真学员自检。

- [ ] **Step 1: controller 充当真学员——通读 s0，答「学完应能讲清」清单**

打开 `steps/s0_pipeline_overview.ipynb` 通读，自答（spec §4）：
1. oneshot 通路三步？（前向收集统计 → 按 recipe 顺序应用 modifier → 保存 compressed-tensors）
2. 五大组件各是什么/为何/格式？
3. 三方法关系（INT 两步补偿 vs FP8 一步 cast）？

答不出/不通透 → 回 Task 1 补。

- [ ] **Step 2: controller 充当真学员——通读 s2，答 AWQ 清单**

打开 `steps/s2_awq_quant.ipynb` 通读，自答（spec §5）：
1. AWQ 在通路哪步？（两段 recipe：搜 scale + 压 INT4）
2. 为什么两段？每段作用？
3. targets/scheme/ignore 各为何？
4. W4A16 产物怎么读、哪些字段证明它？

答不出/不通透 → 回 Task 2 补。

- [ ] **Step 3: 核对范围（s1/s3/脚手架未动）**

Run:
```bash
git diff --stat 1f91441..HEAD -- course/m2-quant-pipeline/
```
Expected: 只有 `s0_pipeline_overview.ipynb`（新增）+ `s2_awq_quant.ipynb`（改）；**s1/s3/pyproject/uv.lock/scripts/README 零改动**。

- [ ] **Step 4: 抽查 s2 代码逻辑未被误改**

Run: `git diff 1f91441..HEAD -- course/m2-quant-pipeline/steps/s2_awq_quant.ipynb`
Expected: diff 只含 markdown cell 新增/修改 + 三个 docstring 的引导段；**`build_awq_recipe`/`build_calibration_dataset`/`awq_config_summary`/`run_awq_quantize` 的函数体代码行未出现在 +/- 侧**（即实现逻辑没动）。若函数体被改 → 回 Task 2 还原。

- [ ] **Step 5: （可选）QA review + 终审**

派一个 reviewer 子 agent 审 s0+s2：准确性（三方法表、组件定义、why 讲解对不对）、连贯性（s0→s2 回扣）、范围（s1/s3/代码逻辑未动）。findings 修完 → 模块就绪，进入 `finishing-a-development-branch` 收尾（merge main）。

---

## Self-Review（写完自查）

**Spec 覆盖**：§3 交付物→T1(s0)+T2(s2)；§4 s0 设计→T1（cell 0-7 对应 §4.1 通路/§4.2 五组件/§4.3 三方法表/§4.4 CONV 例外）；§5 s2 重写→T2（Step1 回顾 s0/Step2 四步 why/Step3 填空 why/Step4 L2L3 说明，对应 §5.1-5.4）；§6 填空哲学→T2 Step3（why 引导，实现不变）；§7.2 验收→T3 Step1-4；§7.3 真学员自检→T3 Step1-2。全覆盖。

**占位符**：无 TBD/TODO；s0 的 7 个 cell 给了完整 markdown/代码；s2 的每个修改点给了具体文本（回顾段/四步 why/三个 docstring 引导/三处验证说明）。

**一致性**：s0 cell 2 路径与 s2/CONV 一致（_find_module_root）；三方法表在 s0(cell 5) 与 spec §4.3、与之前答用户的一致（W4A16/W8A8/FP8 + modifier 组合）；s2 四步 why 的 scheme/modifier 名与 s0 表一致；s2 函数名（build_awq_recipe 等）未变（Task 2 不动实现）。

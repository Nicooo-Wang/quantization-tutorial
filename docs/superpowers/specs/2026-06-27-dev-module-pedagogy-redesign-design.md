# dev-module.js 教学法 gate 重设计 + M2 全 lab 扩展 设计

> 日期：2026-06-27　分支：`m2-pedagogy-redesign`（延续；最后一起 merge）　关联：`docs/superpowers/specs/2026-06-27-m2-pedagogy-redesign-design.md`（M2 AWQ 试点）、`workflows/dev-module.js`

## 1. 背景与动机

用户审核 M2 发现 lab 是「机械化填空」（填 recipe 跑通但不理解通路/组件/关系），而 dev-module.js workflow **没抓出这个教学法问题**。根因诊断（已与用户对齐）：

1. 教学法检查是**模糊词**（"连贯/清晰/概念缺口"），无可判定标准 → QA/student 倾向"看着 OK"。
2. **AI student 读代码就懂**通路，无法模拟真学员「填了但不理解」→ 抓不到理解性缺口。
3. 验收维度偏**结构**（cell 齐/填空够/跑通），无「学没学会」维度。
4. **审角色重叠**：架构师（审+改）与 QA（审）角度重叠，且架构师**既审又改**（自审自改，不干净）。

M2 AWQ 试点（s0 通路总览 + s2 端到端重写，含「摸一摸」实例化演示）已做、用户认可「摸一摸」方向。本次把方向**扩展到全 M2（s1/s3）+ 重设计 workflow 让它能抓教学法**，再跑一遍验收。

## 2. 目标 / 非目标

**目标**
- **重设计 dev-module.js**：架构师+QA 合一成 **reviewer**（审改分离：reviewer 只审 findings，dev 修）；加**学员代理**含**理解探测**（装零基础学员只读讲解答教学目标题）；**教学目标锚点**（每 lab notebook 顶部「学完应能讲清」清单，reviewer/学员代理对照）。
- **扩展 s1/s3**：跟 s0/s2 同一套（教学目标清单 + 摸一摸 + 端到端 why + 填空 why 引导）；代码逻辑保留。
- **跑改好的 workflow 全模块 skipDev 验收** s0/s1/s2/s3（reviewer pass + 学员 satisfied + 闸门 clean）。

**非目标**
- 不改 M1/M3/M4、M2 env（pyproject/uv.lock）/scripts/README。
- 不改 s0/s2 的代码逻辑（只可能补教学目标清单到顶部）。
- 不动 OUTLINE（M2 大纲不变）。

## 3. workflow 新设计（5 阶段，3 角色）

| 阶段 | 角色 | 干啥 | loop |
|---|---|---|---|
| ① 开发 | **dev 开发专家** | 写 notebook（教学目标清单 + 摸一摸 + 讲解 + 填空）+ 验证跑通 + **提供每个填空的参考实现**（供 reviewer 注入跑）。`skipDev` 时只做清单。 | — |
| ② 审核 | **reviewer**（合一原架构师+QA） | 审 结构/准确/教学法（**锚定教学目标清单**）**+ 执行验证**（注入 dev 参考实现 → nbconvert 全量执行每 notebook 的 L1/L2/L3，全过）→ findings → **dev 修** → 再审。**pass = 内容审过 AND 执行验证全过** | ≤3 到 pass |
| ③ 学员试 | **学员代理** | **理解探测**（教学法 gate）：装零基础学员、**只读 markdown 讲解+填空引导**（不读 code cell 实现/测试）、答教学目标题；答不出/要靠偷读代码 = 教学法缺口 → findings → dev 修 → 再试。**不跑代码**（代码跑通由 reviewer 执行验证保证，不重复） | ≤3 到 satisfied |
| ④ 闸门 | 闸门 agent | 核对脚手架未污染（保留现状） | — |
| ⑤ 定稿 | — | acceptance = reviewer.pass && 学员.satisfied && 闸门.clean | — |

**核心变化**：
- **角色合一**：删「架构师」「QA 评审员」「评审员」三个审角色 → 合一成 **reviewer**（少角色、少往返）。
- **审改分离**：reviewer **只审出 findings、不改 notebook**；所有改回退给 **dev**（implementer 模式，避免自审自改）。
- **学员代理加理解探测**（补 AI student 盲区）：见 §5。
- **教学目标锚点**：见 §4。
- **代码跑通 gate**：reviewer 阶段独立执行验证（注入参考实现 + nbconvert 全量执行 L1/L2/L3），**不依赖 dev 自验**——「代码能跑」的硬保证。参考实现由 dev 提供（每填空一份）；reviewer 注入后跑，L1/L2/L3 全过才 pass。

## 4. 教学目标清单机制（notebook 顶部）

- 每个 lab notebook 的**标题 cell 之后**加一个 markdown cell「**## 学完应能讲清**」，列 **3-5 个关键问题**（学完应能口头回答的，不是代码题）。
- reviewer 审、学员代理理解探测都**对照这份清单**（不是泛泛「教学法」）。
- 学员也能当「学习目标/目录」用（一举两得）。
- **s0/s2**：现有「理解检查」段提炼成顶部正式清单（s0 cell 7、s2 的目标段 → 顶部清单）。
- **s1/s3**：新加（见 §6）。

## 5. 学员代理理解探测（核心创新）

学员代理 prompt 关键约束：
- **扮演零基础学员**：不懂 llmcompressor/transformers，第一次见这些类。
- **只许读 markdown 讲解 + 填空 docstring 引导**；**禁止读 code cell 的函数实现/测试/产物代码**（模拟「学员读讲解、还没动手填代码」的状态）。
- **尝试回答教学目标清单的每个问题**，报告：能答 / 要靠偷读代码才能答 / 读了两遍还答不出。
- 「要靠读代码」或「答不出」= **教学法缺口**（讲解不够），写进 findings。
- 兼**跑代码**（L1/L2/L3，注入参考实现后），验证 notebook 可跑（保留原 student 的代码端）。

> 这是逼近「真学员读不懂」的最可行办法：AI 装不懂比装懂难，但「不许读代码」这个硬约束能逼出「光靠讲解够不够」。

## 6. s1/s3 内容扩展（跟 s0/s2 同一套）

**s1_fp8_quant.ipynb**（FP8）：
- 顶部「学完应能讲清」清单（草拟）：① oneshot 通路三步；② FP8 为何只需一个 modifier（浮点 cast、无需补偿）vs INT 两步；③ E4M3(448) vs E5M2(57344) 的来源；④ FP8 产物怎么验证。
- 摸一摸 cell：实例化 `QuantizationModifier(scheme="FP8_DYNAMIC")` 打印 + E4M3/E5M2 dtype cast 演示。
- 端到端 why：FP8 在通路哪步（一步 cast）、为何不要校准（动态范围抗离群）、产物验证。
- 填空 why 引导（保留现有填空实现）。

**s3_smoothquant.ipynb**（SmoothQuant）：
- 顶部清单（草拟）：① oneshot 通路三步；② SmoothQuant 为何两段（SmoothQuantModifier 平滑 + GPTQModifier 量化）；③ α 公式与 `smoothing_strength`（库默认 0.5=论文 α，课程取 0.8）；④ W8A8 产物怎么验证。
- 摸一摸 cell：实例化 `SmoothQuantModifier()` + `GPTQModifier(scheme="W8A8")` 打印 + recipe 构造。
- 端到端 why：两段（先平滑迁移离群点、后量化）、α 为何、产物验证。
- 填空 why 引导（保留现有）。

> s1/s3 的具体清单/why 文本在 plan 阶段细化（参考 s0/s2 已有风格）；**代码逻辑（函数实现/L2/L3）保留不变**，只改/加 markdown + docstring 引导 + 摸一摸 cell。

## 7. 验收（跑改好的 workflow）

1. 改好 dev-module.js（§3-5）+ s1/s3 扩展（§6）+ s0/s2 补顶部清单（§4）。
2. 跑 `dev-module.js`（`skipDev: true`，全模块 s0/s1/s2/s3——notebook 已存在，跳 dev 写，跑 reviewer + 学员代理 + 闸门）。
3. acceptance 全绿：`reviewer.pass（含执行验证：每 notebook 注入参考实现、nbconvert L1/L2/L3 全过）&& 学员.satisfied && 闸门.clean`。
4. **真人（用户）最终审**（真学员，补 AI 盲区）——用户通读 s0/s1/s2/s3，确认「学完能讲清」（spec §7.3 真学员自检的延续）。

## 8. merge 时机

- 继续在 `m2-pedagogy-redesign` 分支（已有 s0/s2 + spec/plan 修正）。
- 实现（dev-module.js 改 + s1/s3 + s0/s2 顶部清单）→ 跑 workflow pass → 真人审满意 → **一次性 merge 全部**（s0/s1/s2/s3 + dev-module.js + spec/plan）回 main。

## 9. 取舍记录

| 决策 | 选择 | 理由 | 备选（未选） |
|---|---|---|---|
| 架构师+QA | 合一 reviewer | 角色重叠，简化 | 保留两角色（冗余） |
| 审改 | 分离（reviewer 审+dev 修） | 客观、避免自审自改 | reviewer 审+改（高效但不干净） |
| 学员代理 | 加理解探测（只读讲解答题） | 补 AI student 盲区，逼近真学员不理解 | 沿用旧 student（读代码就懂，抓不到） |
| 教学目标 | notebook 顶部清单 | 学员当目录 + agent 当检查锚点 | 只放 spec（学员看不到） |
| s1/s3 深度 | 跟 s0/s2 同一套 | 风格一致、彻底解决机械化 | 只加摸一摸（风格不齐） |
| 跑 workflow | 全模块 skipDev | 端到端验 workflow 能抓教学法 | 只跑 s1/s3（没验全模块） |

## 10. 后续

spec 批准 → writing-plans 生实施计划（T1 改 dev-module.js / T2 s1 扩展 / T3 s3 扩展 / T4 s0/s2 补顶部清单 / T5 跑 workflow 验收）→ 执行 → 真人审 → merge。

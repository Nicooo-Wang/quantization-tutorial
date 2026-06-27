# dev-module.js 教学法 gate 重设计 + M2 全 lab 扩展 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 重设计 `dev-module.js`（架构师+QA 合一成 reviewer、审改分离、加学员代理理解探测、加代码跑通执行验证 gate、教学目标锚点），把 s1/s3 扩展到跟 s0/s2 同一套（教学目标+摸一摸+端到端 why+填空 why），给 s0/s2 补顶部教学目标清单，再用改好的 workflow 全模块 skipDev 跑一遍验收。

**Architecture:** workflow 从 6 阶段（dev/架构师/QA loop/student loop/闸门/定稿）改成 5 阶段（dev/reviewer 审核 loop/学员代理 loop/闸门/定稿）。reviewer 只审 + 执行验证（注入参考实现 nbconvert L1/L2/L3）、不改（审改分离，dev 修）。学员代理装零基础学员只读讲解答教学目标题（理解探测，补 AI student 盲区），不跑代码。教学目标清单写每个 notebook 顶部，reviewer/学员代理对照。

**Tech Stack:** workflow JS（`workflows/dev-module.js`）、notebook（nbformat/nbconvert）、M2 env（torch cu128/llmcompressor/transformers v5）、ipytest。

**Spec:** [`docs/superpowers/specs/2026-06-27-dev-module-pedagogy-redesign-design.md`](../specs/2026-06-27-dev-module-pedagogy-redesign-design.md)

## Global Constraints

- 角色合一：删「架构师」「QA 评审员」「评审员」→ 合一成 **reviewer**（审 + 执行验证，不改）；删旧「student 跑代码」→ 学员代理只做**理解探测**（不跑代码）。
- **审改分离**：reviewer 只审出 findings、不改 notebook；所有改回退给 **dev**。
- **代码跑通 gate**：reviewer 注入 dev 提供的参考实现 → nbconvert 全量执行每 notebook L1/L2/L3 → 全过才 pass（独立于 dev 自验）；无 GPU 时 L3 记 skip、L1/L2 必过。
- **教学目标锚点**：每个 lab notebook 标题 cell 后加「## 学完应能讲清」markdown（3-5 问），reviewer/学员代理对照它。
- s1/s3 扩展跟 s0/s2 同一套；**s1/s3/s0/s2 代码逻辑（函数实现/L2/L3/产物）保留不变**，只改/加 markdown + docstring 引导 + 摸一摸 cell + 顶部清单。
- 不改 M2 env（pyproject/uv.lock）/scripts/README、M1/M3/M4、OUTLINE。
- 提交前 nbconvert --clear-output + 删 cell metadata.execution（与 s2 同标准）。
- 在 `m2-pedagogy-redesign` 分支续做（已有 s0/s2 + spec/plan），最后一起 merge。

## File Structure

- Modify: `workflows/dev-module.js`（重写：5 阶段 + reviewer/学员代理 + 执行验证 + 教学目标锚点）
- Modify: `course/m2-quant-pipeline/steps/s1_fp8_quant.ipynb`（顶部清单 + 摸一摸 + 端到端 why + 填空 why；代码不变）
- Modify: `course/m2-quant-pipeline/steps/s3_smoothquant.ipynb`（同上）
- Modify: `course/m2-quant-pipeline/steps/s0_pipeline_overview.ipynb`（顶部教学目标清单——从现有 cell 7「理解检查」提炼到标题后）
- Modify: `course/m2-quant-pipeline/steps/s2_awq_quant.ipynb`（顶部教学目标清单）

---

### Task 1: 重写 dev-module.js（5 阶段 + reviewer + 学员代理 + 执行验证 gate）

**Files:**
- Modify: `workflows/dev-module.js`（整体重写）

**Interfaces:** 产出改好的 workflow，供 Task 5 以 `{module:"m2-quant-pipeline", skipDev:true, spec:"docs/superpowers/specs/2026-06-24-course-development-design.md"}` 调用跑验收。acceptance 含 `reviewerPassed`（含执行验证）/`learnerSatisfied`/`scaffoldClean`/`execVerified`。

- [ ] **Step 1: 用下面完整内容重写 `workflows/dev-module.js`**

```js
export const meta = {
  name: 'dev-module',
  description: '课程模块开发 workflow：开发专家→reviewer 审核(审改分离+执行验证)→学员代理理解探测 loop→闸门→定稿',
  phases: [
    { title: '开发', detail: '开发专家写 notebook + 提供参考实现 + 验证跑通（skipDev 跳过，只清单）' },
    { title: '审核', detail: 'reviewer 审内容(锚教学目标) + 执行验证(注入参考实现 nbconvert L1/L2/L3) → findings → dev 修，loop≤3' },
    { title: '学员试', detail: '学员代理理解探测(只读讲解答教学目标题) → findings → dev 修，loop≤3' },
    { title: '完整性闸门', detail: '核对脚手架未被污染 + uv sync 冒烟' },
    { title: '定稿', detail: '输出 acceptance（reviewer pass[含执行验证] + 学员 satisfied + 闸门 clean）' },
  ],
}

// args 在本环境以 JSON 字符串到达，统一解析
const ARGS = typeof args === 'string' ? JSON.parse(args) : (args || {})
const MODULE = ARGS.module                 // 例 "m2-quant-pipeline"
if (!MODULE || MODULE === 'undefined') {
  throw new Error(
    `dev-module.js: 'module' arg 缺失或 undefined (args=${JSON.stringify(ARGS)})。` +
    `调用：Workflow({scriptPath, args:{module:'<模块目录名>', skipDev:true|false, spec:'<spec路径>'}})。`
  )
}
const MODULE_PATH = `course/${MODULE}`
const MAX_REVIEW_ROUNDS = 3
const MAX_LEARNER_ROUNDS = 3
const CONV = 'course/NOTEBOOK_CONVENTIONS.md'
const SPEC = ARGS.spec || 'docs/superpowers/specs/2026-06-24-course-development-design.md'
const EDIT_SCOPE = `编辑范围：只能创建/修改 ${MODULE_PATH}/steps/*.ipynb。严禁改 pyproject.toml、uv.lock、scripts/、workflows/、README.md、NOTEBOOK_CONVENTIONS.md、顶层任何文件。遇到 env/依赖/CLI 问题写进 findings 报告，不要自己改。`

const DEV_REPORT = {
  type: 'object', additionalProperties: false,
  properties: {
    notebooks: { type: 'array', items: { type: 'string' } },
    blanksTotal: { type: 'number' },
    referenceImpls: { type: 'string' },   // dev 提供的每个填空参考实现（供 reviewer 注入跑），可附在报告里或说明存放位置
    execResults: { type: 'array', items: { type: 'object', additionalProperties: false,
      properties: { notebook:{type:'string'}, l1:{type:'string'}, l2:{type:'string'}, l3:{type:'string'} },
      required: ['notebook','l1','l2','l3'] } },
    notes: { type: 'string' },
  },
  required: ['notebooks','blanksTotal','referenceImpls','execResults','notes'],
}

const REVIEW = {
  type: 'object', additionalProperties: false,
  properties: {
    verdict: { type: 'string', enum: ['pass', 'needs_revision'] },
    execVerification: { type: 'object', additionalProperties: false,
      properties: {
        notebooks: { type: 'array', items: { type: 'object', additionalProperties: false,
          properties: { notebook:{type:'string'}, l1:{type:'string'}, l2:{type:'string'}, l3:{type:'string'}, passed:{type:'boolean'} },
          required: ['notebook','l1','l2','l3','passed'] } },
        allPassed: { type: 'boolean' },   // 每 notebook L1/L2 必过、L3 无 GPU 则 skip 算过
      },
      required: ['notebooks','allPassed'] },
    findings: { type: 'array', items: { type: 'object', additionalProperties: false,
      properties: { severity:{type:'string',enum:['critical','major','minor']}, location:{type:'string'}, issue:{type:'string'}, action:{type:'string'} },
      required: ['severity','location','issue','action'] } },
    summary: { type: 'string' },
  },
  required: ['verdict','execVerification','findings','summary'],
}

const LEARNER_FB = {
  type: 'object', additionalProperties: false,
  properties: {
    comprehension: { type: 'array', items: { type: 'object', additionalProperties: false,
      properties: { notebook:{type:'string'}, question:{type:'string'}, status:{type:'string', enum:['answered','needed_code','could_not_answer']}, note:{type:'string'} },
      required: ['notebook','question','status','note'] } },
    designIssues: { type: 'array', items: { type: 'string' } },
    satisfied: { type: 'boolean' },   // 所有教学目标题都能「只靠讲解」答出
    report: { type: 'string' },
  },
  required: ['comprehension','designIssues','satisfied','report'],
}

const GATE = {
  type: 'object', additionalProperties: false,
  properties: {
    violations: { type: 'array', items: { type: 'string' } },
    notes: { type: 'string' },
    reverted: { type: 'array', items: { type: 'string' } },
    smokeOk: { type: 'boolean' },
  },
  required: ['violations','notes','reverted','smokeOk'],
}

// ---------- ① 开发 ----------
phase('开发')
let devReport
if (ARGS.skipDev) {
  devReport = await agent(`列出模块 ${MODULE}（${MODULE_PATH}/steps/）现有 notebook 清单。对每个 .ipynb：路径 + 填空数（含 NotImplementedError/TODO 的 code cell 数）。按 schema 报 notebooks + blanksTotal（referenceImpls/execResults 给空/'skipDev'）。`,
    { label: '清单(skipDev)', phase: '开发', schema: DEV_REPORT })
  log(`skipDev：清单 ${devReport.notebooks.length} 个 notebook`)
} else {
  devReport = await agent(`你是【课程开发专家】，开发模块 ${MODULE}（${MODULE_PATH}）。
先读 ${CONV} + ${SPEC} + OUTLINE 对应模块 + 模块 spec（若有）。
为本模块每个 Step 创建/改 .ipynb（路径 ${MODULE_PATH}/steps/），严格按 ${CONV} 的 cell 顺序。**每个 notebook 标题 cell 后加「## 学完应能讲清」markdown（3-5 个关键问题，学完应能口头答）**——这是 reviewer/学员代理的检查锚点。
填空设计：从零实现公式/算法，每步 ≥2 填空 + ≥1 判断型，每个填空有 ipytest。
**提供每个填空的参考实现**（写进 referenceImpls 报告字段，或 ${MODULE_PATH}/steps/_solutions/ 旁路文件）——供 reviewer 注入跑执行验证。
写完当场验证：\`cd ${MODULE_PATH} && uv sync\`，注入参考实现跑 L1(ipytest)+L2(tiny)+L3(真模型，无 GPU 则 skip)。提交前 \`nbconvert --clear-output\` + 删 cell metadata.execution。
${EDIT_SCOPE}
按 schema 报告。`, { label: '开发专家', phase: '开发', schema: DEV_REPORT })
}

// ---------- ② reviewer 审核 loop（审改分离：reviewer 审+执行验证，dev 修） ----------
phase('审核')
let reviewerResult = null
for (let i = 0; i < MAX_REVIEW_ROUNDS; i++) {
  const review = await agent(`你是【reviewer】，审模块 ${MODULE}（${MODULE_PATH}/steps/）。合一架构师+QA 角色后你就是唯一审核者。
读 ${CONV} + ${SPEC} + OUTLINE 对应模块。
**审内容**：结构/cell 顺序/准确性/教学法——**锚定每个 notebook 顶部「## 学完应能讲清」清单逐条判**（这条讲清没？），不靠泛泛"连贯"。
**执行验证（代码跑通 gate，必做）**：对每个 notebook，把 dev 提供的参考实现（见开发报告：${JSON.stringify(devReport.referenceImpls)}）注入填空，跑：
  cd ${MODULE_PATH} && uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 steps/<nb>.ipynb
L1(ipytest)+L2(tiny) 必过；L3(真模型) 有 GPU 必过、无 GPU 记 skip（算过）。报 execVerification（每 notebook l1/l2/l3 + passed + allPassed）。
verdict=pass 仅当：无 critical/major findings **AND** execVerification.allPassed=true。
按 schema 报 verdict/execVerification/findings/summary。
${EDIT_SCOPE}（reviewer 只审 + 跑验证，**不改 notebook**——审改分离；改由 dev 在下一 agent 做）`,
    { label: `reviewer(r${i+1})`, phase: '审核', schema: REVIEW })
  reviewerResult = review
  log(`reviewer 第 ${i+1} 轮 verdict=${review.verdict}, execPassed=${review.execVerification.allPassed}, findings=${review.findings.length}`)
  if (review.verdict === 'pass') break
  await agent(`你是【dev】，按 reviewer findings 修模块 ${MODULE}（${MODULE_PATH}/steps/）。
findings：${JSON.stringify(review)}。改掉所有 critical/major（合理采纳 minor）。改完对受影响 notebook 注入参考实现重跑 L1/L2 验证（\`cd ${MODULE_PATH} && uv sync && uv run jupyter nbconvert --execute...\`）。
${EDIT_SCOPE}`,
    { label: `dev 修(r${i+1})`, phase: '审核' })
}

// ---------- ③ 学员代理 理解探测 loop（不跑代码） ----------
phase('学员试')
let learnerResult = null
for (let i = 0; i < MAX_LEARNER_ROUNDS; i++) {
  const fb = await agent(`你是【学员代理】，扮演一个**零基础学员**：从没用过 transformers/llmcompressor，第一次见这些类，不懂 Python 量化库。
通读模块 ${MODULE}（${MODULE_PATH}/steps/）每个 notebook 的 **markdown 讲解 + 填空 docstring 引导 + 「## 学完应能讲清」清单**。
**硬约束：只许读 markdown/docstring；禁止读 code cell 的函数实现、ipytest 测试、L2/L3/产物代码**（模拟"学员读完讲解、还没动手填代码"的状态）。
在该状态下，**尝试回答每个 notebook「## 学完应能讲清」清单的每个问题**，每条报 status：
  - answered：只靠讲解就能答出
  - needed_code：要偷读代码/测试才能答（= 讲解不够）
  - could_not_answer：读了两遍还答不出
needed_code / could_not_answer = 教学法缺口，写进 designIssues（哪个 notebook 哪条、缺什么讲解）。
按 schema 报 comprehension（每条问答状态）/designIssues/satisfied（satisfied = 所有题都 answered）/report。
**不跑代码**（代码跑通由 reviewer 执行验证保证，你不重复）。只读，不改 notebook。`,
    { label: `学员代理(r${i+1})`, phase: '学员试', schema: LEARNER_FB })
  learnerResult = fb
  log(`学员代理 第 ${i+1} 轮 satisfied=${fb.satisfied}`)
  if (fb.satisfied) break
  await agent(`你是【dev】，按学员代理反馈修模块 ${MODULE}（${MODULE_PATH}/steps/）的教学法（补讲解/补 why/补教学目标，不动代码逻辑）。
反馈：${JSON.stringify(fb)}。针对每条 needed_code/could_not_answer 补足 markdown 讲解。
${EDIT_SCOPE}`,
    { label: `dev 修·学员(r${i+1})`, phase: '学员试' })
}

// ---------- ④ 完整性闸门 ----------
phase('完整性闸门')
const gate = await agent(`核对模块 ${MODULE} 开发未污染脚手架。
1. \`git status --porcelain\` 列所有改动。
2. 只有 \`course/${MODULE}/steps/\` 之外的**被 git 跟踪的**改动才算违规；untracked 的 \`.claude/\` 等会话产物不算。
3. 对每个违规文件 \`git checkout HEAD -- <file>\` 还原；steps/ 下不动。
4. 冒烟：\`cd course/${MODULE} && uv sync && uv run python -c "print('ok')"\`。
按 schema 报：violations=只列 steps/ 外被改文件路径（无则 []，说明写进 notes）；notes；reverted；smokeOk。`, { label: '完整性闸门', phase: '完整性闸门', schema: GATE })
log(`完整性闸门: violations=${gate.violations.length}, smokeOk=${gate.smokeOk}`)

// ---------- ⑤ 定稿 ----------
phase('定稿')
const acceptance = {
  module: MODULE,
  notebooks: devReport.notebooks,
  reviewerPassed: reviewerResult && reviewerResult.verdict === 'pass',
  execVerified: reviewerResult && reviewerResult.execVerification && reviewerResult.execVerification.allPassed,
  learnerSatisfied: learnerResult && learnerResult.satisfied,
  scaffoldClean: gate && gate.violations.length === 0 && gate.smokeOk,
}
log(`定稿 ${MODULE}: ${JSON.stringify(acceptance)}`)
return { devReport, reviewerResult, learnerResult, gate, acceptance }
```

- [ ] **Step 2: 语法自检**

Run: `node --check workflows/dev-module.js 2>&1 | head -2 || echo "(ESM export 报错=预期；人工核对 ARGS/MODULE/phases/schemas)"`
Expected: 报 `export const meta` 语法错（ESM，非真错，同 M1/M2 经验）；人工核对 meta 纯字面量、无 Date.now/Math.random、ARGS.module/skipDev/spec 解析正确。

- [ ] **Step 3: 提交**

```bash
git add workflows/dev-module.js
git commit -m "refactor(workflow): pedagogy gate redesign — merge architect+QA into reviewer (review/fix separation), add learner comprehension probe + code-runs exec-verification gate + teach-goal anchor"
```

---

### Task 2: s1_fp8_quant.ipynb 扩展（顶部清单 + 摸一摸 + 端到端 why + 填空 why；代码不变）

**Files:**
- Modify: `course/m2-quant-pipeline/steps/s1_fp8_quant.ipynb`

**Interfaces:** 跟 s0/s2 同一套风格。**代码逻辑（FP8 填空实现、L2、L3、产物检查）保留不变**；只加 markdown + docstring 引导 + 摸一摸 cell + 顶部清单。

- [ ] **Step 1: 标题 cell 后插入「学完应能讲清」清单 cell**

在 s1 标题 cell（cell 0）后插入 markdown cell：
```markdown
## 学完应能讲清（学完本节应能口头回答）

1. `oneshot(model, dataset, recipe)` 内部三步是什么？（前向收集统计 → 按 recipe 顺序应用 modifier → 保存 compressed-tensors）
2. FP8 为何只需**一个** modifier（`QuantizationModifier(FP8_DYNAMIC)`），而 AWQ/SmoothQuant 要两个？（FP8 是浮点、动态范围大、离群点不会让网格崩，直接 cast 无需补偿；INT 要先补偿离群点再量化）
3. E4M3 的 max=448、E5M2 的 max=57344 怎么来的？（指数位→动态范围；E4M3 无 inf、E5M2 有 inf 导致 max exponent 不同）
4. FP8 产物的 `quantization_config` 怎么读、怎么确认它真是 FP8？
```

- [ ] **Step 2: 在原理 cell 后、填空前，插入「端到端 why」markdown cell**

参考 s2 的四步 why 风格，讲 FP8 在通路每步：
```markdown
## 端到端：FP8 在通路上每步为什么这么干

套用 s0 通路（`model + dataset + recipe → oneshot → 产物`）：

**① dataset 步——FP8 为何几乎不要校准？**
FP8 动态量化（激活动态 per-token）靠的是浮点大动态范围**自带抗离群**，不需要像 AWQ/SmoothQuant 那样在校准数据上算 scale。故 FP8 的 `QuantizationModifier(scheme="FP8_DYNAMIC")` 对激活走动态、几乎免校准。

**② recipe 步——为何只要一个 modifier？**
`QuantizationModifier(scheme="FP8_DYNAMIC")` 一步把权重+激活 cast 到 FP8（E4M3）。不需要 SmoothQuant/AWQ 那种「先补偿、后量化」的两段——因为 FP8 浮点本身抗离群。这就是 s0 表里 FP8 只有一行的原因。

**③ oneshot 步——调用时内部发生什么？**
oneshot 前向收集统计（动态量化要 per-token 激活幅值）→ 应用这一个 modifier（cast FP8）→ 保存。比 AWQ/SmoothQuant 短一截（少一步补偿）。

**④ 产物步——怎么验证？**
读 `quantization_config`，确认 `FP8_DYNAMIC`（weights/activations 都是 FP8）。H200 上原生 FP8 kernel，无需降级。
```

- [ ] **Step 3: 插入「亲手摸一摸」code cell（在端到端 why 后）**

```python
## 亲手摸一摸：FP8 的 modifier 长什么样？（运行看输出）
from llmcompressor.modifiers.quantization import QuantizationModifier
import torch

m = QuantizationModifier(scheme="FP8_DYNAMIC")
print("FP8 modifier:", m)
print("  .scheme =", m.scheme, "（动态 FP8，激活 per-token 动态量化）")
print()

# FP8 浮点类型：E4M3 / E5M2
print("E4M3 max =", torch.finfo(torch.float8_e4m3fn).max, "（推理用，无 inf）")
print("E5M2 max =", torch.finfo(torch.float8_e5m2).max, "（有 inf，动态范围更大）")
print("  → 普通权重/激活用 E4M3（448 够覆盖），E5M2 留给极端离群")
```

- [ ] **Step 4: 三个填空 docstring 开头加「为什么这么设计」引导（函数体不动）**

参考 s2 风格，给每个填空 docstring 开头加一段 why 引导（实现提示保留）。具体引导文本由 implementer 按 s2 风格写（指向 s0 通路哪步、参数为何）。**函数体（raise NotImplementedError）不动**。

- [ ] **Step 5: 验证代码逻辑没坏**

注入参考实现跑：`cd course/m2-quant-pipeline && uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 steps/s1_fp8_quant.ipynb`。期望 L1/L2 过、L3（有 GPU）过。核对 `git diff` 只含 markdown + docstring + 摸一摸 cell，函数体未动。

- [ ] **Step 6: 清输出 + 提交**

```bash
uv run --directory course/m2-quant-pipeline jupyter nbconvert --clear-output --inplace course/m2-quant-pipeline/steps/s1_fp8_quant.ipynb
uv run --directory course/m2-quant-pipeline python -c "import nbformat; p='course/m2-quant-pipeline/steps/s1_fp8_quant.ipynb'; nb=nbformat.read(p,as_version=4); [c.get('metadata',{}).pop('execution',None) for c in nb.cells]; nbformat.write(nb,p)"
git add course/m2-quant-pipeline/steps/s1_fp8_quant.ipynb
git commit -m "feat(m2): s1 FP8 pedagogy expansion (teach-goal + hands-on + end-to-end why + fill-in why); code unchanged"
```

---

### Task 3: s3_smoothquant.ipynb 扩展（同 s1 套路；代码不变）

**Files:**
- Modify: `course/m2-quant-pipeline/steps/s3_smoothquant.ipynb`

**Interfaces:** 同 Task 2（s1）套路。**代码逻辑保留不变**。

- [ ] **Step 1: 标题 cell 后插入「学完应能讲清」清单 cell**

```markdown
## 学完应能讲清（学完本节应能口头回答）

1. `oneshot` 通路三步？
2. SmoothQuant 为何是**两段** recipe（`SmoothQuantModifier` + `GPTQModifier(W8A8)`）？（先平滑：把激活离群点等价迁移到权重；后量化：纯 INT8 W8A8）
3. α 公式 `s_j=max(|X_j|)^α/max(|W_j|)^(1-α)` 与 `smoothing_strength` 是什么关系？（库默认 `smoothing_strength=0.5`=论文 α；课程取 0.8 偏向多迁权重）
4. 平滑为何是「等价变换」（前向不变）？
5. W8A8 产物怎么验证？
```

- [ ] **Step 2: 插入「端到端 why」markdown cell（原理后、填空前）**

```markdown
## 端到端：SmoothQuant 在通路上每步为什么这么干

**① dataset 步——为何要校准？** 平滑要算每个 channel 的 `max(|X_j|)`（激活幅值）和 `max(|W_j|)`（权重幅值）定 scale → 需要校准数据前向收集激活。
**② recipe 步——为何两段？** 第一段 `SmoothQuantModifier(smoothing_strength=...)`：算 per-channel scale `s`，做等价变换 `Y=(X/s)·(s·W)`，把激活离群点幅值迁进权重（激活变好量化、权重略难但 INT8 扛得住）。第二段 `GPTQModifier(scheme="W8A8")`：纯 INT8 量化已平滑的模型（GPTQ 用 Hessian 补偿权重，比纯 Round-To-Nearest 准）。
**③ oneshot 步**：前向收集激活统计 → 先平滑、后 GPTQ 量化 → 保存。
**④ 产物步**：读 `quantization_config`，确认 W8A8（权重+激活都 INT8）。
```

- [ ] **Step 3: 插入「亲手摸一摸」code cell**

```python
## 亲手摸一摸：SmoothQuant 两段 modifier + 平滑 scale 长什么样？
from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.gptq import GPTQModifier
import torch

sm = SmoothQuantModifier()              # 默认 smoothing_strength=0.5（=论文 α）
print("SmoothQuantModifier:", sm, "| 默认 smoothing_strength =", sm.smoothing_strength)
gptq = GPTQModifier(scheme="W8A8", targets="Linear", ignore=["lm_head"])
print("GPTQModifier:", gptq, "| .scheme =", gptq.scheme)

recipe = [sm, gptq]                      # 两段：先平滑、后量化
print("recipe:", recipe, "| 两段:", len(recipe))

# 平滑 scale 的直觉（合成激活/权重）
X = torch.randn(2, 8); W = torch.randn(8, 16)
alpha = 0.8
s = (X.abs().max(dim=0).values.pow(alpha) / W.abs().max(dim=0).values.pow(1-alpha)).clamp(min=1e-5)
print("scale s（每 channel 一个）:", s)
print("  → 验证等价：X·W == (X/s)·(s*W)", torch.allclose(X@W, (X/s)@(s[:,None]*W), atol=1e-5))
```

- [ ] **Step 4: 三个填空 docstring 加 why 引导（函数体不动）**

参考 s2 风格。**函数体不动**。

- [ ] **Step 5: 验证代码逻辑没坏（注入参考实现 nbconvert）+ 清输出 + 提交**

同 Task 2 Step 5-6（换成 s3）。commit message: `feat(m2): s3 SmoothQuant pedagogy expansion (teach-goal + hands-on + end-to-end why + fill-in why); code unchanged`

---

### Task 4: s0/s2 补顶部「学完应能讲清」清单

**Files:**
- Modify: `course/m2-quant-pipeline/steps/s0_pipeline_overview.ipynb`（标题 cell 后插清单）
- Modify: `course/m2-quant-pipeline/steps/s2_awq_quant.ipynb`（标题 cell 后插清单）

**Interfaces:** s0/s2 现有「理解检查」/目标段提炼成顶部正式清单（与 s1/s3 同位置：标题 cell 后）。内容/代码不动，只加一个 markdown cell。

- [ ] **Step 1: s0 标题 cell（cell 0）后插入清单 cell**

```markdown
## 学完应能讲清（通路总览，学完应能口头回答）

1. `oneshot(model, dataset, recipe)` 内部三步是什么？
2. 五大组件（dataset/recipe/modifier/scheme/quantization_config）各是什么、为什么需要、什么格式？
3. FP8 为何只要一个 modifier、AWQ/SmoothQuant 要两个？（INT 要补偿离群点，FP8 浮点直接 cast）
```
（s0 现有 cell 7「理解检查」保留——顶部清单是目录用，cell 7 是学完自测，不冲突。）

- [ ] **Step 2: s2 标题 cell 后插入清单 cell**

```markdown
## 学完应能讲清（AWQ，学完应能口头回答）

1. AWQ 在 oneshot 通路哪步？（两段 recipe：AWQModifier 搜 scale + QuantizationModifier(W4A16_ASYM) 压 INT4）
2. 为何两段？每段作用？
3. targets/scheme/ignore 各为何？为何 ignore lm_head？
4. W4A16 产物怎么读、哪些字段证明它？
```

- [ ] **Step 3: 清输出 + 提交**

```bash
for nb in s0_pipeline_overview s2_awq_quant; do
  uv run --directory course/m2-quant-pipeline jupyter nbconvert --clear-output --inplace course/m2-quant-pipeline/steps/${nb}.ipynb
done
git add course/m2-quant-pipeline/steps/s0_pipeline_overview.ipynb course/m2-quant-pipeline/steps/s2_awq_quant.ipynb
git commit -m "feat(m2): add top 'learn-objectives' checklist to s0/s2 (anchor for reviewer/learner-agent)"
```

---

### Task 5: 跑改好的 dev-module.js 全模块 skipDev 验收

**Files:** 由 workflow reviewer/学员代理可能改 steps/*.ipynb（按 findings）。

**Interfaces:** 用 Task 1 改好的 workflow，`{module:"m2-quant-pipeline", skipDev:true}`（notebook 已存在，跑 reviewer 审核 + 执行验证 + 学员代理理解探测 + 闸门）。

- [ ] **Step 1: 跑 workflow（后台，完成通知）**

```
Workflow({ scriptPath: "workflows/dev-module.js",
           args: { module: "m2-quant-pipeline", skipDev: true } })
```
用 `/workflows` 看进度（审核 loop → 学员试 loop → 闸门 → 定稿）。

- [ ] **Step 2: 取回 acceptance 核对**

workflow 完成后读 `acceptance`，全绿：
- `reviewerPassed === true`
- `execVerified === true`（每 notebook 注入参考实现 nbconvert L1/L2 全过）
- `learnerSatisfied === true`（学员代理只读讲解能答所有教学目标题）
- `scaffoldClean === true`

- [ ] **Step 3: 不通过则按 findings 迭代**

若 reviewer/学员代理有 findings：直接改 notebook（补讲解/补 why/补教学目标），再跑对应阶段。封顶（各 3 轮）仍不过 → 升级人工带 findings。

- [ ] **Step 4: 真人（用户）最终审**

用户通读 s0/s1/s2/s3，确认「学完能讲清」（spec §7.3 真学员自检的最终闸，补 AI 学员代理盲区）。

- [ ] **Step 5: 清输出 + 提交（核对只动 steps/）+ 进入 finishing**

```bash
for nb in course/m2-quant-pipeline/steps/*.ipynb; do
  uv run --directory course/m2-quant-pipeline jupyter nbconvert --clear-output --inplace "$nb"
done
git add course/m2-quant-pipeline/steps/
git diff --cached --stat   # 核对只动 steps/*.ipynb
git commit -m "feat(m2): full-module pedagogy pass via redesigned workflow (reviewer exec-verify + learner comprehension probe)"
```
然后 `superpowers:finishing-a-development-branch`：merge `m2-pedagogy-redesign` 回 main（含 s0/s1/s2/s3 + dev-module.js + spec/plan）。

---

## Self-Review（写完自查）

**Spec 覆盖**：§3 workflow 新设计→T1；§4 教学目标清单→T2/T3（s1/s3）+T4（s0/s2）；§5 学员代理理解探测→T1（LEARNER_FB schema + prompt）；§3 代码跑通 gate→T1（reviewer execVerification + DEV_REPORT referenceImpls）；§6 s1/s3 扩展→T2/T3；§7 验收→T5。全覆盖。

**占位符**：T1 给完整 dev-module.js；T2/T3 给清单/why/摸一摸 cell 具体内容（填空引导标"参考 s2 风格"——s2 已有，是引用非占位）；T4 给 s0/s2 清单文本；T5 给 workflow 调用 + acceptance 核对。

**一致性**：教学目标清单在 s0/s1/s2/s3 顶部统一格式（"## 学完应能讲清"）；摸一摸 cell 在 s0（已有）/s1/s3（新）统一风格；reviewer schema（REVIEW.execVerification）+ dev schema（DEV_REPORT.referenceImpls）+ 学员代理 schema（LEARNER_FB.comprehension）在 T1 定义、T5 acceptance 引用一致；三方法 modifier（FP8/AWQ/SmoothQuant）在 s0 表 + s1/s3 一致（AWQ=QuantizationModifier、SmoothQuant=GPTQModifier——与 M2 重设计 spec 一致）。

export const meta = {
  name: 'dev-module',
  description: '课程模块开发 workflow：开发专家→架构师审核→QA 评审 loop→student worktree 试课 loop→定稿',
  phases: [
    { title: '开发', detail: '开发专家写模块全部 notebook（填空+ipytest+tiny+H200 cell）' },
    { title: '架构审核', detail: '架构师审一致性/填空/执行/连贯' },
    { title: 'QA 评审', detail: '评审员严格审，循环到 pass（封顶 3 轮）' },
    { title: 'Student 试课', detail: 'worktree 真跑 H200 + 评课程设计，循环到满意（封顶 3 轮）' },
    { title: '完整性闸门', detail: '核对脚手架未被污染：还原 steps/ 之外改动 + uv sync 冒烟' },
    { title: '定稿', detail: '输出验收结果（merge/commit 由调用方做）' },
  ],
}

// args 在本环境以 JSON 字符串形式到达（probe 实测 argsType==='string'），统一解析成对象
const ARGS = typeof args === 'string' ? JSON.parse(args) : (args || {})
const MODULE = ARGS.module                 // 例 "m2-quant-pipeline"
if (!MODULE || MODULE === 'undefined') {
  // fail-fast：module 缺失时宁可大声失败，也不要静默跑去开发错的模块
  // （历史上 args.module 渲染成 'undefined'，dev agent 误开发了 M1）
  throw new Error(
    `dev-module.js: 'module' arg 缺失或为 undefined (args=${JSON.stringify(ARGS)})。` +
    `调用方式：Workflow({scriptPath, args:{module:'<模块目录名>', skipDev:true}})。`
  )
}
const MODULE_PATH = `course/${MODULE}`
const MAX_QA_ROUNDS = 3
const MAX_STUDENT_ROUNDS = 3
const CONV = 'course/NOTEBOOK_CONVENTIONS.md'   // 约定文档，agent 读取
// SPEC 可按模块透传（args.spec，传 M1/M3/M4 各自 spec 路径）；缺省回退旧总 spec（向后兼容 M2）
const SPEC = ARGS.spec || 'docs/superpowers/specs/2026-06-24-course-development-design.md'

const DEV_REPORT = {
  type: 'object', additionalProperties: false,
  properties: {
    notebooks: { type: 'array', items: { type: 'string' } },          // 写了哪些 .ipynb 路径
    blanksTotal: { type: 'number' },                                   // 总填空数（每步≥2）
    execResults: { type: 'array', items: { type: 'object', additionalProperties: false,
      properties: { notebook:{type:'string'}, l1:{type:'string'}, l2:{type:'string'}, l3:{type:'string'} },
      required: ['notebook','l1','l2','l3'] } },                       // 各 cell 执行结果 pass/skip/fail
    notes: { type: 'string' },
  },
  required: ['notebooks','blanksTotal','execResults','notes'],
}

const REVIEW = {
  type: 'object', additionalProperties: false,
  properties: {
    verdict: { type: 'string', enum: ['pass', 'needs_revision'] },
    findings: { type: 'array', items: { type: 'object', additionalProperties: false,
      properties: { severity:{type:'string',enum:['critical','major','minor']}, location:{type:'string'}, issue:{type:'string'}, action:{type:'string'} },
      required: ['severity','location','issue','action'] } },
    summary: { type: 'string' },
  },
  required: ['verdict','findings','summary'],
}

const STUDENT_FB = {
  type: 'object', additionalProperties: false,
  properties: {
    codeRun: { type: 'object', additionalProperties: false,
      properties: { filledBlanksOk:{type:'boolean'}, notebooksRan:{type:'array',items:{type:'string'}}, errors:{type:'array',items:{type:'string'}} },
      required: ['filledBlanksOk','notebooksRan','errors'] },
    designIssues: { type: 'array', items: { type: 'string' } },        // 连贯/清晰/概念缺口/教学法/填空难度
    satisfied: { type: 'boolean' },                                    // 代码跑通 AND 设计满意
    report: { type: 'string' },
  },
  required: ['codeRun','designIssues','satisfied','report'],
}

const GATE = {
  type: 'object', additionalProperties: false,
  properties: {
    violations: { type: 'array', items: { type: 'string' } },   // 只放 steps/ 之外被改的文件路径（真违规）；无则 []
    notes: { type: 'string' },                                  // 说明性文字（如"无可还原违规"）放这里，绝不进 violations（否则 scaffoldClean 假阴性）
    reverted: { type: 'array', items: { type: 'string' } },     // 已还原的文件
    smokeOk: { type: 'boolean' },                               // uv sync + 冒烟
  },
  required: ['violations','notes','reverted','smokeOk'],
}

// ---------- ① 开发 ----------
phase('开发')
let devReport
if (ARGS.skipDev) {
  // notebook 已存在且 L1/L2 已验过；只做清单，跳过开发期（M2 用）
  devReport = await agent(`列出模块 ${MODULE}（${MODULE_PATH}/steps/）现有 notebook 清单。
对每个 .ipynb：路径 + 填空数（含 NotImplementedError/TODO 的 code cell 数）。
按 schema 报 notebooks + blanksTotal（execResults 给空数组，notes 写 "skipDev"）。`,
    { label: '清单(skipDev)', phase: '开发', schema: DEV_REPORT })
  log(`skipDev：跳过开发，清单 ${devReport.notebooks.length} 个 notebook`)
} else {
  devReport = await agent(`你是【课程开发专家】，开发模块 ${MODULE}（路径 ${MODULE_PATH}）。
先读 ${CONV}（notebook cell 约定）与 ${SPEC}（§4/§5 结构、§11 验收）与 OUTLINE.md 对应模块。
为本模块每个 Step 创建一个 .ipynb（路径 ${MODULE_PATH}/steps/），严格按 ${CONV} 的 cell 顺序：
标题→导入→讲解→填空代码 cell（每步≥2 个 logic 函数 TODO）→ipytest 测试 cell(L1)→tiny 验证 cell(L2)→H200 执行 cell(L3,GPU 守卫)→产物检查 cell。
填空设计由你定（用户全权委托）：选有教学意义的 logic 函数，难度递进，每个填空都有 ipytest 测试。
**写完当场验证跑通**：在模块目录内 \`cd ${MODULE_PATH} && uv sync\`，notebook 用 module-local 路径（\`_find_module_root\` 解析 MODULE_ROOT/MODEL_DIR），跑 L1（ipytest，可用参考实现填空后测）+ L2（tiny 模型，CPU）+ L3（H200，真 Qwen2.5-0.5B 再 7B，若无 GPU 则记录 skip）。运行入口 \`cd ${MODULE_PATH} && uv run jupyter lab\`。
提交前对每个 notebook 跑 \`jupyter nbconvert --clear-output --inplace <nb>\`。
编辑范围：只能创建/修改 ${MODULE_PATH}/steps/*.ipynb。严禁改 pyproject.toml、uv.lock、scripts/、workflows/、README.md、NOTEBOOK_CONVENTIONS.md、顶层任何文件。遇到 env/依赖/CLI 问题写进 findings 报告，不要自己改。
按 schema 报告：写了哪些 notebook、总填空数、各 cell 执行结果、备注。`, { label: '开发专家', phase: '开发', schema: DEV_REPORT })
}

// ---------- ② 架构师审核 ----------
phase('架构审核')
const archReview = await agent(`你是【课程架构师】，审核模块 ${MODULE}（${MODULE_PATH}/steps/）。
读 ${CONV} + OUTLINE 对应模块 + 开发报告：${JSON.stringify(devReport)}。
审：cell 布局是否符合约定、Step 衔接、填空设计合理性（教学意义+难度）、notebook 是否真能跑通（L1/L2/L3，跑法 \`cd ${MODULE_PATH} && uv sync && uv run jupyter lab\`，notebook 用 module-local 路径）、整体连贯。
如需返工，直接在 notebook 里改（你有编辑权），并把 critical/major 改掉。按 schema 输出 verdict/findings/summary。
编辑范围：只能创建/修改 ${MODULE_PATH}/steps/*.ipynb。严禁改 pyproject.toml、uv.lock、scripts/、workflows/、README.md、NOTEBOOK_CONVENTIONS.md、顶层任何文件。遇到 env/依赖/CLI 问题写进 findings 报告，不要自己改。`,
  { label: '课程架构师(审核)', phase: '架构审核', schema: REVIEW })

// ---------- ③ QA 评审 loop ----------
phase('QA 评审')
let qaResult = null
for (let i = 0; i < MAX_QA_ROUNDS; i++) {
  const qa = await agent(`你是【质量保证评审员】，严格审模块 ${MODULE}（${MODULE_PATH}/steps/）。
读 ${CONV} + ${SPEC}(§11 验收) + OUTLINE 对应模块。严格审：准确性/完备/填空+ipytest 覆盖/跑通证据(L1/L2/L3，跑法 \`cd ${MODULE_PATH} && uv sync && uv run jupyter lab\`，notebook 用 module-local 路径)/教学法+连贯。
findings 用 severity 标注，每条给 action。verdict=pass 仅当无 critical/major。按 schema 输出。
编辑范围：只能创建/修改 ${MODULE_PATH}/steps/*.ipynb。严禁改 pyproject.toml、uv.lock、scripts/、workflows/、README.md、NOTEBOOK_CONVENTIONS.md、顶层任何文件。遇到 env/依赖/CLI 问题写进 findings 报告，不要自己改。`,
    { label: `质量保证评审员(r${i+1})`, phase: 'QA 评审', schema: REVIEW })
  qaResult = qa
  log(`QA 第 ${i+1} 轮 verdict=${qa.verdict}, findings=${qa.findings.length}`)
  if (qa.verdict === 'pass') break
  await agent(`你是【课程架构师】，按 QA 评审修模块 ${MODULE}（${MODULE_PATH}/steps/）。
评审：${JSON.stringify(qa)}。改掉所有 critical/major，合理采纳 minor。改完在模块目录内重跑受影响 notebook 的 L1/L2 cell 验证（\`cd ${MODULE_PATH} && uv sync && uv run jupyter lab\`，notebook 用 module-local 路径）。
编辑范围：只能创建/修改 ${MODULE_PATH}/steps/*.ipynb。严禁改 pyproject.toml、uv.lock、scripts/、workflows/、README.md、NOTEBOOK_CONVENTIONS.md、顶层任何文件。遇到 env/依赖/CLI 问题写进 findings 报告，不要自己改。`,
    { label: `课程架构师(修·QA r${i+1})`, phase: 'QA 评审' })
}

// ---------- ④ Student 试课 loop ----------
phase('Student 试课')
let studentResult = null
for (let i = 0; i < MAX_STUDENT_ROUNDS; i++) {
  // 先把当前 steps/ 提交到主工作树分支：worktree 是 git checkout 的副本，untracked/未提交改动不进副本，
  // 故必须 commit，让即将启动的 student worktree（base=head）能拿到最新 notebook。
  await agent(`把 ${MODULE_PATH}/steps/ 当前状态提交到当前分支，让即将启动的 student worktree 能拿到最新 notebook。
在主工作树（repo root）跑：
  git add ${MODULE_PATH}/steps/
  git diff --cached --quiet || git commit -m "chore(${MODULE}): notebook snapshot before student round ${i+1}"
若无可提交改动则跳过。一句话报告结果。`, { label: `快照(r${i+1})`, phase: 'Student 试课' })

  const fb = await agent(`你是【Student/学员】，真实用户代理，试课模块 ${MODULE}。
你会得到一个隔离 git worktree（课程副本）。两端评估：
(代码端) 在隔离 worktree 内跑完整流程：
  cd ${MODULE_PATH}
  uv sync
  # 复用主仓已下基线模型，免重下（worktree 是独立工作树）：
  MAIN="$(git rev-parse --git-common-dir)/.." && MODEL_CACHE="$MAIN/models" bash scripts/download_model.sh
  # （若该路径无模型则脚本会正常下载）
  uv run jupyter lab  # 或逐个 nbconvert --execute
然后自己**填空**(实现 logic 函数) → 跑 L1(ipytest) → L2(tiny) → L3(真 Qwen2.5-0.5B 再 7B H200)。记录能否填出、跑通/报错。
(知识端) 通读全章，评知识连贯/讲解清晰/逻辑跳跃/概念缺口/教学法/填空难度，凡觉得设计不好的都记。
按 schema 输出：codeRun(filledBlanksOk/notebooksRan/errors)、designIssues、satisfied(代码跑通 AND 设计满意)、report。
编辑范围：只能创建/修改 ${MODULE_PATH}/steps/*.ipynb。严禁改 pyproject.toml、uv.lock、scripts/、workflows/、README.md、NOTEBOOK_CONVENTIONS.md、顶层任何文件。遇到 env/依赖/CLI 问题写进 findings/报告，不要自己改。`,
    { label: `Student(r${i+1})`, phase: 'Student 试课', schema: STUDENT_FB, isolation: 'worktree' })
  studentResult = fb
  log(`Student 第 ${i+1} 轮 satisfied=${fb.satisfied}`)
  if (fb.satisfied) break
  await agent(`你是【课程架构师+评审员】联合，按 student 反馈修模块 ${MODULE}。
反馈：${JSON.stringify(fb)}。架构师改代码/结构/填空，评审员定夺教学法/连贯问题。改完在模块目录内重跑 L1/L2 验证（\`cd ${MODULE_PATH} && uv sync && uv run jupyter lab\`，notebook 用 module-local 路径）。
编辑范围：只能创建/修改 ${MODULE_PATH}/steps/*.ipynb。严禁改 pyproject.toml、uv.lock、scripts/、workflows/、README.md、NOTEBOOK_CONVENTIONS.md、顶层任何文件。遇到 env/依赖/CLI 问题写进 findings 报告，不要自己改。`,
    { label: `架构师+评审员(修·student r${i+1})`, phase: 'Student 试课' })
}

// ---------- ⑤ 完整性闸门（定稿前核对脚手架未被污染） ----------
phase('完整性闸门')
const gate = await agent(`核对模块 ${MODULE} 开发未污染脚手架。
1. 跑 \`git status --porcelain\` 列所有改动。
2. 只有 \`course/${MODULE}/steps/\` 之外的**被 git 跟踪的**改动才算违规（尤其 pyproject.toml/uv.lock/scripts/workflows/README/顶层）；untracked 的 \`.claude/\` 等会话产物不算违规。
3. 对每个违规文件 \`git checkout HEAD -- <file>\` 还原；steps/ 下任何文件一律不动。
4. 冒烟：\`cd course/${MODULE} && uv sync && uv run python -c "print('ok')"\`。
按 schema 报：violations=只列 steps/ 外被改的**文件路径**（无则 []；**绝不要**把"无违规""no tracked violations"之类的说明写进 violations——说明写进 notes，否则 scaffoldClean 会假阴性）；notes=说明性文字；reverted；smokeOk。`, { label: '完整性闸门', phase: '完整性闸门', schema: GATE })
log(`完整性闸门: violations=${gate.violations.length}, smokeOk=${gate.smokeOk}`)

// ---------- ⑥ 定稿（输出验收；merge/commit 由调用方做） ----------
phase('定稿')
const acceptance = {
  module: MODULE,
  notebooks: devReport.notebooks,
  blanksPerStepOk: devReport.blanksTotal >= 2 * (devReport.notebooks.length || 1),
  qaPassed: qaResult && qaResult.verdict === 'pass',
  studentSatisfied: studentResult && studentResult.satisfied,
  qaRounds: qaResult ? qaResult.verdict : 'unfinished',
  studentRounds: studentResult ? studentResult.satisfied : 'unfinished',
  scaffoldClean: gate && gate.violations.length === 0 && gate.smokeOk,
}
log(`定稿 ${MODULE}: ${JSON.stringify(acceptance)}`)
return { devReport, archReview, qaResult, studentResult, gate, acceptance }

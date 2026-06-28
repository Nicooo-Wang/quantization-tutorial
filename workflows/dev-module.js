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
// M3 双 env：子项目→env 映射。无 envs（M1/M2/M4）时退化为单根 env（向后兼容，行为不变）。
// envs 项形如 { dir: "steps/quant", smoke: "import llmcompressor,transformers,matplotlib" }
const ENVS = (ARGS.envs && ARGS.envs.length) ? ARGS.envs : [{ dir: '', smoke: "print('ok')" }]
const MULTI_ENV = ENVS.length > 1 || (ENVS.length === 1 && ENVS[0].dir)
// 多 env 时注入 dev/reviewer/gate prompt 的运行说明（单 env 时为空，prompt 行为不变）
const ENV_HELP = MULTI_ENV ? `

【本模块是多 env（子项目隔离，重要）】notebook 在 ${MODULE_PATH}/steps/ 的**子目录**里，每个子目录是独立 uv 项目（自带 .venv）：
${ENVS.map(e => `  - ${e.dir}/（uv 项目根 ${MODULE_PATH}/${e.dir}）`).join('\n')}

运行/执行规则：
- 每个 notebook 按其所在子目录跑：\`uv run --directory ${MODULE_PATH}/<子目录> <cmd>\`，或直接 \`${MODULE_PATH}/<子目录>/.venv/bin/python\`。
- 模块根 ${MODULE_PATH} **没有 pyproject.toml**——绝不在模块根跑 \`uv sync\`（报 no project）；只在各子目录 uv sync。
- setup cell 的 _find_module_root 用「scripts/ + steps/」判据找模块根（子项目自己有 pyproject.toml 但无 steps/，故别用 pyproject 判据）：
  \`\`\`python
  import pathlib
  def _find_module_root(start):
      p = pathlib.Path(start).resolve()
      for cand in [p, *p.parents]:
          if (cand / "scripts").is_dir() and (cand / "steps").is_dir():
              return cand
      raise RuntimeError("找不到模块根（含 scripts/ + steps/ 的目录）")
  MODULE_ROOT = _find_module_root(pathlib.Path.cwd())
  MODEL_DIR      = MODULE_ROOT / "models" / "Qwen2.5-7B-Instruct"
  TINY_MODEL_DIR = MODULE_ROOT / "models" / "Qwen2.5-0.5B-Instruct"
  OUT_ROOT       = MODULE_ROOT / "out"; OUT_ROOT.mkdir(parents=True, exist_ok=True)
  \`\`\`
- models/、out/ 在模块根共享（不在子目录），由上面的 MODULE_ROOT 解析。
- vLLM 子项目（若含）：lm_eval[vllm] extra 不在 uv.lock——执行其 notebook 前先 \`cd ${MODULE_PATH}/<vllm子目录> && uv pip install --python ./.venv/bin/python 'lm_eval[vllm]'\`（幂等）；vllm env 的冒烟/检查一律用 \`./.venv/bin/python\`（不用 uv run，会触发数分钟重装）。
` : ''
// gate 冒烟命令：多 env 遍历各子目录 sync + .venv/bin/python 冒烟；单 env 维持原行为
const SMOKE_CMD = MULTI_ENV
  ? ENVS.map(e => `cd ${MODULE_PATH}/${e.dir} && uv sync && ./.venv/bin/python -c "${e.smoke || "print('ok')"}"`).join(' && ')
  : `cd course/${MODULE} && uv sync && uv run python -c "print('ok')"`
const MAX_REVIEW_ROUNDS = 3
const MAX_LEARNER_ROUNDS = 3
const CONV = 'course/NOTEBOOK_CONVENTIONS.md'
const SPEC = ARGS.spec || 'docs/superpowers/specs/2026-06-24-course-development-design.md'
const EDIT_SCOPE = `编辑范围：只能创建/修改 ${MODULE_PATH}/steps/ 下的 .ipynb（含子目录，如 steps/quant/、steps/vllm/）。严禁改任何 pyproject.toml、uv.lock、scripts/、workflows/、README.md、NOTEBOOK_CONVENTIONS.md、顶层任何文件。遇到 env/依赖/CLI 问题写进 findings 报告，不要自己改。`

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
写完当场验证：\`cd ${MODULE_PATH} && uv sync\`，注入参考实现跑 L1(ipytest)+L2(tiny)+L3(真模型，无 GPU 则 skip)。**L3 cell 除 GPU 守卫外再加 \`os.environ.get('SKIP_L3')\` 守卫**——reviewer 执行验证设 SKIP_L3=1 跳过真 7B（太慢，只验 L1+L2 代码逻辑）；真人/学员跑时不设，L3 实证。形如 \`if torch.cuda.is_available() and not os.environ.get('SKIP_L3'): run_l3(...) else: print('跳过 L3')\`。提交前 \`nbconvert --clear-output\` + 删 cell metadata.execution。
${EDIT_SCOPE}${ENV_HELP}
按 schema 报告。`, { label: '开发专家', phase: '开发', schema: DEV_REPORT })
}

// ---------- ② reviewer 审核 loop（审改分离：reviewer 审+执行验证，dev 修） ----------
phase('审核')
let reviewerResult = null
for (let i = 0; i < MAX_REVIEW_ROUNDS; i++) {
  const review = await agent(`你是【reviewer】，审模块 ${MODULE}（${MODULE_PATH}/steps/）。合一架构师+QA 角色后你就是唯一审核者。
读 ${CONV} + ${SPEC} + OUTLINE 对应模块。
**审内容**：结构/cell 顺序/准确性/教学法——**锚定每个 notebook 顶部「## 学完应能讲清」清单逐条判**（这条讲清没？），不靠泛泛"连贯"。
**执行验证（代码跑通 gate，必做）**：对每个 notebook，把参考实现注入填空（来源：dev 报告 referenceImpls；**若为空/skipDev（skipDev 模式，notebook 已存在），从 notebook 的 ipytest 测试语义反推每个填空的正确实现注入**），跑 nbconvert 全量执行：
${MULTI_ENV
  ? `  多 env：按 notebook 所在子目录跑——\`SKIP_L3=1 uv run --directory ${MODULE_PATH}/<子目录> jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 <子目录>/<nb>.ipynb\`。含 vLLM 子项目的先 \`cd ${MODULE_PATH}/<vllm子目录> && uv pip install --python ./.venv/bin/python 'lm_eval[vllm]'\` 再跑（幂等）。`
  : `  cd ${MODULE_PATH} && SKIP_L3=1 uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 steps/<nb>.ipynb`}
**执行方式（重要，避免卡死）**：用上面的 **foreground** 命令一次跑完一个 notebook 并等其返回（Bash timeout 设 600000ms）。**严禁**把 nbconvert 放 background / 用 Monitor 或 TaskOutput 轮询 / 分离子进程等待——那样无法可靠检测 nbconvert 完成会无限挂起。命令前的 \`SKIP_L3=1\` 让 L3(真 7B) cell 自动跳过（notebook L3 cell 有 \`os.environ.get('SKIP_L3')\` 守卫），只验 L1(ipytest)+L2(tiny)，秒~分钟级。
L1(ipytest)+L2(tiny) 必过；L3(真模型) 由 SKIP_L3=1 跳过（真 7B 太慢，留真人/学员 GPU 实证，workflow 只验代码逻辑）。报 execVerification（每 notebook l1/l2/l3=skip + passed + allPassed）。
verdict=pass 仅当：无 critical/major findings **AND** execVerification.allPassed=true。
按 schema 报 verdict/execVerification/findings/summary。
${EDIT_SCOPE}${ENV_HELP}（reviewer 只审 + 跑验证，**不改 notebook**——审改分离；改由 dev 在下一 agent 做）`,
    { label: `reviewer(r${i+1})`, phase: '审核', schema: REVIEW })
  reviewerResult = review
  log(`reviewer 第 ${i+1} 轮 verdict=${review.verdict}, execPassed=${review.execVerification.allPassed}, findings=${review.findings.length}`)
  if (review.verdict === 'pass') break
  await agent(`你是【dev】，按 reviewer findings 修模块 ${MODULE}（${MODULE_PATH}/steps/）。
findings：${JSON.stringify(review)}。改掉所有 critical/major（合理采纳 minor）。改完对受影响 notebook 注入参考实现重跑 L1/L2 验证（**foreground**，禁 background/Monitor 轮询：\`SKIP_L3=1 uv run --directory ${MODULE_PATH}/<子目录> jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 <子目录>/<nb>.ipynb\`，Bash timeout 600000ms）。
${EDIT_SCOPE}${ENV_HELP}`,
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
4. 冒烟：\`${SMOKE_CMD}\`。
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

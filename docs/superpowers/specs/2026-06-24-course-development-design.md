# 课程开发系统设计（Course Development System）

- 日期：2026-06-24
- 状态：已批准（待用户 spec 复核）
- 依据：已批准的 [`OUTLINE.md`](../../../OUTLINE.md)（4 模块 / 28 课时，H200×8，SmoothQuant + AWQ + FP8，Qwen2.5-7B，vLLM-only）

---

## 1. 目标与背景

按已批准的大纲**开发整套动手课程**：每个模块 = 一个 lesson 交付物。本 spec 定义的是**开发系统**（课程结构 + 测试架构 + 多角色开发 workflow），课程内容由该系统**逐模块**产出。

相比之前"大纲修订 workflow"的关键改进：专家从"提建议"升级为"真开发代码"；新增 student 试课 + git worktree；所有循环跑到"无问题"为止。

## 2. 两条硬约束（用户强调）

- **C1（每步配 UT）**：每个 Step 的代码都要有单元测试。
- **C2（能跑通、早发现框架问题）**：Step 设计逻辑必须能端到端跑通，框架级问题要在开发期暴露，不能攒到最后才卡死。

本设计的所有决策都围绕同时满足 C1 与 C2。

## 3. 关键决策（brainstorming 已定）

| 决策点 | 选择 |
|--------|------|
| 交付形态 | **notebook-only**：每个 Step 一个 `.ipynb`（就是那堂课），测试用 **ipytest** 写在 cell 里；无 `.py` lesson 脚本、无 jupytext |
| student 试课 | **agent**，在 **git worktree** 里**真跑 H200**，评估**代码 + 课程设计** |
| 测试架构 | **三层测试都在 notebook 内**（L1 ipytest / L2 tiny 真模型 / L3 H200 执行 cell） |
| 开发主体 | **单个开发专家**开发整模块（不并行分 Step） |
| student 反馈 | 同时给**架构师和评审员**；循环到**学生满意** |

## 4. 课程目录结构

```
quantization-tutorial/
├── OUTLINE.md                          # 已有，课程总纲
├── envs/{quant,deploy}/                # 已有，uv 双 env
├── scripts/{setup_env,download_model}.sh  # 已有
├── course/                             # 新增：课程内容
│   ├── m2-quant-pipeline/              # 模块 = lesson（M2 为试水模块）
│   │   ├── README.md                   # 本课导读（对应 OUTLINE M2）
│   │   └── steps/
│   │       ├── s1_env.ipynb            # 一个 Step = 一个 notebook（自包含）
│   │       ├── s2_fp8_quant.ipynb
│   │       ├── s3_awq_quant.ipynb
│   │       ├── s4_smoothquant.ipynb
│   │       └── s5_outputs_format.ipynb
│   ├── m1-activation-outliers/
│   ├── m3-tuning-eval/
│   └── m4-deploy-loop/
```

- 每个 `.ipynb` **自包含**：markdown 讲解 cell + 代码 cell（`logic` + `execution`）+ **ipytest 测试 cell**（L1）+ **tiny 模型验证 cell**（L2，CPU）+ **H200 执行 cell**（L3，GPU 守卫）。
- **无** `.py` lesson 脚本、jupytext 配对、`conftest.py` / `tests/` 目录；tiny 模型作为 setup cell 内联在每个 notebook（自包含、便于教学）。
- 测试在 **`envs/quant`** 跑：学员在 Jupyter 里直接跑 cell；headless 用 `uv run --directory envs/quant jupyter nbconvert --execute <notebook>` 或 `pytest --nbval`。
- notebook 提交前清输出（`nbconvert --clear-output`），保 git 干净；学员自己跑产生输出。

## 5. Step + UT 架构（notebook-only，三层测试都在 notebook 内）

每个 Step = 一个 `.ipynb`，含几类 cell：

- **markdown 讲解 cell**：原理 + 这步做什么。
- **代码 cell**：`logic`（纯函数：`build_fp8_recipe()`、`compute_smooth_scale(x, α)`、`validate_quant_config(cfg)`、`parse_kquant_suffix(name)`）+ `execution`（真跑：`run_fp8_quantize(model_id, out_dir)`）。
- **ipytest 测试 cell（L1）**：用 `%%ipytest` 或 `ipytest.run()` 写 `def test_x(): assert ...`，**隔离执行**（避免 cell 间状态串扰导致的假通过）—— 满足 C1"每步配 UT"，学员写完函数直接跑这个 cell 验证。
- **tiny 模型验证 cell（L2）**：内存里 `Qwen2ForCausalLM(Qwen2Config(num_hidden_layers=2, hidden_size=64, ...))`（随机初始化、不下 7B）+ 真跑 llmcompressor + `assert`，CPU 秒级。用**真库跑真架构（只是小）**，框架级 bug 在 CPU 就被抓 —— 正中 C2，且是很好的教学（"上 7B 前先用小模型验证"）。
- **H200 执行 cell（L3）**：真 Qwen2.5-0.5B 再 7B 跑该 Step，`if torch.cuda.is_available():` 守卫（无 GPU 自动跳过），断言跑通 + 输出合理。

三层都在 notebook 内（不再分文件），层层兜底 C2：

- **L2 是 C2 的关键**：真 llmcompressor 流水线跑在 tiny 模型上，库版本不兼容 / API 误用 / recipe 结构错等框架级问题在 CPU 阶段就暴露。
- **FP8 等必须 FP8 硬件的逻辑**：CPU 跑不了 → 归到 L3 cell，在 H200 上跑。
- **开发专家写每个 Step 时当场执行 L1/L2/L3 cell**：C2 的"开发期就验证跑通"由此保证，不攒到最后。

## 6. 开发 Workflow（5 阶段，固化为可复用 Workflow，每模块跑一次）

```
① 开发专家开发 → ② 架构师审核 → ③ QA 评审 loop → ④ student 试课 loop → ⑤ 定稿
```

- **① 一个开发专家开发整模块**：负责整模块全部 Step —— 写每个 Step 的 `.ipynb`（讲解 + 代码 + ipytest 测试 cell + tiny 验证 cell + H200 执行 cell），**当场执行 L1/L2/L3 cell 验证跑通**，输出 `{notebook + 执行结果 + 备注}`。
- **② 架构师审核**：审一致性、Step 衔接、notebook 执行是否过、整体连贯；可要求返工。
- **③ QA 评审 loop**：质量保证评审员严格审（准确性 / 完备 / UT 覆盖 / 跑通证据 / 教学法）→ 架构师修 → 再审 → **循环到 verdict=pass**（封顶 3 轮，否则升级用户）。
- **④ student 试课 loop**（见 §7）。
- **⑤ 定稿**：merge、commit。

①② 确定性跑；③④ 带封顶循环（Workflow 的 `while` 或我重调）。

## 7. Student 试课机制（git worktree）

student agent（学员）是"真实用户"代理，**两端评估**：

- **隔离 worktree**：`git worktree add ../qt-student-mN`（独立分支）= 学员的课程副本，不扰主干。
- **代码端**：`uv sync` 建环境 → 拉模型（或用缓存）→ 按序执行每个 Step 的 notebook（ipytest 测试 cell + tiny 验证 cell + H200 执行 cell）；记录跑通 / 报错 / 摩擦。
- **知识端**：通读全章，评**知识连贯性、讲解清晰度、逻辑跳跃、概念缺口、教学法** —— 凡觉得"课程设计不好"的地方都记下。
- **反馈同时给架构师和评审员**：架构师改代码 / 结构，评审员评估并定夺教学法 / 连贯性问题。
- 架构师 + 评审员修订 → student 重测 + 重评 → **循环到 student 觉得没问题**（代码跑通 AND 课程设计满意）；封顶防死循环，到顶升级用户。

> 因开发期已执行 notebook 验证（L1/L2/L3 cell）+ QA 已审，student 这轮主要抓可用性 / 清晰度（而非框架崩溃），收敛快；GPU 开销靠封顶轮数（如 3 轮）控制。

## 8. 开发顺序

**实现顺序**：先搭系统骨架（`course/` 目录结构、`envs/quant` 加 `ipytest`/`nbval`/`nbconvert`、可复用开发 Workflow 脚本），再用 **M2 试水**跑通整条 workflow。

**M2 试水**（量化流水线 —— 最重、框架风险最高：env / 脚本 / 3 种量化 / compressed-tensors 全在 M2）：M2 跑通整条开发链 = 把"开发系统 + 测试骨架 + workflow"在最难模块上验证一遍；框架"跑不通"在 M2 就暴露、修好 harness，后面顺。

然后 **M1（原理，代码最轻）→ M3（调优，吃 M2 的 SmoothQuant 产物）→ M4（部署，吃 M2/M3 产物）**。

## 9. 测试 / 执行基础设施（notebook-only）

- 测试 = notebook 内 ipytest cell；**headless 执行**：`uv run --directory envs/quant jupyter nbconvert --execute <notebook>`（整本跑通无错即过）或 `pytest --nbval`（cell 输出回归）。
- **H200 cell GPU 守卫**：`if torch.cuda.is_available():` 包裹，无 H200 时自动跳过 → CPU 环境也能跑通整本（只跳过重 cell）。
- **tiny 模型**：作为 setup cell 内联在每个 notebook（自包含，不依赖外部 fixture）。
- **快验证（无 H200）**：执行 notebook，L1（ipytest）+ L2（tiny）cell 全过 = 逻辑 + 库集成 OK，秒级，抓绝大多数 bug。
- **H200 验证**：执行 notebook 的 L3 cell（专家开发期 + student 试课）。
- **清输出**：提交前 `nbconvert --clear-output`。
- **依赖**：`envs/quant` 加 `ipytest`、`nbval`、`jupyter` / `nbconvert`。

## 10. 角色名册（开发 workflow）

| 角色 | 职责 | 阶段 |
|------|------|------|
| **开发专家** | 单 agent 开发整模块（吸收量化技术 + 工程双域；深问可咨询量化技术专家 / 代码导师） | ① |
| **课程架构师** | 审核 / 集成；按评审与 student 反馈修订 | ②③④ |
| **质量保证评审员** | 严格评审（含教学法 / 连贯性） | ③④ |
| **Student / 学员** | worktree 试课，代码 + 课程设计双评估 | ④ |

> 大纲阶段的"量化技术专家""代码导师"在本 workflow 中作为**开发专家可按需咨询的领域顾问**，不作为并行开发主体。

## 11. 单模块"完成"定义（Acceptance）

一个模块判定完成，需同时满足：
1. 全部 Step 有 `.ipynb`（含 ipytest 测试 cell + tiny 验证 cell + H200 执行 cell）。
2. L1（ipytest）+ L2（tiny）cell 全过（CPU）。
3. L3 H200 执行 cell 全过（真 0.5B + 7B）。
4. QA 评审员 verdict = pass。
5. Student 试课 = 满意（代码跑通 AND 课程设计满意），无未决反馈。
6. 模块 merge + commit。

## 12. 范围外 / 开放问题

- 本 spec 只定义**开发系统**；各模块的具体 Step 划分、cell 细节在实现计划（writing-plans）与开发期逐模块落地。
- H200 执行 cell 的真实跑时成本（每模块每轮 review 的 GPU 时间）在 M2 试水时实测，据此定 student loop 封顶轮数（暂定 3）。
- `lm_eval[vllm]` 与 vLLM wheel 的 transformers 共存（见 OUTLINE 附录 B）若冲突，评测走 `local-completions` HTTP 兜底。
- 测试工具链纳入 `envs/quant`：`ipytest` + `nbval` + `jupyter` / `nbconvert`（已定：纳入）。

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
| 交付形态 | **混合**：`.py` + `.ipynb`，经 **jupytext 配对**（`py:percent`）—— 单一源、双产物 |
| student 试课 | **agent**，在 **git worktree** 里**真跑 H200**，评估**代码 + 课程设计** |
| 测试架构 | **逻辑层/执行层分离 + 三层测试（L1/L2/L3）**（方案 A） |
| 开发主体 | **单个开发专家**开发整模块（不并行分 Step） |
| student 反馈 | 同时给**架构师和评审员**；循环到**学生满意** |

## 4. 课程目录结构

```
quantization-tutorial/
├── OUTLINE.md                          # 已有，课程总纲
├── envs/{quant,deploy}/                # 已有，uv 双 env
├── scripts/{setup_env,download_model}.sh  # 已有
├── pyproject.toml                      # 新增（顶层）：[tool.pytest.ini_options] 等
├── course/                             # 新增：课程内容
│   ├── conftest.py                     # 共享 fixtures：tiny 模型 / 合成校准数据 / tmp 路径
│   ├── _tiny/                          # tiny Qwen2 配置（2 层、hidden≈64，内存随机初始化，不下 7B）
│   ├── m2-quant-pipeline/              # 模块 = lesson（M2 为试水模块）
│   │   ├── README.md                   # 本课导读（对应 OUTLINE M2）
│   │   ├── steps/
│   │   │   ├── s1_env.py + s1_env.ipynb        # jupytext 配对
│   │   │   ├── s2_fp8_quant.py + .ipynb
│   │   │   ├── s3_awq_quant.py + .ipynb
│   │   │   ├── s4_smoothquant.py + .ipynb
│   │   │   └── s5_outputs_format.py + .ipynb
│   │   └── tests/
│   │       ├── test_s2_logic.py               # L1 + L2 UT（CPU，快）
│   │       └── test_s2_h200.py                # L3 smoke（@pytest.mark.h200）
│   ├── m1-activation-outliers/
│   ├── m3-tuning-eval/
│   └── m4-deploy-loop/
└── tests/                              # 顶层共享 / 跨模块测试
```

- 测试在 **`envs/quant`** 环境跑：`source envs/quant/.venv/bin/activate && pytest course/`。
- `.py` 与 `.ipynb` 由 jupytext 绑定：改 `.py`（`py:percent`，`# %%` cell 标记）→ `jupytext --sync` 更新 `.ipynb`。`.py` 是合法 Python，直接进 pytest；`.ipynb` 是教学产物。

## 5. Step + UT 架构（三层测试）

每个 Step 的 `.py` 分两部分：
- **`logic`**：纯函数，不碰 GPU / 大模型。如 `build_fp8_recipe()`、`compute_smooth_scale(x, alpha)`、`validate_quant_config(cfg)`、`parse_kquant_suffix(name)`。
- **`execution`**：真跑。如 `run_fp8_quantize(model_id, out_dir)`。

配对的 `.ipynb` = narrative 包裹 `execution` cell。

**三层测试，层层兜底 C2（框架级跑不通）：**

| 层 | 文件 | 环境 | 跑什么 | 目的 | 进 CI |
|----|------|------|--------|------|-------|
| **L1 逻辑 UT** | `test_*_logic.py` | CPU，秒级 | 纯函数 + tiny 张量 | 验证算法/解析/校验逻辑 | ✅ 默认 |
| **L2 tiny 真模型 UT** | `test_*_logic.py` | CPU（可行处） | **内存里 tiny Qwen2ForCausalLM（2 层）+ 真跑 llmcompressor oneshot** + 合成校准数据 | 用**真库跑真架构（只是小）**，库/框架级 bug 在 CPU 就被抓 —— 正中 C2，还不用 H200 | ✅ 默认 |
| **L3 H200 smoke** | `test_*_h200.py` | H200，`@pytest.mark.h200` | 真 Qwen2.5-0.5B 再 7B 跑该 Step，断言跑通 + 输出合理 | 真实规模端到端验证 | ❌ 默认跳过（`-m "not h200"`） |

- **L2 是 C2 的关键**：用 in-memory tiny 模型（`Qwen2Config(num_hidden_layers=2, hidden_size=64, ...)` 随机初始化，不下 7B）跑**真 llmcompressor 流水线**，所以库版本不兼容、API 误用、recipe 结构错等框架级问题在 CPU UT 阶段就暴露。
- **FP8 等必须 FP8 硬件的逻辑**：CPU 跑不了 → 归到 L3（`@h200`），在 H200 上 smoke。
- **专家开发每个 Step 时当场跑 L2/L3**：C2 的"开发期就验证跑通"由此保证，不攒到最后。

## 6. 开发 Workflow（5 阶段，固化为可复用 Workflow，每模块跑一次）

```
① 开发专家开发 → ② 架构师审核 → ③ QA 评审 loop → ④ student 试课 loop → ⑤ 定稿
```

- **① 一个开发专家开发整模块**：负责整模块全部 Step —— 写每个 Step 的 `.py`(logic+execution) + jupytext `.ipynb` + 三层 UT，**当场跑 L2/L3 验证跑通**，输出 `{文件 + smoke 结果 + 备注}`。
- **② 架构师审核**：审一致性、Step 衔接、smoke 是否过、整体连贯；可要求返工。
- **③ QA 评审 loop**：质量保证评审员严格审（准确性 / 完备 / UT 覆盖 / 跑通证据 / 教学法）→ 架构师修 → 再审 → **循环到 verdict=pass**（封顶 3 轮，否则升级用户）。
- **④ student 试课 loop**（见 §7）。
- **⑤ 定稿**：merge、commit。

①② 确定性跑；③④ 带封顶循环（Workflow 的 `while` 或我重调）。

## 7. Student 试课机制（git worktree）

student agent（学员）是"真实用户"代理，**两端评估**：

- **隔离 worktree**：`git worktree add ../qt-student-mN`（独立分支）= 学员的课程副本，不扰主干。
- **代码端**：`uv sync` 建环境 → 拉模型（或用缓存）→ 按序跑每个 Step → 跑 UT + smoke → 执行 notebook；记录跑通 / 报错 / 摩擦。
- **知识端**：通读全章，评**知识连贯性、讲解清晰度、逻辑跳跃、概念缺口、教学法** —— 凡觉得"课程设计不好"的地方都记下。
- **反馈同时给架构师和评审员**：架构师改代码 / 结构，评审员评估并定夺教学法 / 连贯性问题。
- 架构师 + 评审员修订 → student 重测 + 重评 → **循环到 student 觉得没问题**（代码跑通 AND 课程设计满意）；封顶防死循环，到顶升级用户。

> 因开发期已跑 smoke + QA 已审，student 这轮主要抓可用性 / 清晰度（而非框架崩溃），收敛快；GPU 开销靠封顶轮数（如 3 轮）控制。

## 8. 开发顺序

**实现顺序**：先搭系统骨架（`course/` 目录、`conftest.py`、`_tiny/`、顶层 `pyproject.toml` 测试配置、可复用开发 Workflow 脚本），再用 **M2 试水**跑通整条 workflow。

**M2 试水**（量化流水线 —— 最重、框架风险最高：env / 脚本 / 3 种量化 / compressed-tensors 全在 M2）：M2 跑通整条开发链 = 把"开发系统 + 测试骨架 + workflow"在最难模块上验证一遍；框架"跑不通"在 M2 就暴露、修好 harness，后面顺。

然后 **M1（原理，代码最轻）→ M3（调优，吃 M2 的 SmoothQuant 产物）→ M4（部署，吃 M2/M3 产物）**。

## 9. 测试 / CI 基础设施

- 顶层 `pyproject.toml` `[tool.pytest.ini_options]`：注册 marks `h200` / `slow` / `gpu`；默认 `-m "not h200 and not slow"`。
- `course/conftest.py`：fixtures `tiny_model`（in-memory tiny Qwen2）/ `tiny_calib_data`（合成）/ `tmp_out_dir`。
- **快 CI（无 H200）**：跑全部 L1 + L2 → 验证逻辑 + 库集成，秒级，抓绝大多数 bug。
- **H200**：`pytest -m h200` 跑 L3（专家开发期 + student 试课）。
- **notebook 执行测试**：`jupyter nbconvert --execute` / `pytest --nbval` 执行配对 notebook（重的标 `h200`）。

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
1. 全部 Step 有 `.py` + `.ipynb`（jupytext 配对）。
2. L1 + L2 UT 全过（CPU）。
3. L3 H200 smoke 全过（真 0.5B + 7B）。
4. QA 评审员 verdict = pass。
5. Student 试课 = 满意（代码跑通 AND 课程设计满意），无未决反馈。
6. 模块 merge + commit。

## 12. 范围外 / 开放问题

- 本 spec 只定义**开发系统**；各模块的具体 Step 划分、UT 细节在实现计划（writing-plans）与开发期逐模块落地。
- H200 smoke 的真实跑时成本（每模块每轮 review 的 GPU 时间）在 M2 试水时实测，据此定 student loop 封顶轮数（暂定 3）。
- `lm_eval[vllm]` 与 vLLM wheel 的 transformers 共存（见 OUTLINE 附录 B）若冲突，UT/评测走 `local-completions` HTTP 兜底。
- jupytext 是否纳入 `envs/quant` 依赖：是（`jupytext` + `nbval` 加入 quant pyproject）。

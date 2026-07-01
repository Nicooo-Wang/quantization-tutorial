# M3+M4 SmoothQuant 教学法重构（设计 spec）

> 对 [OUTLINE.md](../../../OUTLINE.md) 模块 3（s5/s6/s7）+ 模块 4（s1–s5）的**教学法重构**。不是从零开发，是在已合入 main 的现有 notebook 上改。
>
> 三条主线：① 端到端**只用 SmoothQuant 一种量化算法**（学量化经验，非对比算法）；② **验证型填空归测试**（学员只写业务逻辑）；③ **每填空独立可测**（填完一个立即能测，不要"填完所有才能一起测"）。
>
> 顺带把 M3 末段（s5/s6/s7）重构成一条**对标工业的 SmoothQuant 调优闭环**（产物接力），取代原来"三方法对比"的定位。

---

## 1. 目标与范围

**目标**：重构 M3 s5/s6/s7 + M4 s1–s5，消除三类教学法问题，并把 M3 末段串成 SmoothQuant 工业调优闭环。

**三类问题**（排查已确认，见 §2）：
- **C 算法统一**：端到端只用 SmoothQuant，去掉 FP8/AWQ/GPTQ-standalone 的对比与多分支。
- **A 填空职责**：把"验证型"填空（薄校验/查表/断言角色）从填空移进 ipytest；学员只写"业务型"填空（构造/解析/决策/诊断的核心交付）。
- **B 测试拆分**：每填空一个 ipytest cell，学员填完任一填空即可单独跑该 cell 验证；废除"单 cell 测所有 + cell 末 `assert _ec==0` 绑定"。

**范围**：
- ✅ 改：M3 `s5_pareto_tuning` / `s6_group_size`（quant env）+ `s7_evaluation`（vllm env）+ M4 `s1`–`s5`。
- ❌ 不动：M3 s1–s4（用户明确）、M1、M2、各模块脚手架（pyproject/uv.lock/scripts/README/CONV）、dev-module.js workflow 脚本本身（复用现有 5 阶段）。

### 工业端到端量化评估流程（教程覆盖映射）

完整工业流程（确定 SmoothQuant 后到部署上线）与教程覆盖：

| 阶段 | 环节 | 教程位置 | 覆盖 |
|---|---|---|---|
| A 基线 | FP16 精度(PPL+下游)+性能基线 | s7 / M4-s3 对比基线 | ✓ |
| B 量化配置 | **smoothing_strength(α) 核心旋钮** | **s6（本次补，原 group_size 偏离）** | △→✓ |
| | targets / ignore（不量化层） | s5 + s1–s4 | ✓ |
| | group_size（量化粒度） | s6（次） | ✓ |
| | 校准数据（选择 + 数量 limit） | s5/s6 强化 | △→✓ |
| C 量化执行 | oneshot → compressed-tensors | s5/s6 L3 + M2 | ✓ |
| D 精度评估 | PPL + 下游(gsm8k) | s7 | ✓ |
| | 敏感度分析 + layer fallback | s5 | ✓ |
| | 调参扫描（α / group_size） | s6 | ✓ |
| | 精度门槛 + 达标判定 | s7 强化 | △→✓ |
| E 性能评估 | 吞吐/TTFT/TPOT/显存拆分/加速比 | M4-s3 + s7 | ✓ |
| F 部署 | vLLM 声明式 + 多卡TP + 压测 + 报错 | M4-s1/s2/s3/s4 | ✓ |
| G 交付 | 可复现四件套 | M4-s5 | ✓ |

**核心缺口（本次补）**：**smoothing_strength(α) 调参**——SmoothQuant 的灵魂参数，控制激活离群点向权重迁移的程度，直接决定精度（α 太小激活压不住、α 太大权重误差大），工业上必扫 α 找最优。原 s6 用 group_size（W8A8 下次要，主要影响 scale 开销）偏离了 SmoothQuant 的核心调参。本次 s6 改以 smoothing_strength 为主、group_size 为次。

**次要强化**：校准数据选择/数量影响（B）、精度门槛与达标判定（D）——在 s5/s6/s7 叙事里点明。

---

## 2. 排查结论（改动依据）

### B 类（测试没拆开）— 8/8 全中
所有目标 notebook 都是「1 个 ipytest cell + cell 末 `assert _ec == 0`」，任一填空没填→整 cell 挂。即便填空逻辑解耦，cell 级守卫仍把它们绑成"全有或全无"。M4 s4 还有数据流耦合：`recommend_fix` 内部调 `diagnose_error`。

### A 类（验证型填空该归测试）— 明确 2 处 + 边界 2 处
- 明确：M3 s6 `validate_group_size`（薄校验；其整除逻辑在 `compare_group_sizes` 里还重复一遍）。
- 明确：M4 s1 `pick_vllm_flag_and_kernel`（收敛 SmoothQuant 后只剩单行 scheme→kernel 映射，纯查表）。
- 边界（偏查表/校验，但带二段逻辑，保留为填空）：M4 s4 `recommend_fix`、M4 s5 `build_delivery_bundle`。
- 其余填空均为业务型，保留。

### C 类（多算法对比→统一 SmoothQuant）
- 基准不动：M3 s5（已是 SmoothQuant+GPTQ W8A8）。
- 要改：M3 s6（L3 用纯 QuantizationModifier W4A16，无 SmoothQuant，且跨 step scheme 与 s5 不一致）；M3 s7（finale 对比 FP8/AWQ/SmoothQuant 三方法）；M4 s1（整节三方法识别/构造/速查）；M4 s3（四向压测对比）；M4 s4（诊断样本 AWQ 偏重）；M4 s5（决策树是多算法选型）；M4 s2（仅 L2/L3 演示层提三方法，轻）。

---

## 3. M3 SmoothQuant 工业调优闭环（核心主线）

**前提**：你已经选定 SmoothQuant（M2 产出的全量化 SmoothQuant 模型即调优起点）。现在工业部署前要把它**调到既准又快**。s5/s6/s7 是闭环的三个环节，**产物接力**，不是三个孤立 lab。

### 产物接力（路径约定，写入 `course/m3-tuning-eval/out/`）

| 产物 | 路径 | 含义 | 产出 step |
|---|---|---|---|
| FP16 上界 | `models/Qwen2.5-7B-Instruct`（download） | 精度/性能参照上界 | — |
| s5 全量化基线 | `out/s5_baseline/` | SmoothQuant 全量化（`ignore=()`），调优起点 | s5 L3 |
| s5 调优后 | `out/s5_tuned/` | s5 拐点 k\*（`ignore=top-k* 敏感层`）后的模型 | s5 L3 |
| s6 最终 | `out/s6_final/` | s6 在 s5_tuned 的 ignore 基础上选定 smoothing_strength(α) + group_size 后的模型 | s6 L3 |

s5 产出 baseline + tuned；s6 读 tuned 的 ignore 配方、产出 final；s7 读 baseline/tuned/final + FP16 做四向对比验证。缺产物时友好报错（不崩）。

### 三环节定位

- **s5 — 敏感度比对 → 精度 Pareto**（保证**精度过线**）：SmoothQuant 全量化掉精度；逐层回退敏感层，回退带来的 PPL 恢复 = 该层量化敏感度（"比对各层敏感度"的工程化）；找"最少回退→最大精度"拐点 k\*。
- **s6 — SmoothQuant 调参：smoothing_strength（主）+ group_size（次）**（找精度最优 + 压性能/显存）：smoothing_strength(α) 是 SmoothQuant 灵魂旋钮——控制激活→权重迁移程度，α 太小激活离群点压不住、α 太大权重量化误差大，工业扫 α 找精度最优；group_size 是量化粒度旋钮（W8A8 下次要，主要影响 scale 开销）。工业实践 α 通常在量化初期调好，本 step 在 s5 ignore 基础上深入这个 SmoothQuant 特有旋钮。
- **s7 — 验证调优收益**：对比 baseline（掉点）→ tuned（s5 ignore）→ final（s6 group_size）→ FP16，验证调优闭环的价值。

---

## 4. 逐 notebook 改动

> 统一约定：每个 notebook 顶部「学完应能讲清」清单随主题更新；改完 `nbconvert --clear-output` + 删 cell `metadata.execution`。

### M3 s5 — 敏感度比对 → 精度 Pareto（小改）

- **算法**：SmoothQuantModifier(smoothing_strength=0.8) + GPTQModifier(targets="Linear", scheme="W8A8", ignore=...)（现状已对，**不动**）。
- **填空**（都业务型，保留）：`pareto_step` / `find_pareto_knee`。
- **改**：① **拆测试**（每填空一 ipytest cell）；② 强化工业叙事（"已选定 SmoothQuant，逐层回退看 PPL 恢复 = 比对该层敏感度"）；③ L3 循环找拐点 k\* 后，把 **s5_baseline（ignore=()）+ s5_tuned（ignore=k\*）** 两个产物存到 `out/`，供 s6/s7 接力。

### M3 s6 — SmoothQuant 调参：smoothing_strength（主）+ group_size（次）

- **算法**：L3 纯 QuantizationModifier(W4A16) → **SmoothQuantModifier(smoothing_strength=α) + GPTQModifier(targets="Linear", scheme="W8A8")**，扫 α（如 0.0/0.5/0.8/0.85/1.0）找精度最优；group_size 作为次要旋钮（W8A8 下影响 scale 开销）。
- **填空**：
  - `compare_smoothing_strengths(alphas, ...)`（业务，**新增主填空**：扫 α，返回各 α 的平滑效果代理指标——激活/权重 scale 比值、是否充分平滑；L1/L2 用解析代理验，L3 真扫 α 测 PPL）。
  - `pick_group_size(hidden_size, candidates)`（业务，**保留**：从 group_size 对比表选拐点——scale 开销 vs 精度，W8A8 下主要看开销 + 合法性；原 `compare_group_sizes` 的组装逻辑并入此函数辅助 / ipytest，避免重复）。
  - `validate_group_size` → **归测试**（移出填空，整除/边界校验进 ipytest `pytest.raises`）。
  - 结果：s6 仍 2 个业务填空（α 主 + group_size 次）。
- **改**：① 算法统一 W8A8 + α 扫描；② 主填空 `compare_group_sizes` → `compare_smoothing_strengths`；③ `validate` 归测试；④ **拆测试**；⑤ L3 **加测 PPL**（扫 α + group_size 的精度对比，不只磁盘）；⑥ L3 读 s5_tuned 的 ignore 配方、用最优 α 产出 **s6_final** 到 `out/`；⑦ 叙事：α 是 SmoothQuant 灵魂参数（讲清太小/太大后果 + 工业扫描实践 + 校准数据影响），group_size 为次要旋钮。

### M3 s7 — 验证调优收益（重构对比维度）

- **对比维度**：FP8/AWQ/SmoothQuant 三方法 → **s5_baseline vs s5_tuned vs s6_final vs FP16**（调优前→后 + 上界）。
- **填空**（都业务型，保留）：`run_downstream_eval` / `read_ppl_from_artifacts`（`read_ppl` 聚合对象从三方法改为 baseline/tuned/final）。
- **改**：① `method_dirs` 三键 → 四对比产物；② L2 合成数据、原理段、L3 叙事全改为"调优收益验证"（不再"三方法对比"）；③ **拆测试**。

### M4 s1 — 收敛 SmoothQuant（收敛 + 拆 + 归测试）

- **填空**：
  - `detect_quant_scheme`（业务，保留）：收敛到识别 **SmoothQuant(W8A8)** + FP16→None（删 FP8/AWQ 分支）。
  - `build_quantization_config`（业务，保留）：收敛到只构造 **SmoothQuant(W8A8)** 的 config。
  - `pick_vllm_flag_and_kernel` → **归测试**（收敛后单行映射，纯查表，价值低）。
  - 结果：s1 留 2 个业务填空。
- **改**：① 收敛到 SmoothQuant；② `pick` 归测试；③ **拆测试**；④ 摸一摸/L2/L3 的三方法样本 → SmoothQuant 单方法。声明式边界讲解（两层适配/三条件/vLLM 6 步）**保留**——它们不依赖多算法。

### M4 s2 — 拆测试 + 演示层收敛

- **填空**（都业务型，保留）：`build_serve_cmd` / `recommend_tp`。
- **改**：① **拆测试**；② L2/L3 演示层三方法路径 → SmoothQuant。

### M4 s3 — 四向 → 两向对比

- **填空**（都业务型，保留）：`build_bench_cmd` / `parse_metrics`。
- **改**：① L2/L3 四向（FP16/FP8/AWQ/SmoothQuant）→ **FP16 vs SmoothQuant 两向**；② **拆测试**。

### M4 s4 — 去 AWQ 偏重 + 拆 + 处理耦合

- **填空**：
  - `diagnose_error`（业务，保留）：诊断样本去 AWQ 偏重，补 SmoothQuant/通用类报错（kernel path / NCCL / OOM / 整除等），保持诊断→§5 三条件的映射教学。
  - `recommend_fix`（边界偏查表，但带"先诊断再查表"二段，**保留**）。
- **改**：① 样本调整；② **拆测试**——处理 `recommend_fix`→`diagnose_error` 耦合：`recommend_fix` 的 L1 测试只验"结构化输入→修复动作"（查表），**不依赖 `diagnose_error` 已实现**；str→diagnose→查表 的整合路径放到该 cell 的 docstring 说明 / L2，不作为 L1 必过项；③ 摸一摸 cell 的特征词清单不预写答案（保留挑战）。

### M4 s5 — 删 decision_tree + finale 重构

- **删** `build_decision_tree`（多算法选型，与"只用 SmoothQuant"直接冲突——用户已定删）。
- **保留** `build_delivery_bundle`（边界偏校验打包，但体现"可复现交付"工程约定，保留）。
- **新增** `build_deploy_summary(bench_result)`（业务型：解读 s3 压测结果，生成 SmoothQuant vs FP16 部署对比 + 是否达标的总结）——补足 finale 的"端到端闭环"交付。
  - 结果：s5 留 2 个业务填空。
- **改**：① 删 decision_tree；② 新增 build_deploy_summary；③ **拆测试**；④ finale 叙事聚焦"SmoothQuant 端到端部署交付"（s1 加载→s2 部署→s3 压测→s5 总结），4.7 QAT/NVFP4 前瞻 markdown 保留。

---

## 5. 填空职责原则（A 类判定）

- **业务型**（保留为填空）：本 step 的核心交付——构造命令/config/recipe、解析输入、做决策/诊断。
- **验证型**（归 ipytest）：薄校验、纯查表、断言角色——其逻辑直接写进测试断言效果一样。
- **归测试清单**（本 spec 确定）：M3 s6 `validate_group_size`、M4 s1 `pick_vllm_flag_and_kernel`。
- **边界保留**（偏查表/校验但带实质二段逻辑，有教学价值，保留）：M4 s4 `recommend_fix`、M4 s5 `build_delivery_bundle`。

---

## 6. 测试拆分原则（B 类）

- **每填空一个 ipytest cell**，紧跟该填空定义之后。
- 废除 cell 末 `assert _ec == ipytest.run(...) == 0` 的"cell 级守卫绑定多填空"模式。改为每 cell 独立 `ipytest.run("-qq")`（或保留轻量断言但只守卫本 cell 的填空）。
- 学员填完任一填空，可立即单独跑该填空的 cell 验证（其他填空的 cell 因 `NotImplementedError` 报 error，但不影响本 cell）。
- **耦合处理**（M4 s4 `recommend_fix`→`diagnose_error`）：见 §4 M4 s4——`recommend_fix` 测试不依赖 `diagnose_error` 实现。
- **每 notebook 填空数 ≥ 2**（验收约定）：归测试导致 <2 时，新增业务型填空补足（s6 补 `pick_group_size`、s5 补 `build_deploy_summary`）。

---

## 7. 算法统一细则（C 类）

- **M3 quant env（重量化）**：所有 L3 真量化统一 `recipe = [SmoothQuantModifier(smoothing_strength=α), GPTQModifier(targets="Linear", scheme="W8A8", ignore=...)]`。s5 用默认 α=0.8；s6 扫 α（0.0–1.0）找最优 + 探索 group_size。
- **M4 vllm env（不量化）**：所有"识别/构造/对比/选择/诊断"收敛到 SmoothQuant。讲解/样本里的 FP8/AWQ 多方法对比，降级为一句"其他量化方案见 M1/M2"或删除。
- **跨 step scheme 一致**：M3 s5/s6/s7 的 W8A8 统一（修掉 s6 原 W4A8 与 s5 W8A8 的不一致）。

---

## 8. 验收标准

- **dev-module.js 5 阶段全绿**（复用现有 workflow）：reviewer pass[含执行验证 L1+L2] + learner satisfied + gate smokeOk。
- **三类问题 controller 独立复核**：
  - C：M3 grep L3 recipe 无 `QuantizationModifier(.*W4A16` / 无 FP8/AWQ 独立对比；M4 grep 无三方法分支残留。
  - A：归测试清单的函数已从填空移除（无 `raise NotImplementedError`），逻辑在 ipytest 断言里。
  - B：每 notebook 的 ipytest cell 数 == 业务填空数（每填空独立可测）。
- **M3 闭环产物路径一致**：s5_baseline/s5_tuned/s6_final 路径在 s5/s6/s7 间对齐；缺产物友好降级。
- **清输出**：所有 notebook 无 `outputs`/`execution_count`。

---

## 9. 风险

| 风险 | 缓解 |
|---|---|
| M3 L3 重量化分钟级（s5 循环 + s6 group_size + s7 评测） | 全走 `SKIP_L3=1`；workflow 只验 L1+L2 代码逻辑，运行时留学员 GPU |
| s6 W8A8 group_size 张力小（精度影响幅度不如 W4A16） | 叙事诚实（主要影响 scale 开销）；`pick_group_size` 决策以开销+合法性为主 |
| 产物接力跨 step 耦合（s7 依赖 s5/s6 产物） | 路径约定（§3）+ 缺产物友好报错降级 + 0.5B 兜底（L2 流程独立可跑） |
| 改动大（8 notebook） | 每 notebook 独立 commit + task review；M3 闭环（s5/s6/s7）一组、M4（s1–s5）一组 |
| recommend_fix 拆 cell 后耦合 | L1 测试只测查表（不依赖 diagnose），整合路径放 L2/docstring |

---

## 10. 不在范围

M3 s1–s4（用户明确不动）、M1、M2、各模块脚手架与 dev-module.js workflow 脚本。本 spec 只改 notebook 内容（填空/测试/讲解/L3 recipe/产物路径）。

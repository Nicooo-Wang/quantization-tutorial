# M3+M4 SmoothQuant 教学法重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: 用 dev-module.js workflow（完整 5 阶段）产出/验收 notebook。Steps 用 checkbox（`- [ ]`）跟踪。

**Goal:** 重构 M3 s5/s6/s7 + M4 s1–s5，消除三类教学法问题（算法统一 SmoothQuant / 验证型填空归测试 / 每填空独立可测），并把 M3 末段重构成对标工业的 SmoothQuant 调优闭环（含灵魂参数 smoothing_strength 调参 + 产物接力）。

**Architecture:** 在已合入 main 的现有 notebook 上改（非从零）。**完整跑 dev-module.js workflow**（用户要求，提高质量）：M4 组（单 env，改 s1–s5）+ M3 组（双 env，**只改 s5/s6/s7**，spec hard constraint 禁动 s1–s4）。workflow 内 dev-agent 读 spec 按 §4 改 → reviewer（审内容 + 注入参考实现 nbconvert 跑 L1/L2 执行验证）→ learner（零基础理解探测）→ gate（完整性）→ 定稿。controller 在每组 workflow 后独立复核三类问题 + 闭环产物 + s1–s4 未动。最后 opus 全分支终审。

**Tech Stack:** llmcompressor 0.12.0 + transformers 5.10.1 + torch 2.11.0+cu128（M3 quant env）；vllm 0.23.0 + lm_eval 0.4.12（M3 vllm env）；vllm 0.23.0 + ipytest/nbconvert（M4 env）；Python 3.11；CUDA 12.8/驱动 570。

## Global Constraints

（来自 spec §1–§7；每个 task 隐含包含）

- **分支**：`m3m4-smoothquant-refactor`（已开，base main `b1bee29`）。不在 main 直接提交实现。
- **编辑范围**：只改 `course/m3-tuning-eval/steps/quant/s5_pareto_tuning.ipynb` + `s6_group_size.ipynb` + `course/m3-tuning-eval/steps/vllm/s7_evaluation.ipynb` + `course/m4-deploy-loop/steps/s{1..5}_*.ipynb`。**严禁动 M3 s1–s4**（s1_sensitive_layers / s2_what_not_to_quantize / s3_ignore_syntax / s4_mixed_precision）+ 脚手架 + dev-module.js。
- **C 算法统一**：M3 quant env 所有 L3 真量化统一 `recipe=[SmoothQuantModifier(smoothing_strength=α), GPTQModifier(targets="Linear", scheme="W8A8", ignore=...)]`；s5 α=0.8，s6 扫 α（0.0–1.0）+ group_size。M4 收敛到只 SmoothQuant；s3 四向→FP16 vs SmoothQuant 两向。
- **A 填空职责**：验证型填空归 ipytest——确定清单：M3 s6 `validate_group_size`、M4 s1 `pick_vllm_flag_and_kernel`。业务型保留。
- **B 测试拆分**：每填空一个 ipytest cell（紧跟定义后）；废除 cell 末 `assert _ec==ipytest.run(...)==0` 的 cell 级守卫绑定多填空；学员填完任一填空可单独跑该 cell。每 notebook ≥2 业务填空（归测试致 <2 则补：s6 `compare_smoothing_strengths`、M4 s5 `build_deploy_summary`）。
- **M3 闭环产物接力**（写入 `course/m3-tuning-eval/out/`）：s5 产 `s5_baseline`（α=0.8, ignore=()）+ `s5_tuned`（ignore=拐点 k\*）；s6 读 s5_tuned 的 ignore、用最优 α+group_size 产 `s6_final`；s7 读 `s5_baseline`/`s5_tuned`/`s6_final` + FP16 四向对比。缺产物友好降级。
- **填空签名一致性**：s6 `compare_smoothing_strengths(alphas, ...)` + `pick_group_size(hidden_size, candidates)`；M4 s5 `build_delivery_bundle(...)` + `build_deploy_summary(bench_result)`；s7 `read_ppl_from_artifacts` 聚合对象改 baseline/tuned/final。SKIP_L3 守卫 `torch.cuda.is_available() and not os.environ.get('SKIP_L3')` 保留。
- **提交规则**：notebook commit 前 `nbconvert --clear-output --inplace` + 删 cell `metadata.execution`；commit message 英文。
- **完整跑 workflow**：dev-module.js skipDev=false；reviewer 执行验证 foreground nbconvert + SKIP_L3=1（禁 background/Monitor 轮询，M3 坑已修）。

---

## File Structure

```
改的 notebook（8 个）：
  course/m4-deploy-loop/steps/
    s1_load_quantized.ipynb      # 收敛 SmoothQuant + pick 归测试 + 拆测试
    s2_multigpu_deploy.ipynb     # 拆测试 + 演示层 SmoothQuant
    s3_benchmark.ipynb           # 四向→FP16 vs SmoothQuant 两向 + 拆测试
    s4_error_cheatsheet.ipynb    # 样本去 AWQ 偏重 + recommend_fix 不依赖 diagnose + 拆测试
    s5_e2e_loop.ipynb            # 删 decision_tree + 新增 build_deploy_summary + 拆测试
  course/m3-tuning-eval/steps/quant/
    s5_pareto_tuning.ipynb       # 拆测试 + 存 s5_baseline/s5_tuned + 工业叙事
    s6_group_size.ipynb          # smoothing_strength 主填空 + pick_group_size 次 + validate 归测试 + W8A8+α扫描+PPL + 产 s6_final
  course/m3-tuning-eval/steps/vllm/
    s7_evaluation.ipynb          # 重构四向对比(baseline/tuned/final/FP16) + 拆测试

不改：M3 s1–s4、M1、M2、所有脚手架、workflows/dev-module.js
```

---

### Task 1: M4 组 — 完整跑 dev-module.js workflow（单 env，改 s1–s5）

**Files:**
- Modify: `course/m4-deploy-loop/steps/s{1,2,3,4,5}_*.ipynb`

**Interfaces:**
- Consumes: spec `docs/superpowers/specs/2026-06-30-m3m4-smoothquant-pedagogy-refactor-design.md`（§1.5 工业流程 + §4 M4 各 step + §5/§6/§7 原则）、CONV、现有 s1–s5 notebook。
- Produces: 改后的 s1–s5（收敛 SmoothQuant + 拆测试 + 验证型归测试）+ dev-module acceptance 全绿。

- [ ] **Step 1: 跑 dev-module.js workflow（单 env，skipDev=false）**

controller 执行（Workflow tool，后台跑）：
```javascript
Workflow({
  scriptPath: "workflows/dev-module.js",
  args: {
    module: "m4-deploy-loop",
    skipDev: false,
    spec: "docs/superpowers/specs/2026-06-30-m3m4-smoothquant-pedagogy-refactor-design.md"
  }
})
```
dev-agent 按 spec 改 s1–s5（**核心改动点，dev 读 spec §4 M4 节**）：
- **s1**：`detect_quant_scheme`/`build_quantization_config` 收敛到只 SmoothQuant(W8A8)+FP16→None；`pick_vllm_flag_and_kernel` **归测试**；摸一摸/L2/L3 三方法样本→SmoothQuant；**拆测试**（每填空一 cell）。声明式边界讲解（两层/三条件/6步）保留。
- **s2**：`build_serve_cmd`/`recommend_tp` 保留；L2/L3 演示三方法→SmoothQuant；**拆测试**。
- **s3**：`build_bench_cmd`/`parse_metrics` 保留；L2/L3 四向→**FP16 vs SmoothQuant 两向**；**拆测试**。
- **s4**：`diagnose_error` 样本去 AWQ 偏重、补 SmoothQuant/通用类；`recommend_fix` 的 L1 测试**只测查表（不依赖 diagnose 实现）**，str→diagnose 整合放 L2/docstring；**拆测试**。
- **s5**：**删 `build_decision_tree`**；保留 `build_delivery_bundle`；**新增 `build_deploy_summary(bench_result)`**（解读 s3 压测 SmoothQuant vs FP16 + 是否达标）；**拆测试**；finale 叙事聚焦 SmoothQuant 端到端。

预期：acceptance = `{reviewerPassed:true, execVerified:true, learnerSatisfied:true, scaffoldClean:true}`。

- [ ] **Step 2: 看 acceptance，失败则诊断**

若非全绿：读 workflow log（reviewer findings / learner designIssues / gate violations）。常见：① reviewer 执行验证 L1/L2 不过（填空逻辑/测试问题）→ dev 修；② learner needed_code（讲解不够）→ dev 补讲解；③ gate violations（steps/ 外被改）→ 还原。针对性让 dev 修后重跑 workflow，或 controller 手动补修单点。

- [ ] **Step 3: 清输出 + commit**

```bash
cd course/m4-deploy-loop
for f in steps/s{1,2,3,4,5}_*.ipynb; do ./.venv/bin/jupyter nbconvert --clear-output --inplace "$f"; done
git -C /home/w00809645/repos/quantization-tutorial add course/m4-deploy-loop/steps/
git -C /home/w00809645/repos/quantization-tutorial status --short   # 确认只 s1-s5 改动
```
commit：
```bash
git commit -m "refactor(m4): smoothquant pedagogy — unify algo + split tests + move lookups to ipytest

s1 detect/build converge to SmoothQuant (+pick -> ipytest), s2演示层收敛,
s3 four-way -> FP16 vs SmoothQuant, s4 sample rebalance + recommend_fix decoupled,
s5 drop decision_tree + add build_deploy_summary. One ipytest cell per fill-in.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: M3 组 — 完整跑 dev-module.js workflow（双 env，只改 s5/s6/s7）

**Files:**
- Modify: `course/m3-tuning-eval/steps/quant/s5_pareto_tuning.ipynb`、`s6_group_size.ipynb`、`course/m3-tuning-eval/steps/vllm/s7_evaluation.ipynb`

**Interfaces:**
- Consumes: spec（§1.5 + §3 闭环产物接力 + §4 M3 各 step + §5/§6/§7）、CONV、现有 s5/s6/s7。
- Produces: 改后的 s5/s6/s7（SmoothQuant 工业调优闭环 + 产物接力）+ dev-module acceptance 全绿。

- [ ] **Step 1: 跑 dev-module.js workflow（双 env，skipDev=false）**

controller 执行（Workflow tool，后台跑）：
```javascript
Workflow({
  scriptPath: "workflows/dev-module.js",
  args: {
    module: "m3-tuning-eval",
    envs: [
      { dir: "steps/quant", smoke: "import llmcompressor,transformers,matplotlib" },
      { dir: "steps/vllm",  smoke: "import vllm" }
    ],
    skipDev: false,
    spec: "docs/superpowers/specs/2026-06-30-m3m4-smoothquant-pedagogy-refactor-design.md"
  }
})
```
dev-agent 按 spec 改 s5/s6/s7（**hard constraint：严禁动 s1–s4**；核心改动点 dev 读 spec §3+§4 M3 节）：
- **s5**：算法已是 SmoothQuant+GPTQ W8A8（不动）；`pareto_step`/`find_pareto_knee` 保留；**拆测试**；L3 循环找拐点 k\* 后存 `s5_baseline`(ignore=()) + `s5_tuned`(ignore=k\*)；强化工业叙事（逐层回退看 PPL 恢复=比对敏感度）+ 校准数据/精度门槛点明。
- **s6**：算法 W4A16→**SmoothQuantModifier(α)+GPTQModifier(W8A8)**；主填空 `compare_group_sizes`→**`compare_smoothing_strengths(alphas,...)`**（扫 α，激活/权重 scale 比值代理，L1/L2 解析验、L3 真扫 α 测 PPL）；保留 `pick_group_size(hidden_size,candidates)`；`validate_group_size` **归测试**；**拆测试**；L3 加测 PPL；读 s5_tuned ignore、用最优 α+group_size 产 `s6_final`；叙事讲清 α 太小/太大后果 + 工业扫描。
- **s7**：`run_downstream_eval`/`read_ppl_from_artifacts` 保留（read_ppl 聚合对象→baseline/tuned/final）；`method_dirs` 三方法→**baseline/tuned/final/FP16 四向**；L2/原理/L3 叙事→"调优收益验证"；**拆测试**；强化精度门槛+达标判定。

预期：acceptance 全绿。

- [ ] **Step 2: 看 acceptance，失败则诊断**

同 Task 1 Step 2。额外风险：dev-agent 动了 s1–s4 → Step 3 controller 复原。

- [ ] **Step 3: 清输出 + commit**

```bash
cd course/m3-tuning-eval/steps/quant && for f in s5_pareto_tuning.ipynb s6_group_size.ipynb; do ./.venv/bin/jupyter nbconvert --clear-output --inplace "$f"; done
cd ../vllm && ./.venv/bin/jupyter nbconvert --clear-output --inplace s7_evaluation.ipynb
git -C /home/w00809645/repos/quantization-tutorial add course/m3-tuning-eval/steps/
git -C /home/w00809645/repos/quantization-tutorial status --short   # 确认只 s5/s6/s7 改动（s1-s4 不在）
```
commit：
```bash
git commit -m "refactor(m3): smoothquant industrial tuning closed-loop (s5/s6/s7)

s5 split tests + persist s5_baseline/s5_tuned; s6 W4A16->SmoothQuant W8A8 +
smoothing_strength sweep (main) + pick_group_size (minor) + validate->ipytest +
PPL; s7 four-way compare (baseline/tuned/final/FP16). Artifact handoff s5->s6->s7.
s1-s4 untouched.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: controller 独立复核（三类问题 + s1–s4 未动 + 闭环产物）

**Files:** 无新建（复核现有）。

- [ ] **Step 1: C 算法统一复核**

```bash
cd /home/w00809645/repos/quantization-tutorial
# M3: L3 recipe 无 W4A16/纯 QuantizationModifier 残留
grep -rnE "scheme=\"W4A16\"|QuantizationModifier\(.*W4A16" course/m3-tuning-eval/steps/ && echo "[FAIL] M3 残留 W4A16" || echo "[OK] M3 W8A8 统一"
# M4: 无 FP8/AWQ 三方法分支残留（detect/build/decision_tree）
grep -rnE "\"FP8\"|\"AWQ\"|decision_tree|build_decision_tree" course/m4-deploy-loop/steps/ && echo "[WARN] M4 残留多算法" || echo "[OK] M4 收敛 SmoothQuant"
```
预期全 `[OK]`。

- [ ] **Step 2: A 验证型归测试复核**

```bash
# validate_group_size / pick_vllm_flag_and_kernel 已从填空移除（无 NotImplementedError）
grep -rn "def validate_group_size\|def pick_vllm_flag_and_kernel" course/m3-tuning-eval/steps/ course/m4-deploy-loop/steps/
# 期望：无定义（已移除），或仅在 ipytest 内联
grep -rnE "NotImplementedError" course/m3-tuning-eval/steps/quant/s6_group_size.ipynb course/m4-deploy-loop/steps/s1_load_quantized.ipynb | grep -E "validate_group_size|pick_vllm"
# 期望：无命中（这俩不再是填空）
```

- [ ] **Step 3: B 测试拆分复核**

```bash
# 每 notebook 的 ipytest cell 数 ≈ 业务填空数（grep %%ipytest 或 ipytest.run 出现次数）
for nb in course/m4-deploy-loop/steps/s*.ipynb course/m3-tuning-eval/steps/quant/s{5,6}*.ipynb course/m3-tuning-eval/steps/vllm/s7*.ipynb; do
  cells=$(grep -c "ipytest" "$nb")
  echo "$nb : ipytest 出现 $cells 次"
done
# 人工核对：每填空对应独立 ipytest cell（不再单 cell 测所有）
```

- [ ] **Step 4: M3 s1–s4 未动复核**

```bash
git diff --stat main...HEAD -- course/m3-tuning-eval/steps/quant/s1_sensitive_layers.ipynb course/m3-tuning-eval/steps/quant/s2_what_not_to_quantize.ipynb course/m3-tuning-eval/steps/quant/s3_ignore_syntax.ipynb course/m3-tuning-eval/steps/quant/s4_mixed_precision.ipynb
# 期望：空（s1-s4 零改动）。非空 → git checkout main -- 对应文件 还原
```

- [ ] **Step 5: 闭环产物路径 + 清输出复核**

```bash
# 产物路径在 s5/s6/s7 一致
grep -rn "s5_baseline\|s5_tuned\|s6_final" course/m3-tuning-eval/steps/ | sort
# 清输出
grep -rl '"execution_count": [0-9]' course/m3-tuning-eval/steps/quant/s{5,6}*.ipynb course/m3-tuning-eval/steps/vllm/s7*.ipynb course/m4-deploy-loop/steps/*.ipynb && echo "[WARN] 未清输出" || echo "[OK] 已清输出"
```
任何 `[FAIL]`/`[WARN]` → 回 Task 1/2 让 dev 修。

---

### Task 4: final whole-branch review（opus）+ merge ready

- [ ] **Step 1: review-package**

```bash
cd /home/w00809645/repos/quantization-tutorial
git log --oneline main..HEAD
git diff --stat main...HEAD
```

- [ ] **Step 2: opus 全分支终审**

dispatch opus code-reviewer（review-package main...HEAD），审：
- 三类问题是否改到位（C/A/B）
- 工业流程覆盖（§1.5）：smoothing_strength 调参 + 校准数据 + 精度门槛 是否体现
- M3 闭环产物接力是否一致（s5→s6→s7）
- 教学法质量（"学完应能讲清"锚点 + 讲解是否教会零基础）
- 清输出 + 填空数 ≥2

- [ ] **Step 3: fix loop + merge ready**

opus findings → dispatch ONE fix subagent（全部 findings）。re-review。完成后向用户报告 + 等 merge/push 指示（不自动 merge）。

---

## Self-Review（写完后自审 spec 覆盖）

**1. Spec coverage**：
- §1.5 工业流程（smoothing_strength 缺口）→ T2 s6 主填空 + 强化点 ✅
- §1 三类问题（C/A/B）→ T1/T2 改动 + T3 grep 复核 ✅
- §3 闭环产物接力 → T2 s5/s6/s7 + T3 Step 5 路径复核 ✅
- §4 逐 notebook 改动 → T1（M4 s1-s5）+ T2（M3 s5/s6/s7）核心改动点 ✅
- §5 填空职责（归测试清单）→ T1/T2 + T3 Step 2 ✅
- §6 测试拆分 → T1/T2 拆测试 + T3 Step 3 ✅
- §7 算法统一 → T1/T2 + T3 Step 1 ✅
- §8 验收 → T1/T2 workflow acceptance + T3 复核 + T4 终审 ✅
- §9 风险 → Global Constraints + 各 Step 失败处理 ✅

**2. Placeholder scan**：无 TBD/TODO；T1/T2 给完整 Workflow 调用 + 改动点 + 失败处理 + commit message；T3 给完整 grep 命令。✅

**3. Type consistency**：填空签名跨 spec↔plan 一致——s6 `compare_smoothing_strengths(alphas,...)`/`pick_group_size(hidden_size,candidates)`；M4 s5 `build_delivery_bundle(...)`/`build_deploy_summary(bench_result)`；SKIP_L3 守卫表达式一致；产物路径 s5_baseline/s5_tuned/s6_final 一致。✅

无遗漏，plan 可执行。

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-30-m3m4-smoothquant-pedagogy-refactor.md`。

**执行方式（用户已定）**：完整跑 dev-module.js workflow（skipDev=false）+ controller 复核 + opus 终审。controller 直接顺序执行 T1→T2→T3→T4，workflow 后台跑 + 完成通知，不逐步问用户（用户已授权端到端完成 + 不审核 spec/plan）。

# M3 精度调优与评测（layer fallback 核心）设计

> 日期：2026-06-28　分支：`main`（开发在 feature 分支）　关联：OUTLINE §模块3（3.1-3.7）、`docs/superpowers/specs/2026-06-25-per-module-self-contained-design.md`（自包含架构）、`docs/superpowers/specs/2026-06-27-dev-module-pedagogy-redesign-design.md`（教学法 + workflow 重设计）

## 1. 背景与目标

M3 是课程第 3 模块——OUTLINE 定为「**本课灵魂模块**」。学员已掌握 M2（三种量化方法 oneshot 出产物），本模块教**工业级精度调优全流程**：以 **SmoothQuant W8A8 为范例**，走通「**敏感层识别 → ignore 回退 → mixed-precision → Pareto 调优 → 多维评测**」，最终产出「最少回退 → 最大精度」的调优结果与 Pareto 对比。（同一流程适用 AWQ / FP8。）

**与 M2 的边界**：M2 = 量化方法（怎么把模型压成 FP8/AWQ/SmoothQuant）；M3 = 精度调优（压完发现掉点，怎么靠 layer fallback 把精度救回来）+ 评测（怎么测准）。M3 复用 M2 的 SmoothQuant recipe 作为调优起点。

**核心工程难点（已在 brainstorm 对齐）**：量化要 llmcompressor（transformers v5）、评测 downstream 要 vLLM（自带 transformers）——**一个 env 装不下**。本设计用**模块内双 env**（两个子文件夹）解决，不跨模块复用 M2。

## 2. 目标 / 非目标

**目标**
- 在 `course/m3-tuning-eval/` 下产出 7 个自包含 notebook（s1-s7，对应 OUTLINE 3.1-3.7），严格按 NOTEBOOK_CONVENTIONS。
- 模块内**双 env**：`quant-env`（llmcompressor，量化+PPL）+ `vllm-env`（vLLM+lm_eval，downstream 评测）。
- 每节配 ipytest L1 + tiny/合成 L2 + H200 L3；概念节按需 CPU。
- 套用 M2 重设计验证过的教学法：每 notebook 顶部「学完应能讲清」清单 + 摸一摸 cell + 端到端 why + 填空 why 引导。
- 通过 dev-module.js 全流程（5 阶段：reviewer 执行验证 + 学员代理理解探测 + 闸门）双重验收。

**非目标**
- 不重讲 M2 的量化方法原理（M2 已讲，M3 用 SmoothQuant recipe 作为调优起点）。
- 不做 M4（vLLM 声明式部署/多卡）——M3 的 vLLM 仅用于评测 backend，不做部署。
- 不改 M1/M2/M4 已交付产物。
- 不改 OUTLINE（M3 大纲不变）。

## 3. 模块结构（自包含，双 env）

```
course/m3-tuning-eval/
  README.md                  # 两子项目 setup（steps/quant uv sync、steps/vllm uv sync + lm_eval[vllm] 坑）
  scripts/
    download_model.sh        # 下 Qwen2.5-7B + 0.5B 到 ./models/（两子项目共享；复用 MODEL_CACHE）
  models/                    # 共享（两子项目都读；gitignore）
  out/                       # 共享产物（调优中间模型、PPL 表、评测结果；gitignore）
  steps/
    quant/                   # 量化+PPL step（独立 uv 项目，s1-s6）
      pyproject.toml         # llmcompressor + transformers v5 + datasets + matplotlib + ipytest/nbval/nbconvert/jupyter
      uv.lock                # 提交
      .venv/                 # gitignore
      s1_sensitive_layers.ipynb     # 3.1 敏感层识别
      s2_what_not_to_quantize.ipynb # 3.2 不该量化什么
      s3_ignore_syntax.ipynb        # 3.3 ignore 语法（含正则）
      s4_mixed_precision.ipynb      # 3.4 mixed-precision config_groups
      s5_pareto_tuning.ipynb        # 3.5 Pareto 调优
      s6_group_size.ipynb           # 3.6 group_size 调参
    vllm/                    # downstream 评测 step（独立 uv 项目，s7）
      pyproject.toml         # vLLM + lm_eval[vllm] + ipytest/nbconvert/jupyter（vLLM wheel 自带 torch/transformers）
      uv.lock                # 提交
      .venv/                 # gitignore
      s7_evaluation.ipynb    # 3.7 评测方法论（downstream 评测为主；PPL 数据从 quant 产物读）
```

## 4. env 设计（双 env，关键）

**为何双 env**：llmcompressor（layer fallback 重新量化）要 transformers v5；vLLM（downstream 评测）wheel 自带并锁定一套 transformers——同 env 会让 uv/pip 互锁版本。**在 `steps/` 下划两个子目录，各是一个独立 uv 项目**（自带 pyproject + uv.lock + .venv + 自己的 notebook），env 自然隔离，学员按 step 用对应子项目，不跨 env。

**steps/quant/**（量化 + PPL，s1-s6）：
- `pyproject.toml`：`llmcompressor>=0.9` + `transformers>=5.0`（v5，llmcompressor 0.12 必须）+ `compressed-tensors` + `accelerate` + `datasets` + `matplotlib`（Pareto 曲线/敏感度图）+ `torch`（cu128）+ ipytest/nbval/nbconvert/jupyter。
- 用途：layer fallback 重新量化（ignore 不同层）+ PPL 评测（transformers forward，快，不需生成）。
- 跑法：`cd course/m3-tuning-eval && uv run --directory steps/quant jupyter lab`（s1-s6 在此子项目跑）。
- 冒烟：`uv run --directory course/m3-tuning-eval/steps/quant python -c "import llmcompressor,transformers,matplotlib;print('ok')"`。

**steps/vllm/**（downstream 评测，s7）：
- `pyproject.toml`：`vllm>=0.11` + `lm_eval[vllm]` + ipytest/nbconvert/jupyter（vLLM wheel 自带 torch/transformers，不另装）。
- 用途：downstream 评测（`lm_eval --model vllm`，gsm8k/mmlu/ceval，快）。
- 坑：`lm_eval[vllm]` extra 要 `cd steps/vllm && uv pip install --python ./.venv/bin/python 'lm_eval[vllm]'`（OUTLINE 标注：`uv pip` 不认项目 venv，必须显式 `--python`）。
- 跑法：`uv run --directory steps/vllm jupyter lab`（s7 在此子项目跑）。
- 冒烟：`uv run --directory course/m3-tuning-eval/steps/vllm python -c "import vllm,lm_eval;print('ok')"`。

**notebook 怎么用双 env**：
- s1-s6 在 `steps/quant/` 子项目跑（quant env）；s7 在 `steps/vllm/` 子项目跑（vllm env）。**每个子项目自带 env + notebook，学员用对应子项目启动 jupyter，不跨 env 调 subprocess**。
- `models/`、`out/` 在 `course/m3-tuning-eval/` 根共享——子项目 notebook 用 `_find_module_root` 向上找 m3 根（含 `models/`+`scripts/` 的 `course/m3-tuning-eval/`）解析共享路径（plan 阶段定 `_find_module_root` 适配：从 `steps/quant/` 向上两层找 m3 根）。
- README 说明：先各自 `uv sync` 两子项目、下模型（共享）、s1-s6 用 quant、s7 用 vllm。

## 5. 7 notebook 设计

每个 notebook 按 NOTEBOOK_CONVENTIONS + 套 M2 教学法（顶部「学完应能讲清」清单 + 摸一摸 + 端到端 why + 填空 why）。下表是教学意图 + 候选填空（dev-agent 全权定具体填空）：

| step | OUTLINE | 教学目标 | 候选填空（从零/理解型） | env | L2/L3 |
|---|---|---|---|---|---|
| **s1** sensitive_layers | 3.1 | 敏感层识别两步法（profiling + 逐层量化测 PPL） | `profile_activation_outliers`（hook Linear 输入统计 per-channel 幅值）、`per_layer_quant_ppl`（保持其它层 FP16，只量化目标层测 PPL）| quant | tiny 逐层 PPL；L3 真 7B SmoothQuant 敏感层扫描 |
| **s2** what_not_to_quantize | 3.2 | 工业经验法则：哪些层绝不该量化 | `should_skip_layer`（判断型：lm_head/共享权重/特定 layer id）| quant | CPU 概念（经验法则表）|
| **s3** ignore_syntax | 3.3 | ignore 语法（精确 + `re:` 正则）| `build_ignore_list`（精确 ignore）、`build_regex_ignore`（正则 `re:.*down_proj`）| quant | tiny ignore 跑通；L3 真 7B ignore 前后 PPL |
| **s4** mixed_precision | 3.4 | mixed-precision config_groups（高阶核心）| `build_mixed_precision_recipe`（不同层不同 scheme，如敏感层 W8A16、其余 W8A8）| quant | tiny config_groups 跑通；L3 真 7B mixed-precision |
| **s5** pareto_tuning | 3.5 | 逐步 ignore → PPL 恢复曲线 → Pareto 拐点 | `pareto_step`（按敏感度加一层 ignore、重新量化测 PPL）、`find_pareto_knee`（判断型：PPL 恢复曲线拐点）| quant | tiny Pareto 曲线；L3 真 7B Pareto 调优 |
| **s6** group_size | 3.6 | group_size 对精度/显存影响 | `compare_group_sizes`（group 32/64/128/256 对比 PPL + 磁盘）| quant | tiny group 对比；L3 真 7B group 扫描 |
| **s7** evaluation | 3.7 | 评测方法论：PPL + downstream + 性能 | `run_downstream_eval`（判断型：选 task + lm_eval 参数）、`read_ppl_from_artifacts`（读 s1-s5 产的 PPL 表）| vllm（downstream；PPL 数据从 quant 产物读）| L3 真 7B 三方法（FP8/AWQ/SmoothQuant）四维评测对比 |

**s7 的特殊处理**：s7 跨两 env——PPL cell 在 quant-env，downstream cell subprocess 调 vllm-env。setup cell 明确标注。finale 是三方法（复用 M2 产物或在 s7 重新量化）四维对比表（PPL/gsm8k/显存/吞吐）。

**s2/s6 概念偏重**：s2（经验法则）偏概念，s6（group_size 调参）偏实验。两者 L3 可 CPU 或简化 GPU。

## 6. 教学法（套 M2 重设计验证过的 lesson）

M2 重设计（2026-06-28 合入）验证了这套教学法让学员真懂（不再机械化填空）。M3 直接套：
- **顶部「## 学完应能讲清」清单**：每 notebook 标题 cell 后，3-5 关键问题（学完应口头答），作 reviewer/学员代理检查锚点。
- **摸一摸 cell**：实例化关键对象让学员看到实物（如 s1 hook 输出、s3 ignore 列表、s4 config_groups 结构、s5 Pareto 曲线数据）。
- **端到端 why**：每步讲清「在 layer fallback 流程哪步、为何这么干」。
- **填空 why 引导**：每个填空 docstring 加「为什么这么设计（填前先想）」段，从零实现/判断型。
- 代码逻辑（填空实现 + L2/L3）由 dev-agent 写 + 验证。

## 7. 构建流程（dev-module.js 全流程，skipDev=false）

M3 notebook 不存在 → 全流程：
```
Workflow({ scriptPath: "workflows/dev-module.js",
           args: { module: "m3-tuning-eval", skipDev: false,
                   spec: "docs/superpowers/specs/2026-06-28-m3-tuning-eval-design.md" } })
```
- **前置**：先建好 M3 模块脚手架（双 env pyproject/uv.lock + scripts + README），否则 dev-agent notebook 路径 cell + reviewer 执行验证会失败。脚手架由计划执行者先建（同 M1/M2 T1-T3），再跑 workflow。
- 5 阶段：dev 写 7 notebook（含教学目标+摸一摸+why+填空+参考实现）→ reviewer 审核 loop（审内容锚教学目标 + 执行验证注入参考实现 nbconvert L1/L2/L3）→ 学员代理理解探测 loop（只读讲解答教学目标题）→ 闸门 → 定稿。
- **双 env 执行验证**：reviewer 注入参考实现跑 L1/L2/L3——s1-s6（quant）在 `steps/quant` env 跑、s7（downstream）在 `steps/vllm` env 跑。reviewer prompt 要 aware 双 env 子项目（spec/plan 明确）。

## 8. 验收标准（"完成"）

1. `course/m3-tuning-eval/` 自包含双 env：`quant-env` + `vllm-env` 各 `uv sync` 成功，冒烟（`import llmcompressor/transformers/matplotlib` / `import vllm,lm_eval`）过。
2. `bash scripts/download_model.sh`（MODEL_CACHE 复用）跑通，`./models/` 有 7B+0.5B。
3. 7 notebook 齐全，cell 顺序合 CONV，每步 ≥2 填空（含 ≥1 判断型）+ 教学目标清单 + 摸一摸 + 端到端 why + 填空 why 引导。
4. dev-module.js 全流程跑完，`reviewerPassed && execVerified && learnerSatisfied && scaffoldClean` 全绿。
5. 抽查：s1/s5 的 L3（真 7B SmoothQuant）跑通（s1 见敏感层、s5 Pareto 曲线）。
6. 真人（用户）最终审：通读 s1-s7，确认「学完能讲清」layer fallback 全流程。

## 9. 取舍记录

| 决策 | 选择 | 理由 | 备选（未选） |
|---|---|---|---|
| 范围 | 全 7 节（3.1-3.7） | 与 OUTLINE 对齐、与 M1/M2 一致 | 核心 subset（不完整）|
| env | **模块内双 env**（quant-env + vllm-env） | llmcompressor 与 vLLM transformers 冲突，拆两 uv 项目隔离；自包含不跨模块 | 跨模块复用 M2 quant env（学员要 M2 前置）/ 一个 env（冲突装不下）|
| 评测 | PPL（quant-env，Pareto 主判据）+ downstream（vllm-env，finale）| Pareto 多次评测要快（PPL forward），downstream 一次验证 | 全 downstream（慢，Pareto 不可行）|
| 范例 | SmoothQuant W8A8 | OUTLINE 指定；流程适用 AWQ/FP8 | AWQ（M2 已重写 s2 为范例）|
| 教学法 | 套 M2 lesson（清单+摸一摸+why+填空 why）| M2 已验证能让学员真懂 | 重新设计（浪费 + 风险）|

## 10. 后续

本设计批准 → writing-plans 生实施计划（脚手架 T1-T3 双 env + workflow 全流程 T4 + 验收）→ 执行（feature 分支）→ 验收 → merge。M4（部署）之后按自包含模板。

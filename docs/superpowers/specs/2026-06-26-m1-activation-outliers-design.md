# M1 激活离群点与现代量化方法（原理层）设计

> 日期：2026-06-26　分支：`main`（开发在 feature 分支）　关联：OUTLINE §模块1（1.1-1.7）、`docs/superpowers/specs/2026-06-25-per-module-self-contained-design.md`（自包含架构）

## 1. 背景与目标

M1 是课程第 1 模块——**原理层**。学员已懂基础量化数学（对称/非对称、per-token/channel/group），本模块只讲**为什么 LLM 量化难**与**每种方法在优化什么**，建立 SmoothQuant/AWQ/FP8 的思想模型与选型判断力。

**与 M2 的边界**：M1=**从零用 PyTorch 实现量化数学**（理解 why，可视化、等价变换、成本模型）；M2=**llm-compressor 工具链出可部署产物**（how to ship）。同方法名、不同角度。M1 不调 llm-compressor。

**本模块范围**：OUTLINE 1.1-1.7 全 7 节课，共 7 个自包含 notebook。

## 2. 目标 / 非目标

**目标**
- 在 `course/m1-activation-outliers/` 下产出 7 个自包含 notebook（s1-s7），严格按 NOTEBOOK_CONVENTIONS。
- 每节配 ipytest L1 + tiny L2 +（GPU 步骤）H200 L3；概念节（s6/s7）CPU-only。
- M1 自带精简 env（与 M2 不同）、下载脚本、README。
- 通过 dev-module.js 全流程（dev→架构师→QA→学生试课）双重验收。
- 顺带纠正 OUTLINE 1.3/line462 的 smoothing_strength 事实错误。

**非目标**
- 不讲基础量化数学（学员前置）。
- 不调 llm-compressor（那是 M2）。
- 不做 M3/M4。
- 不改 M2 已交付产物（仅 M1 内部 + OUTLINE 事实纠正）。

## 3. 模块结构（自包含，同 M2 模板）

```
course/m1-activation-outliers/
  README.md                  # 自包含 setup（uv sync + 下模型 + 跑法）
  pyproject.toml             # M1 精简 env（见 §4）
  uv.lock                    # 提交
  scripts/download_model.sh  # 下 7B+0.5B，MODEL_CACHE 复用（同 M2 脚本）
  steps/
    s1_activation_outliers.ipynb
    s2_llmint8.ipynb
    s3_smoothquant.ipynb
    s4_awq.ipynb
    s5_fp8.ipynb
    s6_selection.ipynb
    s7_timeline_ptq.ipynb
  # .venv/ models/ out/ 本模块内，gitignore
```

## 4. M1 env（精简、与 M2 独立）

M1 从零实现量化数学、不调 llm-compressor，故 env 与 M2 不同：

```toml
[project]
name = "m1-activation-outliers"
requires-python = ">=3.10,<3.13"
dependencies = [
    "torch>=2.10,<2.12",            # cu128 build（见 [tool.uv]），匹配 570 驱动
    "transformers>=5.0",            # 加载真 Qwen2.5
    "accelerate>=1.0",              # device_map 多卡加载
    "datasets>=2.20",               # 1.1 取样本文本喂模型看激活
    "matplotlib>=3.8",              # 1.1 画 per-channel 激活幅值分布（M2 无此依赖）
    "safetensors>=0.4",
    "sentencepiece>=0.2",
    "huggingface-hub>=0.26",
    "hf-transfer>=0.1",
    "ipytest>=0.14",
    "nbval>=0.11",
    "nbconvert>=7.0",
    "jupyter>=1.0",
]
# 注意：不含 llmcompressor / compressed-tensors（M1 从零实现，YAGNI）
[tool.uv]  # cu128 index（同 M2）
```

冒烟：`uv run --directory course/m1-activation-outliers python -c "import torch,transformers,matplotlib;print('ok')"`。

## 5. 7 个 notebook 设计

每个 notebook 按 NOTEBOOK_CONVENTIONS 的 cell 顺序：标题→`%%capture` 导入→`_find_module_root` 路径 cell（同 M2）→原理 markdown→≥2 填空 code cell→`%%ipytest` L1→tiny L2→GPU 守卫 L3→产物检查。填空设计由 dev-agent 定（用户全权委托），下面是**教学意图 + 候选填空**（参考已对齐 OUTLINE 的设计）：

| step | OUTLINE | 教学目标 | 候选填空（从零 PyTorch） | L2 / L3 |
|---|---|---|---|---|
| **s1** activation_outliers | 1.1 | 亲眼见 emergent outliers：少数 channel 幅值 20×+、结构性持续 | `per_channel_magnitudes`（hook Linear 输入算每通道幅值）、`outlier_ratio`（算离群比/阈值） | tiny 合成 30× outlier 检出；L3 真 7B 扫描各 Linear 写 outlier_scan.json |
| **s2** llmint8 | 1.2 | LLM.int8() 混合精度分解（基准/对照） | `vectorwise_absmax_quant`（vector-wise INT8）、`mixed_decompose_matmul`（离群维度切 FP16 相加） | tiny 验 mixed 误差 < 全 INT8；L3 真 0.5B gate_proj 对比 |
| **s3** smoothquant | 1.3 | 等价缩放迁移：`Y=(X·diag(s)⁻¹)·(diag(s)·W)`，`s_j=max(|X_j|)^α/max(|W_j|)^(1-α)` | `compute_smooth_scale`（α 公式，课程取 α=0.8 显式传）、`apply_smooth_transform`（X/s 与 s·W） | tiny 验 forward 等价 err~1e-7 + 激活难度下降；L3 真 0.5B v_proj α=0.5 vs 0.8 |
| **s4** awq | 1.4 | 显著通道权重缩放 + per-group INT4（weight-only W4A16） | `awq_weight_scale`（按激活幅值算每通道 s）、`per_group_symmetric_quant`（group INT4） | tiny 验 scaled-W 量化误差 < unscaled；L3 真 0.5B gate_proj group 64/128/256 |
| **s5** fp8 | 1.5 | E4M3 vs E5M2；推理用 E4M3（动态范围自带抗离群） | `fp8_max_value`（E4M3=448/E5M2=57344）、`quantize_to_fp8`（cast+clamp） | tiny 验 E4M3 对正常值误差 << INT8；L3 真 0.5B gate_proj torch.float8_e4m3fn cast |
| **s6** selection | 1.6 | W4A16(访存墙)/W8A8(计算墙)/FP8 选型决策 | `weight_bytes`（按方法算权重字节）、`recommend_method`（按 batch/显存/吞吐推荐） | tiny H200 成本模型交叉点；L3 CPU 概念（成本模型纯算术，无 GPU 守卫，产物 selection_table.json） |
| **s7** timeline_ptq | 1.7 | 量化方法时间线 + 为何 PTQ 为主 | `method_meta`（各方法元数据）、`ptq_vs_qat`（按规模/数据/预算选 PTQ/QAT） | tiny 时间线渲染 + PTQ/QAT 自洽；L3 CPU 概念（产物 method_table.json） |

**s6/s7 约定例外**：纯概念/选型 step，L3 cell 是 CPU 成本模型/元数据表，无 GPU 代码故无 `torch.cuda.is_available()` 守卫（NOTEBOOK_CONVENTIONS 仅约束含 GPU 代码的 L3；与 M2 s6/s7 同处理，已在 markdown 标注"无需 GPU"）。

## 6. 填空哲学（吸取 M2"偏易"教训）

- M1 填空是**从零实现公式/算法**（不是从相邻 cell 抄 API 字面量），天然比 M2 难。
- 额外要求：**每个方法 step ≥1 个判断型空**——不是"取这个字面量"，而是需要推理/选择（例：s1"给定阈值判断哪些通道是 emergent outlier"、s3"为何 α>0.5 偏向多迁权重"、s6"给定 batch/显存约束选方法"）。
- docstring 给方向与坑位，不给逐字答案（沿用 47948c3 的方向式风格）。
- dev-agent 设计、QA/学生 loop 把关难度。

## 7. 顺带纠正（事实性）

OUTLINE 与 M1 s3 涉及的 smoothing_strength：
- **错**："llmcompressor 默认 smoothing_strength=0.8（非论文 0.5）"（OUTLINE line106/462 原文）。
- **对**（实测 llmcompressor 0.12.0）：`SmoothQuantModifier()` 类默认 `smoothing_strength=0.5`（=论文 α）；0.8 是课程刻意取、需显式传。
- M1 s3 讲原理时用**正确**表述；并在 OUTLINE line106/462 同步纠正（与 M2 s3 已修一致）。

## 8. 构建流程（全 dev-module.js，skipDev=false）

M1 notebook 不存在 → 跑全流程：
```
Workflow({ scriptPath: "workflows/dev-module.js",
           args: { module: "m1-activation-outliers", skipDev: false } })
```
（args 现已 JSON.parse 修复，module 会正确解析。）
- 开发：dev-agent 按 §5 + CONV + OUTLINE 生成 7 notebook，当场验证 L1/L2/L3。
- 架构师审核 → QA loop（到 pass）→ 学生试课 loop（worktree 真跑 L3，到 satisfied）→ 完整性闸门 → 定稿。
- **前置**：先建好 M1 模块脚手架（pyproject/uv.lock/scripts/README），否则 dev-agent 的 notebook path cell + 学生 `uv sync` 会失败（上轮 args-bug 误建 M1 时就因无 env 而非功能）。脚手架由计划执行者先建（同 M2 T1-T3），再跑 workflow。

## 9. 验收标准（"完成"）

1. `course/m1-activation-outliers/` 自包含：`uv sync` 成功，`import torch,transformers,matplotlib` 冒烟过。
2. `bash scripts/download_model.sh`（MODEL_CACHE 复用）跑通，`./models/` 有 7B+0.5B config.json。
3. 7 个 notebook 齐全，cell 顺序合 CONV，每步 ≥2 填空（含 ≥1 判断型）、每个填空有 ipytest。
4. dev-module.js 全流程跑完，`qaResult.verdict==="pass"` 且 `studentResult.satisfied===true` 且 `acceptance.scaffoldClean===true`。
5. 抽查：注入参考实现，s1/s3 的 L3（真 7B）跑通（s1 见 outlier、s3 forward 等价）。
6. OUTLINE line106/462 smoothing_strength 已纠正。

## 10. 取舍记录

| 决策 | 选择 | 理由 |
|---|---|---|
| 范围 | 全 7 节（1.1-1.7） | 与 OUTLINE 对齐、完整；用户选定 |
| 构建 | 全 workflow（skipDev=false） | 复用已验过的 dev-module.js，自带双重验收 |
| env | 精简（无 llmcompressor，+matplotlib） | M1 从零实现，YAGNI；与 M2 真独立 |
| 填空 | 从零实现公式 + 判断型空 | 吸取 M2"偏易"教训，提升认知负荷 |
| step/填空划分 | 参考已对齐 OUTLINE 的设计 | 那版内容设计靠谱（只是当时无 env） |

## 11. 后续

本设计批准 → writing-plans 生实施计划（脚手架 T1-T3 + workflow 全流程 T4 + 验收）→ 执行。M3/M4 之后各自按自包含模板。

# 课程大纲：裁剪决策与关键修正记录

> 本文档记录 [`../OUTLINE.md`](../OUTLINE.md) 从「初版 6 模块通识大纲」演进到「H200×8 工业级 4 模块流水线」的**裁剪决策**与**版本查证中发现的关键技术修正**。当前权威大纲以 `OUTLINE.md` 为准；本文档供后续迭代溯源。

---

## 一、演进脉络

1. **v1 初版**（6 模块通识）：数学基础 → 范式与环境 → 离群点与方法 → 评测 → 工程落地 → 前沿。经质量保证评审员评审（结论"小修后通过"），覆盖 AutoRound/LLM-Compressor/TRT-LLM 等遗漏、修正 EXL2/K-quant 等准确性问题。
2. **v2 按用户场景重构**：目标收敛为「**H200×8 上，SmoothQuant + AWQ + FP8 端到端工业闭环**，基线 Qwen2.5-7B，引擎只用 vLLM」。砍掉通识基础与无关路线，压成 4 模块。
3. **v3 定稿**（当前）：经量化技术专家 + 代码导师对 2025–2026 当前版本库 API Web 查证，修正多处版本敏感错误后定稿。

---

## 二、裁剪决策（v1 → v2，按用户明确要求）

### 删除项
- ❌ **前两章基础**：仿射映射、对称/非对称、per-tensor/channel/group 粒度 —— 学员已掌握
- ❌ **llama.cpp / GGUF 转换 / Ollama / EXL2**：CPU·Apple Silicon·单卡极致路线，不在目标场景
- ❌ **所有 ROCm / AMD 内容**：不学
- ❌ **GPTQ**：只保留 SmoothQuant + AWQ + FP8
- ❌ **TensorRT-LLM**：引擎只用 vLLM
- ❌ **全部微调（QLoRA / NF4）**：聚焦推理量化流水线
- ❌ **NVFP4 动手**：H200（Hopper）无 FP4 Tensor Core，仅一句话前瞻

### 结构精简
- 原 M3（LLM.int8 独立模块）并入 M1 原理层（作为 SmoothQuant 对照基准）
- 原 M3+M4+M5+M6（精度调优 / 评测 / 部署 / 闭环）**压成 2 个模块**：M3（精度调优+评测）、M4（部署+闭环）

### 下游单方法范例
- M2 量化覆盖 **SmoothQuant + AWQ + FP8** 三方法（方法差异在此讲清）
- M3 精度调优/评测 与 M4 部署/闭环 **以 SmoothQuant 为唯一范例**走通（评测与部署流程方法无关，避免三方法各跑一遍的重复）；M4 附一张「方法→flag→kernel」cheatsheet 保证单范例可泛化到 AWQ/FP8

### 已确认的开放选择
- 基线模型：**Qwen2.5-7B**（全课统一，保证可比）
- QAT：**仅概念**（M4.7），不动手
- FP8：**保留为 M2 第三种量化方法（动手）**，因 H200 原生支持且为生产首选

---

## 三、关键技术修正（v3 查证中发现，已写入大纲）

> 以下为专家对 **2025–2026 当前版本** Web 查证后纠正的版本敏感错误。这些是课程最易写错、也最核心的工程细节。

### 1. AutoAWQ 已被官方废弃
vLLM 已把 AWQ 能力收编进 **llm-compressor**（输出 compressed-tensors）。AutoAWQ 仅作遗留路径对照，不再是主路径。课程 AWQ 推荐走 llm-compressor 统一工具链。

### 2. vLLM 改 CalVer 版本号
当前为 **`25.x`**（25.09/25.10/25.11），不再是 0.11.x；CUDA 基线 **12.8**。旧版本号描述全部更新。

### 3. FP8 量化 API 已变
- ❌ 旧：`Float8Modifier`（新版 ImportError / 已废弃）
- ✅ 新：`QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])`
- FP8 的 `oneshot` **不传 dataset**（动态激活，无需校准）；权重+激活都用 **E4M3**（"激活 E5M2"是训练场景，推理不用）

### 4. W8A8 INT8 是两段式（非单 modifier）
```python
recipe = [
    SmoothQuantModifier(smoothing_strength=0.8),   # 默认 0.8，非论文 α=0.5
    GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
]
```
- 导入路径：`SmoothQuantModifier` 在 `llmcompressor.modifiers.transform.smoothquant`（不是 `modifiers.smoothquant`）

### 5. AWQ 排除层机制（常见误写）
- AutoAWQ 的 `quant_config` **没有 `exclude_modules` 字段**
- 正确：HF Transformers/vLLM 的 `AwqConfig.modules_to_not_convert`（**加载侧**生效，默认 `['lm_head']`）；量化期 AutoAWQ 硬编码跳过 lm_head
- 课程讲"AWQ 排除层"须区分**量化期 vs 加载期**

### 6. vLLM 量化加载 flag
- `--quantization` **默认 `auto`**（读 `config.json` 的 `quantization_config`，多数情况不传）
- 遗留 AutoAWQ 格式：`--quantization auto_awq`（**带下划线**，不是 `awq`）+ awq_marlin（须 `zero_point=False`）
- compressed-tensors（SmoothQuant / ct-AWQ / FP8）：不传 flag，auto 识别

### 7. layer fallback 的细粒度玩法（M3 灵魂）
- llm-compressor **0.7.0+** 支持 mixed-precision：多个 `QuantizationModifier`（不同 targets+scheme）→ `config_groups`
- **vLLM 0.10.1+ 原生可跑** mixed-precision 模型
- **优于粗粒度 FP16 全回退**；不是手动改 config.json（易漏 `observer` 字段致加载失败）
- `ignore` 支持 `re:` 正则前缀；`targets="Linear"` 是类名匹配

### 8. 评测方法学坑
- V1 引擎下 `llm.generate()` 拿 per-request TTFT **不可靠** → 必须用 `/metrics` + `vllm bench serve`
- 显存差值法（nvidia-smi 峰值−基线）会**高估** KV-Cache（显示已映射显存）→ 推荐 `/metrics`
- **`add_bos_token=True`** 是量化模型评测高频坑（漏加 PPL 虚低/虚高）

---

## 四、需在开课前实测确认的点（见 OUTLINE.md 附录 B）

Qwen2.5-7B 三方法实测精度、逐层敏感度结论、`smoothing_strength` 最优值、SmoothQuant 实际 kernel 名、`llmcompressor`↔vLLM 精确版本配对、lm-eval 中文任务是否默认包含。这些无法从文档坐实，须在目标机器上 `pip freeze` + 实测。

---

## 五、原始评审（v1，已归档）

v1 6 模块通识大纲的完整 QA 评审（覆盖度/准确性/进阶/冗余/必修项）已在 v2 重构中作为输入吸收。其原文（含对 llama.cpp/GPTQ/TRT-LLM 等已裁剪内容的评审）不再适用于当前大纲，故以本文档的「裁剪决策 + 关键修正」替代，避免误导。

---

## 六、v4 严格评审（2026-06-23，三项改动 + 脚本落地轮）

本轮把「uv 双 env + 脚本 / M4 声明式部署 / M4.6 收敛」三项改动整合进大纲，并经量化技术专家 + 代码导师 Web 核实、质量保证评审员**严格**审查（实测 uv 0.8.9 跑脚本 + WebSearch 抽查版本声明）。

**结论：小修后通过**。评审抓到并已修复：

| 级别 | 问题 | 修复 |
|------|------|------|
| **CRITICAL** | `uv pip --directory envs/quant install 'lm_eval[vllm]'` **装不进目标 venv**（`--directory` 对 `uv pip` 只改 cwd、不绑定项目 venv，实测装进系统 anaconda）→ M3.7 评测课会跑不通 | 全部改为 `uv pip install --python envs/quant/.venv/bin/python 'lm_eval[vllm]'`（setup_env.sh / 附录A / uvEnvApproach 坑#5 / 评估题） |
| **MAJOR** | quant env `transformers>=4.45` 与 llmcompressor 0.12（要求 transformers v5）矛盾 | `transformers>=5.0` |
| **MINOR** | "4 脚本已实测可用" 与附录B openQuestion 口径冲突 | 降级为"语法层 uv 0.8.x 解析通过；完整装入目标 venv 待 H200 实测" |
| **MINOR** | `compressed-tensors>=0.9.0` 下限过旧（与技术栈 0.17.x 跨度大） | `>=0.10` |

**版本纠正（核实正确，已写入）**：vLLM PyPI 仍是 **0.x（0.23.x）**，「CalVer 25.x」是 NGC 容器 tag 非 pip 版本；CUDA 12.8 需 **driver ≥570.x**（非 535）；H200 = **141GB HBM3e（4.8TB/s）** vs H100 80GB（3.35TB/s），带宽差异不可忽略；lm-eval 需 `lm_eval[vllm]` extra；transformers 5.12.x / llmcompressor 0.12.x / compressed-tensors 0.17.x / uv 0.8.x。

**覆盖澄清（已落实）**：M2.2 统一基线为 Qwen2.5-7B-Instruct（对齐 download 脚本默认）；M3.7/M4.3 补 `/metrics` 抓取示例；M2.4 Falcon group_size 降为脚注；M4.2 补 TP=2+DP 说明（标注超纲）；M2.3 只讲精确 ignore、正则归 M3.3。

**三项改动落地判定**：uv 脚本（路径 A 两独立 uv 项目，修 C1 后可跑）✅；声明式部署（论断准确、边界清晰、纯文本 Qwen2.5 不需 trust_remote_code 已核实）✅；M4.6 收敛（逐条核对未偷重复 4.1–4.3，真引用）✅。

**交付（落盘）**：`OUTLINE.md`（493 行，4 模块 28 课时）、`envs/quant/pyproject.toml`、`envs/deploy/pyproject.toml`、`scripts/setup_env.sh`、`scripts/download_model.sh`。

**仍需目标 H200 实测（不阻塞定稿）**：driver ≥570 实测、双 uv.lock 跨节点复现、`lm_eval[vllm]` 与 vLLM 0.23 wheel 的 transformers 共存（冲突则用 `--model local-completions` HTTP 兜底）、Qwen2.5-7B 三方法实测精度/逐层敏感度/smoothing_strength 最优值（填 Pareto 表与决策树）。

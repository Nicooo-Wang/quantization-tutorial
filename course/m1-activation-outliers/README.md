# M1 — 激活离群点与现代量化方法（原理层）

对应 [OUTLINE.md](../../../OUTLINE.md) 模块 1（1.1-1.7）。本课是**原理层**：从零用 PyTorch 实现量化数学，建立 SmoothQuant/AWQ/FP8 的思想模型与选型判断力。

> 与 M2 的边界：M1 = **从零实现、理解 why**（可视化、等价变换、成本模型，不调 llm-compressor）；M2 = **llm-compressor 工具链出可部署产物**（how to ship）。同方法名、不同角度。

## Setup（本目录自包含，全部命令在此完成）

1. **建 env**（在本目录下，一次性）：
   ```bash
   cd course/m1-activation-outliers
   uv sync            # 读 uv.lock，建 .venv 并装好依赖（torch cu128 / transformers v5 / matplotlib …）
   ```
2. **拉基线模型**（7B + 0.5B，下到本目录 `./models/`；本模块自包含）：
   ```bash
   bash scripts/download_model.sh
   # 已在别处下过？复用免重下（同卷硬链）：
   # MODEL_CACHE=<repo>/models bash scripts/download_model.sh
   ```

## Steps（每个都是一个自包含 notebook，含填空 + ipytest + tiny 验证 + H200 执行）

- `steps/s1_activation_outliers.ipynb` — 1.1 激活离群点：hook Linear 输入，亲见 emergent outliers（少数 channel 幅值 20×+）
- `steps/s2_llmint8.ipynb` — 1.2 LLM.int8()：混合精度分解（离群维度切 FP16 相加）——SmoothQuant 的对照基准
- `steps/s3_smoothquant.ipynb` — 1.3 SmoothQuant：等价缩放迁移 `Y=(X·diag(s)⁻¹)·(diag(s)·W)`，把激活难度迁到权重 → 纯 INT8 W8A8
- `steps/s4_awq.ipynb` — 1.4 AWQ：显著通道权重缩放 + per-group INT4（weight-only W4A16）
- `steps/s5_fp8.ipynb` — 1.5 FP8：E4M3 vs E5M2，推理用 E4M3（动态范围自带抗离群）
- `steps/s6_selection.ipynb` — 1.6 选型：W4A16（访存墙）/ W8A8（计算墙）/ FP8 决策（**纯 CPU 成本模型，无需 GPU**）
- `steps/s7_timeline_ptq.ipynb` — 1.7 时间线 + 为何 PTQ 为主（**纯 CPU 概念，无需 GPU**）

跑法：在本目录下 `uv run jupyter lab` 打开 notebook；按序填空 → 跑 ipytest 测试 cell（L1）→ 跑 tiny 验证 cell（L2）→ 跑 H200 执行 cell（L3，s1-s5；s6/s7 为 CPU 概念 cell）。

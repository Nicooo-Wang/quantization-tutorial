# Notebook 约定（所有 Step 通用）

每个 Step = 一个自包含 `.ipynb`，cell 顺序与类型固定如下：

1. **标题 markdown cell**：`# Step N: <名称>` + 一句话目标 + 对应 OUTLINE 课时。
2. **导入 cell**：`import`（含 `%%capture` 抑制冗余输出）。
   - **cwd 无关路径解析**：notebooks 通过 `uv run --directory envs/quant jupyter lab` 启动，但 `jupyter lab` 的 cwd 是*所在 shell 的* cwd（**不是** `--directory` 的目标），所以 `models/Qwen2.5-7B-Instruct` 这类仓库根相对路径只有在学员碰巧 `cd` 到仓库根时才工作。notebooks 必须自行解析仓库根，不依赖 cwd：
     ```python
     import subprocess, pathlib
     REPO_ROOT = pathlib.Path(subprocess.check_output(["git","rev-parse","--show-toplevel"], text=True).strip())
     MODEL_DIR = REPO_ROOT / "models" / "Qwen2.5-7B-Instruct"   # 与 download_model.sh 默认 MODEL_DIR 一致
     ```
     notebook 中所有文件路径都从 `REPO_ROOT`（或 `MODEL_DIR`）派生，绝不使用裸相对路径。
3. **讲解 markdown cells**：原理 / 公式 / 这步做什么（分几段）。
4. **填空代码 cells**：每个 `logic` 函数一个 cell，形如：
   ```python
   def build_fp8_recipe(ignore=("lm_head",)):
       # TODO: 返回 QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=list(ignore))
       raise NotImplementedError
   ```
   每 Step ≥2 个填空，逐步搭出本步能力。
5. **ipytest 测试 cell（L1）**：
   ```python
   %%ipytest -qq
   def test_build_fp8_recipe_ignores_lm_head():
       r = build_fp8_recipe()
       assert "lm_head" in r.ignore
       assert r.scheme == "FP8_DYNAMIC"
   ```
   学员填完函数跑此 cell 验证；隔离执行，避免状态串扰。
6. **tiny 模型验证 cell（L2，CPU）**：内存 tiny Qwen2 + 真跑 llmcompressor + assert：
   ```python
   from transformers import Qwen2Config, Qwen2ForCausalLM
   tiny = Qwen2ForCausalLM(Qwen2Config(num_hidden_layers=2, hidden_size=64,
       intermediate_size=128, num_attention_heads=2, num_key_value_heads=2, vocab_size=320))
   # 用学员的函数跑真 llmcompressor（tiny 规模），assert 产物含 quantization_config
   ```
   FP8 等必须 FP8 硬件的：这 cell 标注"CPU 跑不了，见 L3"。
7. **H200 执行 cell（L3，GPU 守卫）**：
   ```python
   import torch
   if torch.cuda.is_available():
       run_fp8_quantize("/shared/models/Qwen2.5-7B-Instruct", "./out/qwen-fp8")
   else:
       print("跳过：无 GPU（CPU 环境只跑 L1/L2）")
   ```
8. **产物检查 cell**（折入原 2.6）：打印 `config.json` 的 `quantization_config` + compressed-tensors 结构 + 显存对比。

**提交规则**：`jupyter nbconvert --clear-output --inplace steps/sN_*.ipynb` 后再 commit（不带输出）。
**填空设计**：由开发专家 + 架构师定（用户全权委托）；硬性 ≥2 个有意义填空、每个有 ipytest 测试。

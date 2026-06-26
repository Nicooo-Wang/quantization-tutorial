# Notebook 约定（所有 Step 通用）

每个 Step = 一个自包含 `.ipynb`，cell 顺序与类型固定如下：

1. **标题 markdown cell**：`# Step N: <名称>` + 一句话目标 + 对应 OUTLINE 课时。
2. **导入 cell**：`import`（含 `%%capture` 抑制冗余输出）。
   - **cwd 无关路径解析**：notebook 通过 `cd course/<module> && uv run jupyter lab` 启动，但 jupyter 的 cwd 是*所在 shell 的* cwd，所以路径绝不依赖裸相对路径。notebook 自己**向上发现模块根**（含 `steps/` + `pyproject.toml` 的目录），从模块根派生 `models/`、`out/`：
     ```python
     import pathlib
     def _find_module_root(start):
         p = pathlib.Path(start).resolve()
         for cand in [p, *p.parents]:
             if (cand / "steps").is_dir() and (cand / "pyproject.toml").exists():
                 return cand
         raise RuntimeError("在模块目录内启动 jupyter")
     MODULE_ROOT   = _find_module_root(pathlib.Path.cwd())
     MODEL_DIR     = MODULE_ROOT / "models" / "Qwen2.5-7B-Instruct"    # 与 scripts/download_model.sh 一致
     TINY_MODEL_DIR = MODULE_ROOT / "models" / "Qwen2.5-0.5B-Instruct"  # L2/L3 先在 0.5B 上验，再上 7B
     OUT_ROOT      = MODULE_ROOT / "out"                               # 已 gitignore
     ```
     notebook 中所有文件路径都从 `MODULE_ROOT`（或 `MODEL_DIR`/`OUT_ROOT`）派生，绝不使用裸相对路径或 `git rev-parse` 仓库根。
3. **讲解 markdown cells**：原理 / 公式 / 这步做什么（分几段）。
4. **填空代码 cells**：每个 `logic` 函数一个 cell，形如：
   ```python
   def build_fp8_recipe(ignore=("lm_head",)):
       # TODO: 构造并返回 FP8 动态量化的 QuantizationModifier——按本函数 docstring 的语义
       #   填 targets/scheme/ignore 三项：scheme 名去「原理」cell 查（别凭记忆），ignore 入参
       #   是 tuple 要先转 list。这里只给方向，不给逐字实参（答案学员自己组合，别照抄）。
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
       run_fp8_quantize(MODEL_DIR, OUT_ROOT / "qwen-fp8")   # MODEL_DIR / OUT_ROOT 见 cell 2；先在 TINY_MODEL_DIR 验，再上 7B
   else:
       print("跳过：无 GPU（CPU 环境只跑 L1/L2）")
   ```
8. **产物检查 cell**（折入原 2.6）：打印 `config.json` 的 `quantization_config` + compressed-tensors 结构 + 显存对比。

**提交规则**：`jupyter nbconvert --clear-output --inplace steps/sN_*.ipynb` 后再 commit（不带输出）。
**填空设计**：由开发专家 + 架构师定（用户全权委托）；硬性 ≥2 个有意义填空、每个有 ipytest 测试。

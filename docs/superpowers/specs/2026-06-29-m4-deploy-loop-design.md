# M4 — vLLM 声明式部署与端到端闭环（设计 spec）

> 对应 [OUTLINE.md](../../../OUTLINE.md) 模块 4。课程终点闭环：把 M2/M3 产出的 compressed-tensors 量化模型部署到 H200×8 的 vLLM，走通「加载识别 → 多卡部署 → 压测 → 报错排查 → 端到端闭环」。引擎只用 vLLM。
>
> **本 spec 是 M3 流程的延续**：自包含 uv 项目 + dev-module.js workflow（5 阶段）+ M2/M3 验证过的教学法（「学完应能讲清」清单 + 摸一摸 cell + 填空 docstring「为什么这么设计」+ SKIP_L3 执行验证）。

---

## 1. 目标与范围

**模块目标**：把调优后的量化模型部署到 H200×8 的 vLLM，完成端到端闭环 mini 项目，补齐部署侧的报错排查。引擎**只用 vLLM**。

**声明式部署主线**（贯穿全模块）：部署是声明式的、**无需改模型代码**——Qwen2.5-7B 的 `architectures` 是 `Qwen2ForCausalLM`，已原生集成进 transformers 与 vLLM，纯文本 Qwen2.5-7B 无需 `--trust-remote-code`；vLLM `--quantization` 默认 `auto`，读 `config.json` 的 `quantization_config` 自动识别 compressed-tensors 格式（FP8/AWQ/SmoothQuant 三产物统一）并选 kernel。学员在 M4 写的代码只有三类且都不是模型适配：① 量化 recipe（M2 已产出）② `vllm serve` 命令 ③ 客户端调用。

**范围边界**（落在声明式范围内）：Qwen2.5-7B 纯文本 + 三种标准 compressed-tensors scheme（FP8/AWQ/SmoothQuant INT8）。多模态变体 / 非标准自定义 scheme / vLLM 未适配的冷门架构 = **边界外**（见 §5 声明式边界，作为教学内容讲清"何时声明式不成立"，不实操）。

**本模块砍 OUTLINE 4.5（HF Hub 发布）**：经用户确认不做独立 notebook。理由：Hub 发布偏运维/分享动作、非核心部署能力；课程目标是"把模型部署上线用起来"，发布到 Hub 不在此核心链路上。（OUTLINE 4.5 的 `huggingface_hub upload_folder` 知识点降级为 s5 finale 末尾的一句"产物可发布到 Hub"提示，不单独成节、不实操。）

---

## 2. 模块结构（单 env，比 M3 简单）

```
course/m4-deploy-loop/
├── pyproject.toml              # 单 env：vllm + ipytest/nbconvert/jupyter（不需要 llmcompressor）
├── uv.lock
├── scripts/download_model.sh   # 拉 Qwen2.5-7B-Instruct + 0.5B-Instruct（M2 模板 + M4 适配）
├── README.md                   # 双前置说明（见 §3）
├── models/                     # gitignore；7B + 0.5B
├── out/                        # gitignore；M4 自己的压测报告/产物
└── steps/
    ├── s1_load_quantized.ipynb      # 4.1 加载识别 + flag/kernel 速查 + 声明式边界
    ├── s2_multigpu_deploy.ipynb     # 4.2 H200×8 多卡 TP 部署
    ├── s3_benchmark.ipynb           # 4.3 性能压测（bench + /metrics）
    ├── s4_error_cheatsheet.ipynb    # 4.4 常见报错诊断
    └── s5_e2e_loop.ipynb            # 4.6 闭环 mini 项目 + 4.7 QAT/NVFP4 前瞻（finale）
```

**单 env**（不需要 llmcompressor，避免 M3 的 quant+vllm 双 env）：`vllm>=0.11`（PyPI 0.x，wheel 自带 torch/transformers，不另装 torch）+ `ipytest>=0.14` + `nbconvert>=7.0` + `jupyter>=1.0`。

**不装 `lm_eval` / `huggingface_hub`（YAGNI）**：M4 是部署模块——压测用 `vllm bench serve` + `/metrics`（vllm 自带），**不跑下游评测**（评测属 M3 s7 的 lm_eval）；Hub 发布已砍（§1），s5 仅 markdown 提示不实操 `upload_folder`。故 env 只需 `vllm + ipytest + nbconvert + jupyter`，**无 extra 旁路坑**（不像 M3 s7 要手动装 lm_eval[vllm]）。

**dev-module.js 单 env 向后兼容**：调用 `Workflow({scriptPath, args:{module:'m4-deploy-loop'}})`（不带 `envs`）→ 退化为单根 env，行为同 M1/M2。

---

## 3. 产物来源与跨模块路径（混合策略）

M4 是纯 vllm env（不自己量化），需要现成的 compressed-tensors 产物。采用**混合**策略：

- **0.5B 兜底**（L2/L3 轻量验证，不依赖任何量化产物）：`scripts/download_model.sh` 拉 `Qwen2.5-0.5B-Instruct`，纯 FP16 直接 vllm 部署——M4 的加载/serve/bench 流程逻辑全可在 0.5B 上验。这保证 M4 的"流程层"独立可跑（不依赖学员先跑 M2/M3）。
- **7B 量化产物跨模块引用**（L3 真部署）：setup cell 解析到 `course/` 父目录，引用兄弟模块 `out/`：
  ```python
  import pathlib
  def _find_module_root(start):
      p = pathlib.Path(start).resolve()
      for cand in [p, *p.parents]:
          if (cand / "scripts").is_dir() and (cand / "steps").is_dir():
              return cand
      raise RuntimeError("找不到模块根（含 scripts/ + steps/ 的目录）")
  MODULE_ROOT = _find_module_root(pathlib.Path.cwd())
  MODEL_DIR      = MODULE_ROOT / "models" / "Qwen2.5-7B-Instruct"
  TINY_MODEL_DIR = MODULE_ROOT / "models" / "Qwen2.5-0.5B-Instruct"
  OUT_ROOT       = MODULE_ROOT / "out"; OUT_ROOT.mkdir(parents=True, exist_ok=True)
  # 跨模块引用 M2/M3 的 7B 量化产物（兄弟模块 out/）
  REPO_COURSE = MODULE_ROOT.parent                       # course/
  M2_OUT = REPO_COURSE / "m2-quant-pipeline" / "out"     # qwen7b-fp8 / qwen7b-awq / qwen7b-smoothquant
  M3_OUT = REPO_COURSE / "m3-tuning-eval" / "out"        # 调优后的 mixed-precision 产物
  ```
  带存在性检查 + 友好报错（"`{M2_OUT}/qwen7b-fp8` 不存在——请先跑完 M2 s1 产出 FP8 7B 量化模型"）。
- **FP16 基线**（s3 压测对比用）：download 脚本拉的 7B 本身，作"三量化 + FP16"四向对比的基线。

**README 双前置说明**：
1. 本 env：`cd course/m4-deploy-loop && uv sync`（M4 用 vllm bench/metrics 压测，不装 lm_eval；env = vllm + ipytest + nbconvert + jupyter）。
2. 跨模块：s1-s5 的 L3 真部署需要 M2/M3 产出的 7B 量化模型——先跑完 M2（FP8/AWQ/SmoothQuant）+ M3（调优）。**0.5B 路径不需此前置**（L2 流程验证独立可跑）。

---

## 4. 教学编排（命令构造 + 离线 LLM()）

M4 核心是"**起 vllm 服务 + 压测**"——vllm serve 是阻塞进程、且 vllm 真跑需要 GPU（不像 M3 的 L2 能在 CPU 上真跑 llmcompressor）。故 notebook 范式为：

- **填空 = 纯逻辑函数**（构造命令 / 解析 config 选 kernel / 诊断报错 / 决策树），ipytest L1 验证（无 GPU 可跑）。
- **vllm 真跑**（0.5B 离线 `vllm.LLM()` 演示 + 7B serve/bench）全归 GPU 守卫的 L2/L3。
- **`vllm bench serve`** 用命令模板呈现（构造命令函数），L3 才真起服务跑——不在 notebook 里后台 `vllm serve &`（避免后台进程残留/端口占用/kill 失败，重蹈 M3 reviewer background 轮询卡死覆辙）。

**M4 特有的测试三层**（vllm 无法 CPU 真跑，跟 M3 略不同）：

| 层 | 内容 | 需 GPU? | workflow 执行验证（SKIP_L3=1）|
|---|---|---|---|
| **L1** ipytest | 命令构造 / config 解析 / kernel 选择 / 报错诊断 / 决策树 —— 纯逻辑 | 否 | ✅ foreground nbconvert 秒级验，**必过** |
| **L2** 结构验证 | 解析真实 config 样本验逻辑（0.5B FP16 → `detect` 返回 None；跨模块 M2/M3 量化产物 → 验 `quantization_config` 结构；缺产物则用内联 config 样本兜底）| 否 | ✅ CPU 可跑，验 |
| **L3** vllm 真跑 | 0.5B `LLM()` 离线加载 generate → 7B serve / bench serve / metrics | 是 | ⏭ SKIP_L3=1 跳过（留学员 GPU 实证）|

> **与 M3 的差异**：M3 的 L2 能在 CPU 上真跑 llmcompressor（tiny Qwen2ForCausalLM）；M4 的 vllm 真跑必须有 GPU（vllm 无实用 CPU 后端跑 FP8/AWQ kernel），故 L2 退化为"真实 config 结构验证"（仍 CPU 可跑、验产物格式正确性），vllm 运行时验证全归 L3。workflow 只验 L1+L2 代码逻辑——正中 M4 教学核心（命令对不对、kernel 选对没、报错诊断对没）。
>
> **L3 守卫**：`if torch.cuda.is_available() and not os.environ.get('SKIP_L3'):` —— workflow 执行验证设 `SKIP_L3=1` 跳过真 vllm 跑（7B serve 分钟级 + 多卡，太重）；学员/真人跑时不设，L3 实证。

---

## 5. 声明式部署的边界（核心教学点）

M4 反复讲"声明式、无需改代码"，但必须讲死其**前提**——否则学员误以为"任何模型量化后都不用改代码"。这是本模块最易被误读处，单列一节作为 s1/s4 的教学锚点。

**谁适配谁（方向性认知，必讲）**：声明式部署 = **vLLM 读你的 `quantization_config` 自动适配你的模型**，不是你去适配 vLLM。你只负责"产出正确的量化产物"（标准 scheme 量化 → 正确 packed 权重 + 把 `quantization_config` 写进 `config.json`），vLLM 读它自动选架构实现 + 量化 kernel——你夹在中间不写任何适配代码。llm-compressor 的 `save_pretrained` 自动把 `quantization_config` 写进 `config.json`，这就是"声明"的来源。

**vLLM 适配全过程（内部 6 步，`vllm serve` 启动时自动做）**：
1. 读 `config.json` 的 `architectures`（如 `Qwen2ForCausalLM`）→ 查 `ModelRegistry` 找架构实现类（attention/MLP/forward）。**层①架构适配**。
2. 读 `quantization_config`（`quant_method`）→ 查 quantization registry 找量化方法类 + 解析 `config_groups` 确定"哪些层用哪种 scheme"。**层②量化适配**。
3. 遍历模型层，给被量化的层套对应 kernel wrapper（`targets` 决定哪些层、`scheme` 决定 FP8/AWQ/INT8 哪个 kernel）。
4. 加载 packed 权重 + scale/zero_point 张量塞进 wrapper（`group_size` 决定分组粒度）。
5. **profile run**（跑 dummy 输入探测 KV-Cache 块数）——**这就是"验证"**：vLLM 启动自带、不是你写的；报错（`No compatible kernel found` / OOM）= 三条件某个不满足。
6. 起服务接请求。

**config 字段 → vLLM 适配步骤的映射**（s1 构造型填空的认知基础——学员懂了这个映射，才能从"量化需求"推导出"该写什么 config"，而非机械抄 JSON）：

| `quantization_config` 字段 | 驱动 vLLM 哪步 |
|---|---|
| `targets` | 步骤③ 哪些层套 kernel wrapper |
| `scheme`（FP8_DYNAMIC / W4A16 / W8A8 等）| 步骤③ 选哪个 kernel |
| `num_bits` / `group_size` | 步骤③-④ 权重打包粒度 |
| `ignore` | 步骤③ 跳过哪些层（保持高精度）|

**两层适配**（vLLM 加载量化模型要两层都通）：
```
config.json
 ├─ architectures: XxxForCausalLM   →  层①架构适配（vLLM 有没有这个模型的实现类）
 └─ quantization_config: {...}       →  层②量化适配（这个 scheme vLLM 认不认）
```

**三条件**（bf16 能跑的模型，量化后直接声明式跑通需满足）：
1. **量化方案是 vLLM 认的标准 scheme**（compressed-tensors 的 FP8/AWQ/INT8、GPTQ 等，用 llm-compressor 产出）。
2. **权重布局符合 kernel 契约**（粒度/scale 形状/group_size 整除/AWQ `zero_point=False` 等）。
3. **vLLM 给该架构补了该量化的 kernel 路径**（新架构常只实现 bf16 forward、未补 quantized 层）。

**关键结论**：
- 声明式承诺 = "**不用写模型适配代码、不用写反量化代码**"（前提架构已通）——成立。
- 但**不是**"随便量化都能跑"——还得"把量化做对"（方案标准 + 布局符合 kernel 契约）。
- **量化不创造架构适配**：vLLM 未适配的架构，量化前后都跑不了原生路径。
- `No compatible kernel found` 通常对应条件③不满足（架构没补 quantized kernel）或条件②（权重布局不符）。

**边界外三条**（声明式不成立，仅讲不实操）：(a) 多模态变体（可能需 trust-remote-code）；(b) 非标准/自定义 quantization scheme；(c) vLLM 未原生支持的冷门架构（走 Transformers fallback `--model-impl transformers`，性能损失大，或写 model 适配 = 改代码）。

---

## 6. 五个 notebook 详细设计

每个 notebook 套 M2/M3 教学法：标题 cell + 一句话目标 + 对应 OUTLINE 课时 → 「## 学完应能讲清」清单（3-5 条）→ 导入/setup cell → 讲解 markdown → 摸一摸 cell → 填空代码 cell（每函数一个，docstring 含「为什么这么设计」）→ ipytest L1 → L2 结构验证 → L3 vllm 真跑（GPU + SKIP_L3 双守卫）。提交前 `nbconvert --clear-output` + 删 cell metadata.execution。

### s1 — vLLM 加载量化模型：auto 识别 + quantization_config 构造 + 声明式边界（4.1，~50min）

> **本节是 M4 声明式核心**：不止让学员"读懂 config"，而是理解 `quantization_config` 结构后**能自己写出来**——从"机械照提示填"升级到"理解字段含义、自己产出正确 config"。

**学完应能讲清**：
1. "适配"是谁适配谁？——是 **vLLM 读 `quantization_config` 适配你的模型**，不是你适配 vLLM。你产出量化产物后，从 `vllm serve` 到服务起来，vLLM 替你做了什么？
2. vLLM 加载量化模型内部经历哪几步？`quantization_config` 的哪个字段驱动 vLLM 给某层选哪个 kernel？（`targets`→哪些层、`scheme`→哪个 kernel、`group_size`→权重分组粒度、`ignore`→跳过哪些层）
3. compressed-tensors 的 `quantization_config` 有哪些关键字段（`config_groups`/`targets`/`scheme`/`num_bits`/`group_size`/`ignore`）？给定一个量化需求（如"FP8 量化除 lm_head 外所有 linear 层"），你能写出对应结构吗？
4. 声明式部署的**两个前提**（架构 vLLM 已支持 + 量化用标准 scheme）+ **三条件**（标准 scheme + 权重布局符合 kernel 契约 + vLLM 补了该架构 quantized 层）；`No compatible kernel found` 对应哪个不满足？
5. 三方法（FP8/AWQ/SmoothQuant）各走哪个 kernel？为什么多数情况**不传 `--quantization` flag** 反而对？遗留 AutoAWQ 为什么 flag 是 `auto_awq`（带下划线）不是 `awq`？

**讲解**（声明式核心，详见 §5）：**谁适配谁**——你只产出正确产物（`config.json` 的 `quantization_config` + 正确 packed 权重），vLLM 读 config 自动适配（选架构实现 + 量化 kernel），你不写适配代码。**vLLM 内部 6 步**：① 读 `architectures` 查 ModelRegistry → ② 读 `quantization_config` 查 quantization registry + 解析 `config_groups` → ③ 遍历层给被量化层套 kernel wrapper（`targets`→哪些层、`scheme`→哪个 kernel）→ ④ 加载 packed 权重（`group_size`→粒度）→ ⑤ profile run 验证（vLLM 自带、非你写；报错=三条件违反）→ ⑥ serve。**字段→步骤映射**是学员"从需求推导 config"的认知基础。

**摸一摸**：打印一个 7B 量化产物（M2/M3 out）完整 `quantization_config`（看 `config_groups` 结构）；对比 FP8/AWQ/SmoothQuant 三种 `config_groups` 的字段差异；列 vllm 支持的 quantization scheme。

**填空**（3 个，理解-构造-速查递进）：
1. `detect_quant_scheme(model_path)` — 读 `config.json` 的 `quantization_config`，返回结构化信息（scheme + 被量化的 targets + 粒度），FP16 返回 None。**为什么这么设计**：先"读懂"——亲手解析 `config_groups`，理解"哪些层用什么 scheme"这个结构是 vLLM 步骤②③适配的依据，而不是把 config 当黑盒。
2. `build_quantization_config(scheme, targets, num_bits=None, group_size=None, ignore=())` — **构造** compressed-tensors 的 `quantization_config` 字典（`config_groups` + `ignore` 结构）。**为什么这么设计（理解型核心）**：这是"自己写出 config"——从量化需求（如"FP8 量化除 lm_head 外所有 linear 层"/"AWQ W4A16 group_size=128"）推导出正确字段结构。**docstring 只给字段语义和约束（`group_size` 须整除 hidden_size、`ignore` 决定哪些层保持高精度、`targets` 用 `re:.*Linear` 之类匹配），不给逐字实参**——学员要理解每个字段驱动 vLLM 哪步、该填什么值，而不是照抄一段 JSON。
3. `pick_vllm_flag_and_kernel(scheme)` — 给 scheme 返回 `(vllm_flag_or_None, kernel_name)` 速查（FP8→(None/CUTLASS scaled_mm)、AWQ→(None/Marlin/Machete)、SmoothQuant INT8→(None/INT8-Marlin)、遗留 AutoAWQ→('auto_awq'/awq_marlin)）。**为什么这么设计**：scheme→flag/kernel 速查逻辑化；多数"不传 flag"体现 auto 的便利，`auto_awq` 下划线是高频坑。

**ipytest**：测 `detect`（FP8/AWQ/SmoothQuant/FP16 四种 config 样本）；`build_quantization_config`（FP8 全量化 / AWQ W4A16 group_size / 带 `ignore` 排除 lm_head 三种场景生成正确结构 + `group_size` 不整除报错）；`pick`（四 scheme 映射）。

**L2**：解析真实 config 样本验 detect 逻辑（0.5B FP16 → None；跨模块 M2/M3 量化产物 → 验 `config_groups` 结构；缺产物用内联样本兜底）。CPU 可跑。
**L3**（GPU+SKIP_L3）：0.5B `vllm.LLM().generate()` 离线加载跑一句；7B 三方法产物各 `LLM()` 加载 generate，验证 auto 识别 + kernel 选择。

### s2 — H200×8 多卡部署（4.2，~50min）

**学完应能讲清**：
1. 单节点 8 卡为什么用**纯 TP**（`--tensor-parallel-size`）不混 PP？PP 什么时候才有意义（跨节点）？
2. continuous batching / PagedAttention 要不要手动开？（V1 引擎默认开启，无需 flag）
3. TP>1 启动 hang / rank timeout（NCCL）怎么排查？`--disable-custom-all-reduce` 什么时候用？`NCCL_DEBUG=INFO` / `VLLM_WORKER_MULTIPROC_METHOD=spawn` 各解什么？
4. Qwen2.5-7B 量化后**单卡就能放**，TP=8 更多是教学演示——真实提吞吐的 TP=2 + 数据并行（多实例 + LB）思路是什么？

**摸一摸**：`nvidia-smi` 拓扑（GPU 数 + P2P）；`vllm serve --help | grep -i parallel`。

**填空**（2 个）：
1. `build_serve_cmd(model_path, tp, gpu_mem_util=0.90, max_model_len=32768, extra=())` — 构造完整 `vllm serve` 命令字符串（含 `--tensor-parallel-size`、`--enable-prefix-caching`、`--disable-custom-all-reduce`）。**为什么这么设计**：4.2 的核心交付是这条命令模板；学员亲手组装 flag 理解每个参数作用，而不是抄。
2. `recommend_tp(num_gpus, model_weight_gb, gpu_mem_gb)` — 判断推荐 TP 数（量化 7B ~7-14GB 单卡能放 → TP 可小（1/2）；教学演示 TP=8；显存不够才升 TP）。**为什么这么设计**：判断型——让学员想清"TP 不是越大越好，是为了放不下或提吞吐才用"。

**ipytest**：测 `build_serve_cmd` 含正确 flag（TP/prefix-cache/disable-custom-all-reduce）；`recommend_tp` 边界（单卡放得下→低 TP、放不下→升 TP）。

**L2**：纯逻辑（命令构造）。
**L3**（GPU+SKIP_L3）：TP=2（或教学 TP=8）真 `vllm serve` + 客户端调一句。

### s3 — 部署性能压测（4.3，~40min）

**学完应能讲清**：
1. `vllm bench serve` / `throughput` / `latency` 三个分别测什么？为什么测 **TTFT 不能用 `llm.generate()`**（V1 引擎下不可靠）？
2. 读 `/metrics` 的 `gpu_cache_usage_perc` / `num_gpu_blocks` 看 KV-Cache 占用，为什么 **nvidia-smi 差值法会高估** KV-Cache？
3. 量化的性能红利（吞吐/显存）怎么压测验证？为什么必须有 **FP16 基线对比**？

**摸一摸**：`vllm bench serve --help`；`/metrics` 字段名（`gpu_cache_usage_perc`、`num_gpu_blocks`）。

**填空**（2 个）：
1. `build_bench_cmd(base_url, model, dataset='sharegpt', num_prompts=1000)` — 构造 `vllm bench serve` 命令。**为什么这么设计**：压测是 4.3 的核心动作；学员组参数（dataset/num_prompts/base_url）理解压测要控什么变量。
2. `parse_metrics(metrics_text)` — 从 `/metrics` Prometheus 文本解析 KV-Cache 占用 + 权重显存拆分，返回 dict。**为什么这么设计**：显存拆分是 4.3 的关键洞察（权重 vs KV-Cache）；学员亲手解析 metrics 文本，理解"为什么不能信 nvidia-smi"。

**ipytest**：测 `build_bench_cmd` 命令构造；`parse_metrics` 喂一段示例 `/metrics` 文本验解析正确。

**L2**：纯逻辑（命令构造 + metrics 解析）。
**L3**（GPU+SKIP_L3）：真压测三量化 + FP16 四向对比，输出吞吐/TTFT/显存对比表。

### s4 — 常见报错 cheatsheet（4.4，~45min）

**学完应能讲清**：
1. `No compatible kernel found` 的根因有哪几类（compressed-tensors 版本不匹配 / 架构没补 quantized kernel / 权重布局不符）？分别对应 §5 三条件的哪个？怎么修？
2. AWQ 报 `--quantization awq` 错（应是 `auto_awq`）/ `zero_point` 错（AutoAWQ 须 `zero_point=False`）分别怎么修？
3. TP 启动 hang / rank timeout 怎么排查 NCCL（`pip show nvidia-nccl-cu12` / `--disable-custom-all-reduce` / `NCCL_DEBUG=INFO`）？
4. OOM at profile run 多半是什么（KV 太大不是权重）？怎么调（降 `--gpu-memory-utilization` 或 `--max-model-len`）？

**摸一摸**：报错文本样本（OUTLINE 4.4 表里的典型报错串）。

**填空**（2 个）：
1. `diagnose_error(error_msg)` — 给报错文本，返回 `(root_cause, category)`，`category` 映射到 §5 三条件之一（scheme/布局/kernel路径）或"环境/版本"。**为什么这么设计**：诊断型——把"看到报错"和"根因 + 属于哪类契约违反"连起来，是 4.4 最高频的工程能力。
2. `recommend_fix(symptom)` — 症状 → 修复命令/动作（如 `uv pip install -U compressed-tensors vllm` / 改 `auto_awq` / 降 gpu_mem_util）。**为什么这么设计**：从诊断到修复的闭环；速查表逻辑化。

**ipytest**：测 `diagnose` + `recommend_fix` 对各典型报错（喂 OUTLINE 4.4 表的报错片段）的正确诊断。

**L2**：纯逻辑（喂报错文本片段诊断）。
**L3**：可选演示一个真报错（或 skip，纯逻辑已够）。

### s5 — 端到端闭环 mini 项目 + 前瞻（4.6 + 4.7，~90+25min，finale）

**学完应能讲清**：
1. 整条链路的**唯一交接物**是什么？（compressed-tensors 产物目录：`config.json` 的 `quantization_config` + 量化权重）为什么这个交接物让量化/调优/部署解耦？
2. 给场景约束（显存紧张 / 算力受限 / 精度优先 / H200），怎么结合 **M1.6 三范式**（W4A16 访存墙 / W8A8 计算墙 / FP8）+ **M3.8 四维 Pareto**（质量/显存/吞吐/TTFT）选方法？
3. 可复现**交付物四件套**是哪四件（recipe + 各模块 uv.lock + serve 命令 + 压测报告）？为什么固定 seed/prompt/batch？
4. QAT 为什么本课不动手（万亿 token 重训成本，仅蒸馏/继续训练）？NVFP4 为什么 H200 跑不了（Blackwell 第五代 Tensor Core 独有）？

**摸一摸**：展示 M2/M3 `out/` 链路（config + 量化权重 + PPL/评测 json），看整条链路产物如何串接。

**填空**（2 个）：
1. `build_decision_tree(constraints)` — 场景约束（显存/吞吐/精度/硬件）→ 推荐方法（FP8/AWQ/SmoothQuant）+ 理由。**为什么这么设计**：决策型——综合 M1.6 三范式 + M3.8 四维 Pareto，是 finale 的核心交付（"给我场景我选路径"）。
2. `build_delivery_bundle(recipe, uv_locks, serve_cmd, bench_report)` — 组装可复现交付物四件套 manifest（dict）。**为什么这么设计**：组装型——把零散交付物结构化，体现"可复现"的工程约定。

**ipytest**：测 `build_decision_tree`（不同约束→不同推荐）；`build_delivery_bundle`（四件套完整性 + 缺件报错）。

**L2**：纯逻辑（决策 + 组装）。
**L3**：可选链路串接（或 skip，纯逻辑已够）。
**末尾 markdown 前瞻**：QAT（fake-quant + 直通估计器 STE，理论精度上限高但 LLM 需万亿 token 重训、成本极高，仅蒸馏/继续训练划算——本课不动手）+ NVFP4（Blackwell 第五代 Tensor Core 独有，H200 不支持——未来硬件升级前瞻）+ 一句"产物可发布到 HF Hub"（OUTLINE 4.5 降级提示，不实操）。

---

## 7. 安全

- **无外发动作**：Hub 发布（4.5）已砍；任何对外发布/上传一律不做。notebook 内不自动起对外服务、不自动上传。`vllm serve` 默认 bind localhost（教学），不对外暴露。
- **跨模块只读**：M4 只**读** M2/M3 `out/` 产物，不写、不改兄弟模块。

---

## 8. 验收标准（dev-module.js workflow 全绿）

一个 notebook 判定完成需：① 有 `.ipynb`（标题 + 学完应能讲清 + 讲解 + 摸一摸 + 填空 + ipytest + L2 + L3 双守卫）；② ≥2 填空，每个有 ipytest；③ L1（ipytest）+ L2（结构验证）CPU 全过；④ reviewer verdict=pass（含执行验证 allPassed）；⑤ 学员代理 satisfied（只读讲解能答「学完应能讲清」每条）。

**模块验收** = dev-module.js 5 阶段全绿：reviewer pass[含执行验证 L1+L2] + learner satisfied + gate smokeOk（单 env `uv sync` 冒烟）。

**controller 独立复核**（M3 教训）：
- grep 查答案泄露：`grep -rn "def _.*_ref\|= _.*_ref\|参考实现" course/m4-deploy-loop/steps/` 必须为 0（参考实现只进 reviewer 注入，不进 notebook）。
- 确认 L3 cell 有 `torch.cuda.is_available() and not os.environ.get('SKIP_L3')` 双守卫。
- 确认跨模块路径有存在性检查 + 友好报错。

---

## 9. 风险

| 风险 | 缓解 |
|---|---|
| 跨模块路径耦合（M4 依赖学员跑过 M2/M3）| 0.5B 兜底（L2 流程独立可跑）+ README 双前置 + 友好报错 |
| vllm 版本敏感（PyPI 0.x，flag/kernel 名易变）| 照 OUTLINE 附录 A 速查（2026-06 实测）；pyproject 下限 `>=0.11`，不写 CalVer |
| L3 真 7B serve 极重（分钟级 + 多卡）| 全走 SKIP_L3=1；workflow 只验 L1+L2 代码逻辑，运行时留学员 GPU |
| reviewer agent 偏离 prompt 用 background nbconvert（M3 坑）| dev-module.js prompt 已强制 foreground + 禁 Monitor/TaskOutput（M3 af3d9be 已修，M4 复用）|

---

## 10. 与 OUTLINE 对应 + 范围调整

| OUTLINE 课时 | M4 notebook | 说明 |
|---|---|---|
| 4.1 加载识别 + flag/kernel | **s1** | 含声明式边界（§5 两层前提 + 三条件）|
| 4.2 H200×8 多卡部署 | **s2** | TP + NCCL 坑 |
| 4.3 部署性能压测 | **s3** | bench + /metrics |
| 4.4 常见报错 cheatsheet | **s4** | 诊断映射到 §5 三条件 |
| **4.5 HF Hub 发布** | — | **砍**（用户决定，降级为 s5 末尾一句提示）|
| 4.6 端到端闭环 mini 项目 | **s5**（finale 前半）| 决策树 + 交付物四件套 |
| 4.7 QAT + NVFP4 前瞻 | **s5**（finale 末尾 markdown）| 纯概念，不填空 |

**填空总计**：11 个（s1 有 3 个、其余各 2），每个有 ipytest。

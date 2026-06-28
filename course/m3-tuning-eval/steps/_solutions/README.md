# M3 填空参考实现（reviewer 注入执行验证用）

每个 notebook 的填空参考实现已**内联在该 notebook 末尾的「参考实现」cell**（紧跟每个填空 cell）。
reviewer 注入执行验证时：把填空 cell 的 `raise NotImplementedError` 替换为下方对应实现，
或直接运行 notebook（参考 cell 用 `xxx = _ref  # noqa` 已覆盖填空名，整 notebook 可直接 execute）。

各填空参考实现清单如下（与 notebook 内参考 cell 一致，此处汇总供 reviewer 快速取用）：

## s1_sensitive_layers
- `profile_activation_outliers(model, sample_inputs)` -> `_profile_activation_outliers_ref`
- `per_layer_quant_ppl(model, tokenizer, target_layer, baseline_ppl)` -> `_per_layer_quant_ppl_ref`

## s2_what_not_to_quantize
- `should_skip_layer(name, module, tie_word_embeddings=False)` -> `_should_skip_layer_ref`
- `classify_layer_risk(name, module)` -> `_classify_layer_risk_ref`

## s3_ignore_syntax
- `build_regex_ignore(suffixes)` -> `_build_regex_ignore_ref`
- `is_layer_ignored(name, ignore)`（判断型）-> `_is_layer_ignored_ref`

## s4_mixed_precision
- `build_mixed_precision_recipe(sensitive_layers, low_scheme, high_scheme)` -> `_build_mixed_precision_recipe_ref`
- `is_mixed_precision(quantization_config)`（判断型）-> `_is_mixed_precision_ref`

## s5_pareto_tuning
- `pareto_step(sensitive_rank, k, fp16_ppl, baseline_ppl, decay)` -> `_pareto_step_ref`
- `find_pareto_knee(curve, fp16_ppl, min_drop_frac)` -> `_find_pareto_knee_ref`

## s6_group_size
- `compare_group_sizes(hidden_size, group_sizes)` -> `_compare_group_sizes_ref`
- `validate_group_size(group_size, hidden_size)` -> `_validate_group_size_ref`

## s7_evaluation (vllm env)
- `run_downstream_eval(model_path, tasks, num_fewshot, limit, execute)` -> `_run_downstream_eval_ref`
- `read_ppl_from_artifacts(out_root, method_tags)` -> `_read_ppl_from_artifacts_ref`

---

## reviewer 注入执行验证流程（双 env）

1. s1-s6（quant env）：`cd course/m3-tuning-eval && SKIP_L3=1 steps/quant/.venv/bin/jupyter-nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 steps/quant/sN_*.ipynb`
   - notebook 内参考 cell 已把填空名 rebind 到 `_ref` 实现，直接 execute 即通过 L1(ipytest)+L2(tiny)。
   - `SKIP_L3=1` 跳过真 7B（reviewer 只验代码逻辑）；真人跑不设，L3 实证。
2. s7（vllm env）：`cd course/m3-tuning-eval && SKIP_L3=1 steps/vllm/.venv/bin/jupyter-nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=300 steps/vllm/s7_evaluation.ipynb`
   - 前置：vllm env 需先 `cd steps/vllm && uv pip install --python ./.venv/bin/python 'lm_eval[vllm]'`（已装）。

所有 notebook 的 L3 cell 均带双守卫：`if torch.cuda.is_available() and not os.environ.get('SKIP_L3'):`。

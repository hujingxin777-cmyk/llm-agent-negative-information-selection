# 敏感性和稳定性检测复刻项目

本目录从 `code-v9-300人` 复刻，只用于后续敏感性、P3 动作消融和网络规模稳健性实验。当前已完成代码与配置准备，尚未运行任何 API 实验。

## 已实现控制

- `extends` 小型配置继承，避免复制并漂移正式材料。
- `--repeat-start/--repeat-end` 与 `--scenario` 过滤。
- 每轮 active agent 序列导出到 `progress/activation_schedule.jsonl`。
- 固定 active-ID 序列回放，用于 S0/S2 直接效应审计。
- P3 动作策略：`full`、`reply_chain`、`post_only`、`like_channel`。
- 模型 usage、累计 token 和估算成本写入 `progress/model_request_log.jsonl`。
- 可选 `runtime.run_protection.max_total_tokens` 预算上限。

## 设计顺序

1. 审计旧 Main repeats 1-10 是否与本代码保持同一行为版本。
2. 使用 `main_300_add_repeats_11_30.yaml` 补 Main repeats 11-30。
3. Main 传递敏感性：S0、S1、S2、S3；S2 优先复用正式 Main。
4. 甲流 5 repeats 固定调度审计：先 S0 导出序列，再由 S2 回放。
5. Main P3 动作消融：Full、Reply-chain、Post-only、Like-channel。
6. 先运行 1000/50/2/8 capacity probe，通过后再运行甲流 1000/50/10/15。

## 运行示例

运行前请将 `.env.example` 复制为当前目录下的 `.env`，并填入所需接口密钥。

```bash
python3 run_experiment.py \
  --config configs/main_300_add_repeats_11_30.yaml \
  --env-file .env \
  p3
```

固定调度必须按顺序：

```bash
python3 run_experiment.py --config configs/main_fixed_schedule_s0_influenza.yaml --env-file .env p3
python3 run_experiment.py --config configs/main_fixed_schedule_s2_influenza.yaml --env-file .env p3
```

## 本地验证

```bash
python3 -m py_compile run_experiment.py src/*.py scripts/*.py tests/*.py
python3 -m unittest discover -s tests -v
```

以上命令不调用模型。

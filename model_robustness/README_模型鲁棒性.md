# 模型鲁棒性实验

本目录用于在不修改实验材料、人格、P3 动力学和指标口径的前提下，仅替换基座模型。

正式比较包含：

- 原正式基准：GLM-4，读取既有 300 人正式主实验结果；
- 国内替代：DeepSeek-V4-Flash，关闭思考模式；
- 国外替代：GPT-5.1 固定快照 `gpt-5.1-2025-11-13`，设置 `reasoning_effort=none`。
- 新版国外替代：GPT-5.4 固定快照 `gpt-5.4-2026-03-05`，设置 `reasoning_effort=none`。

DeepSeek 与 GPT 均运行 Main P1/P2/P3，参数保持为 300 agents、active 15、两个场景、10 repeats、15 timesteps。每个模型先执行 smoke 和 1-repeat probe，门禁通过后才运行正式实验。

DeepSeek 使用官方接口 `https://api.deepseek.com`；GPT-5.1 使用 AnyAPI 的 OpenAI 兼容接口 `https://www.anyapi.vip/v1`。两个接口均在正式运行前记录请求模型名和服务端实际返回模型名。

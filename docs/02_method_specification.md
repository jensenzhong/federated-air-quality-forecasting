# 方法规格

## 预测任务

输入张量 `float32[N,24,31]`，标签 `float32[N,1]`。输入时刻为 `t-23…t`，目标为 `t+1` 的 PM2.5。训练损失在标准化空间中计算，评估在反标准化空间；仅评估时将预测裁剪为非负，并同时记录裁剪前负预测比例。

## 主模型与消融

主模型为 2 层 GRU：`input_dim=31, hidden_size=64, dropout=0.1`；取末时刻隐状态，经 `LayerNorm(64)→Linear(64,32)→GELU→Dropout(0.1)→Linear(32,1)`。损失为 SmoothL1，优化器 AdamW，`batch_size=128`，`weight_decay=1e-4`。MLP 仅将 24×31 展平作为架构敏感性分析，不作为主结论模型。

架构选择网格仅允许 `hidden_size∈{32,64}`、`lr∈{0.0005,0.001}`、seed=42，按验证集站点宏平均 MAE 选择；差异≤0.5% 选小模型。冻结后所有联邦方法使用同一模型。

## 固定联邦方法

- FedAvg：按客户端训练样本数加权。
- FedProx：本地目标加入 `μ/2 ||w-w_global||²`；正式 MAS 与 FedProx 使用相同 μ。
- QFedAvg：公平性基线，q 只由验证集选择。服务端 Lipschitz 估计使用冻结的 `qffl_lr=1.0`（`L=1`），与本地 AdamW `lr` 解耦；不得把客户端优化器学习率传入 Flower 的 `client_learning_rate`。
- FedAdam：客户端 AdamW 不变，服务端自适应更新的 server_lr 只由验证集选择。

客户端固定为 12 站、每轮 12 个、30 轮、默认本地 epoch=1。正式主实验不接受客户端失败，最佳 checkpoint 只按验证集站点宏 MAE。

## 动态聚合

候选权重先计算后归一化：

- `size_only`: `n_i / Σn`
- `perf_only`: `(1/(MAE_i+ε))/Σ(1/(MAE+ε))`
- `hybrid`: `0.5·size + 0.5·performance`
- `fairness_clip`: hybrid 投影到 `w_i∈[0.04,0.16], Σw_i=1`

投影使用有界单纯形二分解，先验证 `n·l≤1≤n·u`，再确保非负、和为 1 与边界同时成立。

## 多智能体协议

v1 的服务器端 `RulePlanningAgent/LLMPlanningAgent` 仅保留为历史基线。v2 中，每个 `StationAgent` 在 ClientApp 本地维护状态胶囊、动作记忆和信用，调用本地/机构 LLM 生成候选，并在本地完成 shadow probe 与安全执行。服务器端 `AggregateCoordinatorAgent` 只读取 SecAgg+ 解密后的群组向量，广播公共阶段与学习率上限。

v2 的模型参数和群组统计向量经过同一次 Flower SecAgg+ 聚合。明文 `num_examples` 固定为 1，因此主聚合是站点等权宏平均；`aggregation_gate` 通过客户端本地缩放 `local-global` 更新实现，服务器不计算或查看单客户端权重。与 v2 对比的 FedProx 必须另建同为站点等权、预算匹配且使用 SecAgg+ 的控制组。

规则优先级：公平性阈值→三轮改善不足 0.5%→更新范数 CV>0.5→size_only。若连续两个间隔都恶化，学习率缩放 0.5、epoch=1；三轮改善不足 0.2%且未连续恶化时缩放 1.5、epoch=2；否则 1.0、1。

## LLM 协议

默认 `deepseek-chat`、temperature=0。第 1–4 轮强制探索四策略，第 5 轮起奇数轮调用、间隔轮复用。只发送训练/验证摘要，拒绝含 test 字段的观测。决策字段固定为 `strategy, lr_scale, local_epochs, reason, prompt_hash, source`；学习率只能为 0.5/1.0/1.5 倍，epoch 只能 1/2。缓存键为规范化 prompt 的 SHA-256。

非法响应/API 失败可在非正式运行回退规则；正式 LLM 运行出现任何预期决策点失败即 invalid 并重跑。密钥只从环境变量读取。Prompt、响应、模型、时间、解析状态、哈希和来源进入可审计缓存，不记录密钥。

v2 客户端 LLM 使用 `llm.client_base_url/client_model`，默认回环地址；公网 `llm.base_url` 只能用于 v1 或未来仅含群组摘要的协调器。客户端 prompt、响应和记忆只保存在 ClientApp 私有状态，不进入服务器工件。复用上一轮提案记录为 `source=cache`，不得计作新的 LLM 调用。

## 计算预算控制

`fedprox_budget_matched` 从配对 MAS 运行的 `decisions.jsonl` 回放每轮 `lr_scale` 与 `local_epochs`，聚合固定为 size_only，并使用同一模型、μ、切分和基础学习率。缺失任一轮轨迹时直接失败，不允许猜测。

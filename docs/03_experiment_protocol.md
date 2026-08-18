# 预注册实验协议

状态：`v1_closed; v2_implementation_pre_P1`。v1 已完成 seed42 test，但结果触发了方法重设计；原冻结协议及结果完整保留，只能作为 v1 结论和 v2 开发审计，不能作为 v2 的未见确认测试。

v2 方法、门槛和评测设计见 `docs/10_llm_mas_v2_research_redesign.md`。在 v2 P2 通过并重新冻结前，暂停原 seeds 123/456/789/2024 正式队列。

v2 已通过 3 客户端 Flower SecAgg+ 合成工程闭环，并完成真实 12 站 pafa_rule 1/3 轮 nonformal P1 smoke（`protocol_frozen=false/evaluation_split=val`，每轮 12/12、0 failures）；该 smoke 仅证明 ClientApp/SecAgg+ 工程闭环，不是正式隐私或性能证据。安全聚合 test 指标协议与机构隔离 ClientApp 未验证前，test 和正式运行均 fail-closed。v2 主要比较将改为同 SecAgg+、站点等权和总 epoch 预算下的 `pafa_llm` 对 `pafa_bandit`、`pafa_rule` 与预算匹配 FedProx，不沿用 v1 的不等价控制组。

冻结记录（2026-08-17，Asia/Shanghai）：Git 基线 `8151de3`；`configs/base.yaml` canonical SHA-256 `9f6b55a6b54e80077e3280b2b7c9b6c3b0fc3bec4699773c9cefa24987faaab0`。各运行的 resolved config、环境和数据 manifest 已随工件保存；当前工作区变更仍需在提交时一并归档，不能把 Git 基线单独解释为全部源代码快照。

## 固定要素

- 数据：`data/manifest.json` 中哈希固定的 12 站版本。
- 任务：24 小时输入预测未来 1 小时 PM2.5。
- 切分、插补、标准化：以 `configs/base.yaml` 和数据卡为准。
- 种子：42、123、456、789、2024。
- 主指标：站点宏平均 MAE。
- 次指标：微 MAE、RMSE、sMAPE、R²、最差站 MAE、站点 MAE std/CV、高污染 MAE、通信轮/字节、时间、峰值内存、LLM 成本。
- 主要比较：MAS-LLM − FedProx-budget-matched。

## 固定执行语义

- 主实验每轮 12 个站点全部参与，服务器收到 12 个成功回复后只聚合一次；任何缺失、重复或错误绑定使该运行 invalid。
- 允许客户端按 `data/cache/metadata.json` 中固定站点顺序逐个执行，以降低峰值内存。顺序执行不得改变客户端样本、初始模型、种子、本地 epoch、学习率、轮数或聚合公式。
- 正式低内存入口必须调用项目的真实 Flower ClientApp 与 Strict Strategy；当前 `scripts/run_sim.py` 只证明资源可行性，不具备正式结果资格。
- 低内存入口在进入正式筛选前必须通过等价性门禁：两客户端合成测试精确一致；12 客户端 FedAvg/FedProx 一轮的模型数组、聚合指标、partition ID 和工件字段一致；浮点比较使用 `rtol≤1e-5, atol≤1e-6`。
- 全并发多进程运行保留为可选的传输层一致性检查，不再是 M4 或正式实验的前置门禁。

当前 W1/W2 工程证据：`scripts/check_flower_equivalence.py` 的 2/12 客户端 FedAvg/FedProx 合成等价性全部通过；真实 12 客户端 FedProx 3 轮基准为 328.94 秒、峰值 RSS 0.392GB。上述证据仍属于工程门禁，不构成正式性能结果。

## 基线与调参预算

确定性基线为 Persistence、Seasonal Naive。学习基线为 Local-only GRU、Centralized GRU、FedAvg、FedProx、QFedAvg、FedAdam、Rule-MAS、MAS-LLM、FedProx-budget-matched。SCAFFOLD、FedDyn、Flash 等强基线在进入主表前必须完成 SecAgg+/本地状态/预算兼容性审计；尚未通过审计的方法不得静默替换为不等价实现。

只用 seed=42 和验证集：FedProx μ `{0.001,0.01,0.1}`；QFedAvg q `{0.1,1,5}`；FedAdam server_lr `{0.01,0.1,1}`。架构网格见方法规格。所有入选参数在主实验前写入决策日志并冻结。

## 执行顺序与门禁

`严格顺序运行时 → 1轮等价性 → 3轮资源基准 → 单种子参数筛选 → 单种子30轮 → 冻结协议 → 5种子主实验 → 消融 → 鲁棒性`。

3 轮低内存资源基准后确定最低可用内存门槛并估计全部队列墙钟时间；不得直接沿用未经测量的 10 GB 阈值。若全部队列预计超过 7 天则停止，不减少种子、轮数或客户端，记录计算资源阻塞。正式运行仍要求 12 客户端全参与且同步聚合；任何客户端失败使运行 invalid。

## 消融

- 动态聚合，但固定 `lr_scale=1, local_epochs=1`。
- LLM 观测中移除公平性指标。
- 展平窗口 MLP。
- FedProx-budget-matched 回放每轮优化预算。

## 压力场景

1. 掉线：每轮固定种子选择 3/12 客户端不可用。
2. 连续缺失：测试输入以连续 6 小时块遮蔽 10% 时间步。
3. 漂移：第 16 轮起，对排序索引 0/3/6/9 站点污染物输入乘 1.10，并在对应测试输入保持。

先单种子检查，再对 FedProx、Rule-MAS、MAS-LLM 跑 3 种子。压力强度不得由测试结果调整。注意：掉线场景是单独鲁棒性协议，主实验的 12 客户端严格门禁不适用于该场景，但必须验证预期 9 客户端和固定掉线轨迹。

## 统计计划

每方法报告 5 种子 mean±std。主要差异采用按“种子—站点—连续 24 小时块”分层的 10,000 次配对 Bootstrap，报告绝对差、相对变化、95% CI 和方向。次要比较用 Holm 校正；不得只报告总体平均。若 paired 运行的种子、站点或时间键不一致，统计程序必须失败。

## 测试集防火墙

模型选择、超参、Prompt、规则阈值和压力强度只能看训练/验证。测试预测在协议冻结后一次性产生；发生实现错误时允许修复并重跑，但必须登记错误、影响范围和新运行 ID，旧运行标 invalid。不得用测试结果选择最佳种子、轮次或结论口径。

### v2 防火墙增补

- seed42 原 test 已被观察且直接影响 v2 的问题定义，因此标为 `development_audit_only_for_v2`。
- v2 的 prompt、状态字段、动作、探针、阈值和消融均不得依据该 test 调整。
- v2 最终确认优先使用新增的 KDD Cup 2018 北京+伦敦跨城市/未来时间封存留出；数据清洗后先固定 manifest、切分和配置哈希，再允许评估。
- 若外部数据不可兼容，必须另立协议采用预登记 rolling-origin nested evaluation，并显式承认无法恢复原 test 未见性的限制。

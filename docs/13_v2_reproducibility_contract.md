# v2 reproducibility contract

状态：`frozen_engineering_contract_pre_confirmation`（2026-08-18）。

这份文件是 v2 主实验的 canonical 合同。它冻结可复现的协议、强隐私兼容基线、预算公平规则和统计门槛；它不把当前 validation smoke 当作正式结果。

方法显示名、比较分组和“哪些方法不能直接放在一起”的规则见 `docs/14_experiment_comparison_map.md`。本合同冻结运行协议；方法目录冻结阅读方式，两者同时生效。

## 1. 方法与协议

- 所有 PAFA 方法都使用真实 Flower ClientApp、四阶段 Secure Aggregation（安全聚合）、12/12 客户端同步参与和完整群组门禁。
- 客户端本地保留原始数据、残差、漂移状态、详细探针结果、LLM 记忆和 task ledger；服务器只能接收固定长度的安全聚合结果。
- 公共 `CohortDirective` 只含无身份、固定枚举字段，并严格绑定上一完成轮；rule、bandit、LLM proposer 使用同一 Schema 和动作库。
- `pafa_llm` 的客户端 LLM 只能是本机或机构内 endpoint；公网 DeepSeek 不得接收客户端级状态。
- 未实现 DP 会计前，任何结果都不得称为差分隐私。

主候选是“本地大模型控制器”（`pafa_llm`）。表 A 的传统联邦主比较包含“平均聚合”（`pafa_fedavg`）、“稳定本地训练”（`pafa_fedprox`）、“自适应服务器更新”（`pafa_fedadam`）和完整方法；表 B 的控制器公平比较包含“预算匹配的稳定本地训练”（`pafa_fedprox_budget_matched`）、“规则控制器”（`pafa_rule`）、“试错控制器”（`pafa_bandit`）和完整方法；“本地大模型控制器（无探针）”（`pafa_llm_no_probe`）只进入消融表。

主差异优先报告完整方法相对当前最强传统基线 `pafa_fedadam` 的差异；LLM 是否有额外价值则只通过与同动作空间 `pafa_bandit` 的受控比较判断。完整方法相对预算匹配控制的差异用于排除额外探针计算造成的假提升。

SCAFFOLD、FedDyn、Flash 等尚未通过 Secure Aggregation（安全聚合）协议审计的实现不得进入正式主表；依赖可链接客户端信号的方法保持协议不兼容或独立上界标签。

## 2. 固定预算

- 数据、模型、初始化、12 客户端集合、通信轮数、checkpoint 规则和 seed 集合 `[42, 123, 456, 789, 2024]` 固定。
- 开发阶段只读 `val`；旧 seed42 `test` 结果只作 v2 开发审计。封存确认集在 manifest、切分和哈希冻结前不得读取。
- 每轮基础本地训练为 1 epoch；动作库只允许 `safe_default(1 epoch)`, `cautious(1)`, `adapt_fast(1)`, `tail_focus(2)`，学习率倍率只允许 `{0.5, 1.0, 1.5}`。
- 每客户端最多 3 个候选，每候选 2 个 probe train batches（最多 6 个 probe batches），probe validation batches 固定为 2；budget-matched FedProx 必须消耗同样的 probe 调用预算但执行固定 `safe_default`。
- Probe batches、local epochs、通信 payload、墙钟、峰值 RSS 和 LLM 调用成本必须逐运行记录；不能用额外 epoch 或额外调参试验换取提升。

## 3. 指标与统计门槛

- 主指标：station-macro MAE（越低越好）。同时报告 worst-station MAE、station-CVaR、站点 MAE CV、漂移恢复、probe calibration、错误干预率、通信/计算成本。
- continual 参照额外报告 AF/AP/AvgPerf；task matrix 的未观测上三角只在客户端本地编码为 benchmark 约定的零。
- 主差异使用按 seed–station–连续 24 小时块分层的 10,000 次配对 bootstrap，报告绝对差、相对变化、95% CI 和方向；次要比较使用 Holm 校正。
- 只有当相对最强合格传统基线的 macro MAE 至少改善 1%，且配对 95% CI 不跨 0，同时 worst-station/CVaR 恶化不超过 0.5%，才允许声称精度胜出。
- 还必须相对同动作空间 contextual bandit 产生可测增益；否则只能声称自适应控制或公平/风险收益，不能归因于 LLM。

## 4. Fail-closed 与停止规则

缺失/重复客户端、安全聚合阶段乱序、身份或轮次不匹配、客户端级字段泄漏、越过 probe/epoch 预算、访问 test、未批准的外部 LLM endpoint，均使运行 invalid。若封存确认集未达到上述门槛，停止“超过强基线”叙事，转向诚实的风险敏感/公平结论。

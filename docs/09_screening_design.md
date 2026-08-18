# W3 Screening Design Review

状态：`approved_for_validation_only_execution`

说明：本文件记录的是早期参数筛选，不是 v2 主实验对比表。当前方法名称和比较分组以 [`docs/14_experiment_comparison_map.md`](14_experiment_comparison_map.md) 为准；筛选阶段的 `fedprox`/`fedadam` 非安全实现不能直接替代 v2 的 `pafa_fedprox`/`pafa_fedadam`。

## Mode

设计阶段；本文件只固定筛选预算、比较规则和证据要求，不生成或推断模型性能结果。

## Venue and assumptions

按 CCF-A/高质量机器学习与数据挖掘论文的最低证据包设计：主比较、机制消融、鲁棒性、失败分析、资源与复现证据必须可追溯。当前阶段只做 M3 参数筛选，不修改论文中心主张。

## Claim-evidence matrix

| Claim | Reviewer question | Screening evidence | Dataset/split | Baselines | Metric | Status |
|---|---|---|---|---|---|---|
| 固定模型/超参数可在联邦任务上稳定训练 | 参数选择是否由验证集预注册规则产生？ | 4 个集中式 GRU 配置 + 3 个 FedProx + 3 个 QFedAvg + 3 个 FedAdam | 12 站训练/验证；禁止 test | Persistence/Seasonal Naive 作为外部 sanity baseline；固定联邦家族内部比较 | `macro_mae`，tie-break `worst_station_mae` → `station_mae_cv` → `elapsed_seconds` | completed_nonformal |
| 联邦筛选保持公平 | 是否改变了客户端数、轮数或本地预算？ | 每个联邦筛选任务 12/12 客户端、固定顺序、3 轮、local epoch=1、seed=42 | 验证集 | 平均聚合/稳定本地训练/公平性加权聚合/自适应服务器更新 | macro MAE、worst-station MAE、station MAE CV、字节量、耗时 | completed_nonformal |
| 低内存实现不改变 Flower 语义 | 顺序调度是否只是工程参数？ | W1/W2 已通过；筛选只调用真实 ClientApp/Strict Strategy | 合成等价门禁 + 真实验证集 | 参考 Grid 与 SequentialGrid | 数组/指标差异、partition 完整性 | passed |

## Fixed screening budget

- Seed：仅 `42`。
- Split：仅 `val`；任何 test 访问、协议冻结标记或测试集选择均视为违规。
- Federated methods：每个候选运行 3 轮，12 个客户端全部参与，每客户端 local epoch=1。
- Centralized GRU：每个候选 1 epoch，使用验证集短预算，仅作为架构/学习率筛选代理。
- Formal confirmation：筛选胜者必须随后用 30 轮运行确认；筛选结果本身不进入正式主结果表。
- Queue：13 个固定任务，禁止临时增加组合或修改参数范围。

## Baseline matrix

| Family | Candidates | Role | Fairness constraints |
|---|---|---|---|
| Centralized GRU | hidden `{32,64}` × lr `{0.0005,0.001}` | 学习上限/架构筛选 | 同数据、同验证切分、同 1 epoch budget |
| FedProx | μ `{0.001,0.01,0.1}` | 主比较的预算匹配固定联邦基线 | 同 12 客户端、3 轮、local epoch=1 |
| QFedAvg | q `{0.1,1,5}`，冻结 `qffl_lr=1.0` | 公平性/效用敏感性基线 | 同客户端集合、轮数和本地预算；Lipschitz η 不进搜索网格 |
| FedAdam | server lr `{0.01,0.1,1}` | 强服务器优化基线 | 同客户端集合、轮数和本地预算 |

## Selection rule

每个 family 独立选择验证集 `macro_mae` 最低者；若相同，依次选择 `worst_station_mae`、`station_mae_cv`、`elapsed_seconds` 较低者。所有选择在任何 test 运行前写入 `docs/07_decision_log.md`，并在 30 轮确认后才允许协议冻结。

## Resource estimate

已观测的真实 3 轮 FedProx 基准为 328.94 秒；若仅作排程线性估算，9 个联邦筛选任务约 49 分钟，另加 4 个中央 GRU 短任务。该估算不是性能结果；若实际队列超过 7 天，停止并登记资源阻塞，不减少预注册任务。

## Screening execution result

13/13 项 screening 已完成，结果与选择日志写入 [`artifacts/reports/screening_results.md`](../artifacts/reports/screening_results.md) 和对应 JSON。筛选只读取验证集；所有 summary 的 `test_metrics` 仍为 `TBD`，且 `protocol_frozen=false`。集中式 GRU 64/0.0005、FedProx μ=0.001、FedAdam server-lr=0.01 仍是有效验证候选。QFedAvg q=0.1 的旧筛选按 D025 作废，必须用冻结 `qffl_lr=1.0` 单独重筛后再确认。

## Execution priority

| Priority | Experiment | Cost | Dependency | Stop condition |
|---|---|---:|---|---|
| P0 | 重新生成 13 项队列并检查命令参数 | low | W1/W2 passed | completed |
| P1 | 执行 4 个中央 GRU 短任务 | low/medium | data cache + validation split | completed_nonformal |
| P1 | 执行 9 个固定联邦 3 轮任务 | medium | strict sequential runner | completed_nonformal；最大记录 RSS 0.393 GB |
| P2 | family 内选择并执行 30 轮确认 | high | P1 全部完成 | 当前下一步；选择规则无法复现或验证工件缺失时停止 |

下一负责人：完成队列执行并将实际结果交给 `ccf-integrity-auditor` 做数字—主张一致性检查；随后进入 W4 MAS 门禁。

# PROJECT_STATUS

最后更新：2026-08-18（Asia/Shanghai）

| 里程碑 | 状态 | 已有证据 | 剩余门禁 |
|---|---|---|---|
| M0 安全归档 | 部分完成 | 85 文件归档总清单；9 文件删除清单；扫描 0 命中 | 用户在服务商控制台撤销旧密钥 |
| M1 项目初始化 | 门禁通过 | `uv sync --extra dev --frozen` 通过；全量验证、ruff、mypy 和 `git diff --check` 持续作为提交门禁；报告依赖已完整锁定 | 无 |
| M2 数据完成 | 门禁通过 | 12 站/420,768 行 manifest；220 个缓存文件；质量报告；全数据测试通过 | 无 |
| M3 模型完成 | v1 已审计；v2 重设计 | v1 的 13/13 screening、7 个 30 轮确认及 seed42 test 已保留；量化确认 LLM-MAS 9.6702 未优于预算匹配 FedProx 9.4665，且动作发生塌缩 | 按 `docs/10_llm_mas_v2_research_redesign.md` 实现并通过 P0--P2 |
| M4 Flower 完成 | 门禁通过 | ServerApp/ClientApp；固定策略；等价门禁 4 案例通过；12 站 FedProx 3 轮真实 ClientApp 基准完成（328.94 秒、峰值 RSS 0.392GB、最低可用内存 1.329GB）；13 项 screening 全部顺序完成 | 保持低内存顺序执行；后续 30 轮确认与正式队列继续记录峰值内存 |
| M5 MAS 完成 | P1 nonformal smoke 通过 | 动作、探针、执行、记忆已下沉 ClientApp；旧服务器逐客户端策略永久 fail-closed；Flower SecAgg+ 3 客户端与 12 客户端四阶段合成闭环通过；严格全客户端、缺失回复 fail-closed、会话重放/乱序、消息身份、数值容量、量化误差和聚合裁剪指示门禁已通过；真实 12 站 pafa_rule 1/3 轮 smoke 均 12/12、0 failures；真实 12 站、11-task、12-round continual smoke 已完成并产生非零 AF/AP/AvgPerf（证据见 `docs/continual_secagg_smoke_20260818.md`） | 该 smoke 不具备机构隔离资格；安全 test 评估、机构隔离和机构签名的 node->physical-station 绑定仍阻塞 |
| M6 正式实验 | 暂停 | 已完成 PAFA FedProx/FedAdam/Bandit/Rule 各 10 轮验证集开发运行；修复后 FedAdam 最佳 macro MAE=10.2817，Bandit=10.5097，Rule=10.7087；详细记录见 `docs/p2_secure_baseline_screening_20260818.md` | 先完成探针预算匹配与最强基线调优；当前仍未达到 active goal 的“相对最强基线至少改善1%且配对CI不跨0”门槛；本地 LLM endpoint 未监听 |
| M7 论文证据包 | 未开始 | 报告和统计生成模块 | 只使用 validated 运行生成表图与结论 |

## 当前最高优先级

- 项目已从 `protocol_v1 frozen/test` 转入 `protocol_v2 draft`。旧结果不删除、不改写，但不再驱动正式主张。
- v2 创新定义为“诊断—候选—局部反事实探针—安全执行—后验信用”加上 `CohortDirective` 安全黑板反馈的可验证智能体控制闭环，而非 LLM 动态调参。C1 已实现：directive 只含固定无身份字段，并严格绑定上一轮聚合结果；rule、bandit、LLM proposer 均消费相同 directive。
- 已加入 `aqfl/evaluation/continual.py`，按 benchmark 论文定义 AF/AP/AvgPerf，并提供固定长度任务矩阵的安全聚合 codec；`aqfl/data/continual_dataset.py` 已把冻结的 T0/base-test/T1--T11 时间窗接到客户端本地 cache 索引和 80/20 任务切分，ClientApp 已在显式 `continual-enabled` 请求下选择当前任务本地训练窗口，并在客户端边界维护本地下三角 ledger；SecAgg+ 已接入固定长度归一化任务矩阵数组，未观测上三角按 supplied benchmark 约定在本地编码为零，服务器仅在最终任务解码聚合指标。真实 12 站 continual smoke 已完成，但仍属于 nonformal validation。
- 已审计用户提供的 benchmark notebook；精确 base/base-test/11-task 边界已冻结到 `aqfl/data/continual_schedule.py`，但 notebook 本身不是可直接导入的安全运行时，且当前尚未运行其任务实验。
- 新确认集首选 KDD Cup 2018 Fresh Air 北京+伦敦多站点数据；必须先冻结 manifest/切分哈希，再产生任何确认结果。
- 隐私威胁模型见 `docs/11_privacy_threat_model.md`。当前实现要求 SecAgg+ 四阶段每个阶段都收到完整唯一客户端集合的一次回复，并把会话绑定到 run/node/round/stage；相关门禁已通过 2026-08-18 全量验证。C1--C4 新增 public directive 的 schema、round/replay、动作消费、摘要质量和固定长度 continual 矩阵门禁；新增隐私回归测试覆盖篡改、重放、身份泄漏和统一 proposer 消费。真实同进程 smoke 仍只是 nonformal 工程证据，test 与正式协议继续 fail-closed。
- Flower 数值路径处理的是加权完整模型参数的逐坐标量化裁剪，不是更新差分的 L2 裁剪。客户端裁剪风险只以布尔指示量安全聚合；一旦群组指示代表至少一个客户端违规，本轮失效且不得进入协调器或有效工件。

## 当前阻塞项

- 当前 `scripts/run_sim.py` 只覆盖 FedAvg/FedProx，保留为早期资源对照；正式低内存入口是 `scripts/run_flower_sequential.py`，新增 `--continual` 只允许 PAFA SecAgg+ 路径且 preflight 已通过（12 轮、training_started=false）。v1 test 工件已验证，但不得转用为 v2 的未见确认结果。
- `aqfl/federated/baseline_contract.py` 已建立强基线协议资格注册：普通 FedAvg/FedProx/FedAdam/QFedAvg 当前仍标为 `pending_secagg_adapter`；SCAFFOLD/FedDyn/Flash 标为 `pending_protocol_audit`；依赖可链接逐客户端信号的方法标为 `incompatible_client_signal`，不得静默改写后进入主表。
- `pafa_fedprox` 已完成 12 站 1 轮 nonformal aggregate-only smoke（run `pafa_fedprox-42-20260818T010041Z-1fc6755`）：probe fraction=0、12/12、0 failures，工件显式 `agentic_v2=false/secure_baseline=true`；该路径只验证安全基线传输和预算语义，不代表性能胜负。
- `pafa_fedadam` 已完成 12 站 1 轮 nonformal aggregate-only smoke（run `pafa_fedadam-42-20260818T010639Z-23007a5`）：probe fraction=0、12/12、0 failures，服务器执行独立 moments 更新；该路径只验证 FedAdam 安全适配，不代表性能胜负。
- `pafa_fedadam` 的首个 10 轮工件 `pafa_fedadam-42-20260818T013129Z-28f691b` 因安全服务器学习率硬编码为 1.0 而排除；代码已修复为透传 runner 的 `--server-lr`，修复后工件为 `pafa_fedadam-42-20260818T015126Z-28f691b`（server_lr=0.1，best validation macro MAE=10.2817）。
- P2 同动作空间开发对照已完成：`pafa_bandit-42-20260818T021014Z-28f691b` best macro MAE=10.5097、平均 probe fraction=0.778；`pafa_rule-42-20260818T022928Z-28f691b` best macro MAE=10.7087、平均 probe fraction=0.800。两者改善了 PAFA FedProx 的平均 MAE，但尚未超过修复后的 FedAdam。
- 受控 hybrid `pafa_bandit_fedadam-42-20260818T025932Z-90200bf` 已完成 3 轮 smoke，best macro MAE=12.4955；早期大量 `adapt_fast` 与服务器动量叠加后劣于静态 FedAdam，已停止其 10 轮扩展并登记为 rejected development candidate。
- 探针预算匹配 FedProx 控制已完成 10 轮：`pafa_fedprox_budget_matched-42-20260818T031349Z-323da88`，best macro MAE=10.5280、高污染 MAE=29.6761、probe fraction=1.0。Bandit 相对该控制仅改善约0.17% macro MAE且高污染指标更差，未达到预注册的1%门槛。
- `pafa_probe_oracle-42-20260818T033649Z-37754fb` 10 轮机制上界 smoke 已完成：best macro MAE=10.5626、高污染 MAE=29.3184；它仍未超过 FedAdam 的平均 MAE，但取得当前最佳尾部指标，支持把 v2 结论转向风险敏感/公平性收益。
- `pafa_llm` 已通过静态隐私 preflight，但配置的 loopback `127.0.0.1:11434` 当前未监听；按政策未启动任何客户端级 LLM 请求，也未退回公网 DeepSeek。
- `pafa_llm-42-20260818T040611Z-c8e2db6` 已用 development-only localhost OpenAI-compatible stub 完成 1 轮真实 ClientApp/SecAgg+ smoke：12/12、0 failures、`source_rate_llm=1.0`、directive compliance=1.0；该工件只证明本地传输与 proposer 闭环，不是 LLM 性能结果。此前 `040320Z` 工件因 HTTP proxy 误拦 localhost 返回 502，已排除。
- Windows 原生 Ray 后端启动 object store 超时；该路径已放弃，不阻塞顺序运行时主线。
- 严格 LLM-MAS 已用当前 `DEEPSEEK_API_KEY` 完成 30 轮验证；密钥不写入仓库，后续重跑仍需在运行环境显式提供。
- v2 顺序 runner 可用于 SecAgg+ 工程验证，但客户端与服务器同进程，因此没有机构隔离资格；新增 probe 的墙钟与峰值 RSS 必须在 P1 重新测量，不能套用 v1 的时长估计。
- 顺序 runner 的唯一 partition/站点配置和 Flower 消息元数据校验不能证明节点对应真实物理站点；正式部署仍需机构签名或等价可信注册的 `Flower node -> physical station` 绑定。

## 下一步

1. 暂停 seeds `123,456,789,2024`；已中断的 centralized seed123 标记 invalid，不续跑 v1 队列。
2. C1--C4 协同闭环与 continual SecAgg+ 固定数组已通过 synthetic/单元门禁；真实 12 站 11-task continual smoke 已通过（run `pafa_rule-42-20260818T043910Z-38c6a2f`），但仅为 validation/nonformal 证据。
3. P2 已完成探针预算匹配与 oracle 控制；由于完整动作空间也未达到相对最强基线的 1% 门槛，下一阶段固定为风险敏感/尾部公平证据包（不再扩展“平均精度超过基线”叙事），并在未见确认集冻结前禁止正式 test、多 seed 和“超过基线”结论。
4. v2 仍需保留 no-probe、probe-only 和相同动作空间对照；SCAFFOLD、FedDyn、Flash 等强基线需先完成协议兼容性审计。

任何未经真实执行与 `validate_run` 验证的结果均为 `TBD`。

# PROJECT_STATUS

最后更新：2026-08-18（Asia/Shanghai）

| 里程碑 | 状态 | 已有证据 | 剩余门禁 |
|---|---|---|---|
| M0 安全归档 | 部分完成 | 85 文件归档总清单；9 文件删除清单；扫描 0 命中 | 用户在服务商控制台撤销旧密钥 |
| M1 项目初始化 | 门禁通过 | `uv sync --extra dev --frozen` 通过；2026-08-18 当前全量验证为 127 项测试通过、1 项完整数据测试因 `BEIJING_AQ_DATA_DIR` 未配置跳过；ruff、mypy（54 个源码入口）和 `git diff --check` 通过；报告依赖已完整锁定 | 无 |
| M2 数据完成 | 门禁通过 | 12 站/420,768 行 manifest；220 个缓存文件；质量报告；全数据测试通过 | 无 |
| M3 模型完成 | v1 已审计；v2 重设计 | v1 的 13/13 screening、7 个 30 轮确认及 seed42 test 已保留；量化确认 LLM-MAS 9.6702 未优于预算匹配 FedProx 9.4665，且动作发生塌缩 | 按 `docs/10_llm_mas_v2_research_redesign.md` 实现并通过 P0--P2 |
| M4 Flower 完成 | 门禁通过 | ServerApp/ClientApp；固定策略；等价门禁 4 案例通过；12 站 FedProx 3 轮真实 ClientApp 基准完成（328.94 秒、峰值 RSS 0.392GB、最低可用内存 1.329GB）；13 项 screening 全部顺序完成 | 保持低内存顺序执行；后续 30 轮确认与正式队列继续记录峰值内存 |
| M5 MAS 完成 | P1 隐私预检通过 | 动作、探针、执行、记忆已下沉 ClientApp；旧服务器逐客户端策略永久 fail-closed；Flower SecAgg+ 3 客户端与 12 客户端四阶段合成闭环通过；严格全客户端、缺失回复 fail-closed、会话重放/乱序、消息身份、数值容量、量化误差和聚合裁剪指示门禁已通过全量验证；真实 12 站配置的无训练 dry-run 已返回 `training_started=false` | 下一步仅运行 12 站 P1 同进程 smoke；安全 test 评估、机构隔离和机构签名的 node->physical-station 绑定仍阻塞 |
| M6 正式实验 | 暂停 | seed42 v1 test 结果完整保留，但因已用于 v2 设计，只能作为 v2 的开发审计证据 | v2 P2 通过后冻结；建立新的未见确认集后才恢复多 seed |
| M7 论文证据包 | 未开始 | 报告和统计生成模块 | 只使用 validated 运行生成表图与结论 |

## 当前最高优先级

- 项目已从 `protocol_v1 frozen/test` 转入 `protocol_v2 draft`。旧结果不删除、不改写，但不再驱动正式主张。
- v2 创新定义为“诊断—候选—局部反事实探针—安全执行—后验信用”加上 `CohortDirective` 安全黑板反馈的可验证智能体控制闭环，而非 LLM 动态调参。C1 已实现：directive 只含固定无身份字段，并严格绑定上一轮聚合结果；rule、bandit、LLM proposer 均消费相同 directive。
- 已加入 `aqfl/evaluation/continual.py`，按 benchmark 论文定义 AF/AP/AvgPerf，并提供固定长度任务矩阵的安全聚合 codec；`aqfl/data/continual_dataset.py` 已把冻结的 T0/base-test/T1--T11 时间窗接到客户端本地 cache 索引和 80/20 任务切分，ClientApp 已在显式 `continual-enabled` 请求下选择当前任务本地训练窗口，并在任务结束时维护本地 ledger；SecAgg+ 已接入固定长度归一化任务矩阵数组，服务器仅在最终任务解码聚合指标。尚未在真实 12 站启动 continual 训练。
- 已审计用户提供的 benchmark notebook；精确 base/base-test/11-task 边界已冻结到 `aqfl/data/continual_schedule.py`，但 notebook 本身不是可直接导入的安全运行时，且当前尚未运行其任务实验。
- 新确认集首选 KDD Cup 2018 Fresh Air 北京+伦敦多站点数据；必须先冻结 manifest/切分哈希，再产生任何确认结果。
- 隐私威胁模型见 `docs/11_privacy_threat_model.md`。当前实现要求 SecAgg+ 四阶段每个阶段都收到完整唯一客户端集合的一次回复，并把会话绑定到 run/node/round/stage；相关门禁已通过 2026-08-18 全量验证。C1--C3 新增 public directive 的 schema、round/replay、动作消费和摘要质量门禁；新增隐私回归测试覆盖篡改、重放、身份泄漏和统一 proposer 消费。下一步允许的同进程验证 smoke 仍只是 nonformal 工程证据，test 与正式协议继续 fail-closed。
- Flower 数值路径处理的是加权完整模型参数的逐坐标量化裁剪，不是更新差分的 L2 裁剪。客户端裁剪风险只以布尔指示量安全聚合；一旦群组指示代表至少一个客户端违规，本轮失效且不得进入协调器或有效工件。

## 当前阻塞项

- 当前 `scripts/run_sim.py` 只覆盖 FedAvg/FedProx，保留为早期资源对照；正式低内存入口是 `scripts/run_flower_sequential.py`，新增 `--continual` 只允许 PAFA SecAgg+ 路径且 preflight 已通过（12 轮、training_started=false）。v1 test 工件已验证，但不得转用为 v2 的未见确认结果。
- Windows 原生 Ray 后端启动 object store 超时；该路径已放弃，不阻塞顺序运行时主线。
- 严格 LLM-MAS 已用当前 `DEEPSEEK_API_KEY` 完成 30 轮验证；密钥不写入仓库，后续重跑仍需在运行环境显式提供。
- v2 顺序 runner 可用于 SecAgg+ 工程验证，但客户端与服务器同进程，因此没有机构隔离资格；新增 probe 的墙钟与峰值 RSS 必须在 P1 重新测量，不能套用 v1 的时长估计。
- 顺序 runner 的唯一 partition/站点配置和 Flower 消息元数据校验不能证明节点对应真实物理站点；正式部署仍需机构签名或等价可信注册的 `Flower node -> physical station` 绑定。

## 下一步

1. 暂停 seeds `123,456,789,2024`；已中断的 centralized seed123 标记 invalid，不续跑 v1 队列。
2. C1--C4 协同与 continual SecAgg+ 固定数组闭环已通过 synthetic/单元门禁；下一步才允许做真实 12 站 1 轮 P1 同进程隐私 smoke，再扩到 3 轮确认真实 ClientApp 会消费上一轮 directive。该 smoke 不作为正式隐私证据；通过后才做 seed42 验证集 10 轮 P2。
3. v2 必须加入相同动作空间的 contextual-bandit、no-probe 和 probe-only 对照，避免把额外计算或动作空间收益误归因于 LLM；SCAFFOLD、FedDyn、Flash 等强基线需先完成协议兼容性审计。

任何未经真实执行与 `validate_run` 验证的结果均为 `TBD`。

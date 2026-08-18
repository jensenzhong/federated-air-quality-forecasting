# LLM-MAS v2 创新性重审与方法设计

状态：`protocol_v2_implementation`（2026-08-17）。本文件是方法重设计与工程边界依据，不代表实验结论。

## 1. 否定性诊断

当前 LLM-MAS v1 不能支撑“LLM 带来优于传统联邦优化的收益”这一中心主张：

- seed=42 验证集最佳 station-macro MAE：LLM-MAS `9.6702`，Rule-MAS `9.6548`，预算匹配 FedProx `9.4665`。
- 30 轮 LLM 决策中，`hybrid` 被选择 27 次；`lr_scale/local_epochs` 始终为 `(1.0, 1)`。动作空间实际上塌缩为近固定聚合器。
- v1 没有客户端级状态、动作后的信用分配或行动前验证。LLM 只能根据聚合统计猜一个全局策略。
- “动态学习率、local epoch、聚合权重、公平目标”本身已有 FedEx、Delta-SGD、FedCompass、AAggFF、FedAWARE 等直接先验工作，不能单独作为核心创新。

因此，v2 不再定义为 policy selector 或高级调参器，而定义为：

> 以联邦学习为主体、由本地站点 LLM 智能体与聚合级协调 LLM 协同的隐私保持联邦优化方法。

暂定名称：**Evidence-Grounded Hierarchical LLM-MAS for Federated Learning**。PAFA 仅表示其中“先本地验证、再执行”的安全子机制，不再作为论文主体。

## 2. 研究问题与可证伪假设

研究问题：在不上传原始数据和不增加总本地训练预算的条件下，LLM 能否利用跨轮语义化诊断提出客户端级干预，并通过本地低成本探针与安全执行器，改善时空漂移下的平均精度—尾部公平—计算成本 Pareto 前沿？

核心假设 H1：相同动作空间、通信轮数和总客户端 epoch 下，`LLM proposer + probe + safe executor` 优于非 LLM contextual bandit/rule controller。

辅助假设 H2：局部探针能预测候选动作的实际下一轮收益；若探针无校准能力，则 LLM 诊断不应被执行。

辅助假设 H3：在错位发生的站点漂移下，v2 可降低 worst-station MAE 或 station-CVaR，同时宏 MAE 不劣于最强传统基线。

反证条件：若相同动作空间的 bandit、规则或 one-step oracle 达到相同/更好 Pareto 结果，则不能把收益归因于 LLM；若只有额外 epoch 带来收益，则中心主张失败。

## 3. 方法闭环

### 3.1 客户端私有状态胶囊

每站点在 `ClientApp Context.state` 内维护以下状态，不上传服务器：

- 当前与 EMA 的 val MAE、斜率、振荡度；
- worst/high-pollution MAE 与残差分位数；
- update norm、与全局更新的 cosine/冲突程度；
- 输入/残差漂移分数及季节位置；
- 上轮动作、探针预测收益、实际后验收益；
- 本轮耗时、客户端 epoch 和通信字节。

本地站点 LLM 智能体读取这些状态并提出动作。原始样本、逐时残差、单客户端状态及提示均不得进入外部 LLM。服务器只接收 SecAgg+ 后的模型和达到最小群组规模的聚合摘要。本项目在未实现 DP 会计前不主张差分隐私。

### 3.2 分层 LLM 多智能体协同

每个客户端的本地 LLM 智能体依据私有状态生成受 Schema 约束的诊断和 2--3 个候选动作。聚合级协调器每轮只依据安全聚合后的群组摘要生成无身份 `CohortDirective`：`phase`、`priority`、`lr_scale_cap`、动作许可掩码和 `directive_round`。该指令在下一轮广播，客户端把它与私有状态结合后重新筛选候选动作；因此公共黑板反馈会实际改变 rule、bandit 和 LLM proposer 的本地动作，而不是只改变一个服务器学习率上限。`directive_round` 必须等于上一轮已完成聚合轮次，过期、重放或跨轮指令 fail-closed。当前实现为确定性、可审计协调器；只有在这一对照稳定后，才允许增加使用相同输入/输出 Schema 的协调 LLM。协调器不接收 client ID、单站点指标或单客户端更新，也不直接输出任意聚合权重。

首版动作库保持小而可审计：

- `lr_scale in {0.5, 1.0, 1.5}`；
- `local_epochs in {1, 2}`；
- `proximal_mu in {0.001, 0.01}`；
- `aggregation_gate in {normal, downweight_conflict, protect_tail}`。

### 3.3 客户端局部反事实探针

客户端在真正训练前，对候选动作运行固定预算的 shadow probe（小批量或截断 mini-epoch），只返回验证效用变化、更新方向摘要和成本。探针数据不用于训练全局模型，所有控制器享有相同探针预算。

这是 v2 与普通 LLM 调参器的关键区别：LLM 提议必须经过本地证据检验，不能凭语言模型置信度直接执行。

### 3.4 安全执行器

客户端安全执行器对候选动作求解受约束选择问题：

`utility = -Delta(macro_MAE) - lambda_tail*Delta(CVaR_station) - lambda_cost*cost`

仅当探针收益的保守下界为正时采纳；拒绝时回退到冻结的 FedProx 安全动作。客户端把 `aggregation_gate` 转换为本地更新贡献缩放，形成新的完整模型参数后，再将完整模型参数与固定长度群组统计向量一并交给 SecAgg+。LLM 无法绕过预算、隐私和数值边界。

这里的数值语义必须与 Flower 1.22 实现一致：`secaggplus_mod` 先以 `num_examples / max_weight` 对客户端返回的完整模型参数逐数组加权，再将每个坐标裁剪到 `[-clipping_range, clipping_range]`、量化并加掩码。当前 PAFA 固定 `num_examples=max_weight=1`，实现站点等权聚合。该操作不是对本地更新差分做 L2 范数裁剪，文中不得将其写成“更新裁剪”或据此推导 DP 保证。

客户端在进入 SecAgg+ 前检查待发送的完整模型参数是否包含非有限值或可能触发上述逐坐标裁剪。非有限值直接在本地拒绝；裁剪风险只编码为一个布尔指示量并随群组统计向量安全聚合。服务器仅看到 `cohort_clipping_violation_rate`；只要该聚合值表明至少一个客户端可能发生裁剪，本轮就在协调器观察和工件落盘前失效，不释放可用于控制的摘要。

### 3.5 跨轮记忆与信用分配

记录 `state -> proposal -> probe -> accepted/rejected -> realized_gain`。后续提示只检索相似的少量历史案例；实际后验收益用于识别无效诊断。该日志同时支持解释性和可复现实验。

### 3.6 严格群组与身份边界

当前 PAFA 工程协议采用严格全客户端语义：SecAgg+ 的 setup、share keys、collect masked vectors、unmask 四个阶段都必须对配置中的完整唯一客户端集合各发送一次请求，并从每个成员取得且只取得一次回复。任一缺失、重复、失败或集合不一致都会使该轮 fail-closed。这个选择是当前实验资格门禁，比 SecAgg+ 可支持的掉线恢复语义更严格；不能把它表述成已经验证了容错训练。

客户端会把每个阶段绑定到 `run_id`、本地 `node_id`、表示服务器轮次的数值 `group_id` 和固定阶段顺序；collect 阶段还绑定 `method` 与 `server-round`。服务器聚合端同时拒绝轮次跳跃/重放和重复或不完整的代理身份。顺序工程 runner 还检查本地 partition 与站点配置是否唯一且完整，但这只是配置一致性检查，不是机构签名的 `Flower node -> physical station` 身份证明。正式跨机构部署在建立并验证该签名绑定前继续阻塞。

同进程顺序 smoke 只能证明 Flower ClientApp/Strategy 调用、四阶段协议流、清洗门禁和聚合数值能够闭环；服务器与客户端共享进程和主机信任域，因此不能作为正式传输隔离、机构隔离或隐私安全的证据。

## 4. 为什么这个设计比“更多状态 + 动态预算”更强

| 维度 | v1 / 直接扩展 | v2 PAFA |
|---|---|---|
| 决策单位 | 全局策略 | 客户端级候选干预 |
| 依据 | 聚合统计与文字判断 | 状态胶囊 + 历史行动收益 |
| 行动前证据 | 无 | 固定预算局部反事实探针 |
| 安全性 | Schema 边界 | 探针门控 + trust region + FedProx fallback |
| 信用分配 | 无 | 候选、预测收益与真实收益闭环 |
| 研究场景 | 静态非 IID | 错位发生的时空概念漂移 |
| LLM 增量证据 | 无同动作空间对照 | rule/bandit/LLM 共享动作和探针预算 |

## 5. 先验工作边界

- FedEx 已覆盖联邦超参数调优；因此不能把动态 lr/epoch 当作主创新。
- Delta-SGD 已覆盖客户端自适应步长；FedCompass 覆盖异构客户端工作量分配。
- AAggFF 将公平聚合统一为在线凸优化；FedAWARE 和 FedAWA 已覆盖基于客户端动态/更新向量的自适应权重。
- Flash 与 distributed concept drift 工作已覆盖联邦漂移检测/适应；v2 必须证明智能体的“诊断—探针—执行”带来额外价值。
- Agentic-FL 2026 提出了 LM agent 编排愿景，但偏范式/路线图。v2 的差异应落在可执行、可验证、可消融的控制机制，而不是仅使用 agent 名称。
- 2025 年已有论文在同一 Beijing 12 站点数据上研究联邦连续时序预测与灾难遗忘。仅凭“空气质量 + 12 客户端 + 漂移”不再新颖。

详细检索表见 `docs/literature-search-20260817-agentic-fl-control/papers.md`。

## 6. 实验设计与晋级门槛

### 6.1 必须比较的控制器

1. FedProx、FedAdam、SCAFFOLD；
2. AAggFF、FedAWARE；Delta-SGD/Flash 至少实现一个客户端自适应和一个漂移基线；
3. LLM-MAS v1；
4. 相同动作空间的 random、rule、contextual bandit；
5. `LLM no-probe`、`probe + non-LLM`、完整 PAFA；
6. one-step probe oracle（作为机制上界，不作为实用方法）。

### 6.2 主要指标

- 主精度：station-macro MAE；
- 尾部公平：worst-station MAE、最差 25% 站点 MAE 的 CVaR；
- 漂移恢复：漂移后面积/恢复轮数；
- 控制质量：probe calibration、proposal acceptance、realized uplift、错误干预率；
- 成本：总客户端 epoch、通信字节、墙钟、峰值 RSS、LLM calls/tokens/cost。

### 6.3 阶段门槛

- P0：单元测试 + 1 轮 2 客户端合成闭环；非法 LLM 动作必须拒绝。
- P1：12 客户端 3 轮低内存 smoke；所有客户端完整，动作和探针预算可回放。
- P2：seed=42、验证集、10 轮淘汰赛。在相同总客户端 epoch 下，完整 v2 必须优于 v1；并且相对最强非 LLM 控制器满足以下之一：macro MAE 至少改善 1%，或 macro 非劣（0.5% 内）同时 station-CVaR 至少改善 3%。
- P3：只有 P2 通过才冻结方法，运行 30 轮验证确认和多 seed。
- P4：使用新的未见确认数据评估；P2 失败则停止扩展 LLM prompt，不进入多 seed。

阈值在运行 v2 前登记，不依据结果修改。

## 7. v2 测试防火墙

seed=42 原 test 已被读取且影响本次方法设计，因此仅能标为 `development_audit_only_for_v2`。v2 不得用它选动作、prompt、阈值或报告最终确认收益。

首选新确认集：KDD Cup 2018 Fresh Air 的北京与伦敦多站点数据，预先固定跨城市/未来时间留出；在下载后先生成 manifest 和切分哈希，再运行任何方法。若该数据不能形成兼容任务，则必须在另一个公开多站点时序数据集上建立封存留出。只有在无法取得外部数据时，才使用预登记 rolling-origin nested evaluation，并明确报告已有数据污染限制。

## 8. 实现顺序

1. `completed`：v2 类型、动作库、探针、预算会计、安全执行器和本地记忆。
2. `completed`：rule、contextual bandit、probe oracle 与本地 LLM proposer 共用同一动作空间。
3. `completed_engineering`：Flower 官方 SecAgg+ 四阶段合成集成测试；客户端回复清洗；群组统计向量与聚合级协调器。
4. `completed_engineering`：四阶段严格全客户端、缺失/重复、会话重放/乱序、消息身份与轮次绑定、数值容量和聚合裁剪指示门禁，以及 12 客户端合成量化误差回归，已通过 2026-08-18 全量验证；新增 `CohortDirective` 轮次绑定、三类 proposer 消费和离线本地 LLM 解析测试（当前定向测试 `26 passed`）。
5. `pending`：运行 12 站 1--3 轮 P1 同进程隐私 smoke，并验证 directive 在真实 ClientApp 下一轮改变动作；该 smoke 仍是 nonformal 工程证据。
6. `blocked`：安全聚合的 test 指标评估、机构隔离部署以及机构签名的 `node -> physical station` 绑定未完成前，不运行正式 test 或多 seed。

这一路线的目标不是保证 LLM 一定获胜，而是让“LLM 是否提供了传统控制器没有的增量价值”成为可以被严格检验的问题。

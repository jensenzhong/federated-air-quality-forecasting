# Goal 模式：隐私保持多智能体协同联邦学习研究路线

状态：`research_goal_active`（2026-08-18）。本文件定义研究目标、胜出条件、方法机制、强基线、开发门禁和失败转向。它不代表已经获得性能提升。

## 1. 目标与完成定义

项目目标不是“实现一个带 LLM 的联邦系统”，而是：

> 在原始数据、单客户端状态和单客户端更新均不离开机构，服务器只能观察 SecAgg+ 聚合结果的条件下，构建可验证的多智能体协同控制机制；在相同模型、数据、通信轮数、总本地优化步数和调参预算下，其封存确认集 station-macro MAE 必须显著优于最强隐私兼容传统/非 LLM 基线，同时不恶化尾部站点公平性。

“超过”采用以下硬定义：

1. 在冻结后的多种子验证上，相对最强合格传统基线，station-macro MAE 至少改善 1%，且分层配对 Bootstrap 的 95% CI 不跨 0；
2. station-CVaR / worst-station MAE 不得恶化超过 0.5%；
3. 在新的封存跨城市/未来时间确认集上方向一致，并再次满足显著性要求；
4. 相对共享相同状态、动作、探针和预算的 contextual bandit，完整 LLM-MAS 仍有可测增益，否则不能把收益归因于 LLM；
5. 不允许以更多客户端 epoch、更多调参试验、读取 test、暴露客户端级信号或关闭 SecAgg+ 换取提升。

无法保证研究结果一定超过基线；Goal 完成只由真实实验是否满足上述条件决定。工程完成、单 seed 更优或某个次要指标更优都不能替代该条件。

## 2. 当前证据与缺口

### 已完成

- 客户端本地 agent、候选动作、局部 probe、安全执行和私有跨轮记忆；
- Flower 1.22 SecAgg+ 四阶段闭环、聚合摘要、严格完整群组、重放/身份/量化/裁剪门禁；
- 聚合协调器可根据群组摘要生成下一轮 `cohort-lr-scale-cap`；
- P1 无训练 12 站 preflight；全量工程测试通过。

### 尚未完成

- C1--C3 已让 coordinator 通过固定无身份 `CohortDirective` 影响下一轮候选动作、动作许可、优先级和学习率上限；C4 已加入冻结 T0/base-test/T1--T11 的客户端本地 cache 适配器、私有 task ledger 和显式 ClientApp 当前任务训练选择，但任务级 federated 调度、任务矩阵 SecAgg 数组和 P1 smoke 仍未完成，因此尚不能宣称完整协议已在真实任务闭环；
- SCAFFOLD、FedDyn、Flash 等隐私兼容强基线尚未在同一协议中实现；
- FedAWARE、FedAWA、AAggFF、选择性协作等依赖客户端级服务器信号的方法尚未完成协议兼容性判定；
- station-CVaR、漂移恢复、probe calibration、错误干预率和严格预算账本未全部进入统一报告；
- 新的未见确认集尚未冻结；旧 seed42 test 已污染，只能作开发审计；
- 没有证明 LLM 相对同动作空间 bandit 的增量价值。

## 3. 优化后的研究问题

### Problem

安全聚合让服务器无法观察单客户端更新、损失和轨迹，而现有自适应聚合、选择性协作和公平控制方法经常依赖这些信号。静态 FedProx/FedAdam 虽符合聚合盲约束，却难以针对错位发生的站点漂移进行客户端级干预。

### Root challenge

系统需要同时满足三个相互冲突的要求：

1. 客户端需要利用私有细粒度证据做差异化动作；
2. 客户端之间需要共享群组目标，不能退化为互不相关的独立控制器；
3. 协调过程不能让服务器恢复或链接单客户端状态。

### Core insight

把协作对象从“客户端明文状态/更新”改为“安全聚合黑板”：每个站点 agent 在本地提出并验证动作，只把固定长度、去标识化的意图/诊断/收益统计加入 SecAgg+；协调 agent 根据群组黑板生成下一轮公共任务与约束；各站点再把公共任务与本站私有记忆结合。这样形成可回放的间接协同闭环，而不是服务器逐客户端控制。

### Contribution type

- 主要：安全聚合可见性约束下的多智能体联邦控制方法；
- 次要：面向非平稳多站点预测的可证伪评测协议与隐私—精度—公平—成本证据包。

新颖性仍需投稿前持续检索。不得声称“首次用 LLM 调联邦参数”或“首次多智能体联邦学习”。

## 4. 完整多智能体协同机制

每轮必须形成以下闭环：

1. **Local observer**：每个站点在本地构造私有状态胶囊，包括趋势、漂移、冲突、尾部误差、成本和历史动作收益；
2. **Local proposer**：LLM/rule/bandit 在完全相同的状态 Schema 和动作库上提出 2--3 个候选；
3. **Counterfactual probe**：固定预算 shadow probe 估计候选收益与不确定性；
4. **Safe executor**：只有保守收益为正且满足公共约束的动作可以执行，否则回退 FedProx；
5. **Secure blackboard write**：模型参数与固定长度的诊断率、候选率、接受率、probe 质量、尾部风险和成本统计一并进入 SecAgg+；
6. **Cohort coordinator**：协调 agent 只能读取群组黑板，输出有界公共指令；
7. **Broadcast coordination**：下一轮广播 `phase`、`priority`、动作许可掩码、学习率上限和公共成本上限，不包含站点或 client ID；
8. **Posterior credit**：本地记录公共指令是否改善实际收益，使后续 agent 学习何时服从、拒绝或回退。

首版公共指令固定为小型可审计 Schema：

```text
phase ∈ {initial, improving, stagnating, volatile}
priority ∈ {accuracy, tail_recovery, conflict_recovery, efficiency}
lr_scale_cap ∈ {0.5, 1.0, 1.5}
allow_adapt_fast ∈ {false, true}
allow_tail_focus ∈ {false, true}
directive_round = previous_completed_round
```

所有控制器接收同一公共指令。协调 LLM 只有在确定性协调器和规则对照稳定后才能加入，且必须使用相同输入/输出 Schema。

## 5. 优化目标与可证伪假设

主结果不使用单一加权分数掩盖退化，分别报告：

- station-macro MAE；
- worst-station MAE 与最差 25% station-CVaR；
- 漂移后误差面积和恢复轮数；
- 总本地优化步数、probe 步数、通信字节、墙钟、峰值内存和 LLM 成本；
- probe calibration、错误干预率、公共指令服从率与 realized uplift。

本地执行器可使用约束效用：

```text
U = predicted_accuracy_gain
    + λ_tail * predicted_tail_gain
    - λ_cost * normalized_extra_cost
    - λ_risk * intervention_uncertainty
```

但主论文结论必须回到分项指标，不能只报告 `U`。

假设：

- H1：安全聚合黑板反馈相对无反馈独立 agent 改善宏平均或尾部性能；
- H2：LLM proposer 相对同动作空间 bandit/rule 产生更高的 probe 后 realized uplift；
- H3：probe + safe executor 降低错误干预率，且收益不是额外训练步数造成；
- H4：完整方法相对最强隐私兼容传统基线在封存确认集达到第 1 节胜出条件。

## 6. 强基线与协议资格

外部 continual-learning 参照协议已单独登记于 `docs/benchmark-2510.21491_catastrophic_forgetting.md`：Hallak & Kem 的 arXiv:2510.21491 在同一北京 12 站任务上比较 Replay、LwF/KD、Online-EWC、EWC 和 SI。其 LSTM、六步预测、(T_0+11) 季节任务和 AF/AP/AvgPerf 不能与当前 GRU PM2.5 一步预测直接混数；必须先完成 manifest、模型/目标、任务调度和 SecAgg-only 评估适配。论文报告的 Replay 结果作为外部定位，不作为本项目已复现实验结果。

### A 级：必须复现并进入主表

| 类别 | 方法 | 作用 | SecAgg+ 兼容要求 |
|---|---|---|---|
| 基础 | FedAvg、FedProx | 静态聚合与异质正则 | 聚合完整模型/更新 |
| 服务器自适应 | FedAdam | 强 FedOpt 基线 | 仅使用聚合更新 |
| 方差/漂移校正 | SCAFFOLD、FedDyn | 强数据异质基线 | 客户端状态留本地，只安全聚合全局校正量 |
| 公平 | q-FedAvg | 客户端本地效用加权 | 安全聚合加权分子与权重和，不暴露单客户端 loss |
| 漂移 | Flash | 漂移感知优化 | 漂移统计必须群组化，不上传单客户端轨迹 |
| 同空间控制 | random、rule、contextual bandit、probe-only、one-step oracle | 识别 LLM、probe 和动作空间贡献 | 与完整方法共享状态/动作/probe/预算 |
| 旧方法 | LLM-MAS v1、FedProx-budget-matched | 证明 v2 相对旧故事的改进 | 固定旧实现与回放轨迹 |

### B 级：强近邻，先做协议兼容性判定

| 方法 | 价值 | 当前风险 | 主表资格 |
|---|---|---|---|
| AAggFF | 强公平 sequential decision | 原方法依赖可链接客户端效用 | 只有等价且不泄漏的实现通过审计才进入主表 |
| FedAWARE / FedAWA | 强自适应权重 | 依赖客户端更新向量/一致性 | 原协议作为能力参考；不得把隐私弱化实现伪装成同条件基线 |
| Selective Collaboration / Merit-based FedAvg | 动态选择有益协作者 | 依赖单客户端相关性且与全客户端门禁冲突 | 作为协议不兼容参考或独立非隐私上界 |
| Fed-TREND | 时序异质强近邻 | 合成知识载体可能改变隐私/通信预算 | 完成代码、许可、数据流和预算审计后决定 |
| Delta-SGD / FedCompass / FedEx | 客户端步长、工作量与超参强先验 | 与部分动作重叠 | 至少选择一个可复现代表进入扩展表 |

### C 级：只作定位

Agentic-FL 2026 是概念路线图，不是已经证明胜出的算法基线；FedMarl 主要解决客户端选择；北京 12 站连续学习 benchmark 约束任务新颖性。它们必须进入相关工作，但不能代替公平可运行对照。

## 7. 公平比较合同

所有主表方法必须满足：

- 相同数据切分、模型容量、初始化、客户端集合、通信轮数和 checkpoint 选择规则；
- 相同总本地优化步数；probe 步数单独计量，并为所有使用 probe 的控制器完全一致；
- 每个算法 family 获得相同数量的验证调参试验，而不是强制相同超参数；
- 同时报告通信 payload、墙钟、峰值内存和 LLM 调用成本；
- 所有 PAFA 主张在 SecAgg+ 下产生；协议不兼容基线单独分组；
- 不引用不同数据/模型/切分的论文报告数值作为本项目胜负证据。

## 8. 数据、统计与泛化

1. **开发数据**：当前 UCI Beijing 12 站只用于开发、验证和机制淘汰；旧 test 不再是未见证据；
2. **封存确认**：KDD Cup 2018 Fresh Air 北京+伦敦，先冻结 manifest、跨城市/未来时间切分和哈希；
3. **Venue gate**：若主张是通用 FL 方法而非环境应用，必须增加一个非空气质量多站点时序数据集；目标 venue 未定前该项保持开放；
4. **统计**：5 个冻结种子；按种子—站点—连续 24 小时块做 10,000 次配对 Bootstrap；次要比较 Holm 校正；
5. **压力场景**：自然季节漂移、预登记合成漂移、连续缺失、系统速度异质；掉线恢复与当前严格全客户端协议分开研究。

## 9. 消融与失败分析

必须包含：

1. 无安全聚合黑板反馈；
2. 只有学习率 cap 的当前实现；
3. 完整公共指令；
4. LLM no-probe；
5. probe + rule / bandit；
6. 无私有记忆；
7. 无 posterior credit；
8. 无 tail objective；
9. 固定总 epoch 的预算回放；
10. 错误公共指令、LLM 无效 JSON、探针误判和漂移误报案例。

## 10. 阶段门禁

### G0：研究合同冻结

- 本文件、强基线检索、基线资格表和胜出阈值进入 Git；
- 明确当前“只广播 lr cap”的实现缺口；
- 禁止在合同冻结前跑 P2 性能实验。

### G1：协同机制工程门禁

- 实现公共指令 Schema、严格 round 绑定、客户端消费和本地信用记录；
- 测试证明改变黑板输入会改变下一轮动作，但不泄漏 client/station 标识；
- 12 站 1 轮、再 3 轮 P1 nonformal smoke 完整通过。

### G2：强基线资格门禁

- A 级算法实现、合成等价测试、预算账本和配置冻结；
- B 级逐项记录 `compatible / adapted / incompatible`，不得静默删去强近邻。

### G3：P2 单种子 10 轮淘汰

- 完整方法优于 v1；
- 相对最强同动作空间非 LLM 控制器，macro MAE 至少改善 1%，或 macro 非劣 0.5% 且 station-CVaR 改善至少 3%；
- probe 预测与 realized gain 必须呈正相关，错误干预率低于 no-probe；
- 不通过则停止扩写 prompt，诊断机制或转向非 LLM MAS。

### G4：30 轮、多种子验证

- 冻结方法、prompt、动作、阈值、基线超参和种子；
- 达到第 1 节在开发验证上的显著性条件；
- 若只改善公平不改善精度，只能转为公平性主张，不能宣布完成当前 Goal。

### G5：封存确认

- 在未见确认集一次性运行冻结方法；
- 相对最强合格传统基线和 contextual bandit 均满足第 1 节条件；
- 完成后才允许写“超过强基线”的论文结论。

## 11. 失败转向

| 结果 | 结论 | 转向 |
|---|---|---|
| 超过传统基线但不超过 bandit | 自适应控制有效，LLM 价值未证实 | 以非 LLM evidence-grounded MAS 为主方法 |
| macro 非劣、CVaR 明显改善 | 公平价值成立，精度胜出未完成 | 转公平/风险敏感 FL 主张 |
| 只靠额外 probe/epoch 提升 | 中心机制失败 | 删除 LLM 增益主张，重做预算控制 |
| 只在北京数据有效 | 泛化不足 | 限定环境应用论文或增加跨域数据 |
| 黑板反馈无增益 | 协同机制未成立 | 回到独立本地控制器，停止使用“协同 MAS”表述 |
| 所有控制器均不优于强基线 | 当前方向未达到 Goal | 保留安全系统贡献，另立研究问题，不继续堆 prompt |

## 12. 开发任务队列

1. `C1`：`completed`，定义 `CohortDirective` 与固定广播 Schema；
2. `C2`：`completed`，让 rule/bandit/LLM proposer 同等消费公共指令；
3. `C3`：`completed_engineering`，扩展安全黑板的 probe calibration、tail/conflict/成本统计；
4. `C4`：`partial`，已加入 directive round/replay/privacy 回归与 continual AF/AP/AvgPerf 安全聚合评估 helper；仍需接入冻结的 (T_0+11) 任务调度；
5. `C5`：运行 P1 1/3 轮 smoke；
6. `B1`：实现 SCAFFOLD、FedDyn；
7. `B2`：实现/审计 Flash、q-FedAvg SecAgg 路径；
8. `B3`：完成 AAggFF、FedAWARE、FedAWA、Fed-TREND 资格判定；
9. `E1`：实现 CVaR、漂移恢复、probe 质量和预算账本；
10. `D1`：冻结 KDD Fresh Air manifest 与确认切分；
11. `X1`：按 G3--G5 执行实验，不越级运行正式 test。

当前下一项是 `C4/P1`：directive replay/privacy 回归与 12 站 1--3 轮 nonformal smoke，不是直接追求多 seed 结果。

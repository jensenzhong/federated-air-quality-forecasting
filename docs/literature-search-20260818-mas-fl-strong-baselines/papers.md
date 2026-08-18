# Literature Search: Secure-Aggregate Multi-Agent Federated Control

Date: 2026-08-18  
Search purpose: Goal 模式的近邻工作、强基线与协议资格核验  
Target family: generic CCF-A AI/ML or KDD; venue not frozen  
Source policy: official proceedings/OpenReview/CVF/PMLR/arXiv preferred; MDPI excluded

## Summary

- 最接近的 Agentic-FL 2026 仍是编排范式与路线图，未替代可执行的算法、同动作空间消融和强基线比较。
- 自适应权重、公平 sequential decision、漂移适应、动态客户端预算和超参调优均已有强工作；这些组件本身不能作为创新。
- 机会位于“服务器处于 SecAgg 可见性约束时的间接多智能体协同”。许多强自适应方法需要客户端级更新/效用，不能未经审计直接放入严格隐私主表。
- 主表应优先复现 FedProx、FedAdam、SCAFFOLD、FedDyn、q-FedAvg、Flash 及同动作空间 rule/bandit；FedAWARE/FedAWA/AAggFF/Fed-TREND 先做协议资格判定。

## Paper Table

| # | Work | Year / source | Type | Quality | What it covers | Consequence for this project |
|---|---|---|---|---|---|---|
| 1 | FedAvg | 2017 AISTATS/PMLR | pure method | A | 标准同步联邦平均 | 必须 sanity baseline |
| 2 | FedProx | 2020 MLSys | pure method | A | 本地 proximal 处理异质性 | 当前最强已验证基线之一 |
| 3 | SCAFFOLD | 2020 ICML/PMLR | pure method | A | control variate 降低 client drift | 必须新增的隐私兼容强基线 |
| 4 | Adaptive Federated Optimization | 2021 ICLR/OpenReview | pure method | A | FedAdam/FedYogi/FedAdagrad | FedAdam 必须公平调参 |
| 5 | FedDyn | 2021 ICLR/OpenReview | pure method | A | 动态正则处理异质目标 | 必须新增强基线 |
| 6 | q-FedAvg / Fair Resource Allocation | 2020 ICLR/OpenReview | pure method | A | 公平目标与客户端效用加权 | 公平主表基线；需 SecAgg 分子/分母审计 |
| 7 | FedEx | 2021 ICLR/OpenReview | pure method | A | 联邦超参数调优 | 动态 lr/epoch 不是新颖性 |
| 8 | FedMarl | 2022 AAAI | pure method | B | MARL 客户端选择与资源目标 | “多 agent + FL”已有直接先例 |
| 9 | Federated Learning under Distributed Concept Drift | 2023 AISTATS/PMLR | pure method | A | 不同客户端/时间的错位漂移 | 漂移协议与恢复指标必须对齐 |
| 10 | Flash | 2023 ICML/PMLR | pure method | A | client early stopping + drift-aware server optimizer | 漂移适应必须复现 |
| 11 | Delta-SGD | 2024 ICLR | pure method | A | 客户端自适应步长 | 独立动态 lr 不是核心创新 |
| 12 | FedCompass | 2024 ICLR/OpenReview | system/method | A | 按速度分配 local work | 动态 epoch/预算已有强先验 |
| 13 | AAggFF | 2024 ICML/PMLR | method + theory | A | 公平聚合的在线凸优化与 regret | 公平 sequential decision 强近邻；检查客户端级效用依赖 |
| 14 | Fed-TREND | 2024 arXiv | pure method | B/Risk | 合成知识载体处理联邦时序异质性 | 时序强近邻；需隐私、通信与代码审计 |
| 15 | FedAWARE | 2025 AISTATS/PMLR | method + theory | A | client consensus dynamics 与自适应权重 | 聚合权重主张的强基线/新颖性风险 |
| 16 | FedAWA | 2025 CVPR | pure method | A | 使用 client update vectors 自适应权重 | 与 aggregate-only 边界冲突，作资格审计 |
| 17 | Beijing 12-client continual forecasting benchmark | 2025 FLTA/IEEE + arXiv | pure benchmark | B/Risk | 同一数据集上的联邦连续时序与遗忘 | 空气质量/12站/漂移不再新颖 |
| 18 | Agentic Federated Learning | 2026 arXiv | roadmap/other | Risk | server/client LM agent 编排愿景 | 不能只靠“agentic”命名；需要算法和证据闭环 |
| 19 | Selective Collaboration for Robust FL | 2026 PMLR | method + theory | A | 基于相关性动态识别有益协作者 | 动态协作强近邻，但与严格全客户端/aggregate-only 有冲突 |

## Stable Sources

- [FedAvg](https://proceedings.mlr.press/v54/mcmahan17a.html)
- [SCAFFOLD](https://proceedings.mlr.press/v119/karimireddy20a.html)
- [Adaptive Federated Optimization](https://openreview.net/forum?id=LkFG3lB13U5)
- [FedDyn](https://openreview.net/forum?id=B7v4QMR6Z9w)
- [q-FedAvg](https://openreview.net/forum?id=ByexElSYDr)
- [FedEx](https://openreview.net/forum?id=p99rWde9fVJ)
- [FedMarl](https://ojs.aaai.org/index.php/AAAI/article/view/20894)
- [Distributed Concept Drift](https://proceedings.mlr.press/v206/jothimurugesan23a.html)
- [Flash](https://proceedings.mlr.press/v202/panchal23a.html)
- [Delta-SGD](https://proceedings.iclr.cc/paper_files/paper/2024/hash/d850b7e0cdc7f1c0820c6ad85405ae94-Abstract-Conference.html)
- [FedCompass](https://openreview.net/forum?id=msXxrttLOi)
- [AAggFF](https://proceedings.mlr.press/v235/hahn24a.html)
- [Fed-TREND](https://arxiv.org/abs/2411.15716)
- [FedAWARE](https://proceedings.mlr.press/v258/zeng25b.html)
- [FedAWA](https://openaccess.thecvf.com/content/CVPR2025/html/Shi_FedAWA_Adaptive_Optimization_of_Aggregation_Weights_in_Federated_Learning_Using_CVPR_2025_paper.html)
- [Beijing continual forecasting benchmark](https://arxiv.org/abs/2510.21491)
- [Agentic Federated Learning](https://arxiv.org/abs/2604.04895)
- [Selective Collaboration](https://proceedings.mlr.press/v328/tupitsa26a.html)

## Opportunity Map

| Cluster | Status | Open gap | Direction | Evidence needed |
|---|---|---|---|---|
| LLM/agent orchestration | crowded but algorithmically open | 缺同动作空间、强基线、可证伪机制 | probe-validated SecAgg+ group coordination | LLM vs bandit/rule/oracle |
| Adaptive weighting/fairness | covered central component | 强方法常依赖客户端级信号 | aggregate-blind adaptive control | 协议资格表 + 隐私兼容主表 |
| Concept drift | crowded but open | 检测/优化已有，安全协同不足 | 公共漂移任务 + 私有本地干预 | Flash 与 distributed drift 对照 |
| Federated time series | benchmark gap narrowing | 已有 Fed-TREND 与 Beijing benchmark | 跨城市封存确认与成本/公平协议 | KDD Fresh Air + manifest/hash |
| Selective collaboration | deployment/privacy gap | 相关性加权与 SecAgg 可见性冲突 | 不暴露身份的 SecAgg+ 群组协调 | 隐私边界审计与负迁移场景 |

## Citation And Positioning Cautions

- 不把 arXiv 路线图描述为同行评审算法结果；
- 不把 FedAWA/FedAWARE 的摘要性“superiority”当成本项目可比较数值；
- 不在不同模型、数据和隐私协议之间比较论文报告数字；
- Fed-TREND、Agentic-FL 及 2026 工作须在投稿前重新核验版本和 venue 状态；
- 论文 PDF 级实现细节、公开代码许可和超参数仍需在基线实现前逐项核验。

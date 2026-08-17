# 联邦学习 × 智能体控制：检索与筛选记录

检索日期：2026-08-17。模式：探索性创新检索；优先官方 proceedings、OpenReview、PMLR、CVF 与 arXiv 原文；MDPI 排除。该清单是设计证据，不等同于完整系统综述。

| 工作 | 年份/来源 | 已覆盖能力 | 对本项目的约束 | 链接 |
|---|---|---|---|---|
| FedEx: Federated Hyperparameter Tuning | 2021, OpenReview | 在相同训练预算下联邦调优 lr、epoch、batch、prox 等 | “LLM 动态调参”不是独立创新 | https://openreview.net/forum?id=p99rWde9fVJ |
| FedMarl | 2022, AAAI | MARL 多目标客户端选择，兼顾精度、延迟、通信 | MAS/资源调度已有直接先例 | https://ojs.aaai.org/index.php/AAAI/article/view/20894 |
| Federated Learning under Distributed Concept Drift | 2023, AISTATS | 客户端与时间上错位发生的概念漂移 | 漂移场景需与既有方法比较 | https://proceedings.mlr.press/v206/jothimurugesan23a.html |
| Flash | 2023, ICML | 基于参数更新检测漂移并进行漂移感知优化 | update norm/漂移分数单独不新颖 | https://proceedings.mlr.press/v202/panchal23a.html |
| Delta-SGD | 2024, ICLR | 客户端按局部光滑性自调步长 | 客户端独立 lr 已有理论与实验 | https://proceedings.iclr.cc/paper_files/paper/2024/hash/d850b7e0cdc7f1c0820c6ad85405ae94-Abstract-Conference.html |
| AAggFF | 2024, ICML | 将公平聚合建模为在线凸优化，给出 regret 界 | 公平权重和 sequential decision 已有强基线 | https://proceedings.mlr.press/v235/hahn24a.html |
| FedCompass | 2024, ICLR | 按客户端计算速度分配不同 local work | 动态 epoch/预算分配不是核心创新 | https://openreview.net/forum?id=msXxrttLOi |
| Fed-TREND | 2024, arXiv | 利用全局更新轨迹处理联邦时序异质性 | 时序更新轨迹是直接相关基线/特征先验 | https://arxiv.org/abs/2411.15716 |
| FedAWARE | 2025, AISTATS | client consensus dynamics 与自适应加权 | 简单动态聚合需面对理论更强的基线 | https://proceedings.mlr.press/v258/zeng25b.html |
| FedAWA | 2025, CVPR | 根据客户端更新向量自适应聚合权重 | update-vector weighting 不能包装成 LLM 创新 | https://openaccess.thecvf.com/content/CVPR2025/html/Shi_FedAWA_Adaptive_Optimization_of_Aggregation_Weights_in_Federated_Learning_Using_CVPR_2025_paper.html |
| Benchmarking CF in Federated Time Series Forecasting | 2025, FLTA/arXiv | 同一 Beijing Multi-site Air Quality、12 客户端、连续时序/遗忘 | 本项目必须超越“同数据集上的 FL 漂移应用” | https://arxiv.org/abs/2510.21491 |
| Agentic Federated Learning | 2026, arXiv | LM agents 作为 server/client 编排器、隐私与资源守护者 | 是最接近的概念先验；其愿景不能替代可验证算法 | https://arxiv.org/abs/2604.04895 |
| Selective Collaboration for Robust FL | 2026, PMLR | 动态选择有益协作者并给出理论 | 动态协作/负迁移控制需纳入相关工作 | https://proceedings.mlr.press/v328/tupitsa26a.html |

## 综合判断

高风险、应放弃的主张：

- “首个用 LLM 为 FL 调学习率/epoch 的方法”；
- “首次做客户端动态预算/公平聚合”；
- “首次在北京 12 站点研究联邦时序漂移”；
- “只因使用多个 agent 就构成 MAS 创新”。

仍有可检验空间：让 LLM 负责跨轮诊断与候选生成，但以客户端反事实探针给出行动前证据，以受约束执行器保证安全，并用同动作空间的 bandit/rule 对照识别 LLM 的真实边际贡献。

## 需要在投稿前继续完成

- 对 Agentic-FL、AO-FRL 等 2026 工作持续更新版本与同行评审状态；
- 在目标会议确定后按 venue 截止日期刷新最近 6 个月论文；
- 对最终纳入 Related Work 的论文逐篇核验 PDF 方法与实验设置，不能只依据摘要。

# Changelog

## Unreleased

- 为低内存顺序 runner 增加 `--preflight-only`：在调用 ServerApp 或开始客户端训练前验证站点数量、PAFA split、同进程 nonformal 边界、SecAgg+ 数值容量、最小群组、本地 LLM 端点与资源门禁，并只输出去标识化 JSON。
- 补充与 `pyproject.toml` 既有 MIT 声明一致的 `LICENSE`；当前全量验证为 `108 passed, 1 skipped`，ruff、mypy（51 个源码入口）和 `git diff --check` 通过。
- C1/C3：加入固定无身份 `CohortDirective`（phase/priority/动作许可/lr cap/上一轮绑定），由安全聚合摘要生成并在下一轮由 rule、contextual bandit、LLM proposer 共同消费；群组摘要新增探针收益、directive 合规、尾部风险和优先级对齐率。
- 增加 continual benchmark AF/AP/AvgPerf 评估与固定长度任务矩阵 codec；仅允许最小群组后的 SecAgg+ 聚合摘要进入服务器评估，尚未声称已复现论文结果。
- 审计用户提供的 benchmark notebook，并冻结其精确 base/base-test/11-task 时间边界；记录源码可复现性与 SecAgg+ 兼容性差异。
- C1--C3：补充 `CohortDirective` 篡改/重放、统一 proposer 消费、聚合摘要身份隔离与 directive 质量回归测试；全量测试 119 passed、1 skipped，mypy 扩展至 52 个源码入口。C4/P1 continual ClientApp 任务适配与真实 smoke 仍未完成。
- C4 partial：新增冻结 benchmark 的 T0/base-test/T1--T11 本地 cache 适配器、chronological 80/20 任务视图和客户端私有 `LocalContinualTaskLedger`；全量测试 125 passed、1 skipped，mypy 扩展至 53 个源码入口。尚未启动 continual ClientApp 训练或 P1 smoke。
- C4 partial：ClientApp 增加显式 `continual-enabled`/`continual-task-id` 门禁并在客户端本地选择当前任务训练窗口；默认 continual 仍关闭，真实 12 站任务调度尚未启动。
- C4：任务结束时的本地 task ledger 与固定长度归一化任务矩阵接入 PAFA SecAgg+；服务器仅在最终任务解码聚合 AF/AP/AvgPerf。新增 3-client synthetic continual SecAgg+ 回归和 12 轮 continual preflight（不启动训练）。
- P1 nonformal：修复 Flower SecAgg+ 参数路径的 tensor state-dict 数组键映射（named/numeric order），并通过真实 12 站 pafa_rule 1 轮与 3 轮 smoke；新增数组 schema 回归测试。该结果不作为正式隐私或性能证据。
- G2 contract：新增 `baseline_contract.py` 与 fail-closed 测试，分离算法 SecAgg 兼容性、当前实现状态、逐客户端信号依赖和预算类别；SCAFFOLD/FedDyn/Flash 等强基线在安全适配前不会进入正式比较。

## 0.3.0 - 2026-08-18

- 将 PAFA SecAgg+ 数值语义明确为：Flower 对客户端返回的加权完整模型参数逐坐标裁剪、量化和加掩码；该过程不是更新差分的 L2 裁剪，也不提供 DP 保证。
- 实现四阶段严格全客户端门禁：setup/share/collect/unmask 均要求完整唯一群组各请求和回复一次，任一缺失、重复、失败或集合不一致均使该轮 fail-closed。
- 增加 run/node/round/stage、collect method/server-round 和聚合轮次绑定，拒绝阶段乱序、会话重放、轮次跳跃以及重复或不完整代理身份。
- 增加 SecAgg+ 数值容量预检和聚合裁剪指示门禁；客户端只通过安全聚合报告完整模型参数可能触发裁剪的布尔量，群组指示发现任一违规时不进入协调器或有效工件。
- 增加 12 客户端合成量化误差回归、collect 阶段缺失回复 fail-closed 集成测试及相关身份/重放测试；2026-08-18 全量结果为 `105 passed, 1 skipped`，ruff、mypy 和 `git diff --check` 通过。
- 明确同进程顺序 smoke 仅为 nonformal 工程证据；正式部署仍缺机构签名或等价可信注册的 `Flower node -> physical station` 绑定。

## 0.2.0 - 2026-08-17

- 将 PAFA 提案、shadow probe、安全执行、状态记忆和信用分配全部下沉到 ClientApp 私有状态。
- 接入 Flower 1.22 `SecAggPlusWorkflow + secaggplus_mod`，模型与固定长度群组统计向量在同一安全聚合中求和。
- 客户端 SecAgg+ 包装器清除应用指标和真实样本数；服务器工件只保留群组摘要，旧逐客户端 AgenticAggregationStrategy 永久 fail-closed。
- 增加聚合级协调器、最小群组门禁、本地/机构 LLM 端点策略和正式 test/机构隔离防火墙。
- 增加客户端 SecAgg+ collect 阶段绑定，拒绝缺少协议记录的伪造 PAFA 请求；旧外部 LLM 开关不再能够放行公网客户端状态。
- 完成 3 客户端官方四阶段 SecAgg+ 合成集成测试；全套 `86 passed, 1 skipped`，ruff 与 mypy 通过；尚未运行 12 站性能实验。

## 0.1.2 - 2026-08-17

- 将 QFedAvg 的 Lipschitz 估计 `qffl_lr` 与本地 AdamW `lr` 解耦，默认冻结为 1.0。
- 旧 QFedAvg screening/30 轮确认标为实现缺陷，不改变其他方法的本地训练或确认结果。
- 增加 QFedAvg 步长回归测试，防止再次把客户端学习率传入 `client_learning_rate`。

## 0.1.1 - 2026-08-17

- 锁定 `tabulate` 报告依赖，报告状态改为只信任 validated 注册项。
- 修复 Flower SuperNode 多字段 `node-config` 引用，并确保资源门禁失败时不会启动任何 Flower 进程。
- 固定 12 客户端启动脚本使用项目 `.venv` 可执行文件，避免环境版本漂移。
- 增加验证集专用短基线 epoch 覆盖和 GRU screening 参数入口。
- 将预注册筛选展开为 13 项参数化队列，并使重复规划保持幂等。
- 完成 Seasonal Naive、Centralized GRU 与 Local GRU 的验证集 smoke；不将其作为正式结果。
- 验证12站 FedProx 低内存顺序一轮可行，并将后续主路线调整为同步顺序执行；全并发降为可选传输层检查。
- 增加正式顺序运行时、Flower 语义等价性和三轮资源基准门禁，当前 debug 结果继续保持 nonformal。

## 0.1.0 - 2026-08-13

- 建立独立 AQ-MAS-FL 子项目和工程配置。
- 增加严格时间切分、因果插补、31 维特征和 24→1 滑动窗口。
- 增加 GRU 主模型、MLP 消融和五类非联邦入口。
- 增加 Flower 1.22 ServerApp/ClientApp、FedAvg、FedProx、QFedAvg、FedAdam。
- 增加 Rule-MAS、DeepSeek LLM-MAS、决策缓存/回退/回放和有界聚合权重。
- 增加运行工件、注册表、校验、报告、统计与鲁棒性模块。
- 归档旧高速公路项目，删除指定旧 CSV/分布图并清理明文密钥。

所有性能结果仍为 `TBD`。
- 2026-08-17: Implemented the real Flower low-memory sequential runner with deterministic all-client scheduling, strict resource preflight, and per-client failure invalidation.

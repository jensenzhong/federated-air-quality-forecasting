# 设计决策日志

| 日期 | ID | 决策 | 原因 | 影响/回滚 |
|---|---|---|---|---|
| 2026-08-13 | D001 | 新建独立 `air_quality_fl` Git 项目 | 旧目录混合高速公路任务，不利于复现 | 旧项目归档，互不复用配置 |
| 2026-08-13 | D002 | 任务固定为 24→1 PM2.5 | 控制研究范围并适配小时数据 | 多步/多目标留待后续新协议 |
| 2026-08-13 | D003 | 严格时间切分 | 随机切分会泄漏未来分布 | 不兼容旧结果；旧结果仅审计 |
| 2026-08-13 | D004 | GRU 主模型、MLP 仅消融 | 保留时序结构并避免多架构寻优 | 若 GRU 实现失败先修复，不改主模型 |
| 2026-08-13 | D005 | 站点宏 MAE 为主指标 | MAPE 对零值不稳，宏平均体现站点公平 | 仍报告微指标但不用于主选择 |
| 2026-08-13 | D006 | 12 客户端权重界 [0.04,0.16] | 旧 3 客户端界不可用于 12 站 | 客户端数变化需新协议验证可行性 |
| 2026-08-13 | D007 | LayerNorm 替代 BatchNorm | 小批次/站点分布差异下更稳定 | 架构冻结后不再变更 |
| 2026-08-13 | D008 | LLM 正式失败即 invalid | 回退会混淆完整 LLM 方法估计 | 非正式调试允许标记 fallback |
| 2026-08-13 | D009 | 增加 FedProx-budget-matched | 隔离 LLM 动态 epoch/lr 的额外计算 | 必须由配对 MAS 决策轨迹回放 |
| 2026-08-13 | D010 | 隐私主张限定为不传原始行 | 当前没有 DP/密码学证明 | 禁止使用“隐私保护已证明”表述 |
| 2026-08-13 | D011 | 资源不足立即停止 | 防止静默降低客户端/并发改变实验 | 资源恢复后使用原协议重跑 |
| 2026-08-17 | D012 | 报告状态只信任实验注册表 | 防止旧 summary 或手工报告把 completed/invalid 运行展示为正式证据 | 所有报告只读取 `validated` 注册项；空注册表明确输出无有效结果 |
| 2026-08-17 | D013 | Flower 启动固定使用项目虚拟环境并显式检查资源门禁退出码 | PowerShell 不会默认把原生进程非零退出码转为异常，旧脚本可能在门禁失败后继续启动 | 缺依赖或门禁失败时在创建 SuperLink/SuperNode 前终止 |
| 2026-08-17 | D014 | Screening 队列按预注册网格展开为 13 项 | 原队列忽略 YAML 中的模型和联邦超参数，无法产生可审计筛选 | 4 个 GRU 组合、3 个 FedProx、3 个 QFedAvg、3 个 FedAdam；仅验证集使用 |
| 2026-08-17 | D015 | 主线采用12客户端同步顺序执行，全并发降为可选检查 | FedProx 12站1轮在约0.40 GB训练进程内存和1.9分钟内完成；并发度不改变同步联邦算法定义 | 协议冻结前实现真实 ClientApp/Strict Strategy 顺序入口并通过等价测试；失败则回滚为资源阻塞，不使用当前 debug 结果 |
| 2026-08-17 | D016 | W1/W2 通过后将严格顺序 Flower 入口作为 screening 默认执行器 | 2/12 客户端 FedAvg/FedProx 等价性全通过；12 客户端 FedProx 3轮约5.48分钟、峰值 RSS 0.392GB | screening 仍只使用验证集；若队列估算超过7天，记录资源阻塞，不减少预注册样本 |
| 2026-08-17 | D017 | W3 筛选固定为 seed42/验证集；联邦3轮、local epoch=1；集中式 GRU 1 epoch | 筛选应是低成本参数代理，30轮/完整训练留给胜者确认；资源基准支持约49分钟的9项联邦筛选排程估算 | 不改变候选范围；所有胜者必须30轮确认后才可冻结协议；test 仍不可访问 |
| 2026-08-17 | D018 | W3 13项筛选完成；各 family 暂选集中式 GRU(hidden=64, lr=0.0005)、FedProx(μ=0.001)、QFedAvg(q=0.1)、FedAdam(server-lr=0.01) | 仅按预注册验证集 `macro_mae`，再以最差站点 MAE、站点 MAE CV、耗时打破平局；完整候选表见 `artifacts/reports/screening_results.md` | 结果仅为 nonformal 代理；不访问 test、不冻结协议；四个候选各运行30轮确认后再决定架构 |
| 2026-08-17 | D019 | W4 先完成 Rule-MAS→FedProx budget replay 的真实顺序回放，再等待严格 LLM-MAS 凭据 | 两个 3 轮运行均 12/12 客户端成功，3 条决策的 `(round, lr_scale, local_epochs)` 完全一致，动态权重在登记边界内 | 没有新 `DEEPSEEK_API_KEY` 时严格 LLM 运行必须 invalid，禁止用 fallback 冒充正式 LLM；W4 暂为 partial pass |
| 2026-08-17 | D020 | FedProx μ=0.001 进入 30 轮验证确认；中断工件不计入证据 | 完整确认 run `fedprox-42-20260817T052846Z-8151de3` 30/30 轮、12/12 客户端成功，验证 macro-MAE 9.4694；峰值 RSS 0.391GB，最低可用内存观测 0.1465GB | 仍为 nonformal 验证结果；其余 family/MAS 候选需同预算确认，资源下限列入风险，不降低客户端或轮数 |
| 2026-08-17 | D021 | 集中式 GRU(hidden=64, lr=0.0005) 完成 30 epoch 验证确认 | Run `centralized_gru-42-20260817T062621Z-8151de3` 的验证 macro-MAE 9.3691，耗时 903.23 秒；起止可用内存已记录 | 现有集中式 runner 没有峰值 RSS 采样；正式冻结前补 telemetry 或在限制中明确说明；test 仍不可访问 |
| 2026-08-17 | D022 | QFedAvg q=0.1 完成 30 轮验证确认并记录为稳定负结果 | Run `qfedavg-42-20260817T064304Z-8151de3` 30/30 轮、12/12 客户端成功，macro-MAE 58.5003；峰值 RSS 0.3903GB | 不提前删除预注册负结果；不进入主候选；最低可用内存观测 0.3197GB，继续记录资源风险 |
| 2026-08-17 | D023 | FedAdam server-lr=0.01 完成 30 轮验证确认 | Run `fedadam-42-20260817T074036Z-8151de3` 30/30 轮、12/12 客户端成功，验证 macro-MAE 9.8053；峰值 RSS 0.3937GB，最低可用内存观测 0.5237GB | 结果仍为 protocol_frozen=false 的 nonformal 验证；Rule-MAS 与预算匹配需完成同预算确认，集中式 runner 峰值遥测仍待补齐 |
| 2026-08-17 | D024 | Rule-MAS 与 FedProx budget replay 完成 30 轮配对确认 | Rule-MAS run `rule_mas-42-20260817T085700Z-8151de3` 的验证 macro-MAE 9.6548；预算回放 run `fedprox_budget_matched-42-20260817T103434Z-8151de3` 的最佳验证 macro-MAE 9.4665；两者 30/30 条 `(round, lr_scale, local_epochs)` 一致 | 两个结果均保持 nonformal、未读取 test；严格 LLM-MAS 仍需新 API key，协议冻结前补齐集中式峰值遥测 |
| 2026-08-17 | D025 | QFedAvg 的 Flower `client_learning_rate` 与本地 AdamW `lr` 解耦，冻结 `qffl_lr=1.0` | 旧接线把 AdamW `lr=0.001` 当作 q-FFL 的 Lipschitz η，导致 `L=1000`、30 轮全局权重几乎不动（init L2=0.015，macro-MAE 58.50）。本地训练、模型、切分、q 搜索网格和其他方法均不变 | 旧 QFedAvg screening/确认标为实现缺陷，不进入主表；只重跑 QFedAvg 家族：先 3 轮重筛 `q∈{0.1,1,5}`，再 30 轮确认胜者。回滚：恢复 `client_learning_rate=base_lr` |
| 2026-08-17 | D026 | 严格 LLM-MAS 30 轮确认完成，但验证集暂不支持优于预算匹配基线的中心主张 | Run `mas_llm-42-20260817T114758Z-8151de3` 30/30 轮、12/12 客户端成功，best validation macro-MAE 9.6702；决策来源 24 llm/4 rule/2 cache/0 fallback；预算匹配 FedProx best validation macro-MAE 9.4665 | 暂不删除或改写中心主张；QFedAvg 先按 D025 修复后重筛，协议冻结和 test 评估前保持中性，必须报告 LLM-MAS 资源风险（最低可用内存 0.3675GB） |
| 2026-08-17 | D027 | 按 D025 使用 qffl_lr=1.0 重筛并确认 QFedAvg q=0.1 | 修复 q-FFL Lipschitz 学习率与本地学习率混用后，30 轮验证集 macro-MAE 为 9.7939；旧 58.5003 运行继续保持实现缺陷状态，不用于方法结论 | 修正版结果替代旧 QFedAvg 确认记录但仍为 nonformal；协议冻结前不读取 test，峰值 RSS 0.3902GB、最低可用内存 0.1483GB 列入资源风险 |
| 2026-08-17 | D028 | 集中式 runner 增加进程级 RSS/可用内存采样并重跑 30 epoch 确认 | 新 run `centralized_gru-42-20260817T135100Z-8151de3` 记录 1533 个采样点，峰值 RSS 1.4936GB、最低可用内存 0.3211GB，验证 macro-MAE 9.3691 | 集中式峰值遥测门禁完成；结果仍为验证集 nonformal，协议冻结前不读取 test |
| 2026-08-17 | D029 | 完成 seed=42 确认工件审计并冻结 test 前协议 | 7 个候选运行必需工件完整、配置/manifest 可复核、低内存和决策门禁通过；冻结记录写入协议（Git 基线 8151de3、base config SHA-256） | 协议状态改为 `frozen_pre_test`；后续只能新建 test 运行 ID，禁止用 test 反向选参 |
| 2026-08-17 | D030 | 在冻结协议下对 7 个 seed=42 验证最佳 checkpoint 各执行一次 test 评估 | 7 个新 test 工件均通过 `validate_artifact_directory`；test macro-MAE 已登记，评估过程不训练、不用 test 选 checkpoint | 标记 seed=42 test 为 validated；五种子与配对统计仍未开始，中心主张保持未测试 |
| 2026-08-17 | D031 | 暂停 v1 多 seed 队列并启动 LLM-MAS v2 重设计 | v1 验证中 LLM-MAS 9.6702 未优于预算匹配 FedProx 9.4665；30轮中 27 次 hybrid 且预算始终 `(1,1)`，表明动作塌缩 | v1 工件全部保留；中断的 centralized seed123 标 invalid；不得续跑 v1 正式队列 |
| 2026-08-17 | D032 | 将 v2 定义为经局部反事实探针验证的智能体联邦控制，而非动态调参 | FedEx、Delta-SGD、FedCompass、AAggFF、FedAWARE 等已覆盖调参、客户端步长、预算和公平权重；纯扩充 prompt 无法建立创新边界 | 实现同动作空间的 rule/bandit/LLM 对照、probe、安全执行和后验信用；详见 `docs/10_llm_mas_v2_research_redesign.md` |
| 2026-08-17 | D033 | seed42 原 test 对 v2 降级为开发审计证据 | 该结果已被观察并用于本次方法设计，不能继续声称为 v2 未见测试 | v2 最终确认必须使用新增封存留出；首选 KDD Cup 2018 北京+伦敦数据，先冻结 manifest/切分哈希 |
| 2026-08-17 | D034 | 否决服务器端逐客户端 LLM 编排作为正式 v2 架构 | 单客户端指标、更新方向和 probe 结果进入服务器/外部 LLM 仍可产生隐私推断风险，与强隐私联邦边界冲突 | 状态胶囊和详细 probe 留在客户端；协调 LLM 只看达到群组阈值的安全聚合摘要 |
| 2026-08-17 | D035 | 所有现有 `pafa_*` 运行在 SecAgg+ 接入并验证前 fail-closed | 当前 `Strategy.start` 能看到逐客户端明文回复，不能用 TLS 或“不传原始行”替代安全聚合主张 | 保留原型代码供重构和单元测试；不得运行 smoke、val 或 test；解除门禁需要安全聚合与隐私日志测试证据 |
| 2026-08-17 | D036 | PAFA 改用客户端本地 agent 与 Flower SecAgg+ 聚合路径，旧服务器逐客户端策略永久禁用 | 官方 SecAgg+ 四阶段合成测试证明只能恢复模型与固定长度群组向量；客户端包装器清除应用指标和真实样本数，聚合协调器不读取身份或单站点轨迹 | 只解除 nonformal 验证 smoke 的工程阻断；12 站 P1、量化/重放/缺失测试未完成，test 评估与同进程正式运行继续 fail-closed |

后续条目必须记录提出者、批准日期、受影响配置/运行和是否需要协议版本升级。

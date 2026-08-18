# AQ-MAS-FL

基于 Flower 的北京多站点空气质量联邦多智能体协同预测研究工程。项目将 UCI Beijing Multi-Site Air Quality Data 的 12 个监测站映射为 12 个联邦客户端，使用过去 24 小时的污染物、气象和时间特征预测未来 1 小时 PM2.5。

本仓库强调可复现、无时间泄漏和可审计。v2 PAFA 已接入 Flower Secure Aggregation（安全聚合），并通过 3 客户端协议闭环与 12 客户端量化/缺失/重放/身份合成工程门禁；尚未完成 12 站真实 smoke 或跨机构隔离。项目不提供差分隐私，也不声称形式化 DP 保证。

## 当前状态

- 工程脚手架、数据管线、GRU/MLP、基线、Flower ServerApp/ClientApp、固定联邦策略、Rule-MAS、LLM-MAS、审计工件与统计模块已实现。
- v2 将站点 agent、prompt、probe 和记忆保留在 ClientApp，本地更新与群组统计通过安全聚合协议处理；旧服务器逐客户端 agent 路径已永久禁用。P1 隐私预检已通过，下一步仅允许 nonformal 的 12 站同进程 smoke；正式 test 和多 seed 仍处于隐私门禁阻塞状态。
- 12 站 FedProx 严格顺序 Flower 入口已通过等价性门禁，并完成 3 轮低内存资源基准；13 项验证集 screening 已完成，但候选仍需 30 轮确认、协议冻结和 `validate_run`。
- 真实数据缓存、测试和静态检查需按 `PROJECT_STATUS.md` 的证据状态确认。
- screening 报告为 nonformal 验证证据，见 `artifacts/reports/screening_results.md`；正式实验结果仍全部保持 `TBD`，只有 `validated` 运行可进入正式报告。
- 实验方法按四层展示：传统联邦主表、同动作空间客户端控制器、机制消融、持续学习独立参照。代码里的 `pafa_*` 只是内部运行 ID；论文显示名、主表和消融关系见 [`docs/14_experiment_comparison_map.md`](docs/14_experiment_comparison_map.md)。

## 环境

- Python 3.12
- PyTorch 2.7
- Flower 1.22
- Windows PowerShell 7 或 Windows PowerShell 5.1

```powershell
cd "D:\科研文档联邦学习\federate_easy -data2\air_quality_fl"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
Copy-Item .env.example .env
```

仓库同时提供 `uv.lock` 作为完整传递依赖锁；推荐复现时使用 `uv sync --extra dev --frozen`。`requirements.lock` 保留为直接依赖的精确约束清单。

设置原始数据目录。原始 CSV 保留在下载目录，不复制到仓库：

```powershell
$env:BEIJING_AQ_DATA_DIR = "C:\Users\23079\Downloads\beijing+multi+site+air+quality+data"
```

只有运行 LLM-MAS 时才设置：

```powershell
$env:DEEPSEEK_API_KEY = "<在本机环境中设置，不写入文件>"
```

旧项目中出现过的密钥必须在服务商控制台撤销；代码清理无法替代服务商侧撤销。

## 快速验证

```powershell
python -m aqfl.data.prepare --config configs/base.yaml
pytest
ruff check .
mypy aqfl
```

非联邦基线：

```powershell
python -m aqfl.experiments.run_baseline --method persistence
python -m aqfl.experiments.run_baseline --method seasonal_naive
python -m aqfl.experiments.run_baseline --method centralized_gru
```

验证集短基线可使用显式 epoch 覆盖；该选项禁止用于冻结协议或测试集：

```powershell
python -m aqfl.experiments.run_baseline --method centralized_gru --split val --max-epochs 1
python -m aqfl.experiments.run_baseline --method local_gru --split val --max-epochs 1
```

Flower 运行：

```powershell
flwr run . local-12 --run-config "method=fedprox seed=42"
```

全并发 12 客户端目前只保留为可选传输层检查；启动前会检查至少 13 个逻辑处理器和 10 GB 可用内存。独立进程入口为：

```powershell
.\scripts\launch_flower_12_clients.ps1 -Method fedprox -Seed 42 -Rounds 1
```

低内存可行性入口会保持12站全部参与并逐客户端执行：

```powershell
python scripts/run_flower_sequential.py --method pafa_rule --rounds 1 --preflight-only
python scripts/run_flower_sequential.py --method fedprox --rounds 1 --seed 42
```

`--preflight-only` 会在调用 ServerApp 或启动客户端训练前退出，并输出不含站点名称的资格报告。实际运行入口直接调用真实 Flower `ClientApp`/`Strict Strategy`，但当前仍属于验证集工程 smoke；正式资格见 `docs/03_experiment_protocol.md` 的顺序运行时等价门禁。`scripts/run_sim.py` 保留为早期资源可行性对照。

实验队列与结果审计：

```powershell
python -m aqfl.experiments.sweep --plan configs/experiments/formal.yaml
python -m aqfl.reporting.validate_run --run-dir artifacts/runs/<run_id>
python -m aqfl.reporting.build_report --registry artifacts/experiment_registry.csv
```

自动报告会同时显示比较组、方法层次、论文显示名和内部运行 ID。不同比较组不直接排名：例如持续学习的 AF/AP/AvgPerf 不能与一步预测的 MAE 混在一张表里；`pafa_llm` 是否真的有额外价值，只能与同状态、同动作、同探针预算的规则/试错控制器比较。

队列命令默认只生成计划；显式传入 `--execute` 才会执行。长实验必须遵循 `docs/03_experiment_protocol.md` 的冻结和资源门禁。
参数筛选使用 `--stage screening`，生成 4 个 GRU、3 个 FedProx、3 个 QFedAvg 和 3 个 FedAdam 任务；筛选固定为 seed=42、验证集、联邦 3 轮/本地 1 epoch、集中式 GRU 1 epoch。执行前同样必须满足 Flower 资源门禁。

筛选完成后可用 `python scripts/summarize_screening.py` 重新匹配 13 个运行工件并生成验证集选择报告；该脚本会拒绝读取 test 指标。Rule-MAS 与预算匹配回放门禁报告见 `artifacts/reports/mas_gate_rule_budget.md`。

## 数据与输出边界

- `data/cache/`：派生窗口缓存，不提交 Git。
- `artifacts/runs/`：逐次运行工件，不提交 Git。
- `artifacts/llm_cache/`：LLM prompt/响应缓存，不提交 Git。
- `data/manifest.json`：原始文件哈希和准备状态，可提交用于审计。
- 每次运行使用 `<method>-<seed>-<UTC时间>-<Git短提交号>`，禁止覆盖。

完整的研究问题、切分、方法、统计检验和结果解释边界见 `docs/`。项目状态见 `PROJECT_STATUS.md`，设计变更只写入 `docs/07_decision_log.md`。

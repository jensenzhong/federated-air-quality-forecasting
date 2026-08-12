# AQ-MAS-FL

基于 Flower 的北京多站点空气质量联邦多智能体协同预测研究工程。项目将 UCI Beijing Multi-Site Air Quality Data 的 12 个监测站映射为 12 个联邦客户端，使用过去 24 小时的污染物、气象和时间特征预测未来 1 小时 PM2.5。

本仓库强调可复现、无时间泄漏和可审计。当前代码不提供差分隐私、密码学安全聚合或真实跨机构隔离；准确表述是：联邦训练通信不传输原始数据行。

## 当前状态

- 工程脚手架、数据管线、GRU/MLP、基线、Flower ServerApp/ClientApp、固定联邦策略、Rule-MAS、LLM-MAS、审计工件与统计模块已实现。
- 真实数据缓存、测试和静态检查需按 `PROJECT_STATUS.md` 的证据状态确认。
- 正式实验结果全部保持 `TBD`，只有 `validated` 运行可进入报告。

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

Flower 运行：

```powershell
flwr run . local-12 --run-config "method=fedprox seed=42"
```

全并发 12 客户端启动前会检查至少 13 个逻辑处理器和 10 GB 可用内存；不满足时直接失败，不会静默降低并发或客户端数。独立进程启动入口为：

```powershell
.\scripts\launch_flower_12_clients.ps1 -Method fedprox -Seed 42 -Rounds 1
```

实验队列与结果审计：

```powershell
python -m aqfl.experiments.sweep --plan configs/experiments/formal.yaml
python -m aqfl.reporting.validate_run --run-dir artifacts/runs/<run_id>
python -m aqfl.reporting.build_report --registry artifacts/experiment_registry.csv
```

队列命令默认只生成计划；显式传入 `--execute` 才会执行。长实验必须遵循 `docs/03_experiment_protocol.md` 的冻结和资源门禁。

## 数据与输出边界

- `data/cache/`：派生窗口缓存，不提交 Git。
- `artifacts/runs/`：逐次运行工件，不提交 Git。
- `artifacts/llm_cache/`：LLM prompt/响应缓存，不提交 Git。
- `data/manifest.json`：原始文件哈希和准备状态，可提交用于审计。
- 每次运行使用 `<method>-<seed>-<UTC时间>-<Git短提交号>`，禁止覆盖。

完整的研究问题、切分、方法、统计检验和结果解释边界见 `docs/`。项目状态见 `PROJECT_STATUS.md`，设计变更只写入 `docs/07_decision_log.md`。

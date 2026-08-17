# 复现运行手册

## 1. 干净环境

```powershell
git status --short
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

若已安装 uv，优先执行 `uv sync --extra dev --frozen`，由 `uv.lock` 恢复完整传递依赖图。

记录 `pip freeze`、Python、平台、CPU、内存和 Git 提交；运行模块会自动写入 `environment.json`。

## 2. 数据准备

```powershell
$env:BEIJING_AQ_DATA_DIR = "<包含12个PRSA_Data_*.csv的上级目录>"
python -m aqfl.data.prepare --config configs/base.yaml
```

核对 `data/manifest.json` 为 prepared、station_count=12、total_rows=420768；核对 `data/cache/metadata.json` feature_dim=31；运行全数据测试。原始文件哈希变化时不得复用旧缓存。

## 3. 工程验证

```powershell
ruff check .
mypy aqfl
pytest --cov=aqfl --cov-report=term-missing
```

## 4. 基线和低内存 Flower 路线

先跑确定性基线，再跑短学习基线；主实验不得提前访问 test。Flower 前执行：

```powershell
python -m aqfl.experiments.run_baseline --method seasonal_naive --split val
python -m aqfl.experiments.run_baseline --method centralized_gru --split val --max-epochs 1
python -m aqfl.experiments.run_baseline --method local_gru --split val --max-epochs 1
```

`--max-epochs`、`--hidden-size` 和 `--learning-rate` 是验证集 smoke/screening 专用参数，代码会拒绝将它们用于测试集或冻结协议。

生成预注册筛选计划：

```powershell
python -m aqfl.experiments.sweep --plan configs/experiments/formal.yaml --stage screening
```

该命令生成 13 项参数化任务但不执行；重复生成同一计划不会覆盖已取得进展的任务。当前筛选预算固定为 seed=42、验证集、联邦3轮/本地1 epoch、集中式 GRU 1 epoch；W1/W2 和 W3 设计复核已通过。

当前可复现的低内存可行性命令：

```powershell
python scripts/run_sim.py fedprox 42 1
```

该命令使用12个站点、逐客户端训练、同步聚合，只允许 `fedavg`/`fedprox`，并强制写入 `evaluation_split=val`、`protocol_frozen=false`、`formal_eligible=false`。它保留为早期资源可行性对照，不替代严格 Flower 入口。

W1 严格顺序 Flower 入口已实现，仍保持12站点全部参与，并直接调用真实 `ClientApp` 与 `Strict Strategy`：

```powershell
python scripts/run_flower_sequential.py --method fedprox --rounds 1 --seed 42
```

该入口不启动 Ray 或多进程；每轮按 `data/cache/metadata.json` 的站点顺序逐客户端执行。当前运行仍需通过 W2 等价性与三轮资源门禁后，才能进入正式筛选。

全并发多进程只作为可选传输层检查。执行前可运行：

```powershell
python -c "from aqfl.config import load_config; from aqfl.federated.resources import enforce_resource_gate; enforce_resource_gate(load_config('configs/base.yaml'))"
```

不满足该门槛时跳过全并发检查，不再阻塞低内存主线。严格顺序入口完成后按“1轮等价性→3轮资源基准→screening→单种子30轮”推进。LLM-MAS 需通过环境变量设置新密钥，先用缓存/规则完成非网络测试。

## 5. 运行验证

每个运行目录应包含 resolved config、dataset manifest、environment、round/client parquet、predictions、decisions、system metrics、checkpoint、summary。执行 `validate_run` 后，注册表状态才能转为 validated；invalid 运行保留但报告器跳过。

协议冻结后的 test 只允许从验证集已选 checkpoint 做一次性评估，避免把 test 指标送回每轮策略选择。使用 `scripts/evaluate_frozen_checkpoints.py --source-run-id <val-run-id>` 生成新的 `evaluation_split=test` 工件；源运行必须是 completed、val-only 且 `protocol_frozen=false`，生成工件必须通过 `validate_artifact_directory` 后才能登记为 validated。

## 6. 恢复与重跑

- 配置变化：新运行 ID。
- API 临时失败：原运行 invalid，使用相同配置和缓存重跑。
- 代码错误：登记决策日志、修复、加回归测试、新运行 ID。
- 数据哈希变化：删除派生缓存后重新准备；不得覆盖旧 manifest 的历史副本（运行目录中已有快照）。

## 7. 论文复核包

保留正式提交所用 Git 提交、lock 文件、协议冻结哈希、validated 注册表、逐样本预测、统计脚本输出、表图生成日志和限制说明。性能数字只由程序从注册表构建。

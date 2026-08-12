# 复现运行手册

## 1. 干净环境

```powershell
git status --short
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

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

## 4. 基线和 Flower

先跑确定性基线，再跑短学习基线；主实验不得提前访问 test。Flower 前执行：

```powershell
python -c "from aqfl.config import load_config; from aqfl.federated.resources import enforce_resource_gate; enforce_resource_gate(load_config('configs/base.yaml'))"
```

不满足资源门槛时停止。满足后执行 1 轮、3 轮、再 30 轮。LLM-MAS 需通过环境变量设置新密钥，先用缓存/规则完成非网络测试。

## 5. 运行验证

每个运行目录应包含 resolved config、dataset manifest、environment、round/client parquet、predictions、decisions、system metrics、checkpoint、summary。执行 `validate_run` 后，注册表状态才能转为 validated；invalid 运行保留但报告器跳过。

## 6. 恢复与重跑

- 配置变化：新运行 ID。
- API 临时失败：原运行 invalid，使用相同配置和缓存重跑。
- 代码错误：登记决策日志、修复、加回归测试、新运行 ID。
- 数据哈希变化：删除派生缓存后重新准备；不得覆盖旧 manifest 的历史副本（运行目录中已有快照）。

## 7. 论文复核包

保留正式提交所用 Git 提交、lock 文件、协议冻结哈希、validated 注册表、逐样本预测、统计脚本输出、表图生成日志和限制说明。性能数字只由程序从注册表构建。

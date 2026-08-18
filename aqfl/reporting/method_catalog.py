"""Paper-facing method names and comparison groups.

The executable method IDs are intentionally kept stable because they are part
of run directories and artifact validation.  This catalog is the single place
where those IDs acquire a paper-facing name and a comparison role.  It keeps
algorithmic mechanism, secure transport, and continual-learning references
from being mixed into one undifferentiated method list.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodInfo:
    method_id: str
    display_name: str
    comparison_group: str
    layer: str
    role: str


_CATALOG: dict[str, MethodInfo] = {
    # Sanity and centralized references.  These are not federated method
    # competitors and should never be used as the main superiority baseline.
    "persistence": MethodInfo("persistence", "持久性预测", "非联邦参考", "预测参考", "sanity"),
    "seasonal_naive": MethodInfo("seasonal_naive", "季节性朴素预测", "非联邦参考", "预测参考", "sanity"),
    "local_gru": MethodInfo("local_gru", "单站点本地 GRU", "非联邦参考", "预测参考", "sanity"),
    "centralized_gru": MethodInfo("centralized_gru", "集中式 GRU", "非联邦参考", "预测参考", "sanity"),
    # Conventional federated algorithms.  The non-pafa IDs are retained as
    # reference implementations but need a SecAgg+ adapter before formal use.
    "fedavg": MethodInfo("fedavg", "平均聚合（FedAvg）", "传统联邦参考", "联邦优化", "traditional"),
    "fedprox": MethodInfo("fedprox", "稳定本地训练（FedProx）", "传统联邦参考", "联邦优化", "traditional"),
    "fedadam": MethodInfo("fedadam", "自适应服务器更新（FedAdam）", "传统联邦参考", "联邦优化", "traditional"),
    "qfedavg": MethodInfo("qfedavg", "公平性加权聚合（q-FedAvg）", "传统联邦参考", "联邦优化", "traditional"),
    "scaffold": MethodInfo("scaffold", "控制变量校正（SCAFFOLD）", "待协议审计的传统参考", "联邦优化", "pending"),
    "feddyn": MethodInfo("feddyn", "动态正则校正（FedDyn）", "待协议审计的传统参考", "联邦优化", "pending"),
    "flash": MethodInfo("flash", "漂移感知更新（FLASH）", "待协议审计的传统参考", "联邦优化", "pending"),
    # Verified SecAgg+ transport paths.  pafa_ is an implementation prefix,
    # not a paper-facing algorithm name.
    "pafa_fedavg": MethodInfo("pafa_fedavg", "平均聚合（FedAvg）", "传统联邦主表", "联邦优化", "secure_traditional"),
    "pafa_fedprox": MethodInfo("pafa_fedprox", "稳定本地训练（FedProx）", "传统联邦主表", "联邦优化", "secure_traditional"),
    "pafa_fedadam": MethodInfo("pafa_fedadam", "自适应服务器更新（FedAdam）", "传统联邦主表", "联邦优化", "secure_traditional"),
    "pafa_fedprox_budget_matched": MethodInfo(
        "pafa_fedprox_budget_matched",
        "预算匹配的稳定本地训练",
        "同动作空间控制器对比",
        "联邦优化",
        "budget_control",
    ),
    # All of these use the same local state schema, action set, probe budget,
    # and group-summary path.  They are the only fair group for attributing a
    # difference to the proposer/controller itself.
    "pafa_rule": MethodInfo("pafa_rule", "规则控制器", "同动作空间控制器对比", "客户端控制器", "controller"),
    "pafa_bandit": MethodInfo("pafa_bandit", "试错控制器", "同动作空间控制器对比", "客户端控制器", "controller"),
    "pafa_llm": MethodInfo("pafa_llm", "本地大模型控制器", "同动作空间控制器对比", "客户端控制器", "proposed"),
    "pafa_llm_no_probe": MethodInfo("pafa_llm_no_probe", "本地大模型控制器（无探针）", "控制器消融", "客户端控制器", "ablation"),
    "pafa_probe_oracle": MethodInfo("pafa_probe_oracle", "探针上界控制器", "机制上界对比", "客户端控制器", "oracle"),
    "pafa_bandit_fedadam": MethodInfo("pafa_bandit_fedadam", "试错控制器 + 自适应服务器更新", "开发候选（不进主表）", "组合候选", "rejected_development"),
    # Historical v1 IDs remain readable in old artifacts but are not silently
    # promoted to the v2 main table.
    "rule_mas": MethodInfo("rule_mas", "旧版规则多智能体", "历史 v1 参考", "历史控制器", "historical"),
    "mas_llm": MethodInfo("mas_llm", "旧版大模型多智能体", "历史 v1 参考", "历史控制器", "historical"),
    "mas_llm_dynamic_only": MethodInfo("mas_llm_dynamic_only", "旧版大模型（仅动态）", "历史 v1 参考", "历史控制器", "historical"),
    "mas_llm_no_fairness": MethodInfo("mas_llm_no_fairness", "旧版大模型（无公平观测）", "历史 v1 参考", "历史控制器", "historical"),
    "fedprox_budget_matched": MethodInfo("fedprox_budget_matched", "旧版预算匹配稳定训练", "历史 v1 参考", "历史控制器", "historical"),
}


def method_info(method_id: str) -> MethodInfo:
    """Return paper-facing metadata, with a safe fallback for new IDs."""

    key = str(method_id).strip().lower()
    if key in _CATALOG:
        return _CATALOG[key]
    return MethodInfo(key, key or "未命名方法", "未分类（需登记）", "未分类", "unknown")


def method_catalog() -> dict[str, MethodInfo]:
    """Return a copy so callers cannot mutate the canonical catalog."""

    return dict(_CATALOG)

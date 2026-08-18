from __future__ import annotations

from aqfl.reporting.method_catalog import method_info


def test_catalog_separates_paper_name_from_internal_id() -> None:
    info = method_info("pafa_llm")
    assert info.display_name == "本地大模型控制器"
    assert info.comparison_group == "同动作空间控制器对比"
    assert info.method_id == "pafa_llm"

    traditional = method_info("pafa_fedadam")
    assert traditional.display_name == "自适应服务器更新（FedAdam）"
    assert traditional.comparison_group == "传统联邦主表"


def test_unknown_method_is_readable_and_flagged() -> None:
    info = method_info("new_experiment")
    assert info.display_name == "new_experiment"
    assert info.comparison_group == "未分类（需登记）"

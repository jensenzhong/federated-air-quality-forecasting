# Search Notes

Date: 2026-08-18

## Purpose

为 Goal 模式确定强基线、协议兼容性和多智能体协同创新边界；不是系统综述，也不产生性能结论。

## Safe Public Queries

- `agentic federated learning LLM orchestration`
- `multi agent federated learning client selection`
- `adaptive federated aggregation fairness worst client`
- `federated learning distributed concept drift`
- `federated time series forecasting heterogeneity`
- `federated optimization adaptive local learning rate`

## Sources Checked

- PMLR official proceedings；
- OpenReview / ICLR official records；
- CVF Open Access；
- AAAI proceedings；
- arXiv stable abstract pages；
- OpenAlex used only for discovery/deduplication, not as the sole evidence for method claims。

## Exclusions

- MDPI and low-signal/untraceable sources excluded；
- generic FL/LLM fine-tuning papers excluded when agents do not control the FL process；
- personalized FL papers excluded from the main baseline set when they output client-personalized models rather than one global forecasting model；
- search snippets were not used as final evidence。

## Verified Updates

- Agentic-FL is arXiv v1 submitted 2026-04-06 and presents an orchestration paradigm/roadmap；
- FedAWARE is AISTATS 2025 PMLR and explicitly introduces client consensus dynamics；
- FedAWA is CVPR 2025 and explicitly optimizes weights from client update vectors；
- Selective Collaboration is PMLR 2026 and assigns merit-based weights using client relevance；
- Beijing continual forecasting work is linked to FLTA 2025/IEEE and uses the same 12-client dataset；
- Fed-TREND remains an arXiv 2024 time-series heterogeneity method in the checked source。

## Unknowns Requiring PDF/Code Audit

- exact public implementations, licenses, tuning ranges and model compatibility for every added baseline；
- whether an exact SecAgg-compatible transformation preserves SCAFFOLD/FedDyn/Flash semantics；
- whether AAggFF/FedAWARE/FedAWA can be evaluated under aggregate-only visibility without changing the algorithm；
- current venue/version status of emerging 2026 papers at submission time；
- Fed-TREND synthetic knowledge carrier privacy and communication accounting。

## Handoff

- Idea optimization: focus on aggregate-blind SecAgg+ group coordination, not dynamic lr/epoch；
- Experiment design: use the A/B/C qualification matrix in `docs/12_goal_driven_mas_fl_roadmap.md`；
- Development: implement the public directive contract before running P2；
- Review: treat any missing A-level baseline or hidden privacy weakening as a major evidence defect。

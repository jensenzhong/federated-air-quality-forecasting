"""Fail-closed privacy policy for federated agent and LLM data flows."""

from __future__ import annotations

from ipaddress import ip_address
from typing import Any
from urllib.parse import urlparse

LOCAL_LLM_HOSTS = {"127.0.0.1", "localhost", "::1"}
FORBIDDEN_PROMPT_KEY_TOKENS = {
    "raw",
    "row",
    "sample",
    "timestamp",
    "prediction",
    "residual_series",
    "target_series",
    "test",
}

def is_loopback_llm_endpoint(base_url: str) -> bool:
    parsed = urlparse(base_url)
    return parsed.scheme in {"http", "https"} and parsed.hostname in LOCAL_LLM_HOSTS


def is_approved_private_llm_endpoint(base_url: str, allowed_hosts: list[str]) -> bool:
    parsed = urlparse(base_url)
    hostname = parsed.hostname
    if parsed.scheme not in {"http", "https"} or hostname is None:
        return False
    if hostname in LOCAL_LLM_HOSTS or hostname in allowed_hosts:
        return True
    try:
        return ip_address(hostname).is_private
    except ValueError:
        return False


def enforce_client_level_llm_policy(config: dict[str, Any]) -> None:
    privacy = config.get("privacy", {})
    if str(privacy.get("mode", "strict_federated")) != "strict_federated":
        raise RuntimeError("PAFA requires privacy.mode=strict_federated")
    base_url = str(config["llm"].get("client_base_url", config["llm"]["base_url"]))
    allowed_hosts = [str(item) for item in privacy.get("client_llm_allowed_hosts", [])]
    if not is_approved_private_llm_endpoint(base_url, allowed_hosts):
        raise RuntimeError(
            "Client-level PAFA state cannot be sent to an external LLM endpoint; "
            "configure a loopback/on-prem client agent or use an aggregate-only coordinator"
        )


def enforce_pafa_run_mode(
    method: str,
    *,
    evaluation_split: str,
    protocol_frozen: bool,
    secure_aggregation_active: bool = False,
    client_state_isolated: bool = False,
) -> None:
    if not method.startswith("pafa_"):
        return
    if not secure_aggregation_active:
        raise RuntimeError(
            "PAFA execution is blocked until SecAgg+ is integrated and verified; "
            "the current individual-reply strategy is code-review-only"
        )
    if evaluation_split == "test":
        raise RuntimeError(
            "PAFA test evaluation remains blocked until cohort metrics are securely "
            "aggregated without per-client EvaluateRes"
        )
    if protocol_frozen and not client_state_isolated:
        raise RuntimeError(
            "Formal PAFA requires institution-isolated ClientApps; an in-process "
            "SecAgg+ verification runner is nonformal only"
        )


def assert_prompt_keys_are_private_safe(value: Any, path: str = "payload") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(token in lowered for token in FORBIDDEN_PROMPT_KEY_TOKENS):
                raise ValueError(f"Forbidden private prompt field at {path}.{key}")
            assert_prompt_keys_are_private_safe(child, f"{path}.{key}")
    elif isinstance(value, list | tuple):
        for index, child in enumerate(value):
            assert_prompt_keys_are_private_safe(child, f"{path}[{index}]")

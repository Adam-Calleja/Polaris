"""Shared request-budget and failure classification helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Mapping

POLARIS_TIMEOUT_HEADER = "X-Polaris-Timeout-Ms"
POLARIS_EVAL_POLICY_HEADER = "X-Polaris-Eval-Policy"

EVAL_POLICY_OFFICIAL = "official"
EVAL_POLICY_DIAGNOSTIC = "diagnostic"
EVAL_POLICY_INTERACTIVE = "interactive"
VALID_EVAL_POLICIES = {
    EVAL_POLICY_OFFICIAL,
    EVAL_POLICY_DIAGNOSTIC,
    EVAL_POLICY_INTERACTIVE,
}

FAILURE_CLASS_RETRIEVAL_TIMEOUT = "retrieval_timeout"
FAILURE_CLASS_GENERATION_TIMEOUT = "generation_timeout"
FAILURE_CLASS_TRANSPORT_ERROR = "transport_error"
FAILURE_CLASS_API_INTERNAL_ERROR = "api_internal_error"
FAILURE_CLASS_EMPTY_RESPONSE = "empty_response"
FAILURE_CLASS_INVALID_INPUT = "invalid_input"

FAILURE_STAGE_RETRIEVAL = "retrieval"
FAILURE_STAGE_GENERATION = "generation"
FAILURE_STAGE_API = "api"
FAILURE_STAGE_DATASET = "dataset"

INFRA_FAILURE_CLASSES = frozenset(
    {
        FAILURE_CLASS_RETRIEVAL_TIMEOUT,
        FAILURE_CLASS_GENERATION_TIMEOUT,
        FAILURE_CLASS_TRANSPORT_ERROR,
        FAILURE_CLASS_API_INTERNAL_ERROR,
    }
)


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def normalize_evaluation_policy(value: str | None, *, default: str = EVAL_POLICY_OFFICIAL) -> str:
    normalized = str(value or default).strip().lower()
    if normalized in VALID_EVAL_POLICIES:
        return normalized
    return default


def timeout_failure_class_for_stage(stage: str) -> str:
    if stage == FAILURE_STAGE_RETRIEVAL:
        return FAILURE_CLASS_RETRIEVAL_TIMEOUT
    return FAILURE_CLASS_GENERATION_TIMEOUT


@dataclass(frozen=True)
class EvaluationDeadlines:
    """Normalized evaluation deadline policy."""

    policy: str
    client_total_seconds: float
    server_total_seconds: float
    retrieval_cap_seconds: float
    cleanup_reserve_seconds: float

    def normalized(self) -> "EvaluationDeadlines":
        client_total = max(1e-3, float(self.client_total_seconds))
        server_total = max(1e-3, float(self.server_total_seconds))
        if server_total >= client_total:
            server_total = max(1e-3, client_total - 1e-3)

        retrieval_cap = max(0.0, float(self.retrieval_cap_seconds))
        cleanup_reserve = max(0.0, float(self.cleanup_reserve_seconds))
        if cleanup_reserve >= server_total:
            cleanup_reserve = max(0.0, server_total - 1e-3)

        return EvaluationDeadlines(
            policy=normalize_evaluation_policy(self.policy),
            client_total_seconds=client_total,
            server_total_seconds=server_total,
            retrieval_cap_seconds=retrieval_cap,
            cleanup_reserve_seconds=cleanup_reserve,
        )

    @property
    def client_total_ms(self) -> int:
        return int(round(self.client_total_seconds * 1000.0))

    @property
    def server_total_ms(self) -> int:
        return int(round(self.server_total_seconds * 1000.0))

    @property
    def retrieval_cap_ms(self) -> int:
        return int(round(self.retrieval_cap_seconds * 1000.0))

    @property
    def cleanup_reserve_ms(self) -> int:
        return int(round(self.cleanup_reserve_seconds * 1000.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "client_total_seconds": float(self.client_total_seconds),
            "server_total_seconds": float(self.server_total_seconds),
            "retrieval_cap_seconds": float(self.retrieval_cap_seconds),
            "cleanup_reserve_seconds": float(self.cleanup_reserve_seconds),
            "client_total_ms": int(self.client_total_ms),
            "server_total_ms": int(self.server_total_ms),
            "retrieval_cap_ms": int(self.retrieval_cap_ms),
            "cleanup_reserve_ms": int(self.cleanup_reserve_ms),
        }


DEFAULT_EVALUATION_DEADLINES = {
    EVAL_POLICY_OFFICIAL: EvaluationDeadlines(
        policy=EVAL_POLICY_OFFICIAL,
        client_total_seconds=120.0,
        server_total_seconds=110.0,
        retrieval_cap_seconds=10.0,
        cleanup_reserve_seconds=5.0,
    ).normalized(),
    EVAL_POLICY_DIAGNOSTIC: EvaluationDeadlines(
        policy=EVAL_POLICY_DIAGNOSTIC,
        client_total_seconds=180.0,
        server_total_seconds=170.0,
        retrieval_cap_seconds=15.0,
        cleanup_reserve_seconds=5.0,
    ).normalized(),
    EVAL_POLICY_INTERACTIVE: EvaluationDeadlines(
        policy=EVAL_POLICY_INTERACTIVE,
        client_total_seconds=120.0,
        server_total_seconds=110.0,
        retrieval_cap_seconds=10.0,
        cleanup_reserve_seconds=5.0,
    ).normalized(),
}


def resolve_evaluation_deadlines(
    generation_cfg: Mapping[str, Any] | None,
    *,
    policy: str,
    client_total_override: float | None = None,
) -> EvaluationDeadlines:
    normalized_policy = normalize_evaluation_policy(policy)
    defaults = DEFAULT_EVALUATION_DEADLINES.get(
        normalized_policy,
        DEFAULT_EVALUATION_DEADLINES[EVAL_POLICY_OFFICIAL],
    )

    generation_map = _as_mapping(generation_cfg)
    deadlines_cfg = _as_mapping(generation_map.get("deadlines", {}))
    policy_cfg = _as_mapping(deadlines_cfg.get(normalized_policy, {}))

    legacy_client_total = _coerce_float(
        generation_map.get("api_timeout"),
        defaults.client_total_seconds,
    )
    client_total_seconds = (
        _coerce_float(client_total_override, legacy_client_total)
        if client_total_override is not None
        else _coerce_float(policy_cfg.get("client_total_seconds"), legacy_client_total)
    )

    resolved = EvaluationDeadlines(
        policy=normalized_policy,
        client_total_seconds=client_total_seconds,
        server_total_seconds=_coerce_float(
            policy_cfg.get("server_total_seconds"),
            defaults.server_total_seconds,
        ),
        retrieval_cap_seconds=_coerce_float(
            policy_cfg.get("retrieval_cap_seconds"),
            defaults.retrieval_cap_seconds,
        ),
        cleanup_reserve_seconds=_coerce_float(
            policy_cfg.get("cleanup_reserve_seconds"),
            defaults.cleanup_reserve_seconds,
        ),
    ).normalized()
    return resolved


@dataclass(frozen=True)
class RequestBudget:
    """Monotonic request budget for a single end-to-end API request."""

    policy: str
    total_ms: int
    retrieval_cap_ms: int
    cleanup_reserve_ms: int
    started_monotonic: float
    deadline_monotonic: float

    @classmethod
    def from_timeout_ms(
        cls,
        *,
        timeout_ms: int,
        policy: str,
        retrieval_cap_ms: int,
        cleanup_reserve_ms: int,
        started_monotonic: float | None = None,
    ) -> "RequestBudget":
        started = time.monotonic() if started_monotonic is None else float(started_monotonic)
        total_ms = max(1, int(timeout_ms))
        return cls(
            policy=normalize_evaluation_policy(policy),
            total_ms=total_ms,
            retrieval_cap_ms=max(0, int(retrieval_cap_ms)),
            cleanup_reserve_ms=max(0, int(cleanup_reserve_ms)),
            started_monotonic=started,
            deadline_monotonic=started + (total_ms / 1000.0),
        )

    def elapsed_ms(self) -> int:
        return max(0, int(round((time.monotonic() - self.started_monotonic) * 1000.0)))

    def remaining_ms(self) -> int:
        return max(0, int(round((self.deadline_monotonic - time.monotonic()) * 1000.0)))

    def remaining_seconds(self) -> float:
        return self.remaining_ms() / 1000.0

    def stage_timeout_ms(
        self,
        *,
        stage: str,
        reserve_ms: int = 0,
        cap_ms: int | None = None,
    ) -> int:
        remaining = max(0, self.remaining_ms() - max(0, int(reserve_ms)))
        limit = remaining
        if cap_ms is not None:
            limit = min(limit, max(0, int(cap_ms)))
        if stage == FAILURE_STAGE_RETRIEVAL:
            limit = min(limit, max(0, int(self.retrieval_cap_ms)))
        return max(0, limit)

    def require_stage_timeout_ms(
        self,
        *,
        stage: str,
        reserve_ms: int = 0,
        cap_ms: int | None = None,
    ) -> int:
        timeout_ms = self.stage_timeout_ms(stage=stage, reserve_ms=reserve_ms, cap_ms=cap_ms)
        if timeout_ms <= 0:
            raise RequestBudgetExceededError(stage=stage)
        return timeout_ms

    def child_timeout_seconds(
        self,
        *,
        stage: str,
        reserve_ms: int = 0,
        cap_ms: int | None = None,
    ) -> float:
        timeout_ms = self.require_stage_timeout_ms(stage=stage, reserve_ms=reserve_ms, cap_ms=cap_ms)
        return timeout_ms / 1000.0

    def to_attributes(self) -> dict[str, Any]:
        return {
            "eval_policy": self.policy,
            "budget_total_ms": int(self.total_ms),
            "budget_remaining_ms": int(self.remaining_ms()),
            "retrieval_cap_ms": int(self.retrieval_cap_ms),
            "cleanup_reserve_ms": int(self.cleanup_reserve_ms),
        }


class PolarisRuntimeError(RuntimeError):
    """Base Polaris runtime failure with structured classification."""

    def __init__(
        self,
        message: str,
        *,
        failure_class: str,
        failure_stage: str,
        response_status: str = "error",
    ) -> None:
        super().__init__(message)
        self.failure_class = str(failure_class)
        self.failure_stage = str(failure_stage)
        self.response_status = str(response_status)


class PolarisTimeoutError(TimeoutError):
    """Base timeout carrying structured failure metadata."""

    def __init__(
        self,
        message: str,
        *,
        failure_class: str,
        failure_stage: str,
        response_status: str = "timeout",
    ) -> None:
        super().__init__(message)
        self.failure_class = str(failure_class)
        self.failure_stage = str(failure_stage)
        self.response_status = str(response_status)


class RetrievalTimeoutError(PolarisTimeoutError):
    def __init__(self, message: str = "retrieval exceeded request deadline") -> None:
        super().__init__(
            message,
            failure_class=FAILURE_CLASS_RETRIEVAL_TIMEOUT,
            failure_stage=FAILURE_STAGE_RETRIEVAL,
            response_status="timeout",
        )


class GenerationTimeoutError(PolarisTimeoutError):
    def __init__(self, message: str = "generation exceeded request deadline") -> None:
        super().__init__(
            message,
            failure_class=FAILURE_CLASS_GENERATION_TIMEOUT,
            failure_stage=FAILURE_STAGE_GENERATION,
            response_status="timeout",
        )


class RequestBudgetExceededError(PolarisTimeoutError):
    def __init__(self, *, stage: str) -> None:
        stage_name = stage if stage in {FAILURE_STAGE_RETRIEVAL, FAILURE_STAGE_GENERATION} else FAILURE_STAGE_GENERATION
        super().__init__(
            f"{stage_name} cannot start because the request budget is exhausted",
            failure_class=timeout_failure_class_for_stage(stage_name),
            failure_stage=stage_name,
            response_status="timeout",
        )


class EmptyResponseError(ValueError):
    def __init__(self, message: str = "generation returned an empty response") -> None:
        super().__init__(message)
        self.failure_class = FAILURE_CLASS_EMPTY_RESPONSE
        self.failure_stage = FAILURE_STAGE_GENERATION
        self.response_status = "empty_response"


def is_timeout_exception(exc: BaseException) -> bool:
    if isinstance(exc, PolarisTimeoutError | TimeoutError):
        return True
    exc_type = type(exc).__name__.lower()
    if "timeout" in exc_type:
        return True
    message = str(exc).lower()
    return any(token in message for token in ("timed out", "deadline exceeded", "read timeout"))


def build_failure_detail(
    exc: BaseException,
    *,
    elapsed_ms: int | None = None,
    http_status: int | None = None,
    request_budget: RequestBudget | None = None,
    response_status: str | None = None,
    traceback_text: str | None = None,
) -> dict[str, Any]:
    detail: dict[str, Any] = {
        "error": f"{type(exc).__name__}: {exc}",
        "failure_class": str(
            getattr(
                exc,
                "failure_class",
                FAILURE_CLASS_GENERATION_TIMEOUT if is_timeout_exception(exc) else FAILURE_CLASS_API_INTERNAL_ERROR,
            )
        ),
        "failure_stage": str(
            getattr(
                exc,
                "failure_stage",
                FAILURE_STAGE_GENERATION if is_timeout_exception(exc) else FAILURE_STAGE_API,
            )
        ),
        "response_status": str(
            response_status
            or getattr(exc, "response_status", "timeout" if is_timeout_exception(exc) else "error")
        ),
    }
    if elapsed_ms is not None:
        detail["elapsed_ms"] = max(0, int(elapsed_ms))
    if http_status is not None:
        detail["http_status"] = int(http_status)
    if request_budget is not None:
        detail["budget_total_ms"] = int(request_budget.total_ms)
        detail["budget_remaining_ms"] = int(request_budget.remaining_ms())
        detail["eval_policy"] = str(request_budget.policy)
    if traceback_text:
        detail["traceback"] = traceback_text
    return detail


__all__ = [
    "DEFAULT_EVALUATION_DEADLINES",
    "EVAL_POLICY_DIAGNOSTIC",
    "EVAL_POLICY_INTERACTIVE",
    "EVAL_POLICY_OFFICIAL",
    "EmptyResponseError",
    "EvaluationDeadlines",
    "FAILURE_CLASS_API_INTERNAL_ERROR",
    "FAILURE_CLASS_EMPTY_RESPONSE",
    "FAILURE_CLASS_GENERATION_TIMEOUT",
    "FAILURE_CLASS_INVALID_INPUT",
    "FAILURE_CLASS_RETRIEVAL_TIMEOUT",
    "FAILURE_CLASS_TRANSPORT_ERROR",
    "FAILURE_STAGE_API",
    "FAILURE_STAGE_DATASET",
    "FAILURE_STAGE_GENERATION",
    "FAILURE_STAGE_RETRIEVAL",
    "GenerationTimeoutError",
    "INFRA_FAILURE_CLASSES",
    "POLARIS_EVAL_POLICY_HEADER",
    "POLARIS_TIMEOUT_HEADER",
    "PolarisRuntimeError",
    "PolarisTimeoutError",
    "RequestBudget",
    "RequestBudgetExceededError",
    "RetrievalTimeoutError",
    "VALID_EVAL_POLICIES",
    "build_failure_detail",
    "is_timeout_exception",
    "normalize_evaluation_policy",
    "resolve_evaluation_deadlines",
    "timeout_failure_class_for_stage",
]

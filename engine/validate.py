"""Typed error taxonomy — the no-silent-failures layer.

Every invalid input raises one of these with a machine-readable `code` and a human
message naming the exact offending value. The UI catches these at its boundary and
renders a banner; the agent returns them to the model as structured errors.
"""
from __future__ import annotations


class CopilotError(Exception):
    """Base class for all surfaced copilot errors."""

    code = "COPILOT_ERROR"

    def __init__(self, message: str, **context):
        super().__init__(message)
        self.message = message
        self.context = context

    def to_dict(self) -> dict:
        return {"error": self.message, "code": self.code, **self.context}


class PriceDataError(CopilotError):
    code = "PRICE_DATA_ERROR"


class UniverseMismatchError(CopilotError):
    code = "UNIVERSE_MISMATCH"


class InsufficientHistoryError(CopilotError):
    code = "INSUFFICIENT_HISTORY"


class WeightError(CopilotError):
    code = "WEIGHT_ERROR"


class NotFoundError(CopilotError):
    code = "NOT_FOUND"


class OptimizationInfeasibleError(CopilotError):
    code = "OPTIMIZATION_INFEASIBLE"


class OptimizerUnavailableError(CopilotError):
    code = "OPTIMIZER_UNAVAILABLE"


class ConfigNotSetError(CopilotError):
    """Raised when a not-yet-supplied config value (e.g. optimizer bounds) is used."""

    code = "CONFIG_NOT_SET"

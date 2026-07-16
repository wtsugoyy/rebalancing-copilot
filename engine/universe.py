"""Fund identity (ISIN) + per-portfolio eligibility + weight validation.

Pure functions, no UI imports.
"""
from __future__ import annotations

import config
from engine.validate import UniverseMismatchError, WeightError


def resolve_isins(headers: list[str]) -> list[str]:
    """Validate that every column header is a known ISIN. Returns them unchanged.

    Raises UniverseMismatchError naming offenders (closes the name-mismatch class of
    bug flagged in unit-trust-aggregator).
    """
    known = set(config.FUND_UNIVERSE)
    unknown = [h for h in headers if h not in known]
    if unknown:
        raise UniverseMismatchError(
            f"Unknown fund column(s) not in the ISIN universe: {unknown}",
            offenders=unknown,
            known=sorted(known),
        )
    return headers


def normalize_weights(raw: dict[str, float]) -> dict[str, float]:
    """Normalize a pasted/entered weight map deterministically.

    - accepts percent or decimal (any value > 1.5 is divided by 100, matching the
      unit-trust-aggregator convention)
    - unknown ISIN -> WeightError
    - missing fund treated as explicit 0 (never dropped silently)
    - sum must be 1 +/- WEIGHT_SUM_TOL (never auto-renormalized silently)
    """
    known = set(config.FUND_UNIVERSE)
    unknown = [k for k in raw if k not in known]
    if unknown:
        raise WeightError(f"Weights reference unknown ISIN(s): {unknown}", offenders=unknown)

    weights = {isin: 0.0 for isin in config.FUND_UNIVERSE}
    for isin, w in raw.items():
        w = float(w)
        if abs(w) > 1.5:
            w = w / 100.0
        weights[isin] = w

    total = sum(weights.values())
    if abs(total - config.WEIGHT_SUM) > config.WEIGHT_SUM_TOL:
        raise WeightError(
            f"Weights sum to {total:.6f}, expected {config.WEIGHT_SUM} "
            f"(tolerance {config.WEIGHT_SUM_TOL}).",
            actual_sum=total,
        )
    return weights


def default_weight_map(code: str) -> dict[str, float]:
    """The Excel's current-live weight vector for a portfolio, as an ISIN->weight map."""
    if code not in config.DEFAULT_WEIGHTS:
        raise WeightError(f"Unknown portfolio code: {code}", offenders=[code])
    return dict(zip(config.FUND_UNIVERSE, config.DEFAULT_WEIGHTS[code]))

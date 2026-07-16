"""Lightweight local observability.

A single rotating file logger for the audit trail: tool invocations, ingest
summaries, and risk-free fallbacks. Proportional to a single-user local tool — no
external sink, no PII (fund-level data only).
"""
from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler

import config

_logger: logging.Logger | None = None


def get_logger() -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger
    log = logging.getLogger("rebalancing_copilot")
    log.setLevel(logging.INFO)
    log.propagate = False
    try:
        os.makedirs(os.path.dirname(config.LOG_PATH), exist_ok=True)
        handler: logging.Handler = RotatingFileHandler(
            config.LOG_PATH, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8")
    except Exception:  # noqa: BLE001 - never let logging setup break the app
        handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    if not log.handlers:
        log.addHandler(handler)
    _logger = log
    return log

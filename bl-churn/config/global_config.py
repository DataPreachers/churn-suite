#!/usr/bin/env python3
from pathlib import Path
import json
from typing import Any, Dict

from .paths_config import ProjectPaths


def _load_ui_settings() -> Dict[str, Any]:
    path = ProjectPaths.ui_settings_file()
    if not path.exists():
        raise FileNotFoundError(f"ui_settings.json nicht gefunden: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_global_config() -> Dict[str, Any]:
    return _load_ui_settings()


def enforce_real_analysis(context: str = "") -> None:
    cfg = get_global_config()
    rt = (cfg.get("runtime") or {})
    if not cfg.get("enforce_real_analysis", False):
        raise RuntimeError(f"Real Analysis erforderlich, Setting fehlt/false. Kontext: {context}")
    if not rt.get("fail_fast", False):
        # Fail-Fast Policy: explizit erzwingen
        raise RuntimeError("Fail-Fast Policy aktivieren: runtime.fail_fast=true")


def enforce_real_backtest(context: str = "") -> None:
    cfg = get_global_config()
    rt = (cfg.get("runtime") or {})
    if not cfg.get("enforce_real_backtest", False):
        raise RuntimeError(f"Real Backtest erforderlich, Setting fehlt/false. Kontext: {context}")
    if not rt.get("fail_fast", False):
        raise RuntimeError("Fail-Fast Policy aktivieren: runtime.fail_fast=true")



#!/usr/bin/env python3
"""
Minimal Leakage-Guard Utilities (Counterfactual Support)
-------------------------------------------------------

Funktionen:
- load_cf_cost_policy(): lädt Policy und expandiert Roh-Feature-Constraints
  auf gemappte engineered Features per config/feature_mapping.json.
- load_feature_mapping(): lädt Mapping (Roh -> [engineered]).
- get_explainable_features(...): passt-through (keine aggressive Excludes).

Hinweis: Minimal-invasive Implementierung, nur für CF-Engine benötigt.
"""
from __future__ import annotations

import json
from typing import Dict, Any, Iterable, List
from pathlib import Path

from config.paths_config import ProjectPaths


def load_feature_mapping() -> Dict[str, List[str]]:
    path = ProjectPaths.feature_mapping_file()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _expand_policy_with_mapping(policy: Dict[str, Any], mapping: Dict[str, List[str]]) -> Dict[str, Any]:
    if not isinstance(policy, dict):
        return {"version": "1.0", "features": {}}
    features = dict(policy.get("features", {}))
    if not isinstance(features, dict):
        features = {}
    # Für jeden Roh-Schlüssel, Constraints an alle engineered übertragen
    for raw_name, engineered_list in mapping.items():
        if raw_name in features and isinstance(engineered_list, list):
            constraints = features.get(raw_name, {})
            for eng in engineered_list:
                if isinstance(eng, str) and eng:
                    # Nur setzen, wenn nicht bereits explizit in Policy vorhanden
                    features.setdefault(eng, constraints)
    out = dict(policy)
    out["features"] = features
    return out


def load_cf_cost_policy() -> Dict[str, Any]:
    path = ProjectPaths.cf_cost_policy_file()
    base = {"version": "1.0", "default_step": 0.1, "features": {}}
    policy = base
    if path.exists():
        try:
            policy = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            policy = base
    mapping = load_feature_mapping()
    return _expand_policy_with_mapping(policy, mapping)


def get_explainable_features(experiment_id: int, available_columns: Iterable[str]) -> List[str]:
    # Minimal: keine aggressiven Excludes; CF-Notebook filtert ggf. separat.
    return [c for c in available_columns]


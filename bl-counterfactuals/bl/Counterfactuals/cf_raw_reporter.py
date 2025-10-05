#!/usr/bin/env python3
"""
CF Raw Reporter: mappt engineered Feature-Namen in CF-Outputs auf Roh-Namen

Eingaben:
- artifacts/<exp_id>/counterfactuals/cf_individual.json
- artifacts/<exp_id>/counterfactuals/cf_aggregate.json
- config/feature_mapping.json (Roh -> [engineered])

Ausgaben:
- cf_individual_raw.json (top_changes feature→raw_feature, aggregiert)
- cf_aggregate_raw.json (feature = raw_feature)
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, Any, List
from config.paths_config import ProjectPaths


def load_json(path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def invert_mapping(mapping: Dict[str, List[str]]) -> Dict[str, str]:
    inv = {}
    for raw, eng_list in mapping.items():
        for eng in eng_list:
            inv[eng] = raw
    return inv


def main(exp_id: int = 1) -> int:
    cf_dir = ProjectPaths.ensure_directory_exists(ProjectPaths.artifacts_for_experiment(exp_id) / "counterfactuals")
    ind = cf_dir / "cf_individual.json"
    agg = cf_dir / "cf_aggregate.json"
    if not ind.exists() or not agg.exists():
        print("❌ Counterfactual-Dateien nicht gefunden:", cf_dir)
        return 1

    mapping_path = ProjectPaths.feature_mapping_file()
    mapping = json.loads(mapping_path.read_text(encoding="utf-8")) if mapping_path.exists() else {}
    inv = invert_mapping(mapping)

    ind_data = load_json(ind)
    agg_data = load_json(agg)

    # Individual: top_changes als Liste {feature, delta, abs_delta}
    ind_out = []
    for rec in ind_data:
        out_rec = {k: v for k, v in rec.items() if k not in ("top_changes",)}
        raw_changes = defaultdict(lambda: {"delta": 0.0, "abs_delta": 0.0})
        for ch in rec.get("top_changes", []):
            feat = ch.get("feature")
            raw = inv.get(feat, feat)
            # aggregiere deltas je raw-feature (summe; abs als summe der abs)
            try:
                raw_changes[raw]["delta"] += float(ch.get("delta", 0.0))
                raw_changes[raw]["abs_delta"] += float(ch.get("abs_delta", 0.0))
            except Exception:
                pass
        out_rec["top_changes_raw"] = [
            {"feature": r, "delta": v["delta"], "abs_delta": v["abs_delta"]}
            for r, v in raw_changes.items()
        ]
        ind_out.append(out_rec)

    # Aggregate: feature→raw_feature, counts und median_abs_delta werden je raw aggregiert (sum counts, max median as proxy)
    agg_by_raw = defaultdict(lambda: {"count": 0, "median_abs_delta": 0.0})
    for a in agg_data:
        raw = inv.get(a.get("feature"), a.get("feature"))
        try:
            agg_by_raw[raw]["count"] += int(a.get("count", 0))
            agg_by_raw[raw]["median_abs_delta"] = max(
                agg_by_raw[raw]["median_abs_delta"], float(a.get("median_abs_delta", 0.0))
            )
        except Exception:
            pass
    agg_out = [
        {"feature": r, "count": v["count"], "median_abs_delta": v["median_abs_delta"]}
        for r, v in agg_by_raw.items()
    ]

    (cf_dir / "cf_individual_raw.json").write_text(json.dumps(ind_out, indent=2, ensure_ascii=False), encoding="utf-8")
    (cf_dir / "cf_aggregate_raw.json").write_text(json.dumps(agg_out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("✅ Raw-Reports geschrieben:")
    print(cf_dir / "cf_individual_raw.json")
    print(cf_dir / "cf_aggregate_raw.json")
    return 0


if __name__ == "__main__":
    import sys
    eid = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    raise SystemExit(main(eid))



Last reviewed: 2025-09-27

# Next Steps – bl-churn

## Befunde (heutiger Stand)
- Feature-Engineering ist zu breit und redundant (viele Fenster: 6/12/18/24/36; viele Operatoren: mean/sum/std/trend/pct_change/cum*). Das verwässert Robustheit und Interpretierbarkeit.
- Trend-Features sind in Teilen deaktiviert, Namenskonventionen uneinheitlich (`m` vs. `p`).
- Leakage-Schutz ist aktiv (Same-Period Business-Features beim Training ausgeschlossen), aber die FE-Breite bleibt hoch.
- CF-Ergebnisse deuten auf wiederkehrende Hebel: `I_SOCIALINSURANCENOTES_*` (Aktivität/Änderung) und `I_MAINTENANCE_*` (Level/Trend). Richtung nicht pauschal, aber häufig: mehr Wartung/Betreuung korreliert mit geringerer Abwanderung.
- `I_MAINTENANCE` (Euro) korreliert mit Größe/Verflechtung → Normalisierung nötig (pro Kunde/Portfolio/Revenue), sonst Bias.

## Empfehlungen (kurzfristig)
- Fenster straffen: primär 6p und 12p; 18/24/36p nur bei nachgewiesenem Zusatznutzen.
- Operatoren reduzieren: beibehalten `mean/sum/trend/pct_change/activity_rate`; weglassen `std/cum*`, außer begründet.
- Whitelist je Roh-Feature (über `config/shared/config/feature_mapping.json`):
  - I_MAINTENANCE (Euro): `rolling_12p_sum`, `trend_12p`, optional `activity_rate_12p`; stets normalisiert.
  - I_SOCIALINSURANCENOTES (Stück): `activity_rate_6p/12p`, `pct_change_6p/12p`, optional `rolling_12p_mean` (Zielkorridor statt blind ↑/↓).
- Konsolidierte Namenskonvention und einheitliche Shift-/Leakage-Logik.
- Frühzeitiges Pruning: Varianz/Korrelation bereits bei FE-Erzeugung anwenden, nicht nur nachgelagert.

## Empfehlungen (mittelfristig)
- SHAP-Integration zur globalen/lokalen Treiberanalyse; CF-Suche auf SHAP-Top-K je Kunde begrenzen (Synergien SHAP↔CF).
- Segmentierung: normalisiertes `I_MAINTENANCE` (Level/Trend) × `I_SOCIALINSURANCENOTES` (Aktivität/Trend) für priorisierte Maßnahmen.
- Stabilität: Zeitliche Stabilität der Treiber prüfen; nur stabile Richtungen in Policies zulassen.

## Konkrete ToDos
1) `feature_mapping.json` auf 6p/12p und reduzierte Operatoren eindampfen; Naming harmonisieren.
2) `churn_feature_engine.py` anpassen: Fenster/Operatoren-Whitelist, konsistente Suffixe, Trend reaktivieren (leakage-sicher).
3) Normalisierung implementieren (Maintenance relativieren: pro Kunde/Portfolio/Revenue).
4) Frühzeitiges FE-Pruning (Varianz/Korrelation) in die Erzeugung integrieren.
5) SHAP-Modul (`bl-shap`) anbinden und Artefakte (global/local) exportieren.
6) Validierung/Kohorten-Checks und Reporting (AUC/Recall, Kosten pro Reduktion, Stabilität).



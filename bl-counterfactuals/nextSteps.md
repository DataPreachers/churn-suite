Last reviewed: 2025-09-29

# Next Steps – bl-counterfactuals

## Befunde aus aktuellen CF-Ergebnissen
- Positive Signale (starke Reduktionen, hoher Net‑Benefit/ROI), aber Kostenmodell oft zu optimistisch (`implementation_cost=0.0` → ROI 999.99).
- `cf_individual.json` enthält `no_cf_found=true` und Fälle mit `p_new > p_old` → zwingende Qualitätsfilter nötig.
- Häufige Hebel: `I_SOCIALINSURANCENOTES_*` (Aktivität/Änderung) und `I_MAINTENANCE_*` (Level/Trend). Richtung nicht pauschal; Rolling‑Mean mit Vorsicht.
- Aggregationen zeigen kleine Counts je Feature‑Delta → Robustheit/Generalisierung verbessern.

## Empfehlungen (kurzfristig)
- Kostenmodell schärfen (`cf_cost_policy.json`): keine 0‑Kosten; fachlich plausibilisieren je Roh‑Feature; Kostenobergrenzen. Für den Start: alle Kosten = 1.0 (neutrale Baseline).
- Qualitätsfilter erzwingen: nur Vorschläge mit `p_new < p_old`, Mindest‑Reduktion, Kosten‑Deckel, `no_cf_found=false`.
- Feature‑Whitelist (aus Churn‑FE ableiten): primär 6p/12p und Operatoren `mean/sum/trend/pct_change/activity_rate`.
- Richtungslogik: Aktivitäts-/Change‑Signale meist ↑; Rolling‑Mean auf Zielkorridor statt blind ↑/↓; Maintenance relativiert interpretieren.

## Empfehlungen (mittelfristig)
- SHAP‑Integration: CF‑Suche auf SHAP‑Top‑K je Kunde begrenzen; Monotonie/Bounds aus SHAP‑Vorzeichen ableiten. Feature‑Ermittlung erfolgt grundsätzlich auf Basis der SHAP‑Ergebnisse (global/lokal), nicht via freier Heuristik.
- Segmentierung: normalisiertes `I_MAINTENANCE` × `I_SOCIALINSURANCENOTES` (Level/Trend × Aktivität/Trend) zur Priorisierung.
- Stabilitäts‑Checks: Zeitliche Stabilität der Treiber und Wirksamkeit der CF‑Maßnahmen (Post‑SHAP‑Delta).

## Konkrete ToDos
1) `cf_cost_policy.json` realistisch kalibrieren (mit Fachseite).
2) CF‑Filter implementieren (p_new < p_old, Min‑Reduktion, Kosten‑Deckel, Ausschluss `no_cf_found`).
3) Feature‑Selektion strikt aus SHAP ableiten (Global Top‑N + Local Top‑K), Richtungs‑/Bounds‑Policies aus SHAP‑Vorzeichen.
4) SHAP‑Artefakte konsumieren (aus `bl-shap`) und CF‑Suche darauf einschränken.
5) Reporting: Segment‑KPIs (Anteil verbessert, Kosten/Reduktion, Top‑Hebel, Stabilität).

## Startkonfiguration
- Feature‑Ermittlung: ausschließlich aus `shap_global`/`shap_local_topk` (und optional aggregierten Varianten).
- Kostenmatrix (`cf_cost_policy.json`): initial jede Maßnahme Kosten = 1.0; spätere Kalibrierung geplant.



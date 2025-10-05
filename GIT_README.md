# Churn Suite

## Overview
Churn Suite bündelt alle Komponenten für unsere Churn-Analytik in einem einzigen Repository. Die Submodule wurden entfernt; jede Domäne lebt nun als normales Verzeichnis mit gemeinsam genutzter Python-Umgebung und zentraler Konfiguration.

## Architecture at a Glance
- **Domain Engines** (`bl-churn`, `bl-cox`, `bl-counterfactuals`, `bl-shap`): enthalten Trainings-, Scoring- und Analysepipelines pro Fachbereich.
- **Data Backbone** (`json-database`): leichtgewichtige JSON-Datenbank als Austausch- und Cache-Layer für alle Domänen.
- **Orchestration** (`runner-service`): FastAPI-Service zum Starten der Pipelines, Sammeln von Logs und Zurückschreiben der Ergebnisse.
- **User Interfaces** (`ui-crud`, `ui-managementstudio`): CRUD-Frontend für Experimente sowie ein leichtes SQL-Front-End auf die JSON-DB.
- **Shared Tooling** (`config`, `bl-workspace`, `Makefile`): zentrale Pfadsteuerung, Makefile-Shortcuts und wiederverwendbare Hilfsskripte.
- **Artifacts & Stage0** (`dynamic_system_outputs/`, `json-database/Views`): definieren die kanonischen Views und speichern Zwischenstände aus Ingestion und Training.

## Repository Layout
| Pfad | Zweck |
| --- | --- |
| `bl-churn/`, `bl-cox/`, `bl-counterfactuals/`, `bl-shap/` | Domänenspezifische Business-Logik und Modelle |
| `json-database/` | JSON-DB Engine, Views und Data Access Layer |
| `runner-service/` | FastAPI-Orchestrator für Pipelines und Logstream |
| `ui-crud/`, `ui-managementstudio/` | Web-UIs für Steuerung und Auswertung |
| `config/paths_config.py` | Zentrale Pfaddefinitionen für alle Komponenten |
| `bl-workspace/Makefile` | Konsolidierte Make Targets (z. B. `make ingest`) |
| `dynamic_system_outputs/` | Outbox, Stage0-Artefakte und Laufzeitdaten |

## Getting Started
1. Python 3.11 installieren.
2. Virtuelle Umgebung anlegen und Abhängigkeiten installieren:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. Optional: Services starten
   ```bash
   python runner-service/app.py      # Port 5050
   python ui-managementstudio/app.py # Port 5051
   ```
4. Daten ingestieren:
   ```bash
   cd bl-workspace
   make ingest
   ```

## Typical Workflow
- Branch erstellen (`git switch -c feature/<name>`), Änderungen umsetzen und lokal committen.
- Pipeline-Läufe über `runner-service` oder `make` anstoßen; Ergebnisse landen in `json-database`.
- Frontends (`ui-crud`, `ui-managementstudio`) für Validierung und Exploration nutzen.
- Änderungen per `git push` ins zentrale Repository spiegeln; keine Submodule oder zusätzlichen Repos nötig.

## Notes
- Alle Komponenten teilen sich dieselbe `requirements.txt`; Submodule und getrennte Umgebungen entfallen.
- Tabellen- und View-Definitionen im Ordner `json-database/Views` werden beim Start der JSON-DB konsistent eingespielt.
- Artefakte oder temporäre Dateien außerhalb von `dynamic_system_outputs/` vermeiden, um den Flatten-Ansatz beizubehalten.

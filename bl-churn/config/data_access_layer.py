#!/usr/bin/env python3
import json
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import sys
from pathlib import Path
# Zentrale ProjectPaths aus Root verwenden
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from config.paths_config import ProjectPaths


@dataclass
class DataAccess:
    input_csv_path: Path
    models_dir: Path
    
    def load_stage0_data(self, customer_reduction: float = 0.0) -> pd.DataFrame:
        """
        Lädt Stage0-Daten aus JSON-Database (rawdata Tabelle)
        
        JSON-Database ist die einzige Quelle der Wahrheit.
        Wenn keine rawdata vorhanden ist, ist das ein Konfigurationsfehler.
        
        Args:
            customer_reduction: Anteil der Kunden zu entfernen (0.0 = alle Kunden, 0.9 = 90% entfernen)
        
        Returns:
            pd.DataFrame: Stage0-Daten aus rawdata Tabelle
            
        Raises:
            ValueError: Wenn keine rawdata in JSON-Database gefunden wird
        """
        # Import hier um Zirkular-Import zu vermeiden
        import sys
        sys.path.insert(0, str(ProjectPaths.project_root()))
        from bl.json_database.churn_json_database import ChurnJSONDatabase
        
        # Verwende zentrale JSON-Database im Root-Verzeichnis
        db = ChurnJSONDatabase()
        
        # Lade rawdata aus JSON-Database
        rawdata_records = db.data.get("tables", {}).get("rawdata", {}).get("records", [])
        
        if not rawdata_records:
            raise ValueError(
                "Keine rawdata in JSON-Database gefunden. "
                "Führe zuerst bl-input Ingestion aus: "
                "python -c 'from input_ingestion import InputIngestionService; "
                "s = InputIngestionService(); s.ingest_csv_to_stage0(...); "
                "db.import_from_outbox_stage0_union()'"
            )
        
        print(f"📂 Lade Stage0-Daten aus JSON-DB rawdata: {len(rawdata_records)} Records")
        df = pd.DataFrame(rawdata_records)
        
        # ✅ ENTFERNE TECHNISCHE METADATEN-FELDER DIREKT BEIM LADEN
        # Diese Felder haben keine Business-Relevanz und können Listen-Probleme verursachen
        technical_columns = ['id', 'id_files', 'dt_inserted']
        columns_to_drop = [col for col in technical_columns if col in df.columns]
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            print(f"🗑️ Technische Metadaten-Felder beim Laden entfernt: {columns_to_drop}")
        
        # ✅ Optional: aus ENV übernehmen, wenn kein expliziter Wert übergeben wurde
        if customer_reduction <= 0.0:
            try:
                import os
                env_val = os.getenv('CHURN_CUSTOMER_REDUCTION')
                if env_val is not None:
                    customer_reduction = max(0.0, min(float(env_val), 0.99))
            except Exception:
                pass

        # ✅ KUNDEN-REDUKTION FÜR SCHNELLERE TESTS
        if customer_reduction > 0.0 and 'Kunde' in df.columns:
            unique_customers = df['Kunde'].unique()
            total_customers = len(unique_customers)
            
            # Berechne Anzahl zu behaltender Kunden
            customers_to_keep = int(total_customers * (1.0 - customer_reduction))
            
            if customers_to_keep < total_customers:
                # Zufällige Auswahl von Kunden (deterministisch mit seed)
                import numpy as np
                np.random.seed(42)  # Reproduzierbare Ergebnisse
                selected_customers = np.random.choice(unique_customers, customers_to_keep, replace=False)
                
                # Filtere DataFrame auf ausgewählte Kunden
                df = df[df['Kunde'].isin(selected_customers)]
                print(f"🔬 TEST-MODUS: {customers_to_keep}/{total_customers} Kunden ({(1-customer_reduction)*100:.0f}%)")
                print(f"   Reduziert von {len(rawdata_records)} auf {len(df)} Records")
        
        print(f"✅ rawdata als DataFrame geladen: {len(df)} Zeilen, {len(df.columns)} Spalten")
        return df


def get_data_access() -> DataAccess:
    return DataAccess(
        input_csv_path=ProjectPaths.get_input_data_path(),
        models_dir=ProjectPaths.models_directory(),
    )



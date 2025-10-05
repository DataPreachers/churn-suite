"""
STEP1 INPUT HANDLER
==================

Verantwortlich für Input-Validierung und -Ladung in Step1

Features:
- Laden der Step0 Metadaten
- Validierung der Training/Backtest Perioden
- Berechnung der Backtest-Perioden
- CSV-Hash-Berechnung
- Input-Validierung und Fehlerbehandlung
"""

import os
import json
import pandas as pd
import hashlib
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

# Project imports
from config.paths_config import ProjectPaths


class Step1InputHandler:
    """
    Verantwortlich für Input-Validierung und -Ladung in Step1
    """
    
    def __init__(self):
        self.stage0_cache_dir = os.path.join(ProjectPaths.dynamic_outputs_directory(), "stage0_cache")
        
    def load_stage0_metadata(self, csv_file_path: str) -> Optional[Dict[str, Any]]:
        """
        Metadaten aus Stufe 0 laden
        
        Args:
            csv_file_path: Pfad zur CSV-Datei
            
        Returns:
            Dict mit Step0 Metadaten oder None bei Fehler
        """
        try:
            # CSV Hash berechnen
            csv_hash = self.calculate_csv_hash(csv_file_path)
            if not csv_hash:
                print(f"❌ Fehler: Konnte Hash für {csv_file_path} nicht berechnen")
                return None
                
            metadata_file = os.path.join(self.stage0_cache_dir, f"{csv_hash}.json")
            
            if os.path.exists(metadata_file):
                print(f"✅ Lade Step0 Metadaten: {metadata_file}")
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    
                # Validierung der geladenen Metadaten
                if self._validate_stage0_metadata(metadata):
                    features_count = len(metadata.get('data_dictionary', {}).get('features', {}))
                    print(f"✅ Step0 Metadaten geladen: {features_count} Features")
                    return metadata
                else:
                    print(f"❌ Step0 Metadaten sind ungültig")
                    return None
            else:
                print(f"⚠️ Keine Step0 Metadaten gefunden für Hash: {csv_hash}")
                print(f"📁 Erwarteter Pfad: {metadata_file}")
                return None
                
        except Exception as e:
            print(f"❌ Fehler beim Laden der Step0 Metadaten: {e}")
            return None
    
    def validate_training_periods(self, training_from: str, training_to: str) -> bool:
        """
        Training-Perioden validieren
        
        Args:
            training_from: Start-Periode (YYYYMM)
            training_to: End-Periode (YYYYMM)
            
        Returns:
            True wenn Perioden gültig sind
        """
        try:
            # Format validieren
            if not self._is_valid_period_format(training_from) or not self._is_valid_period_format(training_to):
                print(f"❌ Ungültiges Perioden-Format: {training_from} - {training_to}")
                return False
            
            # Reihenfolge validieren
            if training_from >= training_to:
                print(f"❌ Ungültige Reihenfolge: {training_from} >= {training_to}")
                return False
            
            # Mindestdauer validieren (mindestens 6 Monate)
            duration = self._calculate_period_duration(training_from, training_to)
            if duration < 6:
                print(f"❌ Training-Periode zu kurz: {duration} Monate (Minimum: 6)")
                return False
            
            print(f"✅ Training-Perioden valid: {training_from} - {training_to} ({duration} Monate)")
            return True
            
        except Exception as e:
            print(f"❌ Fehler bei Perioden-Validierung: {e}")
            return False
    
    def calculate_backtest_period(self, training_to: str) -> str:
        """
        Backtest-Periode nach Training berechnen
        
        Args:
            training_to: Ende der Training-Periode (YYYYMM)
            
        Returns:
            Start der Backtest-Periode (YYYYMM)
        """
        try:
            year = int(training_to[:4])
            month = int(training_to[4:])
            
            if month == 12:
                next_year = year + 1
                next_month = 1
            else:
                next_year = year
                next_month = month + 1
                
            backtest_start = f"{next_year:04d}{next_month:02d}"
            print(f"✅ Backtest-Periode berechnet: {backtest_start} (nach {training_to})")
            return backtest_start
            
        except Exception as e:
            print(f"❌ Fehler bei Backtest-Perioden-Berechnung: {e}")
            return None
    
    def calculate_csv_hash(self, file_path: str) -> Optional[str]:
        """
        CSV Hash berechnen (identisch zu Step0)
        
        Args:
            file_path: Pfad zur CSV-Datei
            
        Returns:
            MD5 Hash der CSV-Struktur
        """
        try:
            if not os.path.exists(file_path):
                print(f"❌ Datei nicht gefunden: {file_path}")
                return None
            
            # Sample lesen für Hash-Berechnung
            try:
                df_sample = pd.read_csv(file_path, nrows=1000, sep=';')
            except:
                df_sample = pd.read_csv(file_path, nrows=1000)
                
            # Struktur-Info erstellen (identisch zu Step0)
            structure_info = {
                'columns': list(df_sample.columns),
                'dtypes': {col: str(dtype) for col, dtype in df_sample.dtypes.items()},
                'null_counts': df_sample.isnull().sum().to_dict(),
                'unique_counts': {col: df_sample[col].nunique() for col in df_sample.columns},
                'file_size': os.path.getsize(file_path)
            }
            
            # Hash berechnen
            structure_str = json.dumps(structure_info, sort_keys=True)
            csv_hash = hashlib.md5(structure_str.encode()).hexdigest()
            
            print(f"✅ CSV Hash berechnet: {csv_hash[:8]}...")
            return csv_hash
            
        except Exception as e:
            print(f"❌ Fehler bei Hash-Berechnung: {e}")
            return None
    
    def validate_backtest_periods(self, backtest_from: str, backtest_to: str) -> bool:
        """
        Backtest-Perioden validieren
        
        Args:
            backtest_from: Start der Backtest-Periode (YYYYMM)
            backtest_to: Ende der Backtest-Periode (YYYYMM)
            
        Returns:
            True wenn Perioden gültig sind
        """
        try:
            # Format validieren
            if not self._is_valid_period_format(backtest_from) or not self._is_valid_period_format(backtest_to):
                print(f"❌ Ungültiges Backtest-Perioden-Format: {backtest_from} - {backtest_to}")
                return False
            
            # Reihenfolge validieren
            if backtest_from >= backtest_to:
                print(f"❌ Ungültige Backtest-Reihenfolge: {backtest_from} >= {backtest_to}")
                return False
            
            # Mindestdauer validieren (mindestens 1 Monat)
            duration = self._calculate_period_duration(backtest_from, backtest_to)
            if duration < 1:
                print(f"❌ Backtest-Periode zu kurz: {duration} Monate (Minimum: 1)")
                return False
            
            print(f"✅ Backtest-Perioden valid: {backtest_from} - {backtest_to} ({duration} Monate)")
            return True
            
        except Exception as e:
            print(f"❌ Fehler bei Backtest-Perioden-Validierung: {e}")
            return False
    
    def validate_csv_file(self, csv_file_path: str) -> bool:
        """
        CSV-Datei validieren
        
        Args:
            csv_file_path: Pfad zur CSV-Datei
            
        Returns:
            True wenn CSV gültig ist
        """
        try:
            if not os.path.exists(csv_file_path):
                print(f"❌ CSV-Datei nicht gefunden: {csv_file_path}")
                return False
            
            # Dateigröße prüfen
            file_size = os.path.getsize(csv_file_path)
            if file_size == 0:
                print(f"❌ CSV-Datei ist leer: {csv_file_path}")
                return False
            
            # CSV-Struktur prüfen
            try:
                df_sample = pd.read_csv(csv_file_path, nrows=10, sep=';')
            except:
                df_sample = pd.read_csv(csv_file_path, nrows=10)
            
            if len(df_sample.columns) == 0:
                print(f"❌ CSV-Datei hat keine Spalten: {csv_file_path}")
                return False
            
            if len(df_sample) == 0:
                print(f"❌ CSV-Datei hat keine Zeilen: {csv_file_path}")
                return False
            
            print(f"✅ CSV-Datei valid: {csv_file_path} ({file_size} bytes, {len(df_sample.columns)} Spalten)")
            return True
            
        except Exception as e:
            print(f"❌ Fehler bei CSV-Validierung: {e}")
            return False
    
    def _validate_stage0_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Step0 Metadaten validieren
        
        Args:
            metadata: Geladene Metadaten
            
        Returns:
            True wenn Metadaten gültig sind
        """
        try:
            required_keys = ['data_dictionary', 'structure']
            
            for key in required_keys:
                if key not in metadata:
                    print(f"❌ Fehlender Schlüssel in Step0 Metadaten: {key}")
                    return False
            
            # Data Dictionary validieren
            data_dict = metadata.get('data_dictionary', {})
            if 'features' not in data_dict:
                print(f"❌ Fehlende Features in Data Dictionary")
                return False
            
            # Structure validieren
            structure = metadata.get('structure', {})
            if 'dtypes' not in structure:
                print(f"❌ Fehlende Datentypen in Structure")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Fehler bei Metadaten-Validierung: {e}")
            return False
    
    def _is_valid_period_format(self, period: str) -> bool:
        """
        Perioden-Format validieren (YYYYMM)
        
        Args:
            period: Zu validierende Periode
            
        Returns:
            True wenn Format gültig ist
        """
        try:
            if len(period) != 6:
                return False
            
            year = int(period[:4])
            month = int(period[4:])
            
            if year < 1900 or year > 2100:
                return False
            
            if month < 1 or month > 12:
                return False
            
            return True
            
        except:
            return False
    
    def _calculate_period_duration(self, period_from: str, period_to: str) -> int:
        """
        Dauer zwischen zwei Perioden berechnen (in Monaten)
        
        Args:
            period_from: Start-Periode (YYYYMM)
            period_to: End-Periode (YYYYMM)
            
        Returns:
            Dauer in Monaten
        """
        try:
            year_from = int(period_from[:4])
            month_from = int(period_from[4:])
            year_to = int(period_to[:4])
            month_to = int(period_to[4:])
            
            duration = (year_to - year_from) * 12 + (month_to - month_from)
            return duration
            
        except:
            return 0


# Convenience Function
def validate_step1_inputs(
    csv_file_path: str,
    training_from: str,
    training_to: str,
    backtest_from: Optional[str] = None,
    backtest_to: str = "202402"
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Alle Step1 Inputs validieren
    
    Args:
        csv_file_path: Pfad zur CSV-Datei
        training_from: Training von (YYYYMM)
        training_to: Training bis (YYYYMM)
        backtest_from: Backtest von (Optional)
        backtest_to: Backtest bis (YYYYMM)
        
    Returns:
        Tuple (is_valid, stage0_metadata)
    """
    
    handler = Step1InputHandler()
    
    # 1. CSV-Datei validieren
    if not handler.validate_csv_file(csv_file_path):
        return False, None
    
    # 2. Training-Perioden validieren
    if not handler.validate_training_periods(training_from, training_to):
        return False, None
    
    # 3. Step0 Metadaten laden
    stage0_metadata = handler.load_stage0_metadata(csv_file_path)
    if not stage0_metadata:
        return False, None
    
    # 4. Backtest-Perioden validieren
    if backtest_from is None:
        backtest_from = handler.calculate_backtest_period(training_to)
        if not backtest_from:
            return False, None
    
    if not handler.validate_backtest_periods(backtest_from, backtest_to):
        return False, None
    
    print(f"✅ Alle Step1 Inputs validiert!")
    return True, stage0_metadata


if __name__ == "__main__":
    # Test des Input-Handlers
    csv_file = ProjectPaths.main_churn_data_file()
    
    is_valid, metadata = validate_step1_inputs(
        csv_file_path=str(csv_file),
        training_from="201701",
        training_to="202312",
        backtest_to="202402"
    )
    
    if is_valid:
        print(f"✅ Input-Validierung erfolgreich!")
        print(f"📊 Features: {len(metadata.get('data_dictionary', {}).get('features', {}))}")
    else:
        print(f"❌ Input-Validierung fehlgeschlagen!") 
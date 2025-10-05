#!/usr/bin/env python3
import json
from typing import Any, Dict
from pathlib import Path

class DataSchema:
    """DataSchema Klasse mit data_dictionary Attribut"""
    
    def __init__(self):
        self.version = "1.0"
        self.data_dictionary = self._load_data_dictionary()
    
    def _load_data_dictionary(self) -> Dict[str, Any]:
        """Lädt das optimierte Data Dictionary aus der Konfiguration"""
        try:
            # Pfad zum optimierten Data Dictionary
            config_path = Path(__file__).parent / "shared" / "config" / "data_dictionary_optimized.json"
            
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    data_dict = json.load(f)
                    print(f"✅ Data Dictionary geladen: {len(data_dict.get('columns', {}))} Features")
                    return data_dict
            else:
                print(f"⚠️ Data Dictionary nicht gefunden: {config_path}")
                return {"columns": {}}
                
        except Exception as e:
            print(f"❌ Fehler beim Laden des Data Dictionary: {e}")
            return {"columns": {}}


def get_data_schema() -> DataSchema:
    """Gibt eine DataSchema Instanz zurück"""
    return DataSchema()



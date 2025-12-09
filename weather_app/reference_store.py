from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


class ReferenceStore:
    """Stores area-wise reference samples: pairs of (provider_temp, ref_temp)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.data: Dict[str, Any] = {}
        self._load()

    def _load(self):
        try:
            if self.path.exists():
                with open(self.path, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
            else:
                self.data = {}
        except Exception:
            self.data = {}

    def _save(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def add_sample(self, area_name: str, provider_temp: float, ref_temp: float, source: str = 'google'):
        """Add a sample for an area."""
        try:
            area = self.data.get(area_name) or {}
            samples: List[Dict[str, Any]] = area.get('samples') or []
            samples.append({
                'provider_temp': float(provider_temp),
                'ref_temp': float(ref_temp),
                'source': source,
                'timestamp': datetime.now().isoformat(timespec='seconds')
            })
            area['samples'] = samples
            self.data[area_name] = area
            self._save()
        except Exception:
            pass

    def get_samples(self, area_name: str) -> List[Dict[str, Any]]:
        try:
            area = self.data.get(area_name) or {}
            samples = area.get('samples') or []
            return samples
        except Exception:
            return []
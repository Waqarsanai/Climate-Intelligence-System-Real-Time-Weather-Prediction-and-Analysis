from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class CalibrationStore:
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

    def _ensure_areas(self):
        try:
            if not isinstance(self.data, dict):
                self.data = {}
            if not isinstance(self.data.get('areas'), dict):
                self.data['areas'] = {}
        except Exception:
            self.data = {'areas': {}}

    def set_bias(self, bias_c: float, source: str = 'google'):
        self.data['bias_c'] = float(bias_c)
        self.data['source'] = source
        self.data['updated_at'] = datetime.now().isoformat(timespec='seconds')
        self._save()

    def get_bias(self) -> Optional[float]:
        b = self.data.get('bias_c')
        try:
            return float(b) if b is not None else None
        except Exception:
            return None

    def info(self) -> Dict[str, Any]:
        return {
            'bias_c': self.get_bias(),
            'source': self.data.get('source'),
            'updated_at': self.data.get('updated_at'),
        }

    # Per-area calibration methods
    def set_area_bias(self, area_name: str, bias_c: float, source: str = 'google', alpha: float | None = None):
        self._ensure_areas()
        self.data['areas'][area_name] = {
            'bias_c': float(bias_c),
            'source': source,
            'updated_at': datetime.now().isoformat(timespec='seconds'),
            'alpha': float(alpha) if alpha is not None else self.data['areas'].get(area_name, {}).get('alpha')
        }
        self._save()

    def get_area_bias(self, area_name: str) -> Optional[float]:
        try:
            self._ensure_areas()
            entry = self.data['areas'].get(area_name)
            if not entry:
                return None
            return float(entry.get('bias_c'))
        except Exception:
            return None

    def area_info(self, area_name: str) -> Dict[str, Any]:
        self._ensure_areas()
        entry = self.data['areas'].get(area_name)
        if not entry:
            return {'bias_c': None, 'source': None, 'updated_at': None, 'alpha': None}
        return {
            'bias_c': float(entry.get('bias_c')) if entry.get('bias_c') is not None else None,
            'source': entry.get('source'),
            'updated_at': entry.get('updated_at'),
            'alpha': float(entry.get('alpha')) if entry.get('alpha') is not None else None,
        }

    def set_area_alpha(self, area_name: str, alpha: float):
        self._ensure_areas()
        entry = self.data['areas'].get(area_name) or {}
        entry['alpha'] = float(alpha)
        entry['updated_at'] = datetime.now().isoformat(timespec='seconds')
        self.data['areas'][area_name] = entry
        self._save()
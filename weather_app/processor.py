import numpy as np
from datetime import datetime

from .logging_utils import logger


class DataProcessor:
    """Lightweight processing utilities for weather data."""

    def add_time_features(self, timestamp: datetime):
        return {
            'hour': timestamp.hour,
            'dayofyear': timestamp.timetuple().tm_yday,
            'dayofweek': timestamp.weekday(),
            'is_weekend': int(timestamp.weekday() >= 5),
        }

    def normalize_temperature(self, temp: float):
        min_t, max_t = -10.0, 50.0
        return (temp - min_t) / (max_t - min_t)
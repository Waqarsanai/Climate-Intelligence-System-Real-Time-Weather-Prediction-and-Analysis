"""
Data Loader Module for Karachi Weather Prediction System
Loads and processes historical weather data from various sources
"""

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None
    np = None

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

import json

from .logging_utils import logger
from .data_cleaner import WeatherDataCleaner
from .config import CONFIG


class WeatherDataLoader:
    """Load weather data from various sources"""
    
    def __init__(self):
        self.cleaner = WeatherDataCleaner()
        self.data_dir = Path(CONFIG.get('data_dir', 'data'))
        self.data_dir.mkdir(exist_ok=True)
    
    def load_from_open_meteo(self, lat: float, lon: float, 
                            start_date: str, end_date: str):
        """
        Load historical data from Open-Meteo API
        
        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        
        Returns:
            DataFrame with weather data
        """
        if not PANDAS_AVAILABLE:
            logger.error("pandas not available")
            return None
        if not REQUESTS_AVAILABLE:
            logger.error("requests not available")
            return None
            
        logger.info(f"Loading data from Open-Meteo: {start_date} to {end_date}")
        
        url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,"
            f"precipitation,cloudcover,pressure_msl"
            f"&timezone=auto&windspeed_unit=kmh"
        )
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            hourly = data.get('hourly', {})
            times = hourly.get('time', [])
            
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(times),
                'temperature': hourly.get('temperature_2m', []),
                'humidity': hourly.get('relative_humidity_2m', []),
                'wind_speed': hourly.get('wind_speed_10m', []),
                'precipitation': hourly.get('precipitation', []),
                'cloud_cover': hourly.get('cloudcover', []),
                'pressure': hourly.get('pressure_msl', [])
            })
            
            # Clean the data
            df = self.cleaner.clean_dataframe(df)
            
            logger.info(f"Loaded {len(df)} records from Open-Meteo")
            return df
            
        except Exception as e:
            logger.error(f"Error loading from Open-Meteo: {e}")
            return pd.DataFrame() if PANDAS_AVAILABLE else None
    
    def load_from_file(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV or text file"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.error(f"File not found: {filepath}")
            return pd.DataFrame()
        
        logger.info(f"Loading data from file: {filepath}")
        
        try:
            if filepath.suffix == '.csv':
                df = pd.read_csv(filepath)
            elif filepath.suffix == '.txt':
                # Try to parse text file (custom format)
                df = self._parse_text_file(filepath)
            else:
                logger.error(f"Unsupported file format: {filepath.suffix}")
                return pd.DataFrame()
            
            # Ensure timestamp column exists
            if 'timestamp' not in df.columns and 'time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['time'])
            
            # Clean the data
            df = self.cleaner.clean_dataframe(df)
            
            logger.info(f"Loaded {len(df)} records from file")
            return df
            
        except Exception as e:
            logger.error(f"Error loading from file: {e}")
            return pd.DataFrame()
    
    def _parse_text_file(self, filepath: Path) -> pd.DataFrame:
        """Parse custom text file format"""
        # This is a simple parser - adjust based on your file format
        data = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Skip header lines
        start_idx = 0
        for i, line in enumerate(lines):
            if 'Hour' in line and 'Temp' in line:
                start_idx = i + 1
                break
        
        for line in lines[start_idx:]:
            line = line.strip()
            if not line or '---' in line:
                continue
            
            # Try to parse the line (adjust based on format)
            parts = line.split('|')
            if len(parts) >= 3:
                try:
                    hour_str = parts[0].strip()
                    temp_str = parts[1].strip().replace('°C', '').replace('C', '')
                    
                    # Extract hour
                    hour = int(hour_str.split(':')[0])
                    
                    # Extract temperature
                    temp = float(temp_str)
                    
                    # Create timestamp (assuming current date)
                    now = datetime.now()
                    timestamp = now.replace(hour=hour, minute=0, second=0, microsecond=0)
                    
                    data.append({
                        'timestamp': timestamp,
                        'temperature': temp
                    })
                except Exception:
                    continue
        
        return pd.DataFrame(data)
    
    def create_training_dataset(self, df: pd.DataFrame, 
                               train_ratio: float = 0.7,
                               val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets (time-series aware)
        
        Args:
            df: Full dataset
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
        
        Returns:
            train_df, val_df, test_df
        """
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        logger.info(f"Dataset split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def prepare_features_target(self, df: pd.DataFrame, 
                               target_col: str = 'temperature',
                               feature_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target arrays
        
        Args:
            df: DataFrame with features and target
            target_col: Name of target column
            feature_cols: List of feature column names (if None, uses all except target and timestamp)
        
        Returns:
            X (features), y (target)
        """
        if feature_cols is None:
            feature_cols = [col for col in df.columns 
                          if col not in ['timestamp', target_col]]
        
        # Remove rows with missing target
        df = df.dropna(subset=[target_col])
        
        # Fill missing features
        for col in feature_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        return X, y
    
    def save_dataset(self, df: pd.DataFrame, filename: str):
        """Save dataset to CSV"""
        filepath = self.data_dir / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Saved dataset to {filepath}")
    
    def load_dataset(self, filename: str) -> pd.DataFrame:
        """Load dataset from CSV"""
        filepath = self.data_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath, parse_dates=['timestamp'])
            logger.info(f"Loaded dataset from {filepath}")
            return df
        else:
            logger.error(f"Dataset file not found: {filepath}")
            return pd.DataFrame()


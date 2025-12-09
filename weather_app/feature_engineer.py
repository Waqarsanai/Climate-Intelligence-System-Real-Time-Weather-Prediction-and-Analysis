"""
Advanced Feature Engineering Module for Karachi Weather Prediction System
Creates comprehensive features including lag features, moving averages, 
meteorological features, and temporal encodings
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from .logging_utils import logger


class AdvancedFeatureEngineer:
    """Advanced feature engineering for weather prediction"""
    
    def __init__(self):
        self.feature_names = []
        self.lag_periods = [1, 3, 6, 12, 24]  # Hours
        self.rolling_windows = [3, 6, 12, 24]  # Hours
        
    def create_features(self, df: pd.DataFrame, target_col: str = 'temperature') -> pd.DataFrame:
        """
        Create comprehensive feature set from weather DataFrame
        
        Args:
            df: DataFrame with timestamp and weather variables
            target_col: Target variable to predict
        
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting advanced feature engineering...")
        df = df.copy()
        
        # Ensure timestamp is datetime and set as index temporarily
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            df_indexed = df.set_index('timestamp')
        else:
            raise ValueError("DataFrame must have 'timestamp' column")
        
        # Step 1: Temporal features
        df_indexed = self._add_temporal_features(df_indexed)
        
        # Step 2: Lag features
        df_indexed = self._add_lag_features(df_indexed, target_col)
        
        # Step 3: Moving averages and rolling statistics
        df_indexed = self._add_rolling_features(df_indexed, target_col)
        
        # Step 4: Meteorological features
        df_indexed = self._add_meteorological_features(df_indexed)
        
        # Step 5: Interaction features
        df_indexed = self._add_interaction_features(df_indexed)
        
        # Step 6: Cyclical encoding
        df_indexed = self._add_cyclical_features(df_indexed)
        
        # Step 7: Weather condition features
        df_indexed = self._add_weather_condition_features(df_indexed)
        
        # Step 8: Trend features
        df_indexed = self._add_trend_features(df_indexed, target_col)
        
        # Reset index
        df = df_indexed.reset_index()
        
        # Store feature names (excluding target and timestamp)
        self.feature_names = [col for col in df.columns 
                             if col not in ['timestamp', target_col]]
        
        logger.info(f"Created {len(self.feature_names)} features")
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features from timestamp"""
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['day_of_year'] = df.index.dayofyear
        df['week_of_year'] = df.index.isocalendar().week
        df['quarter'] = df.index.quarter
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        df['is_month_start'] = df.index.is_month_start.astype(int)
        df['is_month_end'] = df.index.is_month_end.astype(int)
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Add lag features for target variable and other important features"""
        # Lag features for target
        for lag in self.lag_periods:
            df[f'{target_col}_lag_{lag}h'] = df[target_col].shift(lag)
        
        # Lag features for other important variables
        for col in ['humidity', 'wind_speed', 'pressure']:
            if col in df.columns:
                for lag in [1, 3, 6, 12]:
                    df[f'{col}_lag_{lag}h'] = df[col].shift(lag)
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Add rolling statistics (mean, std, min, max)"""
        for window in self.rolling_windows:
            # Rolling mean
            df[f'{target_col}_rolling_mean_{window}h'] = df[target_col].rolling(
                window=window, min_periods=1
            ).mean()
            
            # Rolling std
            df[f'{target_col}_rolling_std_{window}h'] = df[target_col].rolling(
                window=window, min_periods=1
            ).std().fillna(0)
            
            # Rolling min
            df[f'{target_col}_rolling_min_{window}h'] = df[target_col].rolling(
                window=window, min_periods=1
            ).min()
            
            # Rolling max
            df[f'{target_col}_rolling_max_{window}h'] = df[target_col].rolling(
                window=window, min_periods=1
            ).max()
            
            # Rolling range
            df[f'{target_col}_rolling_range_{window}h'] = (
                df[f'{target_col}_rolling_max_{window}h'] - 
                df[f'{target_col}_rolling_min_{window}h']
            )
        
        # Exponential moving averages
        for alpha in [0.1, 0.3, 0.5]:
            df[f'{target_col}_ema_{alpha}'] = df[target_col].ewm(
                alpha=alpha, adjust=False
            ).mean()
        
        return df
    
    def _add_meteorological_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add meteorological derived features"""
        # Heat Index (simplified formula)
        if 'temperature' in df.columns and 'humidity' in df.columns:
            temp = df['temperature']
            hum = df['humidity']
            # Simplified heat index formula (valid for temp > 27°C and hum > 40%)
            df['heat_index'] = (
                0.5 * (temp + 61.0 + ((temp - 68.0) * 1.2) + (hum * 0.094))
            )
            # Only apply when conditions are met
            mask = (temp > 27) & (hum > 40)
            df.loc[~mask, 'heat_index'] = temp[~mask]
        
        # Dew Point (simplified Magnus formula)
        if 'temperature' in df.columns and 'humidity' in df.columns:
            temp = df['temperature']
            hum = df['humidity']
            # Magnus formula approximation
            a = 17.27
            b = 237.7
            alpha = ((a * temp) / (b + temp)) + np.log(hum / 100.0)
            df['dew_point'] = (b * alpha) / (a - alpha)
        
        # Wind Chill (for temperatures < 10°C)
        if 'temperature' in df.columns and 'wind_speed' in df.columns:
            temp = df['temperature']
            wind = df['wind_speed']
            # Wind chill formula (valid for temp < 10°C and wind > 4.8 km/h)
            mask = (temp < 10) & (wind > 4.8)
            df['wind_chill'] = temp.copy()
            df.loc[mask, 'wind_chill'] = (
                13.12 + 0.6215 * temp[mask] - 
                11.37 * (wind[mask] ** 0.16) + 
                0.3965 * temp[mask] * (wind[mask] ** 0.16)
            )
        
        # Pressure gradient (rate of change)
        if 'pressure' in df.columns:
            df['pressure_gradient'] = df['pressure'].diff()
            df['pressure_gradient_3h'] = df['pressure'].diff(3)
            df['pressure_gradient_6h'] = df['pressure'].diff(6)
        
        # Apparent Temperature (feels like)
        if 'temperature' in df.columns and 'wind_speed' in df.columns and 'humidity' in df.columns:
            temp = df['temperature']
            wind = df['wind_speed']
            hum = df['humidity']
            # Simplified apparent temperature
            df['apparent_temp'] = (
                temp - (wind * 0.7) + (hum / 100.0 * 2.0)
            )
        
        # Temperature-Humidity Index (THI)
        if 'temperature' in df.columns and 'humidity' in df.columns:
            temp = df['temperature']
            hum = df['humidity']
            df['thi'] = temp - (0.55 * (1 - hum / 100.0) * (temp - 14.4))
        
        return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between variables"""
        # Temperature-Humidity interaction
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df['temp_humidity_interaction'] = df['temperature'] * df['humidity'] / 100.0
            df['temp_humidity_ratio'] = df['temperature'] / (df['humidity'] + 1)
        
        # Wind-Pressure interaction
        if 'wind_speed' in df.columns and 'pressure' in df.columns:
            df['wind_pressure_interaction'] = df['wind_speed'] * df['pressure'] / 1000.0
        
        # Temperature-Wind interaction
        if 'temperature' in df.columns and 'wind_speed' in df.columns:
            df['temp_wind_interaction'] = df['temperature'] * df['wind_speed']
        
        # Cloud cover and precipitation interaction
        if 'cloud_cover' in df.columns and 'precipitation' in df.columns:
            df['cloud_precip_interaction'] = df['cloud_cover'] * df['precipitation']
        
        return df
    
    def _add_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cyclical encoding for temporal features"""
        # Hour cyclical encoding
        if 'hour' in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
        
        # Day of week cyclical encoding
        if 'day_of_week' in df.columns:
            df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
            df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)
        
        # Month cyclical encoding
        if 'month' in df.columns:
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
        
        # Day of year cyclical encoding
        if 'day_of_year' in df.columns:
            df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.0)
            df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.0)
        
        return df
    
    def _add_weather_condition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features based on weather conditions"""
        # Rain probability (simplified based on humidity, pressure, cloud cover)
        if all(col in df.columns for col in ['humidity', 'pressure', 'cloud_cover']):
            # Higher humidity + lower pressure + higher cloud cover = higher rain probability
            df['rain_probability'] = (
                (df['humidity'] / 100.0) * 0.4 +
                ((1020 - df['pressure']) / 40.0).clip(0, 1) * 0.3 +
                (df['cloud_cover'] / 100.0) * 0.3
            ).clip(0, 1)
        
        # Weather severity index
        if 'temperature' in df.columns:
            # Extreme temperatures
            df['extreme_heat'] = (df['temperature'] > 40).astype(int)
            df['extreme_cold'] = (df['temperature'] < 10).astype(int)
        
        # Comfort index
        if all(col in df.columns for col in ['temperature', 'humidity', 'wind_speed']):
            # Optimal: 20-26°C, 40-60% humidity, moderate wind
            temp_score = 1 - np.abs(df['temperature'] - 23) / 20.0
            hum_score = 1 - np.abs(df['humidity'] - 50) / 50.0
            wind_score = 1 - np.abs(df['wind_speed'] - 8) / 20.0
            df['comfort_index'] = (temp_score + hum_score + wind_score) / 3.0
        
        return df
    
    def _add_trend_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Add trend and momentum features"""
        # First and second derivatives (rate of change)
        df[f'{target_col}_diff_1h'] = df[target_col].diff(1)
        df[f'{target_col}_diff_3h'] = df[target_col].diff(3)
        df[f'{target_col}_diff_6h'] = df[target_col].diff(6)
        
        # Acceleration (second derivative)
        df[f'{target_col}_acceleration'] = df[f'{target_col}_diff_1h'].diff(1)
        
        # Momentum (moving average of differences)
        for window in [3, 6]:
            df[f'{target_col}_momentum_{window}h'] = df[f'{target_col}_diff_1h'].rolling(
                window=window, min_periods=1
            ).mean()
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.feature_names.copy()
    
    def select_features(self, df: pd.DataFrame, method: str = 'correlation', 
                       top_k: int = 50) -> List[str]:
        """
        Select top features based on correlation or importance
        
        Args:
            df: DataFrame with features and target
            method: 'correlation' or 'variance'
            top_k: Number of top features to select
        
        Returns:
            List of selected feature names
        """
        if method == 'correlation' and 'temperature' in df.columns:
            # Select features with highest correlation to target
            correlations = df[self.feature_names].corrwith(df['temperature']).abs()
            top_features = correlations.nlargest(top_k).index.tolist()
            return top_features
        elif method == 'variance':
            # Select features with highest variance
            variances = df[self.feature_names].var()
            top_features = variances.nlargest(top_k).index.tolist()
            return top_features
        else:
            return self.feature_names[:top_k]
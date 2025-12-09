"""
Comprehensive Data Cleaning Module for Karachi Weather Prediction System
Handles missing values, outliers, unit conversion, duplicates, and data quality issues
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from .logging_utils import logger


class WeatherDataCleaner:
    """Comprehensive data cleaning for weather data"""
    
    # Realistic bounds for Karachi weather (in Celsius)
    KARACHI_BOUNDS = {
        'temperature': (-5, 50),  # Karachi can get cold in winter, hot in summer
        'humidity': (0, 100),
        'wind_speed': (0, 100),  # km/h
        'pressure': (980, 1050),  # hPa
        'precipitation': (0, 200),  # mm (for extreme events)
        'cloud_cover': (0, 100),  # percentage
    }
    
    def __init__(self):
        self.stats = {}
        self.cleaning_report = {}
        
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive cleaning pipeline for weather DataFrame
        
        Args:
            df: DataFrame with columns: timestamp, temperature, humidity, wind_speed, 
                pressure, precipitation, cloud_cover (optional)
        
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting comprehensive data cleaning...")
        original_len = len(df)
        df = df.copy()
        
        # Step 1: Ensure timestamp is datetime
        df = self._fix_timestamps(df)
        
        # Step 2: Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Step 3: Remove duplicates
        df = self._remove_duplicates(df)
        
        # Step 4: Fix temperature units (F to C conversion)
        df = self._fix_temperature_units(df)
        
        # Step 5: Handle missing values
        df = self._handle_missing_values(df)
        
        # Step 6: Remove outliers
        df = self._remove_outliers(df)
        
        # Step 7: Fix unrealistic values
        df = self._fix_unrealistic_values(df)
        
        # Step 8: Validate data ranges
        df = self._validate_ranges(df)
        
        # Step 9: Interpolate remaining gaps
        df = self._interpolate_gaps(df)
        
        cleaned_len = len(df)
        removed = original_len - cleaned_len
        
        self.cleaning_report = {
            'original_rows': original_len,
            'cleaned_rows': cleaned_len,
            'removed_rows': removed,
            'removal_percentage': (removed / original_len * 100) if original_len > 0 else 0
        }
        
        logger.info(f"Data cleaning complete: {removed} rows removed ({removed/original_len*100:.2f}%)")
        return df
    
    def _fix_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert timestamp column to datetime and handle timezone issues"""
        if 'timestamp' not in df.columns and 'time' in df.columns:
            df['timestamp'] = df['time']
        
        if 'timestamp' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                except Exception as e:
                    logger.warning(f"Timestamp conversion error: {e}")
                    # Try alternative formats
                    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d']:
                        try:
                            df['timestamp'] = pd.to_datetime(df['timestamp'], format=fmt, errors='coerce')
                            break
                        except:
                            continue
        
        # Remove rows with invalid timestamps
        invalid_timestamps = df['timestamp'].isna()
        if invalid_timestamps.sum() > 0:
            logger.warning(f"Removing {invalid_timestamps.sum()} rows with invalid timestamps")
            df = df[~invalid_timestamps].copy()
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows, keeping the first occurrence"""
        before = len(df)
        df = df.drop_duplicates(subset=['timestamp'], keep='first').reset_index(drop=True)
        after = len(df)
        if before != after:
            logger.info(f"Removed {before - after} duplicate rows")
        return df
    
    def _fix_temperature_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and convert Fahrenheit to Celsius"""
        if 'temperature' not in df.columns:
            return df
        
        # If temperatures are consistently above 50, likely in Fahrenheit
        temp_col = df['temperature'].dropna()
        if len(temp_col) > 0:
            median_temp = temp_col.median()
            # If median is above 50°C, likely Fahrenheit (Karachi rarely exceeds 45°C)
            if median_temp > 50:
                logger.warning(f"Detected likely Fahrenheit temperatures (median: {median_temp:.1f}). Converting to Celsius.")
                df['temperature'] = (df['temperature'] - 32) * 5/9
                # Also convert feels_like if present
                if 'feels_like' in df.columns:
                    df['feels_like'] = (df['feels_like'] - 32) * 5/9
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with intelligent imputation"""
        numeric_cols = ['temperature', 'humidity', 'wind_speed', 'pressure', 
                       'precipitation', 'cloud_cover', 'feels_like']
        
        for col in numeric_cols:
            if col not in df.columns:
                continue
            
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                logger.info(f"Handling {missing_count} missing values in {col}")
                
                # For time-series data, use forward fill then backward fill
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                
                # If still missing, use median
                if df[col].isna().sum() > 0:
                    median_val = df[col].median()
                    if pd.notna(median_val):
                        df[col] = df[col].fillna(median_val)
                    else:
                        # Use default values
                        defaults = {
                            'temperature': 28.0,
                            'humidity': 60.0,
                            'wind_speed': 8.0,
                            'pressure': 1013.0,
                            'precipitation': 0.0,
                            'cloud_cover': 30.0,
                            'feels_like': 28.0
                        }
                        df[col] = df[col].fillna(defaults.get(col, 0.0))
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using IQR method and domain knowledge"""
        numeric_cols = ['temperature', 'humidity', 'wind_speed', 'pressure', 
                       'precipitation', 'cloud_cover']
        
        outliers_removed = 0
        
        for col in numeric_cols:
            if col not in df.columns:
                continue
            
            bounds = self.KARACHI_BOUNDS.get(col, (None, None))
            min_val, max_val = bounds
            
            # Remove values outside realistic bounds
            before = len(df)
            if min_val is not None:
                df = df[df[col] >= min_val].copy()
            if max_val is not None:
                df = df[df[col] <= max_val].copy()
            
            # Also use IQR method for extreme outliers
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR  # More lenient (3*IQR instead of 1.5*IQR)
            upper_bound = Q3 + 3 * IQR
            
            # Apply bounds, but be more lenient for weather data
            if min_val is not None:
                lower_bound = max(lower_bound, min_val)
            if max_val is not None:
                upper_bound = min(upper_bound, max_val)
            
            before_iqr = len(df)
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)].copy()
            outliers_removed += (before_iqr - len(df))
        
        if outliers_removed > 0:
            logger.info(f"Removed {outliers_removed} outlier rows")
        
        return df
    
    def _fix_unrealistic_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix unrealistic spikes and anomalies"""
        # Fix unrealistic temperature spikes (changes > 10°C in 1 hour are suspicious)
        if 'temperature' in df.columns:
            temp_diff = df['temperature'].diff().abs()
            unrealistic_spikes = temp_diff > 10
            if unrealistic_spikes.sum() > 0:
                logger.warning(f"Found {unrealistic_spikes.sum()} unrealistic temperature spikes")
                # Smooth them using rolling median
                df.loc[unrealistic_spikes, 'temperature'] = df['temperature'].rolling(
                    window=3, center=True, min_periods=1
                ).median()[unrealistic_spikes]
        
        # Fix negative precipitation
        if 'precipitation' in df.columns:
            negative_precip = df['precipitation'] < 0
            if negative_precip.sum() > 0:
                df.loc[negative_precip, 'precipitation'] = 0
        
        # Fix humidity > 100%
        if 'humidity' in df.columns:
            over_100 = df['humidity'] > 100
            if over_100.sum() > 0:
                df.loc[over_100, 'humidity'] = 100
        
        return df
    
    def _validate_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final validation of all values within acceptable ranges"""
        for col, (min_val, max_val) in self.KARACHI_BOUNDS.items():
            if col not in df.columns:
                continue
            
            # Clip values to bounds
            if min_val is not None:
                df[col] = df[col].clip(lower=min_val)
            if max_val is not None:
                df[col] = df[col].clip(upper=max_val)
        
        return df
    
    def _interpolate_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolate small gaps in time series"""
        numeric_cols = ['temperature', 'humidity', 'wind_speed', 'pressure', 
                       'precipitation', 'cloud_cover']
        
        # Ensure timestamp is the index for interpolation
        df_indexed = df.set_index('timestamp')
        
        for col in numeric_cols:
            if col not in df_indexed.columns:
                continue
            
            # Interpolate using time-aware interpolation
            df_indexed[col] = df_indexed[col].interpolate(method='time', limit=3)
        
        # Reset index
        df = df_indexed.reset_index()
        
        return df
    
    def get_cleaning_report(self) -> Dict[str, Any]:
        """Get detailed cleaning report"""
        return self.cleaning_report.copy()


"""
Advanced Predictor Module for Karachi Weather Prediction System
Uses ensemble models and advanced features for high-accuracy predictions
"""

import numpy as np
import pandas as pd
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

from .logging_utils import logger
from .config import CONFIG
from .feature_engineer import AdvancedFeatureEngineer
from .ensemble import EnsembleModel, ModelBlender
from .data_cleaner import WeatherDataCleaner


class AdvancedKarachiPredictor:
    """Advanced predictor using ensemble models and comprehensive features"""
    
    def __init__(self):
        self.ensemble = None
        self.model_trainer = None
        self.feature_engineer = AdvancedFeatureEngineer()
        self.data_cleaner = WeatherDataCleaner()
        self.feature_names = []
        self.metrics = {}  # Store model metrics
        self.is_trained = False
        self.baseline_weather = None
        
    def load_ensemble(self, model_path: str):
        """Load trained ensemble model"""
        model_path = Path(model_path)
        
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return False
        
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
            
            self.ensemble = data.get('ensemble')
            self.model_trainer = data.get('model_trainer')
            self.feature_names = data.get('feature_names', [])
            self.metrics = data.get('metrics', {})  # Store metrics
            self.is_trained = True
            
            logger.info(f"Loaded ensemble model from {model_path}")
            logger.info(f"Feature count: {len(self.feature_names)}")
            if self.metrics:
                logger.info(f"Model metrics: R²={self.metrics.get('r2', 'N/A'):.4f}, RMSE={self.metrics.get('rmse', 'N/A'):.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    
    def load_latest_model(self):
        """Load the latest trained model from models directory"""
        model_dir = Path(CONFIG['model_dir'])
        ensemble_files = sorted(model_dir.glob('ensemble_v*.pkl'), reverse=True)
        
        if ensemble_files:
            return self.load_ensemble(ensemble_files[0])
        else:
            logger.warning("No ensemble models found")
            return False
    
    def _prepare_features_for_prediction(self, current_weather: Dict, 
                                        historical_data: Optional[pd.DataFrame] = None,
                                        hours: int = 24) -> np.ndarray:
        """
        Prepare features for prediction
        
        Args:
            current_weather: Current weather conditions
            historical_data: Historical weather data (if available)
            hours: Number of hours to predict
        
        Returns:
            Feature matrix
        """
        now = datetime.now()
        
        # Create DataFrame with future timestamps
        timestamps = [now + timedelta(hours=i) for i in range(hours)]
        
        # Build base DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': [current_weather.get('temperature', 28.0)] * hours,
            'humidity': [current_weather.get('humidity', 60.0)] * hours,
            'wind_speed': [current_weather.get('wind_speed', 8.0)] * hours,
            'pressure': [current_weather.get('pressure', 1013.0)] * hours,
            'precipitation': [current_weather.get('precipitation', 0.0)] * hours,
            'cloud_cover': [current_weather.get('cloud_cover', 30.0)] * hours,
        })
        
        # If historical data available, use it to create lag features
        if historical_data is not None and len(historical_data) > 0:
            # Combine historical and future data
            hist_df = historical_data.copy()
            hist_df = hist_df.sort_values('timestamp').reset_index(drop=True)
            
            # Append future data
            combined_df = pd.concat([hist_df, df], ignore_index=True)
            combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
            
            # Create features
            df_features = self.feature_engineer.create_features(
                combined_df, target_col='temperature'
            )
            
            # Extract only future features
            future_start_idx = len(hist_df)
            future_features = df_features.iloc[future_start_idx:]
        else:
            # Create features without historical context
            df_features = self.feature_engineer.create_features(
                df, target_col='temperature'
            )
            future_features = df_features
        
        # Select only the features used during training
        available_features = [f for f in self.feature_names if f in future_features.columns]
        missing_features = [f for f in self.feature_names if f not in future_features.columns]
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features[:10]}...")
            # Fill missing features with zeros or defaults
            for feat in missing_features:
                future_features[feat] = 0.0
        
        # Ensure correct order
        X = future_features[self.feature_names].values
        
        return X
    
    def predict_temperature(self, current_weather: Dict, hours: int = 24,
                          historical_data: Optional[pd.DataFrame] = None) -> List[Dict]:
        """
        Predict temperature for next N hours
        
        Args:
            current_weather: Current weather conditions
            hours: Number of hours to predict
            historical_data: Historical weather data (optional, improves accuracy)
        
        Returns:
            List of predictions with 'time' and 'temp' keys
        """
        if not self.is_trained or self.ensemble is None or self.model_trainer is None:
            logger.warning("Model not trained. Using fallback prediction.")
            return self._fallback_prediction(current_weather, hours)
        
        try:
            # Prepare features
            X = self._prepare_features_for_prediction(
                current_weather, historical_data, hours
            )
            
            # Get predictions from all models
            model_predictions = {}
            for model_name in self.model_trainer.models.keys():
                try:
                    pred = self.model_trainer.predict(X, model_name=model_name)
                    model_predictions[model_name] = pred
                except Exception as e:
                    logger.warning(f"Prediction failed for {model_name}: {e}")
            
            if not model_predictions:
                return self._fallback_prediction(current_weather, hours)
            
            # Get ensemble prediction
            ensemble_pred = self.ensemble.predict_weighted_average(model_predictions)
            
            # Apply smoothing
            blender = ModelBlender()
            smoothed_pred = blender._apply_smoothing(ensemble_pred)
            
            # Create output
            now = datetime.now()
            predictions = []
            for i, temp in enumerate(smoothed_pred):
                t = now + timedelta(hours=i)
                predictions.append({
                    'time': t.strftime('%Y-%m-%d %H:%M'),
                    'temp': float(temp)
                })
            
            logger.info(f"Generated {len(predictions)} predictions using ensemble model")
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._fallback_prediction(current_weather, hours)
    
    def _fallback_prediction(self, current_weather: Dict, hours: int) -> List[Dict]:
        """Fallback prediction using simple model"""
        base_temp = current_weather.get('temperature', 28.0)
        now = datetime.now()
        predictions = []
        
        for i in range(hours):
            t = now + timedelta(hours=i)
            hour = t.hour
            
            # Simple diurnal cycle
            daily_cycle = 3.5 * np.sin(2 * np.pi * (hour - 6) / 24.0)
            temp = base_temp + daily_cycle + np.random.normal(0, 0.5)
            
            predictions.append({
                'time': t.strftime('%Y-%m-%d %H:%M'),
                'temp': float(temp)
            })
        
        return predictions
    
    def predict_extended_forecast(self, current_weather: Dict, days: int = 7,
                                 historical_data: Optional[pd.DataFrame] = None) -> List[Dict]:
        """Predict extended forecast for multiple days"""
        hours = days * 24
        return self.predict_temperature(current_weather, hours, historical_data)
    
    def predict_multiple_variables(self, current_weather: Dict, hours: int = 24,
                                  historical_data: Optional[pd.DataFrame] = None) -> List[Dict]:
        """
        Predict multiple weather variables
        
        Args:
            current_weather: Current weather conditions
            hours: Number of hours to predict
            historical_data: Historical weather data
        
        Returns:
            List of predictions with multiple variables
        """
        # For now, focus on temperature. Can be extended to predict other variables
        temp_predictions = self.predict_temperature(current_weather, hours, historical_data)
        
        # Add other variables based on temperature and current conditions
        predictions = []
        base_humidity = current_weather.get('humidity', 60.0)
        base_wind = current_weather.get('wind_speed', 8.0)
        base_pressure = current_weather.get('pressure', 1013.0)
        
        for pred in temp_predictions:
            temp = pred['temp']
            hour = datetime.strptime(pred['time'], '%Y-%m-%d %H:%M').hour
            
            # Estimate humidity (higher at night, lower during day)
            if 6 <= hour <= 18:
                humidity = base_humidity - 10 + np.random.normal(0, 3)
            else:
                humidity = base_humidity + 5 + np.random.normal(0, 3)
            humidity = np.clip(humidity, 20, 95)
            
            # Estimate wind speed (higher during day)
            if 12 <= hour <= 17:
                wind = base_wind + 2 + np.random.normal(0, 1)
            else:
                wind = base_wind + np.random.normal(0, 1)
            wind = np.clip(wind, 0, 30)
            
            # Estimate pressure (small variations)
            pressure = base_pressure + np.random.normal(0, 1)
            pressure = np.clip(pressure, 980, 1050)
            
            predictions.append({
                'time': pred['time'],
                'temperature': temp,
                'humidity': float(humidity),
                'wind_speed': float(wind),
                'pressure': float(pressure),
                'feels_like': float(temp + (0.3 * (humidity/100) * temp))
            })
        
        return predictions
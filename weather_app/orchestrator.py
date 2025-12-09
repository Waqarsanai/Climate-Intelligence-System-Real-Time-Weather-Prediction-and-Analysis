"""
Main Orchestrator - Controls the entire weather prediction system
Coordinates data loading, cleaning, feature engineering, training, and prediction
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
import traceback
import numpy as np

from .config import CONFIG
from .logging_utils import logger
from .data_loader import WeatherDataLoader
from .data_cleaner import WeatherDataCleaner
from .feature_engineer import AdvancedFeatureEngineer
from .model_trainer import ModelTrainer
from .ensemble import EnsembleModel
from .advanced_predictor import AdvancedKarachiPredictor
from .fetcher import RealTimeWeatherDataFetcher

class WeatherSystemOrchestrator:
    """
    Main controller that orchestrates all components of the weather prediction system
    """
    
    def __init__(self):
        """Initialize all components"""
        self.data_loader = WeatherDataLoader()
        self.cleaner = WeatherDataCleaner()
        self.feature_engineer = AdvancedFeatureEngineer()
        self.model_trainer = ModelTrainer()
        self.ensemble = EnsembleModel()
        self.predictor = AdvancedKarachiPredictor()
        self.fetcher = RealTimeWeatherDataFetcher()
        
        self.is_trained = False
        self.training_status = {
            'status': 'not_started',
            'progress': 0,
            'message': '',
            'error': None
        }
    
    def train_pipeline(self, 
                      data_source: str = 'api',
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      retrain: bool = False) -> Dict[str, Any]:
        """
        Complete training pipeline: data_loader → cleaner → features → model_trainer
        
        Args:
            data_source: 'api' or 'file'
            start_date: Start date for data (YYYY-MM-DD)
            end_date: End date for data (YYYY-MM-DD)
            retrain: Whether to retrain even if model exists
        
        Returns:
            Dictionary with training results and metrics
        """
        try:
            self.training_status = {
                'status': 'in_progress',
                'progress': 0,
                'message': 'Starting training pipeline...',
                'error': None
            }
            
            # Step 1: Load Data
            logger.info("Step 1: Loading data...")
            self.training_status['progress'] = 10
            self.training_status['message'] = 'Loading data...'
            
            if data_source == 'api':
                if not start_date:
                    end_date = datetime.now().strftime('%Y-%m-%d')
                    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
                
                df = self.data_loader.load_from_open_meteo(
                    CONFIG['coordinates']['lat'],
                    CONFIG['coordinates']['lon'],
                    start_date,
                    end_date
                )
            else:
                # Load from file
                df = self.data_loader.load_dataset('karachi_weather_historical.csv')
            
            if df is None or df.empty:
                logger.warning("No data from source; generating synthetic training data")
                try:
                    if not CONFIG.get('allow_synthetic_training', False):
                        raise ValueError("Synthetic training disabled by config")
                    import pandas as pd
                    base = self.fetcher.fetch() or {}
                    base_temp = float(base.get('temperature') or 30.0)
                    base_hum = float(base.get('humidity') or 60.0)
                    base_wind = float(base.get('wind_speed') or 8.0)
                    base_press = float(base.get('pressure') or 1013.0)
                    base_prec = float(base.get('precipitation') or 0.0)
                    base_cloud = float(base.get('cloud_cover') or 30.0)
                    start_dt = datetime.now() - timedelta(days=60)
                    stamps = [start_dt + timedelta(hours=i) for i in range(60*24)]
                    temps = []
                    hums = []
                    winds = []
                    press = []
                    precs = []
                    clouds = []
                    for i, ts in enumerate(stamps):
                        hour = ts.hour
                        daily = 3.0 * np.sin(2 * np.pi * (hour - 6) / 24.0)
                        seasonal = 1.0 * np.sin(2 * np.pi * (i / (24*14)))
                        t = base_temp + daily + seasonal + np.random.normal(0, 0.4)
                        h = np.clip(base_hum + (20 if hour < 6 or hour > 18 else -10) + np.random.normal(0, 3), 20, 95)
                        w = np.clip(base_wind + np.random.normal(0, 1.5), 0, 40)
                        p = base_press + np.random.normal(0, 1.2)
                        c = np.clip(base_cloud + (15 if hour < 10 else -5) + np.random.normal(0, 10), 0, 100)
                        pr = max(0.0, base_prec + np.random.normal(0, 0.2))
                        temps.append(t)
                        hums.append(h)
                        winds.append(w)
                        press.append(p)
                        clouds.append(c)
                        precs.append(pr)
                    df = pd.DataFrame({
                        'timestamp': stamps,
                        'temperature': temps,
                        'humidity': hums,
                        'wind_speed': winds,
                        'precipitation': precs,
                        'cloud_cover': clouds,
                        'pressure': press,
                    })
                    logger.info(f"Synthesized {len(df)} records for training")
                except Exception as se:
                    raise ValueError("No data available. Please check data source.")
            
            logger.info(f"Loaded {len(df)} records")
            self.training_status['progress'] = 20
            
            # Step 2: Clean Data
            logger.info("Step 2: Cleaning data...")
            self.training_status['message'] = 'Cleaning data...'
            df_cleaned = self.cleaner.clean_dataframe(df)
            logger.info(f"Cleaned data: {len(df_cleaned)} records")
            self.training_status['progress'] = 30
            
            # Step 3: Feature Engineering
            logger.info("Step 3: Creating features...")
            self.training_status['message'] = 'Creating features...'
            df_features = self.feature_engineer.create_features(
                df_cleaned, target_col='temperature'
            )
            feature_names = self.feature_engineer.get_feature_names()
            logger.info(f"Created {len(feature_names)} features")
            self.training_status['progress'] = 40
            
            # Step 4: Prepare Train/Val/Test Split
            logger.info("Step 4: Preparing train/validation/test split...")
            self.training_status['message'] = 'Preparing datasets...'
            train_df, val_df, test_df = self.data_loader.create_training_dataset(
                df_features, train_ratio=0.7, val_ratio=0.15
            )
            
            X_train, y_train = self.data_loader.prepare_features_target(
                train_df, target_col='temperature', feature_cols=feature_names
            )
            X_val, y_val = self.data_loader.prepare_features_target(
                val_df, target_col='temperature', feature_cols=feature_names
            )
            X_test, y_test = self.data_loader.prepare_features_target(
                test_df, target_col='temperature', feature_cols=feature_names
            )
            self.training_status['progress'] = 50
            
            # Step 5: Train Models
            logger.info("Step 5: Training models...")
            self.training_status['message'] = 'Training models...'
            model_results = self.model_trainer.train_all_models(
                X_train, y_train, X_val, y_val, feature_names
            )
            self.training_status['progress'] = 70
            
            # Step 6: Create Ensemble
            logger.info("Step 6: Creating ensemble...")
            self.training_status['message'] = 'Creating ensemble...'
            model_predictions = {}
            for model_name in self.model_trainer.models.keys():
                try:
                    pred = self.model_trainer.predict(X_val, model_name=model_name)
                    model_predictions[model_name] = pred
                except Exception as e:
                    logger.warning(f"Could not get predictions from {model_name}: {e}")
            
            if model_predictions:
                self.ensemble.fit_weighted_average(
                    X_train, y_train, X_val, y_val, model_predictions
                )
            
            self.training_status['progress'] = 85
            
            # Step 7: Evaluate on Test Set
            logger.info("Step 7: Evaluating on test set...")
            self.training_status['message'] = 'Evaluating models...'
            test_predictions = {}
            for model_name in self.model_trainer.models.keys():
                try:
                    pred = self.model_trainer.predict(X_test, model_name=model_name)
                    test_predictions[model_name] = pred
                except Exception as e:
                    logger.warning(f"Could not get test predictions from {model_name}: {e}")
            
            ensemble_test_pred = self.ensemble.predict_weighted_average(test_predictions)
            test_metrics = self.model_trainer._calculate_metrics(y_test, ensemble_test_pred)
            self.training_status['progress'] = 95
            
            # Step 8: Save Models
            logger.info("Step 8: Saving models...")
            self.training_status['message'] = 'Saving models...'
            self._save_models(feature_names, test_metrics)
            self.training_status['progress'] = 100
            
            # Update predictor
            self.predictor.ensemble = self.ensemble
            self.predictor.model_trainer = self.model_trainer
            self.predictor.feature_names = feature_names
            self.predictor.is_trained = True
            # Ensure UI can read metrics immediately without requiring reload
            try:
                self.predictor.metrics = {k: float(v) for k, v in test_metrics.items()}
            except Exception:
                self.predictor.metrics = test_metrics
            
            self.is_trained = True
            self.training_status = {
                'status': 'completed',
                'progress': 100,
                'message': 'Training completed successfully!',
                'error': None
            }
            
            return {
                'success': True,
                'metrics': test_metrics,
                'model_results': model_results,
                'feature_count': len(feature_names),
                'train_size': len(X_train),
                'val_size': len(X_val),
                'test_size': len(X_test)
            }
            
        except Exception as e:
            error_msg = str(e)
            error_trace = traceback.format_exc()
            logger.error(f"Training pipeline error: {error_msg}\n{error_trace}")
            
            self.training_status = {
                'status': 'failed',
                'progress': 0,
                'message': f'Training failed: {error_msg}',
                'error': error_msg
            }
            
            return {
                'success': False,
                'error': error_msg,
                'traceback': error_trace
            }
    
    def _save_models(self, feature_names: list, metrics: Dict):
        """Save trained models"""
        import pickle
        import json
        
        model_dir = Path(CONFIG['model_dir'])
        model_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save individual models
        for model_name, model in self.model_trainer.models.items():
            if model_name in ['lstm', 'bilstm', 'gru', 'cnn_lstm']:
                # TensorFlow models
                model_path = model_dir / f"{model_name}_v{timestamp}.h5"
                try:
                    model.save(str(model_path))
                except Exception as e:
                    logger.warning(f"Could not save {model_name}: {e}")
            else:
                # Scikit-learn/XGBoost models
                model_path = model_dir / f"{model_name}_v{timestamp}.pkl"
                try:
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                except Exception as e:
                    logger.warning(f"Could not save {model_name}: {e}")
        
        # Save ensemble
        ensemble_path = model_dir / f"ensemble_v{timestamp}.pkl"
        with open(ensemble_path, 'wb') as f:
            pickle.dump({
                'ensemble': self.ensemble,
                'model_trainer': self.model_trainer,
                'feature_names': feature_names,
                'metrics': metrics,
                'timestamp': timestamp
            }, f)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'metrics': {k: float(v) for k, v in metrics.items()},
            'feature_count': len(feature_names),
            'best_model': self.model_trainer.best_model_name
        }
        
        metadata_path = model_dir / f"training_metadata_v{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Models saved to {model_dir}")
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load trained model
        
        Args:
            model_path: Path to model file (if None, loads latest)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if model_path:
                success = self.predictor.load_ensemble(model_path)
            else:
                success = self.predictor.load_latest_model()
            
            if success:
                self.is_trained = True
                logger.info("Model loaded successfully")
                return True
            else:
                logger.warning("Failed to load model")
                return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict(self, hours: int = 24, historical_data: Optional[Any] = None) -> Dict[str, Any]:
        """
        Generate weather predictions
        
        Args:
            hours: Number of hours to predict
            historical_data: Optional historical data for better predictions
        
        Returns:
            Dictionary with predictions and metadata
        """
        try:
            if not self.is_trained:
                # Try to load model
                if not self.load_model():
                    return {
                        'success': False,
                        'error': 'Model not trained. Please train the model first.'
                    }
            
            # Get current weather
            current_weather = self.fetcher.fetch()
            
            if not current_weather:
                return {
                    'success': False,
                    'error': 'Could not fetch current weather data'
                }
            
            # Generate predictions
            predictions = self.predictor.predict_temperature(
                current_weather, hours=hours, historical_data=historical_data
            )
            
            return {
                'success': True,
                'predictions': predictions,
                'current_weather': current_weather,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        return self.training_status.copy()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        status = {
            'is_trained': self.is_trained,
            'training_status': self.training_status,
            'model_loaded': self.predictor.is_trained if self.predictor else False,
            'target_accuracy_min': 93.0,
            'target_accuracy_max': 95.0
        }
        
        # Add model metrics if available
        if self.predictor and self.predictor.is_trained:
            if hasattr(self.predictor, 'metrics') and self.predictor.metrics:
                status['model_metrics'] = self.predictor.metrics

        return status

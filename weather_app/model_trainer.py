"""
Comprehensive Model Training Module for Karachi Weather Prediction System
Includes multiple ML models (Random Forest, XGBoost, LightGBM, CatBoost, Gradient Boosting)
and Deep Learning models (LSTM, BiLSTM, GRU, CNN-LSTM)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from .logging_utils import logger

# ML Models
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available")

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.warning("CatBoost not available")

# Deep Learning Models
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Bidirectional, GRU, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available")


class ModelTrainer:
    """Comprehensive model trainer with multiple algorithms"""
    
    def __init__(self):
        self.models = {}
        self.model_scores = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = None
        
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        feature_names: List[str]) -> Dict[str, Any]:
        """
        Train all available models and return results
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            feature_names: List of feature names
        
        Returns:
            Dictionary with model results
        """
        logger.info("Training all available models...")
        results = {}
        
        # Train ML models
        if SKLEARN_AVAILABLE:
            results['random_forest'] = self._train_random_forest(X_train, y_train, X_val, y_val)
            results['gradient_boosting'] = self._train_gradient_boosting(X_train, y_train, X_val, y_val)
        
        if XGBOOST_AVAILABLE:
            results['xgboost'] = self._train_xgboost(X_train, y_train, X_val, y_val)
        
        if LIGHTGBM_AVAILABLE:
            results['lightgbm'] = self._train_lightgbm(X_train, y_train, X_val, y_val)
        
        if CATBOOST_AVAILABLE:
            results['catboost'] = self._train_catboost(X_train, y_train, X_val, y_val)
        
        # Train DL models (if data is suitable)
        if TF_AVAILABLE and len(X_train) > 100:
            try:
                results['lstm'] = self._train_lstm(X_train, y_train, X_val, y_val)
                results['bilstm'] = self._train_bilstm(X_train, y_train, X_val, y_val)
                results['gru'] = self._train_gru(X_train, y_train, X_val, y_val)
                results['cnn_lstm'] = self._train_cnn_lstm(X_train, y_train, X_val, y_val)
            except Exception as e:
                logger.warning(f"Deep learning training failed: {e}")
        
        # Find best model
        if results:
            best_model_name = min(results.keys(), key=lambda k: results[k]['mae'])
            self.best_model_name = best_model_name
            self.best_model = self.models.get(best_model_name)
            logger.info(f"Best model: {best_model_name} (MAE: {results[best_model_name]['mae']:.4f})")
        
        return results
    
    def _train_random_forest(self, X_train, y_train, X_val, y_val):
        """Train Random Forest model"""
        logger.info("Training Random Forest...")
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        
        y_pred = model.predict(X_val)
        metrics = self._calculate_metrics(y_val, y_pred)
        
        return metrics
    
    
    
    def _train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model"""
        logger.info("Training XGBoost...")
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        model.fit(X_train, y_train, 
                 eval_set=[(X_val, y_val)],
                 verbose=False)
        self.models['xgboost'] = model
        
        y_pred = model.predict(X_val)
        metrics = self._calculate_metrics(y_val, y_pred)
        
        return metrics
    
    def _train_lightgbm(self, X_train, y_train, X_val, y_val):
        """Train LightGBM model"""
        logger.info("Training LightGBM...")
        model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        model.fit(X_train, y_train,
                 eval_set=[(X_val, y_val)],
                 callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)])
        self.models['lightgbm'] = model
        
        y_pred = model.predict(X_val)
        metrics = self._calculate_metrics(y_val, y_pred)
        
        return metrics
    
    def _train_catboost(self, X_train, y_train, X_val, y_val):
        """Train CatBoost model"""
        logger.info("Training CatBoost...")
        model = CatBoostRegressor(
            iterations=200,
            depth=6,
            learning_rate=0.05,
            loss_function='RMSE',
            random_seed=42,
            verbose=False
        )
        model.fit(X_train, y_train,
                 eval_set=(X_val, y_val),
                 early_stopping_rounds=20)
        self.models['catboost'] = model
        
        y_pred = model.predict(X_val)
        metrics = self._calculate_metrics(y_val, y_pred)
        
        return metrics
    
    def _train_lstm(self, X_train, y_train, X_val, y_val, lookback=24):
        """Train LSTM model"""
        logger.info("Training LSTM...")
        
        # Reshape data for LSTM (samples, timesteps, features)
        X_train_seq = self._create_sequences(X_train, lookback)
        X_val_seq = self._create_sequences(X_val, lookback)
        y_train_seq = y_train[lookback:]
        y_val_seq = y_val[lookback:]
        
        if len(X_train_seq) == 0:
            return {'mae': float('inf'), 'rmse': float('inf'), 'r2': -float('inf')}
        
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(lookback, X_train_seq.shape[2])),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        history = model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        self.models['lstm'] = model
        
        y_pred = model.predict(X_val_seq, verbose=0).flatten()
        metrics = self._calculate_metrics(y_val_seq, y_pred)
        
        return metrics
    
    def _train_bilstm(self, X_train, y_train, X_val, y_val, lookback=24):
        """Train Bidirectional LSTM model"""
        logger.info("Training BiLSTM...")
        
        X_train_seq = self._create_sequences(X_train, lookback)
        X_val_seq = self._create_sequences(X_val, lookback)
        y_train_seq = y_train[lookback:]
        y_val_seq = y_val[lookback:]
        
        if len(X_train_seq) == 0:
            return {'mae': float('inf'), 'rmse': float('inf'), 'r2': -float('inf')}
        
        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True), input_shape=(lookback, X_train_seq.shape[2])),
            Dropout(0.2),
            Bidirectional(LSTM(32, return_sequences=False)),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        history = model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        self.models['bilstm'] = model
        
        y_pred = model.predict(X_val_seq, verbose=0).flatten()
        metrics = self._calculate_metrics(y_val_seq, y_pred)
        
        return metrics
    
    def _train_gru(self, X_train, y_train, X_val, y_val, lookback=24):
        """Train GRU model"""
        logger.info("Training GRU...")
        
        X_train_seq = self._create_sequences(X_train, lookback)
        X_val_seq = self._create_sequences(X_val, lookback)
        y_train_seq = y_train[lookback:]
        y_val_seq = y_val[lookback:]
        
        if len(X_train_seq) == 0:
            return {'mae': float('inf'), 'rmse': float('inf'), 'r2': -float('inf')}
        
        model = Sequential([
            GRU(64, return_sequences=True, input_shape=(lookback, X_train_seq.shape[2])),
            Dropout(0.2),
            GRU(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        history = model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        self.models['gru'] = model
        
        y_pred = model.predict(X_val_seq, verbose=0).flatten()
        metrics = self._calculate_metrics(y_val_seq, y_pred)
        
        return metrics
    
    def _train_cnn_lstm(self, X_train, y_train, X_val, y_val, lookback=24):
        """Train CNN-LSTM hybrid model"""
        logger.info("Training CNN-LSTM...")
        
        X_train_seq = self._create_sequences(X_train, lookback)
        X_val_seq = self._create_sequences(X_val, lookback)
        y_train_seq = y_train[lookback:]
        y_val_seq = y_val[lookback:]
        
        if len(X_train_seq) == 0:
            return {'mae': float('inf'), 'rmse': float('inf'), 'r2': -float('inf')}
        
        model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(lookback, X_train_seq.shape[2])),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        ]
        
        history = model.fit(
            X_train_seq, y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        self.models['cnn_lstm'] = model
        
        y_pred = model.predict(X_val_seq, verbose=0).flatten()
        metrics = self._calculate_metrics(y_val_seq, y_pred)
        
        return metrics
    
    def _create_sequences(self, data, lookback):
        """Create sequences for LSTM/GRU models"""
        if len(data) < lookback:
            return np.array([]).reshape(0, lookback, data.shape[1])
        
        X = []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
        return np.array(X)
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Percentage within thresholds
        errors = np.abs(y_true - y_pred)
        within_half = (errors <= 0.5).mean() * 100
        within_1c = (errors <= 1.0).mean() * 100
        within_2c = (errors <= 2.0).mean() * 100
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'within_half': within_half,
            'within_1c': within_1c,
            'within_2c': within_2c
        }
    
    def get_best_model(self):
        """Get the best performing model"""
        return self.best_model, self.best_model_name
    
    def predict(self, X, model_name=None):
        """Make predictions using specified model or best model"""
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Handle DL models differently
        if model_name in ['lstm', 'bilstm', 'gru', 'cnn_lstm']:
            lookback = 24
            X_seq = self._create_sequences(X, lookback)
            if len(X_seq) == 0:
                # If not enough data, pad with zeros
                X_seq = np.zeros((1, lookback, X.shape[1]))
                X_seq[0, -min(lookback, len(X)):] = X[-min(lookback, len(X)):]
            pred = model.predict(X_seq, verbose=0).flatten()
            # Return predictions aligned with input
            if len(pred) < len(X):
                # Pad with first prediction
                pred = np.concatenate([np.full(lookback, pred[0]), pred])
            return pred[-len(X):]
        else:
            return model.predict(X)


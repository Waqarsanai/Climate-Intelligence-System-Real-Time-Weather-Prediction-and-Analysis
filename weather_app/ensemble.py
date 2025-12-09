"""
Ensemble Model Module for Karachi Weather Prediction System
Implements weighted averaging, stacking, and blending of multiple models
"""

import numpy as np
from typing import Dict, List, Optional, Any
from sklearn.linear_model import LinearRegression
from .logging_utils import logger


class EnsembleModel:
    """Ensemble model combining multiple base models"""
    
    def __init__(self):
        self.base_models = {}
        self.meta_model = None
        self.weights = {}
        self.is_fitted = False
    
    def add_model(self, name: str, model: Any, weight: float = 1.0):
        """Add a base model to the ensemble"""
        self.base_models[name] = model
        self.weights[name] = weight
        logger.info(f"Added model {name} with weight {weight}")
    
    def fit_weighted_average(self, X_train, y_train, X_val, y_val, 
                            model_predictions: Dict[str, np.ndarray]):
        """
        Fit weighted average ensemble using validation performance
        
        Args:
            X_train: Training features (not used, kept for interface consistency)
            y_train: Training target (not used)
            X_val: Validation features (not used)
            y_val: Validation target
            model_predictions: Dictionary of model_name -> predictions on validation set
        """
        logger.info("Fitting weighted average ensemble...")
        
        # Calculate weights based on inverse MAE
        errors = {}
        for name, pred in model_predictions.items():
            mae = np.mean(np.abs(y_val - pred))
            errors[name] = mae
        
        # Weight inversely proportional to error
        total_inv_error = sum(1.0 / (e + 1e-8) for e in errors.values())
        for name in errors:
            self.weights[name] = (1.0 / (errors[name] + 1e-8)) / total_inv_error
        
        logger.info(f"Ensemble weights: {self.weights}")
        self.is_fitted = True
    
    def fit_stacking(self, X_train, y_train, X_val, y_val,
                    model_predictions: Dict[str, np.ndarray]):
        """
        Fit stacking ensemble with meta-learner
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            model_predictions: Dictionary of model_name -> predictions on validation set
        """
        logger.info("Fitting stacking ensemble...")
        
        # Create meta-features from base model predictions
        meta_features = np.column_stack([pred for pred in model_predictions.values()])
        
        # Train meta-learner (Ridge regression)
        self.meta_model = Ridge(alpha=1.0)
        self.meta_model.fit(meta_features, y_val)
        
        logger.info(f"Stacking meta-model coefficients: {self.meta_model.coef_}")
        self.is_fitted = True
    
    def predict_weighted_average(self, model_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Predict using weighted average"""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit_weighted_average or fit_stacking first.")
        
        weighted_sum = np.zeros(len(list(model_predictions.values())[0]))
        total_weight = 0
        
        for name, pred in model_predictions.items():
            weight = self.weights.get(name, 1.0 / len(model_predictions))
            weighted_sum += weight * pred
            total_weight += weight
        
        return weighted_sum / total_weight
    
    def predict_stacking(self, model_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Predict using stacking"""
        if not self.is_fitted or self.meta_model is None:
            raise ValueError("Stacking ensemble not fitted. Call fit_stacking first.")
        
        meta_features = np.column_stack([pred for pred in model_predictions.values()])
        return self.meta_model.predict(meta_features)
    
    def predict(self, model_predictions: Dict[str, np.ndarray], method: str = 'weighted') -> np.ndarray:
        """
        Predict using ensemble
        
        Args:
            model_predictions: Dictionary of model_name -> predictions
            method: 'weighted' or 'stacking'
        
        Returns:
            Ensemble predictions
        """
        if method == 'weighted':
            return self.predict_weighted_average(model_predictions)
        elif method == 'stacking':
            return self.predict_stacking(model_predictions)
        else:
            raise ValueError(f"Unknown method: {method}")


class ModelBlender:
    """Advanced model blending with post-processing"""
    
    def __init__(self):
        self.ensemble = EnsembleModel()
        self.smoothing_window = 3
    
    def blend_predictions(self, predictions: Dict[str, np.ndarray], 
                         method: str = 'weighted') -> np.ndarray:
        """
        Blend multiple model predictions with post-processing
        
        Args:
            predictions: Dictionary of model_name -> predictions
            method: Blending method ('weighted', 'stacking', or 'simple_average')
        
        Returns:
            Blended predictions
        """
        if method == 'simple_average':
            # Simple average
            pred_array = np.array(list(predictions.values()))
            blended = np.mean(pred_array, axis=0)
        elif method == 'weighted':
            blended = self.ensemble.predict_weighted_average(predictions)
        elif method == 'stacking':
            blended = self.ensemble.predict_stacking(predictions)
        else:
            raise ValueError(f"Unknown blending method: {method}")
        
        # Apply smoothing
        blended = self._apply_smoothing(blended)
        
        return blended
    
    def _apply_smoothing(self, predictions: np.ndarray) -> np.ndarray:
        """Apply moving average smoothing to predictions"""
        if len(predictions) < self.smoothing_window:
            return predictions
        
        smoothed = np.convolve(
            predictions, 
            np.ones(self.smoothing_window) / self.smoothing_window, 
            mode='same'
        )
        
        # Keep first and last values unchanged
        smoothed[0] = predictions[0]
        smoothed[-1] = predictions[-1]
        
        return smoothed
    
    def calibrate_predictions(self, predictions: np.ndarray, 
                             historical_bias: float = 0.0) -> np.ndarray:
        """
        Calibrate predictions by adjusting for known bias
        
        Args:
            predictions: Raw predictions
            historical_bias: Average bias from historical performance
        
        Returns:
            Calibrated predictions
        """
        return predictions - historical_bias


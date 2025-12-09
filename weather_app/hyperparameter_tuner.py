"""
Hyperparameter Tuning Module for Karachi Weather Prediction System
Implements Bayesian Optimization, Grid Search, and Random Search
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from .logging_utils import logger
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit



from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args


class HyperparameterTuner:
    """Hyperparameter tuning for weather prediction models"""
    
    def __init__(self):
        self.best_params = {}
        self.best_score = {}
    
    def tune_random_forest(self, X_train, y_train, method='grid', cv_folds=3):
        """Tune Random Forest hyperparameters"""
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available for hyperparameter tuning")
            return {}
        
        from sklearn.ensemble import RandomForestRegressor
        
        logger.info(f"Tuning Random Forest using {method} search...")
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        model = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        if method == 'grid':
            search = GridSearchCV(
                model, param_grid, cv=tscv, 
                scoring='neg_mean_absolute_error',
                n_jobs=-1, verbose=0
            )
        else:  # random
            search = RandomizedSearchCV(
                model, param_grid, n_iter=20, cv=tscv,
                scoring='neg_mean_absolute_error',
                n_jobs=-1, random_state=42, verbose=0
            )
        
        search.fit(X_train, y_train)
        
        self.best_params['random_forest'] = search.best_params_
        self.best_score['random_forest'] = -search.best_score_
        
        logger.info(f"Best Random Forest params: {search.best_params_}")
        logger.info(f"Best score: {-search.best_score_:.4f}")
        
        return search.best_params_, search.best_estimator_
    
    def tune_xgboost(self, X_train, y_train, method='grid', cv_folds=3):
        """Tune XGBoost hyperparameters"""
        if not SKLEARN_AVAILABLE or not hasattr(__import__('xgboost'), 'XGBRegressor'):
            logger.warning("XGBoost not available for hyperparameter tuning")
            return {}
        
        import xgboost as xgb
        
        logger.info(f"Tuning XGBoost using {method} search...")
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        model = xgb.XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
        
        if method == 'grid':
            search = GridSearchCV(
                model, param_grid, cv=tscv,
                scoring='neg_mean_absolute_error',
                n_jobs=-1, verbose=0
            )
        else:
            search = RandomizedSearchCV(
                model, param_grid, n_iter=20, cv=tscv,
                scoring='neg_mean_absolute_error',
                n_jobs=-1, random_state=42, verbose=0
            )
        
        search.fit(X_train, y_train)
        
        self.best_params['xgboost'] = search.best_params_
        self.best_score['xgboost'] = -search.best_score_
        
        logger.info(f"Best XGBoost params: {search.best_params_}")
        logger.info(f"Best score: {-search.best_score_:.4f}")
        
        return search.best_params_, search.best_estimator_
    
    def tune_bayesian(self, X_train, y_train, model_type='xgboost', n_calls=20):
        """Bayesian optimization for hyperparameters"""
        if not SKOPT_AVAILABLE:
            logger.warning("scikit-optimize not available for Bayesian optimization")
            return {}
        
        logger.info(f"Bayesian optimization for {model_type}...")
        
        if model_type == 'xgboost':
            import xgboost as xgb
            
            # Define search space
            dimensions = [
                Integer(50, 300, name='n_estimators'),
                Integer(3, 10, name='max_depth'),
                Real(0.01, 0.2, name='learning_rate'),
                Integer(1, 10, name='min_child_weight'),
                Real(0.6, 1.0, name='subsample'),
                Real(0.6, 1.0, name='colsample_bytree')
            ]
            
            tscv = TimeSeriesSplit(n_splits=3)
            
            @use_named_args(dimensions=dimensions)
            def objective(**params):
                model = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1, verbosity=0)
                scores = []
                for train_idx, val_idx in tscv.split(X_train):
                    X_tr, X_val = X_train[train_idx], X_train[val_idx]
                    y_tr, y_val = y_train[train_idx], y_train[val_idx]
                    model.fit(X_tr, y_tr)
                    pred = model.predict(X_val)
                    mae = np.mean(np.abs(y_val - pred))
                    scores.append(mae)
                return np.mean(scores)
            
            result = gp_minimize(
                func=objective,
                dimensions=dimensions,
                n_calls=n_calls,
                random_state=42,
                verbose=False
            )
            
            best_params = {
                'n_estimators': result.x[0],
                'max_depth': result.x[1],
                'learning_rate': result.x[2],
                'min_child_weight': result.x[3],
                'subsample': result.x[4],
                'colsample_bytree': result.x[5]
            }
            
            self.best_params[f'{model_type}_bayesian'] = best_params
            self.best_score[f'{model_type}_bayesian'] = result.fun
            
            logger.info(f"Best {model_type} params (Bayesian): {best_params}")
            logger.info(f"Best score: {result.fun:.4f}")
            
            return best_params
        
        return {}
    
    def get_best_params(self, model_name: str) -> Dict:
        """Get best parameters for a model"""
        return self.best_params.get(model_name, {})
    
    def get_best_score(self, model_name: str) -> float:
        """Get best score for a model"""
        return self.best_score.get(model_name, float('inf'))


import numpy as np
import optuna
import logging
logger = logging.getLogger(__name__)
from typing import Dict, Any, List, Tuple, Callable
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from xgboost import XGBClassifier, XGBRegressor

# Suppress Optuna logs unless warning/error
optuna.logging.set_verbosity(optuna.logging.WARNING)

import pandas as pd

def validate_numeric_matrix(X: Any) -> None:
    """
    Defensive check to ensure X contains only numeric values before fitting/predicting.
    Raises ValueError with diagnostic details if non-numeric values are found.
    """
    if isinstance(X, pd.DataFrame):
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                non_numeric_samples = []
                for val in X[col].head(20):
                    try:
                        float(val)
                    except (ValueError, TypeError):
                        non_numeric_samples.append(val)
                if non_numeric_samples:
                    raise ValueError(
                        f"Column '{col}' is not numeric. Type: {X[col].dtype}. "
                        f"Sample offending values: {non_numeric_samples[:5]}"
                    )
    else:
        arr = np.asarray(X)
        if not np.issubdtype(arr.dtype, np.number):
            for col_idx in range(arr.shape[1]):
                col_data = arr[:, col_idx]
                offending = []
                for val in col_data:
                    try:
                        float(val)
                    except (ValueError, TypeError):
                        offending.append(val)
                    if len(offending) >= 5:
                        break
                if offending:
                    raise ValueError(
                        f"Transformed matrix column index {col_idx} contains non-numeric values. "
                        f"Sample offending values: {offending}"
                    )

class ModelTrainer:
    def __init__(self, problem_type: str, primary_metric: str, mode: str = "standard"):
        self.problem_type = problem_type
        self.primary_metric = primary_metric
        self.mode = mode
        
        # Trial settings based on study modes
        if mode == "quick":
            self.n_trials = 5
        elif mode == "research":
            self.n_trials = 25
        else:
            self.n_trials = 12

    def train_candidates(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_val: np.ndarray, 
        y_val: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Runs training and HPO on candidate algorithms.
        Returns details of each model training execution.
        """
        # Defensive validation check
        validate_numeric_matrix(X_train)
        validate_numeric_matrix(X_val)

        results = []
        
        if self.problem_type == "classification":
            candidates = ["logistic_regression", "random_forest", "xgboost"]
        else:
            candidates = ["ridge_regression", "random_forest", "xgboost"]
            
        for model_name in candidates:
            try:
                # 1. Optimize hyperparameters
                best_params = self._run_hpo(model_name, X_train, y_train)
                
                # 2. Train final model with best params
                model = self._create_model(model_name, best_params)
                model.fit(X_train, y_train)
                
                # 3. Double-check CV stability
                cv_scores = self._evaluate_cross_val(model_name, best_params, X_train, y_train)
                
                # 4. Evaluate on Train/Val
                metrics_res = self._compute_split_metrics(model, X_train, y_train, X_val, y_val)
                
                # Assess overfitting risk
                train_score = metrics_res["train"][self.primary_metric]
                val_score = metrics_res["val"][self.primary_metric]
                
                # Higher gap -> higher overfitting risk
                if self.problem_type == "classification":
                    gap = train_score - val_score
                    if gap > 0.15:
                        overfitting_risk = "high"
                    elif gap > 0.07:
                        overfitting_risk = "moderate"
                    else:
                        overfitting_risk = "low"
                else: # regression
                    # In regression R2: higher R2 on train vs val. Or lower RMSE.
                    # Let's check R2 score gap
                    train_r2 = metrics_res["train"].get("r2", 1.0)
                    val_r2 = metrics_res["val"].get("r2", 0.0)
                    r2_gap = train_r2 - val_r2
                    if r2_gap > 0.20:
                        overfitting_risk = "high"
                    elif r2_gap > 0.08:
                        overfitting_risk = "moderate"
                    else:
                        overfitting_risk = "low"
                
                results.append({
                    "model_name": model_name,
                    "model_instance": model,
                    "hyperparameters": best_params,
                    "metrics": metrics_res,
                    "cv_metrics": cv_scores,
                    "overfitting_risk": overfitting_risk,
                    "status": "success"
                })
            except Exception as e:
                results.append({
                    "model_name": model_name,
                    "status": "failed",
                    "error_message": str(e)
                })
                
        return results

    def _create_model(self, model_name: str, params: Dict[str, Any]) -> Any:
        if self.problem_type == "classification":
            if model_name == "logistic_regression":
                return LogisticRegression(max_iter=1000, random_state=42, **params)
            elif model_name == "random_forest":
                return RandomForestClassifier(random_state=42, **params)
            elif model_name == "xgboost":
                # Ensure classification parameters
                return XGBClassifier(random_seed=42, random_state=42, eval_metric="logloss", **params)
        else:
            if model_name == "ridge_regression":
                return Ridge(random_state=42, **params)
            elif model_name == "random_forest":
                return RandomForestRegressor(random_state=42, **params)
            elif model_name == "xgboost":
                return XGBRegressor(random_seed=42, random_state=42, **params)
        raise ValueError(f"Unknown model name: {model_name}")

    def _run_hpo(self, model_name: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Runs Optuna study for the matching candidate model.
        """
        validate_numeric_matrix(X)
        def objective(trial: optuna.Trial) -> float:
            params = {}
            if self.problem_type == "classification":
                if model_name == "logistic_regression":
                    params = {
                        "C": trial.suggest_float("C", 0.01, 10.0, log=True),
                        "penalty": trial.suggest_categorical("penalty", ["l2"])
                    }
                elif model_name == "random_forest":
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                        "max_depth": trial.suggest_int("max_depth", 3, 15),
                        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10)
                    }
                elif model_name == "xgboost":
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 50, 150),
                        "max_depth": trial.suggest_int("max_depth", 3, 8),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
                    }
            else: # regression
                if model_name == "ridge_regression":
                    params = {
                        "alpha": trial.suggest_float("alpha", 0.1, 100.0, log=True)
                    }
                elif model_name == "random_forest":
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                        "max_depth": trial.suggest_int("max_depth", 3, 15),
                        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10)
                    }
                elif model_name == "xgboost":
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 50, 150),
                        "max_depth": trial.suggest_int("max_depth", 3, 8),
                        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
                    }
            
            # Simple 3-fold CV inside the hyperparameter loop to save training time
            cv = KFold(n_splits=3, shuffle=True, random_state=42)
            scores = []
            
            for train_idx, val_idx in cv.split(X, y):
                X_t, X_v = X[train_idx], X[val_idx]
                y_t, y_v = y[train_idx], y[val_idx]
                
                m = self._create_model(model_name, params)
                m.fit(X_t, y_t)
                
                # Check target metric
                preds = m.predict(X_v)
                if self.problem_type == "classification":
                    if self.primary_metric == "f1":
                        scores.append(f1_score(y_v, preds, average="binary" if len(np.unique(y)) <= 2 else "macro"))
                    elif self.primary_metric == "roc_auc":
                        if hasattr(m, "predict_proba"):
                            probs = m.predict_proba(X_v)
                            scores.append(roc_auc_score(y_v, probs[:, 1] if probs.shape[1] > 1 else probs))
                        else:
                            scores.append(accuracy_score(y_v, preds))
                    else:
                        scores.append(accuracy_score(y_v, preds))
                else: # regression
                    if self.primary_metric == "r2":
                        scores.append(r2_score(y_v, preds))
                    elif self.primary_metric == "rmse":
                        scores.append(-np.sqrt(mean_squared_error(y_v, preds))) # Minimize negative RMSE
                    else:
                        scores.append(-mean_absolute_error(y_v, preds)) # Minimize negative MAE
                        
            return float(np.mean(scores))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials)
        return study.best_params

    def _evaluate_cross_val(self, model_name: str, params: Dict[str, Any], X: np.ndarray, y: np.ndarray) -> List[float]:
        """
        Runs comprehensive 5-fold CV to evaluate model stability.
        """
        validate_numeric_matrix(X)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if self.problem_type == "classification" else KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in cv.split(X, y):
            X_t, X_v = X[train_idx], X[val_idx]
            y_t, y_v = y[train_idx], y[val_idx]
            
            m = self._create_model(model_name, params)
            m.fit(X_t, y_t)
            preds = m.predict(X_v)
            
            if self.problem_type == "classification":
                if self.primary_metric == "f1":
                    scores.append(f1_score(y_v, preds, average="binary" if len(np.unique(y)) <= 2 else "macro"))
                elif self.primary_metric == "roc_auc":
                    if hasattr(m, "predict_proba"):
                        probs = m.predict_proba(X_v)
                        scores.append(roc_auc_score(y_v, probs[:, 1] if probs.shape[1] > 1 else probs))
                    else:
                        scores.append(accuracy_score(y_v, preds))
                else:
                    scores.append(accuracy_score(y_v, preds))
            else: # regression
                if self.primary_metric == "r2":
                    scores.append(r2_score(y_v, preds))
                else:
                    scores.append(np.sqrt(mean_squared_error(y_v, preds)))
        return [float(s) for s in scores]

    def _compute_split_metrics(self, model: Any, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Computes standard metrics on both training and validation sets.
        """
        validate_numeric_matrix(X_train)
        validate_numeric_matrix(X_val)
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
            average_precision_score, mean_absolute_error, mean_squared_error, r2_score
        )
        
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)
        
        has_proba = hasattr(model, "predict_proba")
        
        def safe_score(score_fn, *args, **kwargs):
            try:
                val = score_fn(*args, **kwargs)
                import math
                if isinstance(val, (int, float)):
                    if math.isnan(val) or math.isinf(val):
                        return None
                    return float(val)
                return val
            except Exception as e:
                logger.warning(f"Error computing metric: {e}")
                return None

        if self.problem_type == "classification":
            is_binary = len(np.unique(y_train)) <= 2
            avg_mode = "binary" if is_binary else "macro"
            
            train_metrics = {
                "accuracy": safe_score(accuracy_score, y_train, train_preds),
                "precision": safe_score(precision_score, y_train, train_preds, average=avg_mode, zero_division=0),
                "recall": safe_score(recall_score, y_train, train_preds, average=avg_mode, zero_division=0),
                "f1": safe_score(f1_score, y_train, train_preds, average=avg_mode, zero_division=0),
            }
            val_metrics = {
                "accuracy": safe_score(accuracy_score, y_val, val_preds),
                "precision": safe_score(precision_score, y_val, val_preds, average=avg_mode, zero_division=0),
                "recall": safe_score(recall_score, y_val, val_preds, average=avg_mode, zero_division=0),
                "f1": safe_score(f1_score, y_val, val_preds, average=avg_mode, zero_division=0),
            }
            
            if has_proba:
                try:
                    train_probs = model.predict_proba(X_train)
                    val_probs = model.predict_proba(X_val)
                    
                    train_metrics["roc_auc"] = safe_score(roc_auc_score, y_train, train_probs[:, 1] if is_binary else train_probs, multi_class="ovr")
                    val_metrics["roc_auc"] = safe_score(roc_auc_score, y_val, val_probs[:, 1] if is_binary else val_probs, multi_class="ovr")
                    
                    # pr_auc using average precision score
                    if is_binary:
                        train_metrics["pr_auc"] = safe_score(average_precision_score, y_train, train_probs[:, 1])
                        val_metrics["pr_auc"] = safe_score(average_precision_score, y_val, val_probs[:, 1])
                    else:
                        from sklearn.preprocessing import LabelBinarizer
                        lb = LabelBinarizer()
                        y_train_bin = lb.fit_transform(y_train)
                        y_val_bin = lb.transform(y_val)
                        train_metrics["pr_auc"] = safe_score(average_precision_score, y_train_bin, train_probs, average="macro")
                        val_metrics["pr_auc"] = safe_score(average_precision_score, y_val_bin, val_probs, average="macro")
                except Exception as e:
                    logger.warning(f"Error computing probability metrics: {e}")
                    train_metrics["roc_auc"] = None
                    val_metrics["roc_auc"] = None
                    train_metrics["pr_auc"] = None
                    val_metrics["pr_auc"] = None
            else:
                train_metrics["roc_auc"] = train_metrics["accuracy"]
                val_metrics["roc_auc"] = val_metrics["accuracy"]
                train_metrics["pr_auc"] = None
                val_metrics["pr_auc"] = None
                
        else: # regression
            train_metrics = {
                "mae": safe_score(mean_absolute_error, y_train, train_preds),
                "mse": safe_score(mean_squared_error, y_train, train_preds),
                "rmse": safe_score(lambda *a: np.sqrt(mean_squared_error(*a)), y_train, train_preds),
                "r2": safe_score(r2_score, y_train, train_preds)
            }
            val_metrics = {
                "mae": safe_score(mean_absolute_error, y_val, val_preds),
                "mse": safe_score(mean_squared_error, y_val, val_preds),
                "rmse": safe_score(lambda *a: np.sqrt(mean_squared_error(*a)), y_val, val_preds),
                "r2": safe_score(r2_score, y_val, val_preds)
            }
            
        return {
            "train": train_metrics,
            "val": val_metrics
        }

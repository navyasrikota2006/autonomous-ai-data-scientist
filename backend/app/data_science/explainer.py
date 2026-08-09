import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any, List

class ModelExplainer:
    def __init__(self, model: Any, feature_names: List[str], problem_type: str):
        self.model = model
        self.feature_names = feature_names
        self.problem_type = problem_type

    def generate_explanations(self, X_train: np.ndarray, output_dir: str) -> Dict[str, Any]:
        """
        Computes SHAP scores or uses fallback coefficients/importances to explain the local weights.
        Generates and saves visual feature importance charts.
        Returns: Dict of important features and paths to saved charts.
        """
        os.makedirs(output_dir, exist_ok=True)
        importance_chart_path = os.path.join(output_dir, "feature_importance.png")
        sns.set_theme(style="whitegrid")
        
        feature_importance_map = {}
        used_fallback = False
        
        # 1. Try SHAP first
        try:
            import shap
            
            # Simple sample size to run SHAP fast
            sample_size = min(100, X_train.shape[0])
            rng = np.random.default_rng(42)
            sample_idx = rng.choice(X_train.shape[0], sample_size, replace=False)
            X_sample = X_train[sample_idx]
            
            # Select Explainer matching model types
            model_class_name = self.model.__class__.__name__.lower()
            if "randomforest" in model_class_name or "xgb" in model_class_name or "gradientboosting" in model_class_name:
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_sample)
            else:
                # Fallback generalized kernel/linear explainer
                explainer = shap.Explainer(self.model, X_sample)
                shap_values = explainer(X_sample)
                
            # If multi-class or list returned, take mean absolute values
            if isinstance(shap_values, list):
                # For classification random forest, shap returns a list of classes
                # We can average feature importances across positive/negative class
                mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            elif hasattr(shap_values, "values"):
                # explainer(X) returns an Explanation object
                mean_shap = np.abs(shap_values.values).mean(axis=0)
            else:
                mean_shap = np.abs(shap_values).mean(axis=0)
                
            # Ensure mean_shap matches feature count
            if len(mean_shap) == len(self.feature_names):
                for name, score in zip(self.feature_names, mean_shap):
                    feature_importance_map[name] = float(score)
            else:
                raise ValueError("SHAP shape mismatch, falling back to heuristics")
                
            # Save SHAP summary representation
            plt.figure(figsize=(8, 6))
            # Create a simple vertical/horizontal bar chart of SHAP values
            sorted_idx = np.argsort(mean_shap)[::-1][:15] # Top 15 features
            sorted_features = [self.feature_names[i] for i in sorted_idx]
            sorted_scores = [mean_shap[i] for i in sorted_idx]
            
            sns.barplot(x=sorted_scores, y=sorted_features, palette="viridis")
            plt.title("SHAP Global Feature Importance")
            plt.xlabel("mean(|SHAP value|) (average impact on model output)")
            plt.tight_layout()
            plt.savefig(importance_chart_path, dpi=120)
            plt.close()
            
        except Exception as e:
            print(f"SHAP calculations failed or not installed ({e}). Using native model importances.")
            used_fallback = True
            
        # 2. Native Importances Fallback
        if used_fallback:
            try:
                # Double-check model coef_ or feature_importances_
                if hasattr(self.model, "feature_importances_"):
                    scores = self.model.feature_importances_
                elif hasattr(self.model, "coef_"):
                    scores = np.abs(self.model.coef_)
                    # If multi-class coef_ is 2D, average across target classes
                    if len(scores.shape) > 1:
                        scores = np.mean(scores, axis=0)
                else:
                    # Generic uniform importances if neither exists
                    scores = np.ones(len(self.feature_names)) / len(self.feature_names)
                    
                # Re-align
                for name, score in zip(self.feature_names, scores):
                    feature_importance_map[name] = float(score)
                    
                # Render importances plot
                plt.figure(figsize=(8, 6))
                sorted_idx = np.argsort(scores)[::-1][:15]
                sorted_features = [self.feature_names[i] for i in sorted_idx]
                sorted_scores = [scores[i] for i in sorted_idx]
                
                sns.barplot(x=sorted_scores, y=sorted_features, palette="rocket")
                plt.title("Feature Importance (Model Specific)")
                plt.xlabel("Weights / Importances")
                plt.tight_layout()
                plt.savefig(importance_chart_path, dpi=120)
                plt.close()
                
            except Exception as ex:
                plt.close()
                print(f"Failed to generate fallback feature importance: {ex}")
                
        # Sort values
        sorted_importance = dict(sorted(feature_importance_map.items(), key=lambda item: item[1], reverse=True))
        
        return {
            "importance": sorted_importance,
            "chart_path": importance_chart_path,
            "used_shap": not used_fallback
        }

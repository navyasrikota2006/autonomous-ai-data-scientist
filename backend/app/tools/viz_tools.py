import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

def generate_eda_plots(
    df: pd.DataFrame, 
    target_col: Optional[str], 
    numerical_cols: List[str], 
    categorical_cols: List[str], 
    output_dir: str
) -> Dict[str, str]:
    """
    Generates and saves standard diagnostic EDA plots.
    Returns: Dict of feature name to absolute file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = {}
    
    # Set standard styling
    sns.set_theme(style="whitegrid")
    
    # 1. Target Variable Distribution (only runs if target_col is provided and exists)
    if target_col and target_col in df.columns:
        try:
            plt.figure(figsize=(7, 5))
            if df[target_col].nunique() <= 10:
                sns.countplot(x=target_col, data=df, palette="viridis")
                plt.title(f"Target Distribution: '{target_col}' (Classification)")
            else:
                sns.histplot(df[target_col].dropna(), kde=True, color="teal")
                plt.title(f"Target Distribution: '{target_col}' (Regression)")
            plt.tight_layout()
            target_path = os.path.join(output_dir, "target_distribution.png")
            plt.savefig(target_path, dpi=120)
            plt.close()
            paths["target_distribution"] = target_path
        except Exception as e:
            plt.close()
            print(f"Failed to plot target distribution: {e}")
    else:
        print("Skipping target variable distribution plot: No target_col provided or not in DataFrame.")
 
    # 2. Correlation Heatmap
    try:
        valid_numeric = [c for c in numerical_cols if c in df.columns]
        if len(valid_numeric) > 1:
            plt.figure(figsize=(8, 6))
            corr = df[valid_numeric].corr().fillna(0.0)
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1.0, vmax=1.0)
            plt.title("Numerical Feature Correlations")
            plt.tight_layout()
            heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
            plt.savefig(heatmap_path, dpi=120)
            plt.close()
            paths["correlation_heatmap"] = heatmap_path
    except Exception as e:
        plt.close()
        print(f"Failed to plot correlation matrix: {e}")
 
    # 3. Numerical Column Outlier Check (Box plots for top 4 numeric columns)
    try:
        top_numeric = [c for c in numerical_cols if c != target_col][:4] if target_col else numerical_cols[:4]
        top_numeric = [c for c in top_numeric if c in df.columns]
        if top_numeric:
            fig, axes = plt.subplots(1, len(top_numeric), figsize=(4 * len(top_numeric), 4))
            if len(top_numeric) == 1:
                axes = [axes]
            for i, col in enumerate(top_numeric):
                sns.boxplot(y=df[col], ax=axes[i], color="salmon")
                axes[i].set_title(f"Outlier Analysis: {col}")
            plt.tight_layout()
            boxplot_path = os.path.join(output_dir, "outliers_boxplots.png")
            plt.savefig(boxplot_path, dpi=120)
            plt.close()
            paths["outliers_boxplots"] = boxplot_path
    except Exception as e:
        plt.close()
        print(f"Failed to plot boxplots: {e}")
 
    # 4. Feature-Target Relationships (e.g. bar charts or scatter plots for top features)
    if target_col and target_col in df.columns:
        try:
            # Check target type again
            unique_t_vals = df[target_col].nunique()
            if unique_t_vals <= 10: # Classification
                # Plot the top categorical variable vs Target
                top_cats = [c for c in categorical_cols if c != target_col][:2]
                if top_cats:
                    fig, axes = plt.subplots(1, len(top_cats), figsize=(6 * len(top_cats), 4))
                    if len(top_cats) == 1:
                        axes = [axes]
                    for i, col in enumerate(top_cats):
                        sns.countplot(x=col, hue=target_col, data=df, ax=axes[i], palette="muted")
                        axes[i].set_title(f"Distribution of {col} grouped by {target_col}")
                        axes[i].tick_params(axis='x', rotation=30)
                    plt.tight_layout()
                    relation_path = os.path.join(output_dir, "relationships_countplot.png")
                    plt.savefig(relation_path, dpi=120)
                    plt.close()
                    paths["relationships"] = relation_path
            else: # Regression
                # Top numerical feature vs Continuous Target
                top_num = [c for c in numerical_cols if c != target_col][:2]
                if top_num:
                    fig, axes = plt.subplots(1, len(top_num), figsize=(6 * len(top_num), 4))
                    if len(top_num) == 1:
                        axes = [axes]
                    for i, col in enumerate(top_num):
                        sns.scatterplot(x=df[col], y=df[target_col], ax=axes[i], alpha=0.6, color="purple")
                        axes[i].set_title(f"{col} vs {target_col}")
                    plt.tight_layout()
                    relation_path = os.path.join(output_dir, "relationships_scatterplot.png")
                    plt.savefig(relation_path, dpi=120)
                    plt.close()
                    paths["relationships"] = relation_path
        except Exception as e:
            plt.close()
            print(f"Failed to plot relationship charts: {e}")
 
    return paths

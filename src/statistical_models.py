"""
Statistical Analysis and Cross-Validation for Prospect Theory Models

This script performs rigorous statistical analysis and cross-validation
for all modeling approaches (deep learning, classical, and hybrid) to
ensure reliable and scientifically valid results.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import argparse
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
import time

# Default paths
DEFAULT_RESULTS_DIR = "results"
DEFAULT_OUTPUT_DIR = "results/statistical_analysis"
DEFAULT_SEED = 42


def load_results(results_dir: str) -> Dict:
    """
    Load results from all modeling approaches.
    
    Args:
        results_dir: Directory containing results
        
    Returns:
        Dictionary of results
    """
    results = {
        "deep_learning": None,
        "classical": None,
        "hybrid": None
    }
    
    # Load deep learning results
    dl_results_path = os.path.join(results_dir, "comparison", "benchmark_results.json")
    if os.path.exists(dl_results_path):
        with open(dl_results_path, "r") as f:
            results["deep_learning"] = json.load(f)
    
    # Load classical results
    cl_results_path = os.path.join(results_dir, "classical_comparison", "classical_comparison_results.json")
    if os.path.exists(cl_results_path):
        with open(cl_results_path, "r") as f:
            results["classical"] = json.load(f)
    
    # Load hybrid results
    hy_results_path = os.path.join(results_dir, "hybrid_comparison", "hybrid_comparison_results.json")
    if os.path.exists(hy_results_path):
        with open(hy_results_path, "r") as f:
            results["hybrid"] = json.load(f)
    
    return results


def extract_metrics(results: Dict) -> pd.DataFrame:
    """
    Extract metrics from results for statistical analysis.
    
    Args:
        results: Dictionary of results
        
    Returns:
        DataFrame of metrics
    """
    metrics = []
    
    # Extract deep learning metrics
    if results["deep_learning"] is not None:
        for model_name, model_results in results["deep_learning"]["models"].items():
            if "error" in model_results:
                continue
            
            for layers, layer_results in model_results["layer_strategies"].items():
                if "error" in layer_results:
                    continue
                
                metrics.append({
                    "approach": "deep_learning",
                    "model": model_name,
                    "variant": f"layers_{layers}",
                    "accuracy": layer_results["accuracy"],
                    "std_accuracy": layer_results["std_accuracy"],
                    "training_time": layer_results["training_time"],
                    "inference_time": layer_results["inference_time"]
                })
    
    # Extract classical metrics
    if results["classical"] is not None:
        for model_name, model_results in results["classical"]["models"].items():
            for feature_set, feature_results in model_results["feature_sets"].items():
                for strategy, strategy_results in feature_results["engineering_strategies"].items():
                    metrics.append({
                        "approach": "classical",
                        "model": model_name,
                        "variant": f"{feature_set}_{strategy}",
                        "accuracy": strategy_results["accuracy"],
                        "std_accuracy": strategy_results["std_accuracy"],
                        "training_time": strategy_results["training_time"],
                        "inference_time": strategy_results["inference_time"]
                    })
    
    # Extract hybrid metrics
    if results["hybrid"] is not None:
        for approach_name, approach_results in results["hybrid"]["approaches"].items():
            for classical_model, model_results in approach_results["classical_models"].items():
                metrics.append({
                    "approach": "hybrid",
                    "model": approach_name,
                    "variant": classical_model,
                    "accuracy": model_results["accuracy"],
                    "std_accuracy": model_results["std_accuracy"],
                    "training_time": model_results["training_time"],
                    "inference_time": model_results["inference_time"]
                })
    
    return pd.DataFrame(metrics)


def perform_statistical_tests(metrics_df: pd.DataFrame) -> Dict:
    """
    Perform statistical tests to compare different approaches.
    
    Args:
        metrics_df: DataFrame of metrics
        
    Returns:
        Dictionary of statistical test results
    """
    statistical_tests = {}
    
    # Group by approach
    approach_groups = metrics_df.groupby("approach")
    
    # ANOVA test for accuracy across approaches
    if len(approach_groups) > 1:
        approach_accuracies = [group["accuracy"].values for _, group in approach_groups]
        f_stat, p_value = stats.f_oneway(*approach_accuracies)
        
        statistical_tests["anova_approaches"] = {
            "f_statistic": f_stat,
            "p_value": p_value,
            "significant": p_value < 0.05
        }
    
    # T-tests for pairwise comparisons
    approach_pairs = []
    for i, approach1 in enumerate(approach_groups.groups.keys()):
        for approach2 in list(approach_groups.groups.keys())[i+1:]:
            approach_pairs.append((approach1, approach2))
    
    for approach1, approach2 in approach_pairs:
        accuracies1 = metrics_df[metrics_df["approach"] == approach1]["accuracy"].values
        accuracies2 = metrics_df[metrics_df["approach"] == approach2]["accuracy"].values
        
        t_stat, p_value = stats.ttest_ind(accuracies1, accuracies2, equal_var=False)
        
        statistical_tests[f"ttest_{approach1}_vs_{approach2}"] = {
            "t_statistic": t_stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "better_approach": approach1 if np.mean(accuracies1) > np.mean(accuracies2) else approach2
        }
    
    # Find best model in each approach
    for approach in approach_groups.groups.keys():
        approach_df = metrics_df[metrics_df["approach"] == approach]
        best_idx = approach_df["accuracy"].idxmax()
        best_model = approach_df.loc[best_idx]
        
        statistical_tests[f"best_model_{approach}"] = {
            "model": best_model["model"],
            "variant": best_model["variant"],
            "accuracy": best_model["accuracy"],
            "std_accuracy": best_model["std_accuracy"],
            "training_time": best_model["training_time"],
            "inference_time": best_model["inference_time"]
        }
    
    # Overall best model
    best_idx = metrics_df["accuracy"].idxmax()
    best_model = metrics_df.loc[best_idx]
    
    statistical_tests["overall_best_model"] = {
        "approach": best_model["approach"],
        "model": best_model["model"],
        "variant": best_model["variant"],
        "accuracy": best_model["accuracy"],
        "std_accuracy": best_model["std_accuracy"],
        "training_time": best_model["training_time"],
        "inference_time": best_model["inference_time"]
    }
    
    return statistical_tests


def generate_statistical_visualizations(metrics_df: pd.DataFrame, output_dir: str):
    """
    Generate visualizations for statistical analysis.
    
    Args:
        metrics_df: DataFrame of metrics
        output_dir: Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot accuracy by approach
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="approach", y="accuracy", data=metrics_df)
    plt.title("Accuracy by Approach")
    plt.xlabel("Approach")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_by_approach.png"))
    
    # Plot accuracy by approach and model
    plt.figure(figsize=(16, 10))
    sns.boxplot(x="approach", y="accuracy", hue="model", data=metrics_df)
    plt.title("Accuracy by Approach and Model")
    plt.xlabel("Approach")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_by_approach_and_model.png"))
    
    # Plot training time by approach
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="approach", y="training_time", data=metrics_df)
    plt.title("Training Time by Approach")
    plt.xlabel("Approach")
    plt.ylabel("Training Time (s)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_time_by_approach.png"))
    
    # Plot inference time by approach
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="approach", y="inference_time", data=metrics_df)
    plt.title("Inference Time by Approach")
    plt.xlabel("Approach")
    plt.ylabel("Inference Time (s)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "inference_time_by_approach.png"))
    
    # Plot accuracy vs. training time
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x="training_time", y="accuracy", hue="approach", style="model", s=100, data=metrics_df)
    plt.title("Accuracy vs. Training Time")
    plt.xlabel("Training Time (s)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_vs_training_time.png"))
    
    # Plot accuracy vs. inference time
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x="inference_time", y="accuracy", hue="approach", style="model", s=100, data=metrics_df)
    plt.title("Accuracy vs. Inference Time")
    plt.xlabel("Inference Time (s)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_vs_inference_time.png"))
    
    # Plot top 10 models by accuracy
    top_models = metrics_df.sort_values("accuracy", ascending=False).head(10)
    plt.figure(figsize=(14, 10))
    sns.barplot(x="accuracy", y="model", hue="approach", data=top_models)
    plt.title("Top 10 Models by Accuracy")
    plt.xlabel("Accuracy")
    plt.ylabel("Model")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_models_by_accuracy.png"))
    
    # Plot accuracy distribution
    plt.figure(figsize=(12, 8))
    sns.histplot(data=metrics_df, x="accuracy", hue="approach", kde=True, bins=20)
    plt.title("Accuracy Distribution by Approach")
    plt.xlabel("Accuracy")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_distribution.png"))


def generate_statistical_report(metrics_df: pd.DataFrame, statistical_tests: Dict, output_dir: str):
    """
    Generate a comprehensive statistical analysis report.
    
    Args:
        metrics_df: DataFrame of metrics
        statistical_tests: Dictionary of statistical test results
        output_dir: Directory to save report
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Start report
    report = "# Statistical Analysis of Prospect Theory Models\n\n"
    
    # Add summary statistics
    report += "## Summary Statistics\n\n"
    
    # Overall statistics
    report += "### Overall Statistics\n\n"
    report += "| Metric | Value |\n"
    report += "|--------|------|\n"
    report += f"| Number of Models | {len(metrics_df)} |\n"
    report += f"| Average Accuracy | {metrics_df['accuracy'].mean():.4f} |\n"
    report += f"| Std Accuracy | {metrics_df['accuracy'].std():.4f} |\n"
    report += f"| Min Accuracy | {metrics_df['accuracy'].min():.4f} |\n"
    report += f"| Max Accuracy | {metrics_df['accuracy'].max():.4f} |\n"
    report += f"| Average Training Time | {metrics_df['training_time'].mean():.2f} s |\n"
    report += f"| Average Inference Time | {metrics_df['inference_time'].mean():.4f} s |\n"
    report += "\n"
    
    # Statistics by approach
    report += "### Statistics by Approach\n\n"
    approach_stats = metrics_df.groupby("approach").agg({
        "accuracy": ["mean", "std", "min", "max"],
        "training_time": "mean",
        "inference_time": "mean",
        "model": "count"
    }).reset_index()
    
    report += "| Approach | Count | Avg Accuracy | Std Accuracy | Min Accuracy | Max Accuracy | Avg Training Time (s) | Avg Inference Time (s) |\n"
    report += "|----------|-------|--------------|--------------|--------------|--------------|----------------------|------------------------|\n"
    
    for _, row in approach_stats.iterrows():
        report += f"| {row['approach']} | {row[('model', 'count')]} | {row[('accuracy', 'mean')]:.4f} | {row[('accuracy', 'std')]:.4f} | {row[('accuracy', 'min')]:.4f} | {row[('accuracy', 'max')]:.4f} | {row[('training_time', 'mean')]:.2f} | {row[('inference_time', 'mean')]:.4f} |\n"
    
    report += "\n"
    
    # Add statistical test results
    report += "## Statistical Tests\n\n"
    
    # ANOVA test
    if "anova_approaches" in statistical_tests:
        anova = statistical_tests["anova_approaches"]
        report += "### ANOVA Test for Approaches\n\n"
        report += "| Test | F-statistic | p-value | Significant |\n"
        report += "|------|-------------|---------|-------------|\n"
        report += f"| ANOVA | {anova['f_statistic']:.4f} | {anova['p_value']:.4f} | {'Yes' if anova['significant'] else 'No'} |\n\n"
    
    # T-tests
    report += "### T-tests for Pairwise Comparisons\n\n"
    report += "| Comparison | T-statistic | p-value | Significant | Better Approach |\n"
    report += "|------------|-------------|---------|-------------|----------------|\n"
    
    for test_name, test_results in statistical_tests.items():
        if test_name.startswith("ttest_"):
            report += f"| {test_name.replace('ttest_', '').replace('_vs_', ' vs. ')} | {test_results['t_statistic']:.4f} | {test_results['p_value']:.4f} | {'Yes' if test_results['significant'] else 'No'} | {test_results['better_approach']} |\n"
    
    report += "\n"
    
    # Best models
    report += "## Best Models\n\n"
    
    # Best model by approach
    report += "### Best Model by Approach\n\n"
    report += "| Approach | Model | Variant | Accuracy | Std Accuracy | Training Time (s) | Inference Time (s) |\n"
    report += "|----------|-------|---------|----------|--------------|-------------------|-------------------|\n"
    
    for test_name, test_results in statistical_tests.items():
        if test_name.startswith("best_model_"):
            approach = test_name.replace("best_model_", "")
            report += f"| {approach} | {test_results['model']} | {test_results['variant']} | {test_results['accuracy']:.4f} | {test_results['std_accuracy']:.4f} | {test_results['training_time']:.2f} | {test_results['inference_time']:.4f} |\n"
    
    report += "\n"
    
    # Overall best model
    report += "### Overall Best Model\n\n"
    best_model = statistical_tests["overall_best_model"]
    report += f"- **Approach**: {best_model['approach']}\n"
    report += f"- **Model**: {best_model['model']}\n"
    report += f"- **Variant**: {best_model['variant']}\n"
    report += f"- **Accuracy**: {best_model['accuracy']:.4f}\n"
    report += f"- **Std Accuracy**: {best_model['std_accuracy']:.4f}\n"
    report += f"- **Training Time**: {best_model['training_time']:.2f} s\n"
    report += f"- **Inference Time**: {best_model['inference_time']:.4f} s\n\n"
    
    # Add visualizations
    report += "## Visualizations\n\n"
    report += "### Accuracy by Approach\n\n"
    report += "![Accuracy by Approach](accuracy_by_approach.png)\n\n"
    report += "### Accuracy by Approach and Model\n\n"
    report += "![Accuracy by Approach and Model](accuracy_by_approach_and_model.png)\n\n"
    report += "### Training Time by Approach\n\n"
    report += "![Training Time by Approach](training_time_by_approach.png)\n\n"
    report += "### Inference Time by Approach\n\n"
    report += "![Inference Time by Approach](inference_time_by_approach.png)\n\n"
    report += "### Accuracy vs. Training Time\n\n"
    report += "![Accuracy vs. Training Time](accuracy_vs_training_time.png)\n\n"
    report += "### Accuracy vs. Inference Time\n\n"
    report += "![Accuracy vs. Inference Time](accuracy_vs_inference_time.png)\n\n"
    report += "### Top 10 Models by Accuracy\n\n"
    report += "![Top 10 Models by Accuracy](top_models_by_accuracy.png)\n\n"
    report += "### Accuracy Distribution\n\n"
    report += "![Accuracy Distribution](accuracy_distribution.png)\n\n"
    
    # Add conclusion
    report += "## Conclusion\n\n"
    
    # Determine if there are significant differences
    significant_differences = any(test["significant"] for test_name, test in statistical_tests.items() if test_name.startswith("ttest_"))
    
    if significant_differences:
        report += "The statistical analysis reveals significant differences between the modeling approaches. "
    else:
        report += "The statistical analysis does not reveal significant differences between the modeling approaches. "
    
    best_approach = approach_stats.loc[approach_stats[('accuracy', 'mean')].idxmax()]["approach"]
    report += f"The {best_approach} approach achieves the highest average accuracy. "
    
    report += f"The overall best model is a {best_model['approach']} approach using {best_model['model']} with variant {best_model['variant']}, "
    report += f"achieving an accuracy of {best_model['accuracy']:.4f}.\n\n"
    
    # Add recommendations
    report += "## Recommendations\n\n"
    report += f"1. **Model Selection**: Use the {best_model['approach']} approach with {best_model['model']} and variant {best_model['variant']} for the final pipeline.\n"
    report += "2. **Ensemble Methods**: Consider ensemble methods combining the best models from different approaches for even better results.\n"
    report += "3. **Trade-offs**: Consider the trade-off between accuracy, training time, and inference time when selecting a model for production.\n"
    report += "4. **Further Exploration**: Experiment with hyperparameter tuning for the best models to further improve performance.\n"
    report += "5. **Interpretability**: Leverage the interpretability of the best models to gain insights into the importance of different features.\n\n"
    
    # Save report
    report_path = os.path.join(output_dir, "statistical_analysis_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    
    print(f"Statistical analysis report saved to {report_path}")


def perform_cross_validation_analysis(results: Dict, output_dir: str):
    """
    Analyze cross-validation results to assess model stability and generalization.
    
    Args:
        results: Dictionary of results
        output_dir: Directory to save analysis
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    cv_analysis = {
        "deep_learning": {},
        "classical": {},
        "hybrid": {}
    }
    
    # Analyze deep learning cross-validation
    if results["deep_learning"] is not None:
        for model_name, model_results in results["deep_learning"]["models"].items():
            if "error" in model_results:
                continue
            
            for layers, layer_results in model_results["layer_strategies"].items():
                if "error" in layer_results or "fold_accuracies" not in layer_results:
                    continue
                
                fold_accuracies = layer_results.get("fold_accuracies", [])
                
                if fold_accuracies:
                    cv_analysis["deep_learning"][f"{model_name}_{layers}"] = {
                        "fold_accuracies": fold_accuracies,
                        "mean_accuracy": np.mean(fold_accuracies),
                        "std_accuracy": np.std(fold_accuracies),
                        "min_accuracy": np.min(fold_accuracies),
                        "max_accuracy": np.max(fold_accuracies),
                        "range": np.max(fold_accuracies) - np.min(fold_accuracies)
                    }
    
    # Analyze classical cross-validation
    if results["classical"] is not None:
        for model_name, model_results in results["classical"]["models"].items():
            for feature_set, feature_results in model_results["feature_sets"].items():
                for strategy, strategy_results in feature_results["engineering_strategies"].items():
                    fold_accuracies = strategy_results.get("fold_accuracies", [])
                    
                    if fold_accuracies:
                        cv_analysis["classical"][f"{model_name}_{feature_set}_{strategy}"] = {
                            "fold_accuracies": fold_accuracies,
                            "mean_accuracy": np.mean(fold_accuracies),
                            "std_accuracy": np.std(fold_accuracies),
                            "min_accuracy": np.min(fold_accuracies),
                            "max_accuracy": np.max(fold_accuracies),
                            "range": np.max(fold_accuracies) - np.min(fold_accuracies)
                        }
    
    # Analyze hybrid cross-validation
    if results["hybrid"] is not None:
        for approach_name, approach_results in results["hybrid"]["approaches"].items():
            for classical_model, model_results in approach_results["classical_models"].items():
                fold_accuracies = model_results.get("fold_accuracies", [])
                
                if fold_accuracies:
                    cv_analysis["hybrid"][f"{approach_name}_{classical_model}"] = {
                        "fold_accuracies": fold_accuracies,
                        "mean_accuracy": np.mean(fold_accuracies),
                        "std_accuracy": np.std(fold_accuracies),
                        "min_accuracy": np.min(fold_accuracies),
                        "max_accuracy": np.max(fold_accuracies),
                        "range": np.max(fold_accuracies) - np.min(fold_accuracies)
                    }
    
    # Generate cross-validation visualizations
    generate_cv_visualizations(cv_analysis, output_dir)
    
    # Generate cross-validation report
    generate_cv_report(cv_analysis, output_dir)
    
    return cv_analysis


def generate_cv_visualizations(cv_analysis: Dict, output_dir: str):
    """
    Generate visualizations for cross-validation analysis.
    
    Args:
        cv_analysis: Dictionary of cross-validation analysis
        output_dir: Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for plotting
    cv_data = []
    
    for approach, models in cv_analysis.items():
        for model_name, model_results in models.items():
            for i, acc in enumerate(model_results["fold_accuracies"]):
                cv_data.append({
                    "approach": approach,
                    "model": model_name,
                    "fold": i + 1,
                    "accuracy": acc
                })
    
    if not cv_data:
        print("No cross-validation data available for visualization")
        return
    
    cv_df = pd.DataFrame(cv_data)
    
    # Plot cross-validation accuracy by approach
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="approach", y="accuracy", data=cv_df)
    plt.title("Cross-Validation Accuracy by Approach")
    plt.xlabel("Approach")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cv_accuracy_by_approach.png"))
    
    # Plot cross-validation stability (range) by approach
    stability_data = []
    
    for approach, models in cv_analysis.items():
        for model_name, model_results in models.items():
            stability_data.append({
                "approach": approach,
                "model": model_name,
                "range": model_results["range"],
                "std": model_results["std_accuracy"]
            })
    
    stability_df = pd.DataFrame(stability_data)
    
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="approach", y="range", data=stability_df)
    plt.title("Cross-Validation Stability (Range) by Approach")
    plt.xlabel("Approach")
    plt.ylabel("Accuracy Range (Max - Min)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cv_stability_range_by_approach.png"))
    
    # Plot cross-validation stability (std) by approach
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="approach", y="std", data=stability_df)
    plt.title("Cross-Validation Stability (Std) by Approach")
    plt.xlabel("Approach")
    plt.ylabel("Accuracy Standard Deviation")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cv_stability_std_by_approach.png"))
    
    # Plot top 10 most stable models (lowest std)
    top_stable = stability_df.sort_values("std").head(10)
    plt.figure(figsize=(14, 10))
    sns.barplot(x="std", y="model", hue="approach", data=top_stable)
    plt.title("Top 10 Most Stable Models (Lowest Std)")
    plt.xlabel("Accuracy Standard Deviation")
    plt.ylabel("Model")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_stable_models.png"))
    
    # Plot accuracy vs. stability
    plt.figure(figsize=(12, 10))
    for approach, models in cv_analysis.items():
        x = []
        y = []
        labels = []
        
        for model_name, model_results in models.items():
            x.append(model_results["mean_accuracy"])
            y.append(model_results["std_accuracy"])
            labels.append(model_name)
        
        plt.scatter(x, y, label=approach, alpha=0.7, s=100)
    
    plt.title("Accuracy vs. Stability")
    plt.xlabel("Mean Accuracy")
    plt.ylabel("Accuracy Standard Deviation")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_vs_stability.png"))


def generate_cv_report(cv_analysis: Dict, output_dir: str):
    """
    Generate a comprehensive cross-validation analysis report.
    
    Args:
        cv_analysis: Dictionary of cross-validation analysis
        output_dir: Directory to save report
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Start report
    report = "# Cross-Validation Analysis of Prospect Theory Models\n\n"
    
    # Add summary statistics
    report += "## Summary Statistics\n\n"
    
    # Statistics by approach
    for approach, models in cv_analysis.items():
        if not models:
            continue
        
        report += f"### {approach.replace('_', ' ').title()} Approach\n\n"
        report += "| Model | Mean Accuracy | Std Accuracy | Min Accuracy | Max Accuracy | Range |\n"
        report += "|-------|--------------|--------------|--------------|--------------|-------|\n"
        
        for model_name, model_results in models.items():
            report += f"| {model_name} | {model_results['mean_accuracy']:.4f} | {model_results['std_accuracy']:.4f} | {model_results['min_accuracy']:.4f} | {model_results['max_accuracy']:.4f} | {model_results['range']:.4f} |\n"
        
        report += "\n"
    
    # Add stability analysis
    report += "## Stability Analysis\n\n"
    
    # Find most stable models
    all_models = []
    
    for approach, models in cv_analysis.items():
        for model_name, model_results in models.items():
            all_models.append({
                "approach": approach,
                "model": model_name,
                "mean_accuracy": model_results["mean_accuracy"],
                "std_accuracy": model_results["std_accuracy"],
                "range": model_results["range"]
            })
    
    if all_models:
        # Sort by stability (std)
        stable_models = sorted(all_models, key=lambda x: x["std_accuracy"])
        
        report += "### Most Stable Models (Lowest Std)\n\n"
        report += "| Approach | Model | Mean Accuracy | Std Accuracy | Range |\n"
        report += "|----------|-------|--------------|--------------|-------|\n"
        
        for model in stable_models[:10]:
            report += f"| {model['approach']} | {model['model']} | {model['mean_accuracy']:.4f} | {model['std_accuracy']:.4f} | {model['range']:.4f} |\n"
        
        report += "\n"
        
        # Sort by accuracy
        accurate_models = sorted(all_models, key=lambda x: x["mean_accuracy"], reverse=True)
        
        report += "### Most Accurate Models\n\n"
        report += "| Approach | Model | Mean Accuracy | Std Accuracy | Range |\n"
        report += "|----------|-------|--------------|--------------|-------|\n"
        
        for model in accurate_models[:10]:
            report += f"| {model['approach']} | {model['model']} | {model['mean_accuracy']:.4f} | {model['std_accuracy']:.4f} | {model['range']:.4f} |\n"
        
        report += "\n"
        
        # Find best trade-off between accuracy and stability
        # Using a simple score: accuracy - 2 * std
        for model in all_models:
            model["score"] = model["mean_accuracy"] - 2 * model["std_accuracy"]
        
        best_tradeoff = sorted(all_models, key=lambda x: x["score"], reverse=True)
        
        report += "### Best Trade-off between Accuracy and Stability\n\n"
        report += "| Approach | Model | Mean Accuracy | Std Accuracy | Score |\n"
        report += "|----------|-------|--------------|--------------|-------|\n"
        
        for model in best_tradeoff[:10]:
            report += f"| {model['approach']} | {model['model']} | {model['mean_accuracy']:.4f} | {model['std_accuracy']:.4f} | {model['score']:.4f} |\n"
        
        report += "\n"
    
    # Add visualizations
    report += "## Visualizations\n\n"
    report += "### Cross-Validation Accuracy by Approach\n\n"
    report += "![Cross-Validation Accuracy by Approach](cv_accuracy_by_approach.png)\n\n"
    report += "### Cross-Validation Stability (Range) by Approach\n\n"
    report += "![Cross-Validation Stability (Range) by Approach](cv_stability_range_by_approach.png)\n\n"
    report += "### Cross-Validation Stability (Std) by Approach\n\n"
    report += "![Cross-Validation Stability (Std) by Approach](cv_stability_std_by_approach.png)\n\n"
    report += "### Top 10 Most Stable Models\n\n"
    report += "![Top 10 Most Stable Models](top_stable_models.png)\n\n"
    report += "### Accuracy vs. Stability\n\n"
    report += "![Accuracy vs. Stability](accuracy_vs_stability.png)\n\n"
    
    # Add conclusion
    report += "## Conclusion\n\n"
    
    if all_models:
        best_model = best_tradeoff[0]
        report += f"Based on the cross-validation analysis, the model with the best trade-off between accuracy and stability is a {best_model['approach']} approach using {best_model['model']}, "
        report += f"achieving a mean accuracy of {best_model['mean_accuracy']:.4f} with a standard deviation of {best_model['std_accuracy']:.4f}.\n\n"
        
        # Compare approaches
        approach_stats = {}
        
        for model in all_models:
            approach = model["approach"]
            if approach not in approach_stats:
                approach_stats[approach] = {
                    "count": 0,
                    "mean_accuracy": 0,
                    "mean_std": 0
                }
            
            approach_stats[approach]["count"] += 1
            approach_stats[approach]["mean_accuracy"] += model["mean_accuracy"]
            approach_stats[approach]["mean_std"] += model["std_accuracy"]
        
        for approach, stats in approach_stats.items():
            stats["mean_accuracy"] /= stats["count"]
            stats["mean_std"] /= stats["count"]
        
        # Find most stable approach
        most_stable_approach = min(approach_stats.items(), key=lambda x: x[1]["mean_std"])[0]
        
        # Find most accurate approach
        most_accurate_approach = max(approach_stats.items(), key=lambda x: x[1]["mean_accuracy"])[0]
        
        report += f"The {most_stable_approach} approach shows the highest stability across models, while the {most_accurate_approach} approach achieves the highest average accuracy. "
        
        if most_stable_approach == most_accurate_approach:
            report += f"The {most_stable_approach} approach provides the best combination of accuracy and stability.\n\n"
        else:
            report += f"There is a trade-off between accuracy and stability when choosing between these approaches.\n\n"
    
    # Add recommendations
    report += "## Recommendations\n\n"
    
    if all_models:
        report += f"1. **Model Selection**: Use the {best_model['approach']} approach with {best_model['model']} for the best trade-off between accuracy and stability.\n"
        report += "2. **Ensemble Methods**: Consider ensemble methods combining stable models to further improve stability.\n"
        report += "3. **Hyperparameter Tuning**: Focus hyperparameter tuning efforts on the most promising models identified in this analysis.\n"
        report += "4. **Data Augmentation**: Consider data augmentation techniques to improve model stability.\n"
        report += "5. **Feature Engineering**: Experiment with different feature engineering techniques to improve both accuracy and stability.\n\n"
    
    # Save report
    report_path = os.path.join(output_dir, "cross_validation_analysis_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    
    print(f"Cross-validation analysis report saved to {report_path}")


def main():
    """Main function to parse arguments and run the analysis."""
    parser = argparse.ArgumentParser(description="Statistical Analysis and Cross-Validation for Prospect Theory Models")
    
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR,
                        help="Directory containing results")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save analysis")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    results = load_results(args.results_dir)
    
    # Extract metrics
    metrics_df = extract_metrics(results)
    
    # Save metrics
    metrics_df.to_csv(os.path.join(args.output_dir, "metrics.csv"), index=False)
    
    # Perform statistical tests
    statistical_tests = perform_statistical_tests(metrics_df)
    
    # Save statistical tests
    with open(os.path.join(args.output_dir, "statistical_tests.json"), "w") as f:
        json.dump(statistical_tests, f, indent=2)
    
    # Generate statistical visualizations
    generate_statistical_visualizations(metrics_df, args.output_dir)
    
    # Generate statistical report
    generate_statistical_report(metrics_df, statistical_tests, args.output_dir)
    
    # Perform cross-validation analysis
    cv_analysis = perform_cross_validation_analysis(results, os.path.join(args.output_dir, "cross_validation"))
    
    print("Statistical analysis and cross-validation complete!")


if __name__ == "__main__":
    main()

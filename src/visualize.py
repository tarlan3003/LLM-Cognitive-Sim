"""
Advanced visualization and interpretability tools for Prospect Theory LLM Pipeline.

This module provides visualization functions to create meaningful and interpretable
results for the Prospect Theory LLM Pipeline, focusing on cognitive biases and
their relationship to voting behavior.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional, Union


def set_plotting_style():
    """Set consistent plotting style for all visualizations."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 18


def plot_bias_scores_by_class(
    bias_scores: np.ndarray,
    targets: np.ndarray,
    bias_names: List[str],
    target_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Cognitive Bias Scores by Voting Preference"
):
    """
    Plot cognitive bias scores by class.
    
    Args:
        bias_scores: Bias scores [num_samples, num_biases]
        targets: Target labels [num_samples]
        bias_names: Names of biases
        target_names: Names of target classes
        save_path: Path to save the plot
        title: Plot title
    """
    set_plotting_style()
    
    # Calculate mean bias scores by class
    bias_by_class = {}
    for i, class_name in enumerate(target_names):
        mask = targets == i
        if mask.sum() > 0:
            bias_by_class[class_name] = bias_scores[mask].mean(axis=0)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Set width of bars
    bar_width = 0.35
    index = np.arange(len(bias_names))
    
    # Plot bars for each class
    for i, (class_name, scores) in enumerate(bias_by_class.items()):
        ax.bar(
            index + i * bar_width, 
            scores, 
            bar_width, 
            label=class_name,
            alpha=0.8
        )
    
    # Add labels and legend
    ax.set_xlabel('Cognitive Bias Type')
    ax.set_ylabel('Bias Score')
    ax.set_title(title)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(bias_names, rotation=45, ha='right')
    ax.legend()
    
    # Add annotations for significant differences
    for i, bias_name in enumerate(bias_names):
        scores = [bias_by_class[class_name][i] for class_name in target_names if class_name in bias_by_class]
        if len(scores) == 2:
            diff = abs(scores[0] - scores[1])
            if diff > 0.1:  # Threshold for significance
                max_score = max(scores)
                ax.annotate(
                    f'Δ = {diff:.2f}',
                    xy=(i, max_score + 0.05),
                    ha='center',
                    va='bottom',
                    fontweight='bold',
                    color='red' if diff > 0.2 else 'black'
                )
    
    # Add interpretation text
    plt.figtext(
        0.5, 0.01, 
        "Interpretation: Higher scores indicate stronger presence of cognitive bias.\n"
        "Red annotations highlight significant differences between voter groups.",
        ha='center', 
        fontsize=12, 
        bbox={'facecolor': 'lightgray', 'alpha': 0.5, 'pad': 5}
    )
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved bias scores plot to {save_path}")
    else:
        plt.show()


def plot_system_weights_by_class(
    system_weights: np.ndarray,
    targets: np.ndarray,
    target_names: List[str],
    save_path: Optional[str] = None,
    title: str = "System 1 vs System 2 Thinking by Voting Preference"
):
    """
    Plot System 1 vs System 2 thinking weights by class.
    
    Args:
        system_weights: System weights [num_samples, 2]
        targets: Target labels [num_samples]
        target_names: Names of target classes
        save_path: Path to save the plot
        title: Plot title
    """
    set_plotting_style()
    
    # Calculate mean system weights by class
    system_by_class = {}
    for i, class_name in enumerate(target_names):
        mask = targets == i
        if mask.sum() > 0:
            system_by_class[class_name] = system_weights[mask].mean(axis=0)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Bar chart of average system weights by class
    index = np.arange(2)
    bar_width = 0.35
    
    for i, (class_name, weights) in enumerate(system_by_class.items()):
        ax1.bar(
            index + i * bar_width, 
            weights, 
            bar_width, 
            label=class_name,
            alpha=0.8
        )
    
    ax1.set_xlabel('Thinking System')
    ax1.set_ylabel('Weight')
    ax1.set_title('Average System Weights by Voting Preference')
    ax1.set_xticks(index + bar_width / 2)
    ax1.set_xticklabels(['System 1 (Fast)', 'System 2 (Slow)'])
    ax1.legend()
    
    # Add annotations for differences
    for i in range(2):
        weights = [system_by_class[class_name][i] for class_name in target_names if class_name in system_by_class]
        if len(weights) == 2:
            diff = abs(weights[0] - weights[1])
            if diff > 0.05:  # Threshold for significance
                max_weight = max(weights)
                ax1.annotate(
                    f'Δ = {diff:.2f}',
                    xy=(i, max_weight + 0.05),
                    ha='center',
                    va='bottom',
                    fontweight='bold',
                    color='red' if diff > 0.1 else 'black'
                )
    
    # Plot 2: System 1 vs System 2 scatter plot with density
    for i, class_name in enumerate(target_names):
        mask = targets == i
        if mask.sum() > 0:
            ax2.scatter(
                system_weights[mask, 0],
                system_weights[mask, 1],
                alpha=0.5,
                label=class_name
            )
    
    # Add diagonal line
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    
    # Add regions
    ax2.fill_between([0, 1], [0, 0], [0, 1], color='blue', alpha=0.1)
    ax2.fill_between([0, 1], [0, 1], [1, 1], color='red', alpha=0.1)
    ax2.annotate('System 1 Dominant', xy=(0.15, 0.85), ha='center', fontsize=12)
    ax2.annotate('System 2 Dominant', xy=(0.85, 0.15), ha='center', fontsize=12)
    
    ax2.set_xlabel('System 1 Weight (Fast, Intuitive)')
    ax2.set_ylabel('System 2 Weight (Slow, Deliberative)')
    ax2.set_title('System 1 vs System 2 Thinking Distribution')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.legend()
    
    # Add interpretation text
    plt.figtext(
        0.5, 0.01, 
        "Interpretation: System 1 represents fast, intuitive thinking while System 2 represents slow, deliberative thinking.\n"
        "According to Prospect Theory, System 1 is more susceptible to cognitive biases like framing effects.",
        ha='center', 
        fontsize=12, 
        bbox={'facecolor': 'lightgray', 'alpha': 0.5, 'pad': 5}
    )
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved system weights plot to {save_path}")
    else:
        plt.show()


def plot_bias_correlation_matrix(
    bias_scores: np.ndarray,
    bias_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Cognitive Bias Correlation Matrix"
):
    """
    Plot correlation matrix between different cognitive biases.
    
    Args:
        bias_scores: Bias scores [num_samples, num_biases]
        bias_names: Names of biases
        save_path: Path to save the plot
        title: Plot title
    """
    set_plotting_style()
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(bias_scores.T)
    
    # Create plot
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=1,
        vmin=-1,
        center=0,
        square=True,
        linewidths=.5,
        annot=True,
        fmt=".2f",
        xticklabels=bias_names,
        yticklabels=bias_names
    )
    
    plt.title(title)
    
    # Add interpretation text
    plt.figtext(
        0.5, 0.01, 
        "Interpretation: This matrix shows how different cognitive biases correlate with each other.\n"
        "Strong positive correlations suggest biases that often occur together, while negative correlations suggest opposing biases.",
        ha='center', 
        fontsize=12, 
        bbox={'facecolor': 'lightgray', 'alpha': 0.5, 'pad': 5}
    )
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved bias correlation matrix to {save_path}")
    else:
        plt.show()


def plot_feature_importance(
    classifier,
    feature_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Feature Importance for Voting Prediction"
):
    """
    Plot feature importance for the ANES classifier.
    
    Args:
        classifier: Trained classifier model
        feature_names: Names of features
        save_path: Path to save the plot
        title: Plot title
    """
    set_plotting_style()
    
    # Extract weights from the classifier
    if hasattr(classifier, 'classifier'):
        weights = classifier.classifier.weight.data.cpu().numpy()
    else:
        print("Classifier does not have expected structure. Cannot extract weights.")
        return
    
    # Calculate absolute importance
    importance = np.abs(weights).mean(axis=0)
    
    # Sort features by importance
    sorted_idx = np.argsort(importance)
    sorted_importance = importance[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]
    
    # Create plot
    plt.figure(figsize=(12, 10))
    y_pos = np.arange(len(sorted_names))
    
    # Plot horizontal bars
    bars = plt.barh(y_pos, sorted_importance, align='center')
    
    # Color bars by importance
    norm = plt.Normalize(sorted_importance.min(), sorted_importance.max())
    colors = plt.cm.viridis(norm(sorted_importance))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.yticks(y_pos, sorted_names)
    plt.xlabel('Importance')
    plt.title(title)
    
    # Add interpretation text
    plt.figtext(
        0.5, 0.01, 
        "Interpretation: Features with higher importance have stronger influence on voting predictions.\n"
        "This shows which aspects of Prospect Theory are most predictive of voting behavior.",
        ha='center', 
        fontsize=12, 
        bbox={'facecolor': 'lightgray', 'alpha': 0.5, 'pad': 5}
    )
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved feature importance plot to {save_path}")
    else:
        plt.show()


def plot_bias_embedding_visualization(
    bias_scores: np.ndarray,
    targets: np.ndarray,
    bias_names: List[str],
    target_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Cognitive Bias Embedding Visualization",
    method: str = "tsne"
):
    """
    Create 2D visualization of cognitive bias embeddings using t-SNE or PCA.
    
    Args:
        bias_scores: Bias scores [num_samples, num_biases]
        targets: Target labels [num_samples]
        bias_names: Names of biases
        target_names: Names of target classes
        save_path: Path to save the plot
        title: Plot title
        method: Dimensionality reduction method ('tsne' or 'pca')
    """
    set_plotting_style()
    
    # Apply dimensionality reduction
    if method.lower() == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
        embedding = reducer.fit_transform(bias_scores)
        method_name = "t-SNE"
    else:
        reducer = PCA(n_components=2, random_state=42)
        embedding = reducer.fit_transform(bias_scores)
        method_name = "PCA"
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Plot points colored by target class
    for i, class_name in enumerate(target_names):
        mask = targets == i
        if mask.sum() > 0:
            plt.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                alpha=0.7,
                label=class_name
            )
    
    # Add labels and legend
    plt.xlabel(f'{method_name} Dimension 1')
    plt.ylabel(f'{method_name} Dimension 2')
    plt.title(f'{title} ({method_name})')
    plt.legend()
    
    # Add interpretation text
    plt.figtext(
        0.5, 0.01, 
        f"Interpretation: This {method_name} visualization shows how voters cluster based on their cognitive bias patterns.\n"
        "Distinct clusters suggest different cognitive bias profiles between voter groups.",
        ha='center', 
        fontsize=12, 
        bbox={'facecolor': 'lightgray', 'alpha': 0.5, 'pad': 5}
    )
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved bias embedding visualization to {save_path}")
    else:
        plt.show()


def plot_threshold_performance(
    results: Dict,
    save_path: Optional[str] = None,
    title: str = "Classification Performance vs. Threshold"
):
    """
    Plot classification performance metrics across different thresholds.
    
    Args:
        results: Dictionary of results from evaluate_anes_classifier
        save_path: Path to save the plot
        title: Plot title
    """
    set_plotting_style()
    
    # Extract thresholds and metrics
    thresholds = []
    accuracy = []
    precision = []
    recall = []
    f1_scores = []
    
    for key, value in results['thresholded_results'].items():
        threshold = float(key.split('_')[1])
        thresholds.append(threshold)
        accuracy.append(value['accuracy'])
        
        # Get precision, recall, f1 for both classes
        report = value['classification_report']
        precision.append([report[class_name]['precision'] for class_name in ['Donald Trump', 'Kamala Harris']])
        recall.append([report[class_name]['recall'] for class_name in ['Donald Trump', 'Kamala Harris']])
        f1_scores.append([report[class_name]['f1-score'] for class_name in ['Donald Trump', 'Kamala Harris']])
    
    # Sort by threshold
    sorted_idx = np.argsort(thresholds)
    thresholds = [thresholds[i] for i in sorted_idx]
    accuracy = [accuracy[i] for i in sorted_idx]
    precision = [precision[i] for i in sorted_idx]
    recall = [recall[i] for i in sorted_idx]
    f1_scores = [f1_scores[i] for i in sorted_idx]
    
    # Convert to numpy arrays for easier manipulation
    precision = np.array(precision)
    recall = np.array(recall)
    f1_scores = np.array(f1_scores)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Accuracy vs. threshold
    ax1.plot(thresholds, accuracy, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Classification Threshold')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy vs. Threshold')
    ax1.grid(True)
    
    # Add best threshold
    if 'best_threshold' in results and 'best_accuracy' in results:
        best_threshold = results['best_threshold']
        best_accuracy = results['best_accuracy']
        ax1.axvline(x=best_threshold, color='r', linestyle='--', alpha=0.7)
        ax1.axhline(y=best_accuracy, color='r', linestyle='--', alpha=0.7)
        ax1.plot(best_threshold, best_accuracy, 'ro', markersize=10)
        ax1.annotate(
            f'Best: {best_threshold:.2f}, {best_accuracy:.2f}',
            xy=(best_threshold, best_accuracy),
            xytext=(best_threshold + 0.05, best_accuracy - 0.05),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
            fontsize=12
        )
    
    # Plot 2: Precision and recall vs. threshold
    ax2.plot(thresholds, precision[:, 0], 'b-', label='Trump Precision', linewidth=2)
    ax2.plot(thresholds, recall[:, 0], 'b--', label='Trump Recall', linewidth=2)
    ax2.plot(thresholds, precision[:, 1], 'r-', label='Harris Precision', linewidth=2)
    ax2.plot(thresholds, recall[:, 1], 'r--', label='Harris Recall', linewidth=2)
    
    ax2.set_xlabel('Classification Threshold')
    ax2.set_ylabel('Score')
    ax2.set_title('Precision and Recall vs. Threshold')
    ax2.legend()
    ax2.grid(True)
    
    # Add interpretation text
    plt.figtext(
        0.5, 0.01, 
        "Interpretation: The threshold controls the trade-off between precision and recall for each class.\n"
        "The optimal threshold balances performance across both voter groups.",
        ha='center', 
        fontsize=12, 
        bbox={'facecolor': 'lightgray', 'alpha': 0.5, 'pad': 5}
    )
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved threshold performance plot to {save_path}")
    else:
        plt.show()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    target_names: List[str],
    threshold: float,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix"
):
    """
    Plot confusion matrix for classification results.
    
    Args:
        confusion_matrix: Confusion matrix [num_classes, num_classes]
        target_names: Names of target classes
        threshold: Classification threshold used
        save_path: Path to save the plot
        title: Plot title
    """
    set_plotting_style()
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Calculate percentages
    cm_sum = np.sum(confusion_matrix, axis=1, keepdims=True)
    cm_perc = confusion_matrix / cm_sum * 100
    annot = np.empty_like(confusion_matrix, dtype=str)
    
    # Format annotation text
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            annot[i, j] = f'{confusion_matrix[i, j]}\n{cm_perc[i, j]:.1f}%'
    
    # Plot heatmap
    sns.heatmap(
        confusion_matrix,
        annot=annot,
        fmt='',
        cmap='Blues',
        square=True,
        xticklabels=target_names,
        yticklabels=target_names
    )
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{title} (Threshold: {threshold:.2f})')
    
    # Add interpretation text
    plt.figtext(
        0.5, 0.01, 
        "Interpretation: The confusion matrix shows how many voters were correctly and incorrectly classified.\n"
        "Diagonal elements represent correct predictions, while off-diagonal elements represent errors.",
        ha='center', 
        fontsize=12, 
        bbox={'facecolor': 'lightgray', 'alpha': 0.5, 'pad': 5}
    )
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved confusion matrix to {save_path}")
    else:
        plt.show()


def create_prospect_theory_summary_figure(
    bias_scores: np.ndarray,
    system_weights: np.ndarray,
    targets: np.ndarray,
    bias_names: List[str],
    target_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Prospect Theory Analysis of Voting Behavior"
):
    """
    Create a comprehensive summary figure showing key Prospect Theory insights.
    
    Args:
        bias_scores: Bias scores [num_samples, num_biases]
        system_weights: System weights [num_samples, 2]
        targets: Target labels [num_samples]
        bias_names: Names of biases
        target_names: Names of target classes
        save_path: Path to save the plot
        title: Plot title
    """
    set_plotting_style()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(2, 2)
    
    # Plot 1: Bias scores by class
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Calculate mean bias scores by class
    bias_by_class = {}
    for i, class_name in enumerate(target_names):
        mask = targets == i
        if mask.sum() > 0:
            bias_by_class[class_name] = bias_scores[mask].mean(axis=0)
    
    # Set width of bars
    bar_width = 0.35
    index = np.arange(len(bias_names))
    
    # Plot bars for each class
    for i, (class_name, scores) in enumerate(bias_by_class.items()):
        ax1.bar(
            index + i * bar_width, 
            scores, 
            bar_width, 
            label=class_name,
            alpha=0.8
        )
    
    # Add labels and legend
    ax1.set_xlabel('Cognitive Bias Type')
    ax1.set_ylabel('Bias Score')
    ax1.set_title('Cognitive Bias Scores by Voting Preference')
    ax1.set_xticks(index + bar_width / 2)
    ax1.set_xticklabels(bias_names, rotation=45, ha='right')
    ax1.legend()
    
    # Plot 2: System weights by class
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Calculate mean system weights by class
    system_by_class = {}
    for i, class_name in enumerate(target_names):
        mask = targets == i
        if mask.sum() > 0:
            system_by_class[class_name] = system_weights[mask].mean(axis=0)
    
    # Plot bars
    index = np.arange(2)
    for i, (class_name, weights) in enumerate(system_by_class.items()):
        ax2.bar(
            index + i * bar_width, 
            weights, 
            bar_width, 
            label=class_name,
            alpha=0.8
        )
    
    ax2.set_xlabel('Thinking System')
    ax2.set_ylabel('Weight')
    ax2.set_title('System 1 vs System 2 Thinking by Voting Preference')
    ax2.set_xticks(index + bar_width / 2)
    ax2.set_xticklabels(['System 1 (Fast)', 'System 2 (Slow)'])
    ax2.legend()
    
    # Plot 3: Bias embedding visualization
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Apply PCA for dimensionality reduction
    reducer = PCA(n_components=2, random_state=42)
    embedding = reducer.fit_transform(bias_scores)
    
    # Plot points colored by target class
    for i, class_name in enumerate(target_names):
        mask = targets == i
        if mask.sum() > 0:
            ax3.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                alpha=0.7,
                label=class_name
            )
    
    ax3.set_xlabel('PCA Dimension 1')
    ax3.set_ylabel('PCA Dimension 2')
    ax3.set_title('Cognitive Bias Embedding Visualization')
    ax3.legend()
    
    # Plot 4: System 1 vs System 2 scatter plot
    ax4 = fig.add_subplot(gs[1, 1])
    
    for i, class_name in enumerate(target_names):
        mask = targets == i
        if mask.sum() > 0:
            ax4.scatter(
                system_weights[mask, 0],
                system_weights[mask, 1],
                alpha=0.5,
                label=class_name
            )
    
    # Add diagonal line
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    
    # Add regions
    ax4.fill_between([0, 1], [0, 0], [0, 1], color='blue', alpha=0.1)
    ax4.fill_between([0, 1], [0, 1], [1, 1], color='red', alpha=0.1)
    ax4.annotate('System 1 Dominant', xy=(0.15, 0.85), ha='center', fontsize=12)
    ax4.annotate('System 2 Dominant', xy=(0.85, 0.15), ha='center', fontsize=12)
    
    ax4.set_xlabel('System 1 Weight (Fast, Intuitive)')
    ax4.set_ylabel('System 2 Weight (Slow, Deliberative)')
    ax4.set_title('System 1 vs System 2 Thinking Distribution')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.legend()
    
    # Add main title
    plt.suptitle(title, fontsize=20)
    
    # Add interpretation text
    plt.figtext(
        0.5, 0.01, 
        "Interpretation: This figure summarizes key insights from Prospect Theory applied to voting behavior.\n"
        "Different cognitive biases and thinking systems show distinct patterns between voter groups,\n"
        "supporting Kahneman and Tversky's theory that decision-making is influenced by cognitive shortcuts and framing effects.",
        ha='center', 
        fontsize=14, 
        bbox={'facecolor': 'lightgray', 'alpha': 0.5, 'pad': 10}
    )
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved Prospect Theory summary figure to {save_path}")
    else:
        plt.show()


def generate_all_visualizations(
    results: Dict,
    bias_names: List[str],
    target_names: List[str],
    feature_names: List[str],
    classifier,
    output_dir: str = "results"
):
    """
    Generate all visualizations for the Prospect Theory LLM Pipeline.
    
    Args:
        results: Dictionary of results from evaluate_anes_classifier
        bias_names: Names of biases
        target_names: Names of target classes
        feature_names: Names of features
        classifier: Trained classifier model
        output_dir: Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data from results
    all_bias_scores = np.concatenate([results['bias_by_class'][class_name] for class_name in target_names])
    all_system_weights = np.concatenate([results['system_by_class'][class_name] for class_name in target_names])
    all_targets = np.concatenate([np.full(len(results['bias_by_class'][class_name]), i) for i, class_name in enumerate(target_names)])
    
    # Generate visualizations
    plot_bias_scores_by_class(
        all_bias_scores,
        all_targets,
        bias_names,
        target_names,
        save_path=os.path.join(output_dir, "bias_scores_by_class.png")
    )
    
    plot_system_weights_by_class(
        all_system_weights,
        all_targets,
        target_names,
        save_path=os.path.join(output_dir, "system_weights_by_class.png")
    )
    
    plot_bias_correlation_matrix(
        all_bias_scores,
        bias_names,
        save_path=os.path.join(output_dir, "bias_correlation_matrix.png")
    )
    
    plot_feature_importance(
        classifier,
        feature_names,
        save_path=os.path.join(output_dir, "feature_importance.png")
    )
    
    plot_bias_embedding_visualization(
        all_bias_scores,
        all_targets,
        bias_names,
        target_names,
        save_path=os.path.join(output_dir, "bias_embedding_tsne.png"),
        method="tsne"
    )
    
    plot_bias_embedding_visualization(
        all_bias_scores,
        all_targets,
        bias_names,
        target_names,
        save_path=os.path.join(output_dir, "bias_embedding_pca.png"),
        method="pca"
    )
    
    plot_threshold_performance(
        results,
        save_path=os.path.join(output_dir, "threshold_performance.png")
    )
    
    # Plot confusion matrices for different thresholds
    for threshold, result in results['thresholded_results'].items():
        threshold_value = float(threshold.split('_')[1])
        plot_confusion_matrix(
            result['confusion_matrix'],
            target_names,
            threshold_value,
            save_path=os.path.join(output_dir, f"confusion_matrix_{threshold_value:.1f}.png")
        )
    
    # Create summary figure
    create_prospect_theory_summary_figure(
        all_bias_scores,
        all_system_weights,
        all_targets,
        bias_names,
        target_names,
        save_path=os.path.join(output_dir, "prospect_theory_summary.png")
    )
    
    print(f"All visualizations saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Create dummy data
    num_samples = 100
    num_biases = 5
    bias_names = ["anchoring", "framing", "availability", "confirmation_bias", "loss_aversion"]
    target_names = ["Donald Trump", "Kamala Harris"]
    
    # Generate random bias scores
    bias_scores = np.random.rand(num_samples, num_biases)
    
    # Generate random system weights
    system_weights = np.random.rand(num_samples, 2)
    system_weights = system_weights / system_weights.sum(axis=1, keepdims=True)
    
    # Generate random targets
    targets = np.random.randint(0, 2, num_samples)
    
    # Create output directory
    os.makedirs("results", exist_ok=True)
    
    # Generate example visualizations
    plot_bias_scores_by_class(
        bias_scores,
        targets,
        bias_names,
        target_names,
        save_path="results/bias_scores_by_class.png"
    )
    
    plot_system_weights_by_class(
        system_weights,
        targets,
        target_names,
        save_path="results/system_weights_by_class.png"
    )
    
    create_prospect_theory_summary_figure(
        bias_scores,
        system_weights,
        targets,
        bias_names,
        target_names,
        save_path="results/prospect_theory_summary.png"
    )

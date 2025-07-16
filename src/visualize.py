
"""
Advanced visualization and interpretability tools for Prospect Theory LLM Pipeline - Fixed Version.

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
    plt.style.use("default")  # Use default style for compatibility
    plt.rcParams["figure.figsize"] = (12, 8)
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["figure.titlesize"] = 18


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
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate mean bias scores for each class
    class_bias_means = []
    class_bias_stds = []
    
    for class_idx in range(len(target_names)):
        mask = targets == class_idx
        if mask.sum() > 0:
            class_means = bias_scores[mask].mean(axis=0)
            class_stds = bias_scores[mask].std(axis=0)
        else:
            class_means = np.zeros(len(bias_names))
            class_stds = np.zeros(len(bias_names))
        
        class_bias_means.append(class_means)
        class_bias_stds.append(class_stds)
    
    # Create grouped bar plot
    x = np.arange(len(bias_names))
    width = 0.35
    
    for i, (class_name, means, stds) in enumerate(zip(target_names, class_bias_means, class_bias_stds)):
        offset = (i - len(target_names)/2 + 0.5) * width
        ax.bar(x + offset, means, width, label=class_name, yerr=stds, capsize=5, alpha=0.8)
    
    ax.set_xlabel("Cognitive Biases")
    ax.set_ylabel("Average Bias Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(bias_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved bias scores plot to {save_path}")
    
    plt.show()


def plot_system_weights_distribution(
    system_weights: np.ndarray,
    targets: np.ndarray,
    target_names: List[str],
    save_path: Optional[str] = None,
    title: str = "System 1/2 Weight Distribution by Voting Preference"
):
    """
    Plot System 1/2 weight distributions by class.
    
    Args:
        system_weights: System weights [num_samples, 2]
        targets: Target labels [num_samples]
        target_names: Names of target classes
        save_path: Path to save the plot
        title: Plot title
    """
    set_plotting_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # System 1 weights
    for class_idx, class_name in enumerate(target_names):
        mask = targets == class_idx
        if mask.sum() > 0:
            weights = system_weights[mask, 0]  # System 1 weights
            ax1.hist(weights, bins=30, alpha=0.7, label=class_name, density=True)
    
    ax1.set_xlabel("System 1 Weight")
    ax1.set_ylabel("Density")
    ax1.set_title("System 1 Weight Distribution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # System 2 weights
    for class_idx, class_name in enumerate(target_names):
        mask = targets == class_idx
        if mask.sum() > 0:
            weights = system_weights[mask, 1]  # System 2 weights
            ax2.hist(weights, bins=30, alpha=0.7, label=class_name, density=True)
    
    ax2.set_xlabel("System 2 Weight")
    ax2.set_ylabel("Density")
    ax2.set_title("System 2 Weight Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    plt.show()


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    target_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix"
):
    """
    Plot confusion matrix.
    
    Args:
        confusion_matrix: Confusion matrix
        target_names: Names of target classes
        save_path: Path to save the plot
        title: Plot title
    """
    set_plotting_style()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Normalize confusion matrix, handle division by zero
    cm_sum = confusion_matrix.sum(axis=1, keepdims=True)
    cm_normalized = np.where(cm_sum == 0, 0, confusion_matrix.astype("float") / cm_sum)
    
    # Create heatmap
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=target_names, yticklabels=target_names, ax=ax)
    
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved confusion matrix to {save_path}")
    
    plt.show()


def plot_training_curves(
    training_metrics: Dict,
    save_path: Optional[str] = None,
    title: str = "Training Curves"
):
    """
    Plot training loss and accuracy curves.
    
    Args:
        training_metrics: Dictionary containing training metrics
        save_path: Path to save the plot
        title: Plot title
    """
    set_plotting_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    epochs = range(1, len(training_metrics["epoch_losses"]) + 1)
    
    # Loss curve
    ax1.plot(epochs, training_metrics["epoch_losses"], "b-", label="Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Accuracy curve
    ax2.plot(epochs, training_metrics["epoch_accuracies"], "r-", label="Training Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training Accuracy")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved training curves to {save_path}")
    
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
    correlation_matrix = np.corrcoef(bias_scores.T)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                xticklabels=bias_names, yticklabels=bias_names, ax=ax)
    
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved bias correlation matrix to {save_path}")
    
    plt.show()


def plot_threshold_performance(
    thresholded_results: Dict,
    save_path: Optional[str] = None,
    title: str = "Performance vs Classification Threshold"
):
    """
    Plot performance metrics vs classification threshold.
    
    Args:
        thresholded_results: Dictionary of results for different thresholds
        save_path: Path to save the plot
        title: Plot title
    """
    set_plotting_style()
    
    thresholds = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for key, metrics in thresholded_results.items():
        if key.startswith("threshold_") and "accuracy" in metrics:
            threshold = float(key.split("_")[1])
            thresholds.append(threshold)
            accuracies.append(metrics["accuracy"])
            
            if "macro_precision" in metrics:
                precisions.append(metrics["macro_precision"])
                recalls.append(metrics["macro_recall"])
                f1_scores.append(metrics["macro_f1"])
            else:
                precisions.append(0)
                recalls.append(0)
                f1_scores.append(0)
    
    # Sort by threshold
    sorted_indices = np.argsort(thresholds)
    thresholds = [thresholds[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    precisions = [precisions[i] for i in sorted_indices]
    recalls = [recalls[i] for i in sorted_indices]
    f1_scores = [f1_scores[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(thresholds, accuracies, "o-", label="Accuracy", linewidth=2)
    ax.plot(thresholds, precisions, "s-", label="Precision", linewidth=2)
    ax.plot(thresholds, recalls, "^--", label="Recall", linewidth=2) # Fixed: Changed '^- ' to '^--'
    ax.plot(thresholds, f1_scores, "d-", label="F1-Score", linewidth=2)
    
    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Performance Metric")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min(thresholds) - 0.05, max(thresholds) + 0.05)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved threshold performance plot to {save_path}")
    
    plt.show()


def generate_visualizations(
    val_dataloader,
    extractor,
    bias_representer,
    classifier,
    metrics: Dict,
    save_dir: str,
    bias_names: Optional[List[str]] = None,
    target_names: List[str] = None,
    use_bert_classifier: bool = False
):
    """
    Generate all visualizations for the Prospect Theory LLM Pipeline.
    
    Args:
        val_dataloader: Validation data loader
        extractor: Hidden layer extractor
        bias_representer: Cognitive bias representer
        classifier: Trained classifier
        metrics: Evaluation metrics
        save_dir: Directory to save visualizations
        bias_names: List of bias names (only for non-BERT classifier)
        target_names: Names of target classes
        use_bert_classifier: Whether the classifier is BERT-based
    """
    if target_names is None:
        target_names = ["Trump", "Harris"]
    
    print("Generating comprehensive visualizations...")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    if not use_bert_classifier:
        all_bias_scores = []
        all_system_weights = []
        all_targets = []
        
        if classifier is not None:
            classifier.eval()
            device = next(classifier.parameters()).device
            
            with torch.no_grad():
                for batch in val_dataloader:
                    texts = batch["text"]
                    targets = batch["target"]
                    
                    # Extract activations
                    activations = extractor.extract_activations(texts)
                    
                    # Get bias scores and system representations
                    bias_scores = bias_representer.get_bias_scores(activations)
                    _, system_weights = bias_representer.get_system_representations(activations)
                    
                    all_bias_scores.append(bias_scores.cpu().numpy())
                    all_system_weights.append(system_weights.cpu().numpy())
                    all_targets.append(targets.numpy())
            
            # Concatenate all data
            all_bias_scores = np.concatenate(all_bias_scores, axis=0)
            all_system_weights = np.concatenate(all_system_weights, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            
            # Generate individual plots for non-BERT classifier
            if bias_names:
                try:
                    # 1. Bias scores by class
                    plot_bias_scores_by_class(
                        all_bias_scores, all_targets, bias_names, target_names,
                        save_path=os.path.join(save_dir, "bias_scores_by_class.png")
                    )
                except Exception as e:
                    print(f"Error generating bias scores plot: {e}")
                
                try:
                    # 2. System weights distribution
                    plot_system_weights_distribution(
                        all_system_weights, all_targets, target_names,
                        save_path=os.path.join(save_dir, "system_weights_distribution.png")
                    )
                except Exception as e:
                    print(f"Error generating system weights plot: {e}")
                
                try:
                    # 3. Bias correlation matrix
                    plot_bias_correlation_matrix(
                        all_bias_scores, bias_names,
                        save_path=os.path.join(save_dir, "bias_correlation_matrix.png")
                    )
                except Exception as e:
                    print(f"Error generating bias correlation plot: {e}")
    else:
        print("Skipping bias and system weight visualizations for BERT classifier.")
    
    # 4. Training curves
    if "training_metrics" in metrics:
        try:
            plot_training_curves(
                metrics["training_metrics"],
                save_path=os.path.join(save_dir, "training_curves.png")
            )
        except Exception as e:
            print(f"Error generating training curves: {e}")
    
    # 5. Threshold performance
    if "thresholded_results" in metrics:
        try:
            plot_threshold_performance(
                metrics["thresholded_results"],
                save_path=os.path.join(save_dir, "threshold_performance.png")
            )
        except Exception as e:
            print(f"Error generating threshold performance plot: {e}")
    
    # 6. Confusion matrices for different thresholds
    if "thresholded_results" in metrics:
        for threshold_key, threshold_metrics in metrics["thresholded_results"].items():
            if "confusion_matrix" in threshold_metrics:
                try:
                    threshold_value = threshold_key.split("_")[1]
                    plot_confusion_matrix(
                        threshold_metrics["confusion_matrix"], target_names,
                        save_path=os.path.join(save_dir, f"confusion_matrix_{threshold_value}.png"),
                        title=f"Confusion Matrix (Threshold = {threshold_value})"
                    )
                except Exception as e:
                    print(f"Error generating confusion matrix for {threshold_key}: {e}")
    
    print(f"Visualizations saved to {save_dir}")


if __name__ == "__main__":
    # Example usage
    print("Visualization module loaded successfully!")
    print("Available functions:")
    print("- plot_bias_scores_by_class: Compare bias scores across voting preferences")
    print("- plot_system_weights_distribution: Analyze System 1/2 thinking patterns")
    print("- plot_confusion_matrix: Visualize classification performance")
    print("- plot_training_curves: Monitor training progress")
    print("- plot_bias_correlation_matrix: Understand bias relationships")
    print("- plot_threshold_performance: Optimize classification thresholds")
    print("- generate_visualizations: Create comprehensive visualization suite")
    
    # Test plotting style
    set_plotting_style()
    print("Plotting style configured successfully!")










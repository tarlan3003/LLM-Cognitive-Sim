"""
Utility functions for Prospect Theory LLM - Best Performing Version

This module provides utility functions for the Prospect Theory LLM pipeline.

Author: Tarlan Sultanov
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union

def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(directory: str):
    """
    Ensure directory exists.
    
    Args:
        directory: Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_directory_structure():
    """
    Create directory structure for the project.
    """
    directories = [
        "data/prospect_theory",
        "data/anes",
        "models",
        "results"
    ]
    
    for directory in directories:
        ensure_dir(directory)
    
    print("Directory structure created.")

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = ["Donald Trump", "Kamala Harris"],
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix"
):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Class names
        save_path: Path to save the plot
        title: Plot title
    """
    # Calculate confusion matrix
    cm = np.zeros((len(class_names), len(class_names)))
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1
    
    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved confusion matrix to {save_path}")
    else:
        plt.show()

def plot_bias_scores_by_class(
    bias_scores: np.ndarray,
    targets: np.ndarray,
    bias_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Cognitive Bias Scores by Voting Preference"
):
    """
    Plot bias scores by class.
    
    Args:
        bias_scores: Bias scores
        targets: Target labels
        bias_names: Bias names
        save_path: Path to save the plot
        title: Plot title
    """
    # Calculate mean bias scores for each class
    class_0_indices = targets == 0  # Trump
    class_1_indices = targets == 1  # Harris
    
    class_0_scores = bias_scores[class_0_indices].mean(axis=0)
    class_1_scores = bias_scores[class_1_indices].mean(axis=0)
    
    # Plot
    x = np.arange(len(bias_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, class_0_scores, width, label='Donald Trump Voters')
    rects2 = ax.bar(x + width/2, class_1_scores, width, label='Kamala Harris Voters')
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Cognitive Bias Type', fontsize=14)
    ax.set_ylabel('Average Bias Score', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(bias_names, rotation=45, ha='right')
    ax.legend()
    
    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved bias scores plot to {save_path}")
    else:
        plt.show()

def plot_system_weights_by_class(
    system_weights: np.ndarray,
    targets: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "System 1 vs System 2 Thinking by Voting Preference"
):
    """
    Plot system weights by class.
    
    Args:
        system_weights: System weights
        targets: Target labels
        save_path: Path to save the plot
        title: Plot title
    """
    # Calculate mean system weights for each class
    class_0_indices = targets == 0  # Trump
    class_1_indices = targets == 1  # Harris
    
    class_0_system = system_weights[class_0_indices].mean(axis=0)
    class_1_system = system_weights[class_1_indices].mean(axis=0)
    
    # Plot
    system_names = ['System 1 (Fast)', 'System 2 (Slow)']
    x = np.arange(len(system_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, class_0_system, width, label='Donald Trump Voters')
    rects2 = ax.bar(x + width/2, class_1_system, width, label='Kamala Harris Voters')
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Thinking System', fontsize=14)
    ax.set_ylabel('Average Weight', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(system_names)
    ax.legend()
    
    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved system weights plot to {save_path}")
    else:
        plt.show()

def plot_bias_correlation_matrix(
    bias_scores: np.ndarray,
    bias_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Correlation Between Cognitive Biases"
):
    """
    Plot bias correlation matrix.
    
    Args:
        bias_scores: Bias scores
        bias_names: Bias names
        save_path: Path to save the plot
        title: Plot title
    """
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(bias_scores.T)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, xticklabels=bias_names, yticklabels=bias_names)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved correlation matrix to {save_path}")
    else:
        plt.show()

def plot_feature_importance(
    feature_names: List[str],
    importance: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Feature Importance"
):
    """
    Plot feature importance.
    
    Args:
        feature_names: Feature names
        importance: Feature importance values
        save_path: Path to save the plot
        title: Plot title
    """
    # Sort by importance
    indices = np.argsort(importance)[::-1]
    sorted_names = [feature_names[i] for i in indices]
    sorted_importance = importance[indices]
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(sorted_names)), sorted_importance)
    plt.xticks(range(len(sorted_names)), sorted_names, rotation=45, ha='right')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved feature importance plot to {save_path}")
    else:
        plt.show()

def plot_threshold_performance(
    thresholds: List[float],
    metrics: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Performance Across Thresholds"
):
    """
    Plot performance across thresholds.
    
    Args:
        thresholds: Threshold values
        metrics: Dictionary of metrics
        save_path: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    for metric_name, values in metrics.items():
        plt.plot(thresholds, values, marker='o', label=metric_name)
    
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title(title, fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved threshold performance plot to {save_path}")
    else:
        plt.show()

def plot_comprehensive_summary(
    bias_scores: np.ndarray,
    system_weights: np.ndarray,
    targets: np.ndarray,
    bias_names: List[str],
    feature_importance: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "Prospect Theory Analysis Summary"
):
    """
    Plot comprehensive summary figure.
    
    Args:
        bias_scores: Bias scores
        system_weights: System weights
        targets: Target labels
        bias_names: Bias names
        feature_importance: Feature importance values
        feature_names: Feature names
        save_path: Path to save the plot
        title: Plot title
    """
    # Calculate mean bias scores for each class
    class_0_indices = targets == 0  # Trump
    class_1_indices = targets == 1  # Harris
    
    class_0_scores = bias_scores[class_0_indices].mean(axis=0)
    class_1_scores = bias_scores[class_1_indices].mean(axis=0)
    
    class_0_system = system_weights[class_0_indices].mean(axis=0)
    class_1_system = system_weights[class_1_indices].mean(axis=0)
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(bias_scores.T)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Bias scores by class
    x = np.arange(len(bias_names))
    width = 0.35
    
    rects1 = axes[0, 0].bar(x - width/2, class_0_scores, width, label='Donald Trump Voters')
    rects2 = axes[0, 0].bar(x + width/2, class_1_scores, width, label='Kamala Harris Voters')
    
    axes[0, 0].set_title('Cognitive Bias Scores by Voting Preference', fontsize=16)
    axes[0, 0].set_xlabel('Cognitive Bias Type', fontsize=14)
    axes[0, 0].set_ylabel('Average Bias Score', fontsize=14)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(bias_names, rotation=45, ha='right')
    axes[0, 0].legend()
    
    # System weights by class
    system_names = ['System 1 (Fast)', 'System 2 (Slow)']
    x = np.arange(len(system_names))
    
    rects3 = axes[0, 1].bar(x - width/2, class_0_system, width, label='Donald Trump Voters')
    rects4 = axes[0, 1].bar(x + width/2, class_1_system, width, label='Kamala Harris Voters')
    
    axes[0, 1].set_title('System 1 vs System 2 Thinking by Voting Preference', fontsize=16)
    axes[0, 1].set_xlabel('Thinking System', fontsize=14)
    axes[0, 1].set_ylabel('Average Weight', fontsize=14)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(system_names)
    axes[0, 1].legend()
    
    # Bias correlation matrix
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                xticklabels=bias_names, yticklabels=bias_names, ax=axes[1, 0])
    axes[1, 0].set_title('Correlation Between Cognitive Biases', fontsize=16)
    
    # Feature importance
    if feature_importance is not None and feature_names is not None:
        # Sort by importance
        indices = np.argsort(feature_importance)[::-1]
        sorted_names = [feature_names[i] for i in indices]
        sorted_importance = feature_importance[indices]
        
        # Plot top 10 features
        top_n = min(10, len(sorted_names))
        axes[1, 1].bar(range(top_n), sorted_importance[:top_n])
        axes[1, 1].set_xticks(range(top_n))
        axes[1, 1].set_xticklabels(sorted_names[:top_n], rotation=45, ha='right')
        axes[1, 1].set_title('Top Features for Prediction', fontsize=16)
        axes[1, 1].set_xlabel('Feature', fontsize=14)
        axes[1, 1].set_ylabel('Importance', fontsize=14)
    else:
        # Plot system weight distribution
        axes[1, 1].hist(system_weights[:, 0], bins=20, alpha=0.7, label="System 1")
        axes[1, 1].hist(system_weights[:, 1], bins=20, alpha=0.7, label="System 2")
        axes[1, 1].set_title("Distribution of System 1/2 Thinking Weights", fontsize=16)
        axes[1, 1].set_xlabel("Weight", fontsize=14)
        axes[1, 1].set_ylabel("Frequency", fontsize=14)
        axes[1, 1].legend()
    
    plt.suptitle(title, fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved comprehensive summary to {save_path}")
    else:
        plt.show()

def get_device_info():
    """
    Get device information.
    
    Returns:
        Dictionary with device information
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    info = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available()
    }
    
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_device_capability"] = torch.cuda.get_device_capability(0)
    
    return info

def optimize_batch_size(model_name: str, max_batch_size: int = 32):
    """
    Optimize batch size based on available memory.
    
    Args:
        model_name: Name of the model
        max_batch_size: Maximum batch size to try
        
    Returns:
        Optimal batch size
    """
    if not torch.cuda.is_available():
        return 8  # Default for CPU
    
    # Try different batch sizes
    for batch_size in [max_batch_size, 16, 8, 4, 2, 1]:
        try:
            # Try to load model and create a dummy batch
            model = torch.hub.load('huggingface/pytorch-transformers', 'model', model_name)
            model.to("cuda")
            
            # Create dummy input
            dummy_input = torch.ones((batch_size, 128), dtype=torch.long).to("cuda")
            
            # Try forward pass
            with torch.no_grad():
                model(dummy_input)
            
            # If we get here, the batch size works
            del model
            torch.cuda.empty_cache()
            return batch_size
        
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"Batch size {batch_size} too large, trying smaller...")
                torch.cuda.empty_cache()
            else:
                raise e
    
    return 1  # Fallback to minimum batch size

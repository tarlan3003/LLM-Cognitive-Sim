"""
Utility functions for the Prospect Theory LLM Pipeline.

This module provides helper functions for data processing, visualization,
and other common tasks used across the pipeline.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union


def plot_training_curves(
    system_metrics: Dict, 
    anes_metrics: Dict, 
    save_path: str = None
) -> None:
    """
    Plot training curves for system components and ANES classifier.
    
    Args:
        system_metrics: Dictionary of system training metrics
        anes_metrics: Dictionary of ANES training metrics
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(system_metrics['epoch_losses'], label='Loss')
    plt.plot(system_metrics['epoch_accuracies'], label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('System Components Training')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(anes_metrics['epoch_losses'], label='Loss')
    plt.plot(anes_metrics['epoch_accuracies'], label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('ANES Classifier Training')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training curves saved to {save_path}")
    else:
        plt.show()


def plot_bias_scores_by_class(
    bias_by_class: Dict, 
    bias_names: List[str], 
    save_path: str = None
) -> None:
    """
    Plot bias scores by class.
    
    Args:
        bias_by_class: Dictionary mapping class names to bias scores
        bias_names: List of bias names
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(bias_names))
    width = 0.35
    
    for i, (class_name, scores) in enumerate(bias_by_class.items()):
        plt.bar(
            x + i*width - width/2, 
            scores, 
            width, 
            label=class_name
        )
    
    plt.xlabel('Cognitive Bias')
    plt.ylabel('Average Score')
    plt.title('Cognitive Bias Scores by Class')
    plt.xticks(x, bias_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Bias scores plot saved to {save_path}")
    else:
        plt.show()


def plot_system_weights_by_class(
    system_by_class: Dict, 
    save_path: str = None
) -> None:
    """
    Plot system weights by class.
    
    Args:
        system_by_class: Dictionary mapping class names to system weights
        save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    
    system_labels = ['System 1', 'System 2']
    x = np.arange(len(system_labels))
    width = 0.35
    
    for i, (class_name, weights) in enumerate(system_by_class.items()):
        plt.bar(
            x + i*width - width/2, 
            weights, 
            width, 
            label=class_name
        )
    
    plt.xlabel('Thinking System')
    plt.ylabel('Average Weight')
    plt.title('System 1/2 Weights by Class')
    plt.xticks(x, system_labels)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"System weights plot saved to {save_path}")
    else:
        plt.show()


def save_evaluation_results(
    eval_metrics: Dict, 
    output_dir: str,
    target_names: List[str] = None
) -> None:
    """
    Save evaluation results to files.
    
    Args:
        eval_metrics: Dictionary of evaluation metrics
        output_dir: Directory to save results
        target_names: Names of target classes
    """
    if target_names is None:
        target_names = ["Donald Trump", "Kamala Harris"]
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results for each threshold
    for threshold, metrics in eval_metrics['thresholded_results'].items():
        # Save classification report
        with open(os.path.join(output_dir, f'classification_report_{threshold}.txt'), 'w') as f:
            f.write(f"Accuracy: {metrics['accuracy']:.4f}\n\n")
            f.write("Classification Report:\n")
            for class_name, class_metrics in metrics['classification_report'].items():
                if isinstance(class_metrics, dict):
                    f.write(f"  {class_name}: Precision={class_metrics['precision']:.4f}, ")
                    f.write(f"Recall={class_metrics['recall']:.4f}, F1={class_metrics['f1-score']:.4f}\n")
        
        # Save confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            metrics['confusion_matrix'], 
            annot=True, 
            fmt='d', 
            xticklabels=target_names,
            yticklabels=target_names
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix (Threshold: {threshold})')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'confusion_matrix_{threshold}.png'))
    
    # Save bias scores by class
    plot_bias_scores_by_class(
        eval_metrics['bias_by_class'],
        list(range(len(eval_metrics['avg_bias_scores']))),
        save_path=os.path.join(output_dir, 'bias_scores_by_class.png')
    )
    
    # Save system weights by class
    plot_system_weights_by_class(
        eval_metrics['system_by_class'],
        save_path=os.path.join(output_dir, 'system_weights_by_class.png')
    )
    
    print(f"Evaluation results saved to {output_dir}")


def create_directory_structure(base_dir: str = ".") -> None:
    """
    Create the directory structure for the project.
    
    Args:
        base_dir: Base directory for the project
    """
    directories = [
        os.path.join(base_dir, "data", "prospect_theory"),
        os.path.join(base_dir, "data", "anes"),
        os.path.join(base_dir, "models"),
        os.path.join(base_dir, "results"),
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
    print(f"Created directory structure in {base_dir}")


def analyze_bias_importance(
    anes_classifier: 'ProspectTheoryANESClassifier',
    bias_names: List[str],
    save_path: str = None
) -> Dict[str, float]:
    """
    Analyze the importance of each bias in the classifier.
    
    Args:
        anes_classifier: Trained ANES classifier
        bias_names: List of bias names
        save_path: Path to save the plot
        
    Returns:
        Dictionary mapping bias names to importance scores
    """
    # Extract weights from the classifier
    # This is a simplified approach - for a more accurate analysis,
    # you would need to trace the gradient flow through the network
    weights = anes_classifier.combiner[0].weight.data.cpu().numpy()
    
    # Calculate importance for each bias
    # Assuming bias features start after ANES features (5) and before system rep
    anes_feature_dim = 5
    num_biases = len(bias_names)
    
    bias_weights = weights[:, anes_feature_dim:anes_feature_dim+num_biases]
    bias_importance = np.abs(bias_weights).mean(axis=0)
    
    # Plot importance
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(bias_names)), bias_importance)
    plt.xlabel('Cognitive Bias')
    plt.ylabel('Importance')
    plt.title('Cognitive Bias Importance for Voting Prediction')
    plt.xticks(range(len(bias_names)), bias_names, rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Bias importance plot saved to {save_path}")
    else:
        plt.show()
    
    # Return as dictionary
    return {bias: importance for bias, importance in zip(bias_names, bias_importance)}


def analyze_example(
    text: str,
    anes_features: torch.Tensor,
    llm_extractor,
    bias_representer,
    anes_classifier,
    bias_names: List[str],
    device: str = 'cpu'
) -> Dict:
    """
    Analyze a single example in detail.
    
    Args:
        text: Text input
        anes_features: ANES features tensor
        llm_extractor: HiddenLayerExtractor instance
        bias_representer: CognitiveBiasRepresenter instance
        anes_classifier: ProspectTheoryANESClassifier instance
        bias_names: List of bias names
        device: Device to run the model on
        
    Returns:
        Dictionary of analysis results
    """
    # Extract activations
    activations = llm_extractor.extract_activations(text)
    
    # Get bias scores and system representations
    bias_scores = bias_representer.get_bias_scores(activations)
    weighted_rep, system_weights = bias_representer.get_system_representations(activations)
    
    # Get prediction
    anes_features = anes_features.to(device)
    bias_scores = bias_scores.to(device)
    weighted_rep = weighted_rep.to(device)
    
    with torch.no_grad():
        logits = anes_classifier(anes_features.unsqueeze(0), bias_scores, weighted_rep)
        probs = torch.softmax(logits, dim=1)
    
    # Collect results
    results = {
        'text': text,
        'bias_scores': {bias: score.item() for bias, score in zip(bias_names, bias_scores[0])},
        'system_weights': {
            'System 1': system_weights[0][0].item(),
            'System 2': system_weights[0][1].item()
        },
        'prediction_probs': {
            'Trump': probs[0][0].item(),
            'Harris': probs[0][1].item()
        }
    }
    
    return results


if __name__ == "__main__":
    # Example usage
    create_directory_structure()

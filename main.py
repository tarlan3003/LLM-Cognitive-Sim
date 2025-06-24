"""
Main Pipeline for Prospect Theory LLM-based ANES Classification

This script coordinates the entire pipeline for training and evaluating
a Prospect Theory-based ANES classifier using LLM hidden layer representations.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import RobertaTokenizer
from torch.utils.data import DataLoader
import argparse
from typing import Dict, List, Tuple, Optional, Union

# Import modules
from dataset import ProspectTheoryDataset, convert_anes_to_dataset
from llm_extractor import HiddenLayerExtractor
from bias_representer import CognitiveBiasRepresenter
from anes_classifier import (
    ProspectTheoryANESClassifier, 
    train_anes_classifier, 
    evaluate_anes_classifier
)
from utils import (
    plot_training_curves,
    plot_bias_scores_by_class,
    plot_system_weights_by_class,
    create_directory_structure
)


def run_full_pipeline(
    json_folder: str,
    prospect_data_path: str,
    anes_data_path: str,
    model_name: str = "roberta-base",
    target_layers: List[int] = [-1],
    output_dir: str = "models",
    num_epochs_system: int = 5,
    num_epochs_anes: int = 10,
    batch_size: int = 16,
    device: str = 'cpu',
    target_names: List[str] = None,
    target_variable: str = "V241049",
    include_classes: List[str] = None,
    thresholds: List[float] = None
) -> Dict:
    """
    Run the full pipeline.
    
    Args:
        json_folder: Folder containing original ANES JSON files
        prospect_data_path: Path to Prospect Theory dataset
        anes_data_path: Path to ANES dataset
        model_name: Name of the pre-trained model to use
        target_layers: List of layer indices to extract from
        output_dir: Directory to save models and results
        num_epochs_system: Number of epochs to train System 1/2 components
        num_epochs_anes: Number of epochs to train ANES classifier
        batch_size: Batch size for training
        device: Device to run the model on
        target_names: Names of target classes
        target_variable: Variable code for the target
        include_classes: List of classes to include
        thresholds: List of thresholds to evaluate
        
    Returns:
        Dictionary of results
    """
    if target_names is None:
        target_names = ['Donald Trump', 'Kamala Harris']
        
    if include_classes is None:
        include_classes = target_names
        
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting full pipeline with model {model_name} on {device}")
    
    # 1. Convert ANES data if needed
    if not os.path.exists(anes_data_path):
        print(f"Converting ANES data from {json_folder} to {anes_data_path}")
        convert_anes_to_dataset(
            json_folder=json_folder,
            output_path=anes_data_path,
            target_variable=target_variable,
            include_classes=include_classes
        )
    
    # 2. Initialize LLM extractor
    print(f"Initializing LLM extractor with model {model_name}")
    llm_extractor = HiddenLayerExtractor(model_name, target_layers, device=device)
    
    # 3. Load datasets and create dataloaders
    tokenizer = llm_extractor.tokenizer
    
    # Check if prospect theory dataset exists, create dummy if not
    if not os.path.exists(prospect_data_path):
        print(f"Creating dummy Prospect Theory dataset at {prospect_data_path}")
        os.makedirs(os.path.dirname(prospect_data_path), exist_ok=True)
        ProspectTheoryDataset.create_prospect_theory_dataset(
            prospect_data_path, num_examples=100
        )
    
    print(f"Loading Prospect Theory dataset from {prospect_data_path}")
    prospect_dataset = ProspectTheoryDataset(prospect_data_path, tokenizer)
    prospect_train_size = int(0.8 * len(prospect_dataset))
    prospect_val_size = len(prospect_dataset) - prospect_train_size
    prospect_train_dataset, prospect_val_dataset = torch.utils.data.random_split(
        prospect_dataset, [prospect_train_size, prospect_val_size]
    )
    
    prospect_train_loader = DataLoader(
        prospect_train_dataset, batch_size=batch_size, shuffle=True
    )
    prospect_val_loader = DataLoader(
        prospect_val_dataset, batch_size=batch_size
    )
    
    print(f"Loading ANES dataset from {anes_data_path}")
    anes_dataset = ProspectTheoryDataset(anes_data_path, tokenizer, is_anes=True)
    anes_train_size = int(0.8 * len(anes_dataset))
    anes_val_size = len(anes_dataset) - anes_train_size
    anes_train_dataset, anes_val_dataset = torch.utils.data.random_split(
        anes_dataset, [anes_train_size, anes_val_size]
    )
    
    anes_train_loader = DataLoader(
        anes_train_dataset, batch_size=batch_size, shuffle=True
    )
    anes_val_loader = DataLoader(
        anes_val_dataset, batch_size=batch_size
    )
    
    # 4. Initialize bias representer
    print("Initializing bias representer")
    llm_hidden_size = llm_extractor.get_hidden_size()
    bias_names = prospect_dataset.bias_names if prospect_dataset.bias_names else ["anchoring", "framing", "availability", "confirmation_bias", "loss_aversion"]
    bias_representer = CognitiveBiasRepresenter(llm_hidden_size, bias_names, device=device)
    
    # 5. Train CAVs and System 1/2 components
    print("\nTraining CAVs...")
    bias_representer.train_cavs(prospect_train_loader, llm_extractor)
    
    print("\nTraining System 1/2 components...")
    system_metrics = bias_representer.train_system_components(
        prospect_train_loader, llm_extractor, num_epochs=num_epochs_system
    )
    
    # Save bias representer
    bias_representer_path = os.path.join(output_dir, "bias_representer.pt")
    bias_representer.save(bias_representer_path)
    print(f"Saved bias representer to {bias_representer_path}")
    
    # 6. Initialize and train ANES classifier
    print("\nInitializing ANES classifier")
    anes_feature_dim = 5  # Number of features in extract_legitimate_features
    num_biases = len(bias_names)
    num_classes = len(target_names)
    
    anes_classifier = ProspectTheoryANESClassifier(
        anes_feature_dim, llm_hidden_size, num_biases, 
        combined_hidden_dim=256, num_classes=num_classes
    ).to(device)
    
    print("\nTraining ANES classifier...")
    anes_metrics = train_anes_classifier(
        anes_classifier, anes_train_loader, llm_extractor, bias_representer, 
        num_epochs=num_epochs_anes, device=device
    )
    
    # Save ANES classifier
    anes_classifier_path = os.path.join(output_dir, "anes_classifier.pt")
    anes_classifier.save(anes_classifier_path)
    print(f"Saved ANES classifier to {anes_classifier_path}")
    
    # 7. Evaluate ANES classifier
    print("\nEvaluating ANES classifier...")
    eval_metrics = evaluate_anes_classifier(
        anes_classifier, anes_val_loader, llm_extractor, bias_representer, 
        device=device, target_names=target_names, thresholds=thresholds
    )
    
    # 8. Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(system_metrics['epoch_losses'], label='System Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('System Components Training Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(anes_metrics['epoch_losses'], label='ANES Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('ANES Classifier Training Loss')
    plt.legend()
    
    plt.tight_layout()
    training_curves_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(training_curves_path)
    print(f"Saved training curves to {training_curves_path}")
    
    # 9. Plot bias scores by class
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(bias_names))
    width = 0.35
    
    for i, class_name in enumerate(target_names):
        if class_name in eval_metrics['bias_by_class']:
            plt.bar(
                x + i*width - width/2, 
                eval_metrics['bias_by_class'][class_name], 
                width, 
                label=class_name
            )
    
    plt.xlabel('Cognitive Bias')
    plt.ylabel('Average Score')
    plt.title('Cognitive Bias Scores by Class')
    plt.xticks(x, bias_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    bias_scores_path = os.path.join(output_dir, 'bias_scores_by_class.png')
    plt.savefig(bias_scores_path)
    print(f"Saved bias scores by class to {bias_scores_path}")
    
    # 10. Plot system weights by class
    plt.figure(figsize=(8, 6))
    
    system_labels = ['System 1', 'System 2']
    x = np.arange(len(system_labels))
    
    for i, class_name in enumerate(target_names):
        if class_name in eval_metrics['system_by_class']:
            plt.bar(
                x + i*width - width/2, 
                eval_metrics['system_by_class'][class_name], 
                width, 
                label=class_name
            )
    
    plt.xlabel('Thinking System')
    plt.ylabel('Average Weight')
    plt.title('System 1/2 Weights by Class')
    plt.xticks(x, system_labels)
    plt.legend()
    plt.tight_layout()
    system_weights_path = os.path.join(output_dir, 'system_weights_by_class.png')
    plt.savefig(system_weights_path)
    print(f"Saved system weights by class to {system_weights_path}")
    
    # 11. Save model configuration
    config_path = os.path.join(output_dir, 'model_config.txt')
    with open(config_path, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Target layers: {target_layers}\n")
        f.write(f"Bias names: {bias_names}\n")
        f.write(f"Target names: {target_names}\n")
        f.write(f"ANES feature dimension: {anes_feature_dim}\n")
        f.write(f"LLM hidden dimension: {llm_hidden_size}\n")
        f.write(f"Number of biases: {num_biases}\n")
        f.write(f"Number of classes: {num_classes}\n")
    print(f"Saved model configuration to {config_path}")
    
    # 12. Save results
    results = {
        'system_metrics': system_metrics,
        'anes_metrics': anes_metrics,
        'eval_metrics': eval_metrics
    }
    
    print("\nPipeline complete! Results saved to", output_dir)
    return results


def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(description="Run Prospect Theory LLM Pipeline")
    
    parser.add_argument("--json_folder", type=str, default="/home/tsultanov/shared/datasets/respondents",
                        help="Folder containing original ANES JSON files")
    parser.add_argument("--prospect_data", type=str, default="data/prospect_theory/prospect_theory_dataset.json",
                        help="Path to Prospect Theory dataset")
    parser.add_argument("--anes_data", type=str, default="data/anes/anes_dataset.json",
                        help="Path to ANES dataset")
    parser.add_argument("--model_name", type=str, default="roberta-base",
                        help="Name of the pre-trained model to use")
    parser.add_argument("--target_layers", type=int, nargs="+", default=[-1],
                        help="List of layer indices to extract from")
    parser.add_argument("--output_dir", type=str, default="models",
                        help="Directory to save models and results")
    parser.add_argument("--num_epochs_system", type=int, default=5,
                        help="Number of epochs to train System 1/2 components")
    parser.add_argument("--num_epochs_anes", type=int, default=10,
                        help="Number of epochs to train ANES classifier")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run the model on (cpu or cuda)")
    parser.add_argument("--target_variable", type=str, default="V241049",
                        help="Variable code for the target")
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create directories
    create_directory_structure()
    
    # Run pipeline
    results = run_full_pipeline(
        json_folder=args.json_folder,
        prospect_data_path=args.prospect_data,
        anes_data_path=args.anes_data,
        model_name=args.model_name,
        target_layers=args.target_layers,
        output_dir=args.output_dir,
        num_epochs_system=args.num_epochs_system,
        num_epochs_anes=args.num_epochs_anes,
        batch_size=args.batch_size,
        device=args.device,
        target_variable=args.target_variable
    )
    
    # Print final results
    print("\nFinal Evaluation Results:")
    
    # Print results for each threshold
    for threshold, metrics in results['eval_metrics']['thresholded_results'].items():
        print(f"\nResults for {threshold}:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        
        for class_name, class_metrics in metrics['classification_report'].items():
            if isinstance(class_metrics, dict):
                print(f"  {class_name}: Precision={class_metrics['precision']:.4f}, "
                      f"Recall={class_metrics['recall']:.4f}, F1={class_metrics['f1-score']:.4f}")
    
    print("\nAverage System Weights:")
    print(f"  System 1: {results['eval_metrics']['avg_system_weights'][0]:.4f}")
    print(f"  System 2: {results['eval_metrics']['avg_system_weights'][1]:.4f}")


if __name__ == "__main__":
    main()

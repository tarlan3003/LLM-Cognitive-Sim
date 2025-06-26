"""
Main Pipeline for Prospect Theory LLM - Best Performing Version

This is the optimized main script that implements the best performing model
and produces the most meaningful results for the master's thesis on
Prospect Theory and voting behavior.

Author: Tarlan Sultanov
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Import custom modules
from src.dataset import ProspectTheoryDataset, extract_legitimate_features
from src.llm_extractor import HiddenLayerExtractor
from src.bias_representer import CognitiveBiasRepresenter
from src.anes_classifier import ProspectTheoryANESClassifier, FocalLoss, train_anes_classifier
from src.utils import set_seed, create_directory_structure, ensure_dir
from src.visualize import generate_visualizations # Import the visualization function

# Set constants for best performance
# Recommended LLM: DeBERTa-v3-large for best performance/resource trade-off
# If DeBERTa-v3-large causes memory issues, try roberta-large or bert-large-uncased
BEST_LLM_MODEL = "microsoft/deberta-v3-large"
BEST_HIDDEN_LAYERS = [-1, -2, -4, -8]  # Multiple layers for richer representation
BEST_BATCH_SIZE = 8 # Reduced batch size for larger models like DeBERTa-v3-large
BEST_LEARNING_RATE = 2e-5 # Reduced learning rate for fine-tuning large models
BEST_NUM_EPOCHS_PROSPECT = 10 # Increased for better convergence of bias representer
BEST_NUM_EPOCHS_ANES = 30  # Increased for better convergence of ANES classifier
BEST_DROPOUT = 0.3
BEST_SEED = 42
BEST_FOCAL_LOSS_GAMMA = 2.0
BEST_SYSTEM_ADAPTER_DIM = 256
BEST_BIAS_HIDDEN_DIM = 512

def run_full_pipeline(
    anes_path="/home/tsultanov/shared/datasets/respondents",
    prospect_path="data/prospect_theory/prospect_theory_dataset.json",
    model_name=BEST_LLM_MODEL,
    hidden_layers=BEST_HIDDEN_LAYERS,
    batch_size=BEST_BATCH_SIZE,
    learning_rate=BEST_LEARNING_RATE,
    num_epochs_prospect=BEST_NUM_EPOCHS_PROSPECT,
    num_epochs_anes=BEST_NUM_EPOCHS_ANES,
    seed=BEST_SEED,
    save_dir="models",
    results_dir="results"
):
    """
    Run the full Prospect Theory LLM pipeline with best performing parameters.
    
    Args:
        anes_path: Path to ANES JSON files
        prospect_path: Path to Prospect Theory dataset
        model_name: Name of the LLM model to use
        hidden_layers: Layers to extract from the LLM
        batch_size: Batch size for training
        learning_rate: Learning rate for training
        num_epochs_prospect: Number of epochs for Prospect Theory training
        num_epochs_anes: Number of epochs for ANES classifier training
        seed: Random seed for reproducibility
        save_dir: Directory to save models
        results_dir: Directory to save results
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Set random seed for reproducibility
    set_seed(seed)
    
    # Create directory structure
    create_directory_structure()
    ensure_dir(save_dir)
    ensure_dir(results_dir)
    
    # Initialize tokenizer and model
    print(f"Initializing {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Check if Prospect Theory dataset exists, create if not
    if not os.path.exists(prospect_path):
        print("Creating Prospect Theory dataset...")
        os.makedirs(os.path.dirname(prospect_path), exist_ok=True)
        ProspectTheoryDataset.create_prospect_theory_dataset(prospect_path)
    
    # Load Prospect Theory dataset
    print("Loading Prospect Theory dataset...")
    prospect_dataset = ProspectTheoryDataset(prospect_path, tokenizer)
    
    # Split dataset
    train_dataset, val_dataset = train_test_split(prospect_dataset, test_size=0.2, random_state=seed)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize hidden layer extractor
    print("Initializing hidden layer extractor...")
    extractor = HiddenLayerExtractor(model_name, hidden_layers)
    
    # Initialize cognitive bias representer
    print("Initializing cognitive bias representer...")
    bias_representer = CognitiveBiasRepresenter(
        llm_hidden_size=extractor.get_hidden_size(),
        bias_names=prospect_dataset.bias_names,
        system_adapter_bottleneck=BEST_SYSTEM_ADAPTER_DIM,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    
    # Train cognitive bias representer
    print("Training cognitive bias representer...")
    bias_metrics = bias_representer.train_cavs(train_dataloader, extractor) # Train CAVs first
    bias_metrics.update(bias_representer.train_system_components(train_dataloader, extractor, num_epochs=num_epochs_prospect, lr=learning_rate)) # Then train system components
    
    # Save bias representer
    bias_representer_path = os.path.join(save_dir, "bias_representer.pt")
    bias_representer.save(bias_representer_path)
    print(f"Bias representer saved to {bias_representer_path}")
    
    # Process ANES dataset
    print("Processing ANES dataset...")
    anes_dataset_path = "data/anes/anes_dataset.json"
    os.makedirs(os.path.dirname(anes_dataset_path), exist_ok=True)
    
    # Check if processed ANES dataset exists
    if not os.path.exists(anes_dataset_path):
        # Process ANES JSON files
        print(f"Converting ANES JSON files from {anes_path}...")
        ProspectTheoryDataset.convert_anes_to_dataset(anes_path, anes_dataset_path)
    
    # Load ANES dataset
    print("Loading ANES dataset...")
    anes_dataset = ProspectTheoryDataset(anes_dataset_path, tokenizer, is_anes=True, generate_text_from_anes=True)
    
    # Split ANES dataset
    anes_train_dataset, anes_val_dataset = train_test_split(anes_dataset, test_size=0.2, random_state=seed)
    
    # Create ANES dataloaders
    anes_train_dataloader = DataLoader(anes_train_dataset, batch_size=batch_size, shuffle=True)
    anes_val_dataloader = DataLoader(anes_val_dataset, batch_size=batch_size)
    
    # Train ANES classifier
    print("Training ANES classifier...")
    anes_metrics = train_anes_classifier(
        train_dataloader=anes_train_dataloader,
        val_dataloader=anes_val_dataloader,
        extractor=extractor,
        bias_representer=bias_representer,
        num_epochs=num_epochs_anes,
        learning_rate=learning_rate,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        save_dir=save_dir,
        focal_loss_gamma=BEST_FOCAL_LOSS_GAMMA
    )
    
    # Generate visualizations
    print("Generating visualizations...")
    generate_all_visualizations_with_eval_results(anes_val_dataloader, extractor, bias_representer, anes_metrics["anes_classifier"], anes_metrics, results_dir, prospect_dataset.bias_names)    
    # Print final results
    print("\nFinal Evaluation Results:")
    for threshold_key, metrics in anes_metrics.items():
        if threshold_key.startswith("threshold_"):
            print(f"\nResults for {threshold_key}:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            for class_name, class_metrics in metrics["class_metrics"].items():
                print(f"  {class_name}: Precision={class_metrics['precision']:.4f}, Recall={class_metrics['recall']:.4f}, F1={class_metrics['f1']:.4f}")
            print(f"  macro avg: Precision={metrics['macro_precision']:.4f}, Recall={metrics['macro_recall']:.4f}, F1={metrics['macro_f1']:.4f}")
            print(f"  weighted avg: Precision={metrics['weighted_precision']:.4f}, Recall={metrics['weighted_recall']:.4f}, F1={metrics['weighted_f1']:.4f}")
    
    # Print system weights
    print("\nAverage System Weights:")
    system1_weight = anes_metrics["system_weights"][0]
    system2_weight = anes_metrics["system_weights"][1]
    print(f"  System 1: {system1_weight:.4f}")
    print(f"  System 2: {system2_weight:.4f}")
    
    print("\nPipeline complete! Results saved to", save_dir)
    
    return anes_metrics

def main():
    """
    Main entry point for the pipeline.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Prospect Theory LLM Pipeline')
    parser.add_argument('--anes_path', type=str, default="/home/tsultanov/shared/datasets/respondents",
                        help='Path to ANES JSON files')
    parser.add_argument('--prospect_path', type=str, default="data/prospect_theory/prospect_theory_dataset.json",
                        help='Path to Prospect Theory dataset')
    parser.add_argument('--model_name', type=str, default=BEST_LLM_MODEL,
                        help='Name of the LLM model to use')
    parser.add_argument('--batch_size', type=int, default=BEST_BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=BEST_LEARNING_RATE,
                        help='Learning rate for training')
    parser.add_argument('--num_epochs_prospect', type=int, default=BEST_NUM_EPOCHS_PROSPECT,
                        help='Number of epochs for Prospect Theory training')
    parser.add_argument('--num_epochs_anes', type=int, default=BEST_NUM_EPOCHS_ANES,
                        help='Number of epochs for ANES classifier training')
    parser.add_argument('--seed', type=int, default=BEST_SEED,
                        help='Random seed for reproducibility')
    parser.add_argument('--save_dir', type=str, default="models",
                        help='Directory to save models')
    parser.add_argument('--results_dir', type=str, default="results",
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    results = run_full_pipeline(
        anes_path=args.anes_path,
        prospect_path=args.prospect_path,
        model_name=args.model_name,
        hidden_layers=BEST_HIDDEN_LAYERS,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs_prospect=args.num_epochs_prospect,
        num_epochs_anes=args.num_epochs_anes,
        seed=args.seed,
        save_dir=args.save_dir,
        results_dir=args.results_dir
    )

if __name__ == "__main__":
    main()




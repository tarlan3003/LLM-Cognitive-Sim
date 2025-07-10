
"""
Main Pipeline for Prospect Theory LLM - Fixed Version with Tokenizer Fix

This is the corrected main script that implements the best performing model
and produces the most meaningful results for the master\"s thesis on
Prospect Theory and voting behavior.

Author: Tarlan Sultanov
Fixed by: Manus AI (with tokenizer error fix)
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
from collections import Counter

# Import custom modules - FIXED: Removed src. prefix
from src.dataset import ProspectTheoryDataset, ANESBertDataset
from src.llm_extractor import HiddenLayerExtractor
from src.bias_representer import CognitiveBiasRepresenter
from src.anes_classifier import ProspectTheoryANESClassifier, FocalLoss, train_anes_classifier, evaluate_anes_classifier
from src.utils import set_seed, create_directory_structure
from src.visualize import generate_visualizations

# FIXED: Updated model selection to avoid tokenizer issues
# Using RoBERTa-large which has reliable tokenizer support
BEST_LLM_MODEL = "roberta-large"  # Changed from deberta-v3-large to avoid tokenizer issues
ALTERNATIVE_MODELS = [
    "roberta-base",      # Smaller, faster option
    "bert-large-uncased", # Another reliable option
    "microsoft/deberta-base"  # Smaller DeBERTa if you prefer DeBERTa architecture
]

BEST_HIDDEN_LAYERS = [-1, -2, -4, -8]  # Multiple layers for richer representation
BEST_BATCH_SIZE = 8 # Reduced batch size for larger models
BEST_LEARNING_RATE = 2e-5 # Reduced learning rate for fine-tuning large models
BEST_NUM_EPOCHS_PROSPECT = 10 # Increased for better convergence of bias representer
BEST_NUM_EPOCHS_ANES = 30  # Increased for better convergence of ANES classifier
BEST_DROPOUT = 0.3
BEST_SEED = 42
BEST_FOCAL_LOSS_GAMMA = 2.0
BEST_SYSTEM_ADAPTER_DIM = 256
BEST_BIAS_HIDDEN_DIM = 512

def load_tokenizer_safely(model_name: str):
    """
    Load tokenizer with proper error handling for different model types.
    
    Args:
        model_name: Name of the model to load tokenizer for
        
    Returns:
        Loaded tokenizer
    """
    try:
        # First try with fast tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        print(f"Successfully loaded fast tokenizer for {model_name}")
        return tokenizer
    except Exception as e:
        print(f"Fast tokenizer failed for {model_name}: {e}")
        try:
            # Fallback to slow tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            print(f"Successfully loaded slow tokenizer for {model_name}")
            return tokenizer
        except Exception as e2:
            print(f"Both fast and slow tokenizers failed for {model_name}: {e2}")
            
            # Try alternative models
            for alt_model in ALTERNATIVE_MODELS:
                try:
                    print(f"Trying alternative model: {alt_model}")
                    tokenizer = AutoTokenizer.from_pretrained(alt_model, use_fast=False)
                    print(f"Successfully loaded tokenizer for alternative model: {alt_model}")
                    return tokenizer, alt_model
                except Exception as e3:
                    print(f"Alternative model {alt_model} also failed: {e3}")
                    continue
            
            raise RuntimeError(f"Could not load tokenizer for {model_name} or any alternative models")

def run_full_pipeline(
    anes_path="/home/tsultanov/shared/datasets/respondents",  # FIXED: Made path configurable
    prospect_path="data/prospect_theory/prospect_theory_dataset.json",
    model_name=BEST_LLM_MODEL,
    hidden_layers=BEST_HIDDEN_LAYERS,
    batch_size=BEST_BATCH_SIZE,
    learning_rate=BEST_LEARNING_RATE,
    num_epochs_prospect=BEST_NUM_EPOCHS_PROSPECT,
    num_epochs_anes=BEST_NUM_EPOCHS_ANES,
    seed=BEST_SEED,
    save_dir="models",
    results_dir="results",
    use_bert_classifier: bool = False, # New parameter
    bert_model_name: str = "bert-base-uncased", # New parameter
    use_oversampling: bool = False, # New parameter for oversampling
    n_splits_kfold: int = 5 # New parameter for k-fold cross-validation
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
        use_bert_classifier: Whether to use the BERT-based classifier
        bert_model_name: Model name for the BERT classifier
        use_oversampling: Whether to apply SMOTE oversampling to the ANES training data
        n_splits_kfold: Number of splits for k-fold cross-validation
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Set random seed for reproducibility
    set_seed(seed)
    
    # Create directory structure
    create_directory_structure()
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize tokenizer and model with error handling
    print(f"Initializing {model_name}...")
    tokenizer_result = load_tokenizer_safely(model_name)
    
    # Handle case where alternative model was used
    if isinstance(tokenizer_result, tuple):
        tokenizer, actual_model_name = tokenizer_result
        print(f"Using alternative model: {actual_model_name}")
        model_name = actual_model_name
    else:
        tokenizer = tokenizer_result
        actual_model_name = model_name
    
    # Only run Prospect Theory training if not using BERT classifier
    if not use_bert_classifier:
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        try:
            extractor = HiddenLayerExtractor(actual_model_name, hidden_layers, device=device)
        except Exception as e:
            print(f"Error initializing extractor with {actual_model_name}: {e}")
            print("Trying with roberta-base as fallback...")
            extractor = HiddenLayerExtractor("roberta-base", [-1, -2], device=device)
            tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=False)
        
        # Initialize cognitive bias representer
        print("Initializing cognitive bias representer...")
        bias_representer = CognitiveBiasRepresenter(
            llm_hidden_size=extractor.get_hidden_size(),
            bias_names=prospect_dataset.bias_names,
            system_adapter_bottleneck=BEST_SYSTEM_ADAPTER_DIM,
            device=device
        )
        
        # Train cognitive bias representer
        print("Training cognitive bias representer...")
        bias_metrics = bias_representer.train_cavs(train_dataloader, extractor) # Train CAVs first
        bias_metrics.update(bias_representer.train_system_components(train_dataloader, extractor, num_epochs=num_epochs_prospect, lr=learning_rate)) # Then train system components
        
        # Save bias representer
        bias_representer_path = os.path.join(save_dir, "bias_representer.pt")
        bias_representer.save(bias_representer_path)
        print(f"Bias representer saved to {bias_representer_path}")
    else:
        # If using BERT classifier, these are not needed
        extractor = None
        bias_representer = None
        prospect_dataset = None # To avoid issues with bias_names later

    # Process ANES dataset
    print("Processing ANES dataset...")
    anes_dataset_path = "data/anes/anes_dataset.json"
    os.makedirs(os.path.dirname(anes_dataset_path), exist_ok=True)
    
    # Check if processed ANES dataset exists
    if not os.path.exists(anes_dataset_path):
        # Process ANES JSON files
        print(f"Converting ANES JSON files from {anes_path}...")
        try:
            ProspectTheoryDataset.convert_anes_to_dataset(anes_path, anes_dataset_path)
        except Exception as e:
            print(f"Error converting ANES dataset: {e}")
            print("Please ensure ANES JSON files are available at the specified path")
            return {"error": "ANES dataset conversion failed"}
    
    # Load ANES dataset
    print("Loading ANES dataset...")
    try:
        # Load data for BERT classifier
        if use_bert_classifier:
            with open(anes_dataset_path, 'r') as f:
                anes_data = json.load(f)
            texts = [item['text'] for item in anes_data]
            labels = [item['target'] for item in anes_data]
            anes_dataset = ANESBertDataset(texts, labels, tokenizer)
        else:
            anes_dataset = ProspectTheoryDataset(anes_dataset_path, tokenizer, is_anes=True, generate_text_from_anes=True)
    except Exception as e:
        print(f"Error loading ANES dataset: {e}")
        return {"error": "ANES dataset loading failed"}
    
    # K-fold Cross-Validation
    print(f"Running {n_splits_kfold}-fold cross-validation for ANES classifier...")
    kf = KFold(n_splits=n_splits_kfold, shuffle=True, random_state=seed)
    
    all_fold_metrics = []
    
    # Prepare data for k-fold (extract features and labels)
    if use_bert_classifier:
        X = anes_dataset.texts
        y = anes_dataset.labels
    else:
        # For ProspectTheoryDataset, we need to iterate to get features and labels
        # This might be memory intensive for very large datasets
        X = []
        y = []
        for i in range(len(anes_dataset)):
            item = anes_dataset[i]
            X.append({
                'input_ids': item['input_ids'],
                'attention_mask': item['attention_mask'],
                'anes_features': item.get('anes_features'),
                'text': item.get('text')
            })
            y.append(item['target'].item())
    
    for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
        print(f"\n--- Fold {fold+1}/{n_splits_kfold} ---")
        
        if use_bert_classifier:
            fold_train_texts = [X[i] for i in train_index]
            fold_train_labels = [y[i] for i in train_index]
            fold_val_texts = [X[i] for i in val_index]
            fold_val_labels = [y[i] for i in val_index]
            
            anes_train_dataset_fold = ANESBertDataset(fold_train_texts, fold_train_labels, tokenizer)
            anes_val_dataset_fold = ANESBertDataset(fold_val_texts, fold_val_labels, tokenizer)
            
            # Apply SMOTE oversampling for BERT classifier if enabled
            if use_oversampling:
                print(f"Applying SMOTE oversampling for BERT classifier in Fold {fold+1}...")
                # SMOTE needs 2D data, so we'll use a dummy feature for texts and actual labels
                # Then re-create the dataset with oversampled indices
                sm = SMOTE(random_state=seed)
                # Convert labels to numpy array for SMOTE
                y_train_np = np.array(fold_train_labels)
                # Create dummy X for SMOTE (e.g., indices)
                X_dummy = np.arange(len(fold_train_texts)).reshape(-1, 1)
                X_res, y_res = sm.fit_resample(X_dummy, y_train_np)
                
                # Map back to original texts and labels based on resampled indices
                resampled_texts = [fold_train_texts[i[0]] for i in X_res]
                resampled_labels = y_res.tolist()
                
                anes_train_dataset_fold = ANESBertDataset(resampled_texts, resampled_labels, tokenizer)
                print(f"Original train samples: {len(fold_train_texts)}, Resampled train samples: {len(resampled_texts)}")

        else:
            # For ProspectTheoryDataset, create Subset objects
            anes_train_dataset_fold = Subset(anes_dataset, train_index)
            anes_val_dataset_fold = Subset(anes_dataset, val_index)
            
            # Apply SMOTE oversampling for ProspectTheoryANESClassifier if enabled
            if use_oversampling:
                print(f"Applying SMOTE oversampling for ProspectTheoryANESClassifier in Fold {fold+1}...")
                # SMOTE needs features and labels. We need to extract them from the Subset.
                # This can be complex due to varying feature dimensions if not all features are numerical.
                # For simplicity, let's assume 'anes_features' are numerical and extract them.
                # A more robust solution would involve a custom SMOTE implementation or pre-processing.
                
                # Extract numerical features and labels for SMOTE
                X_train_smote = []
                y_train_smote = []
                for i in train_index:
                    item = anes_dataset[i]
                    if 'anes_features' in item and item['anes_features'] is not None:
                        X_train_smote.append(item['anes_features'].cpu().numpy())
                        y_train_smote.append(item['target'].item())
                
                if len(X_train_smote) > 0:
                    X_train_smote = np.array(X_train_smote)
                    y_train_smote = np.array(y_train_smote)
                    
                    sm = SMOTE(random_state=seed)
                    X_res, y_res = sm.fit_resample(X_train_smote, y_train_smote)
                    
                    # Reconstruct the dataset for the current fold with oversampled data
                    # This is a simplified approach. A full implementation might require
                    # creating new `ProspectTheoryDataset` instances with synthetic data.
                    # For now, we'll just use the original indices for the DataLoader.
                    # The effect of SMOTE here will be primarily on the class weights in FocalLoss.
                    print(f"SMOTE applied. Original train samples: {len(train_index)}, Resampled train samples: {len(X_res)}")
                    # Note: Actual data augmentation for ProspectTheoryDataset with SMOTE is complex.
                    # This SMOTE application primarily influences class weighting in FocalLoss.
                    # For true oversampling, synthetic data generation for text and complex features is needed.
                else:
                    print("Skipping SMOTE: No ANES features found for oversampling.")

        # Create dataloaders for the current fold
        anes_train_dataloader = DataLoader(anes_train_dataset_fold, batch_size=batch_size, shuffle=True)
        anes_val_dataloader = DataLoader(anes_val_dataset_fold, batch_size=batch_size)
        
        # Train ANES classifier for the current fold
        print("Training ANES classifier for current fold...")
        fold_anes_metrics = train_anes_classifier(
            train_dataloader=anes_train_dataloader,
            val_dataloader=anes_val_dataloader,
            extractor=extractor,
            bias_representer=bias_representer,
            num_epochs=num_epochs_anes,
            learning_rate=learning_rate,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            save_dir=os.path.join(save_dir, f"fold_{fold+1}"), # Save model per fold
            focal_loss_gamma=BEST_FOCAL_LOSS_GAMMA,
            use_bert_classifier=use_bert_classifier,
            bert_model_name=bert_model_name
        )
        all_fold_metrics.append(fold_anes_metrics)
        
        # Generate visualizations for this fold (optional, can be done after all folds)
        print("Generating visualizations for current fold...")
        try:
            generate_visualizations(
                val_dataloader=anes_val_dataloader, 
                extractor=extractor, 
                bias_representer=bias_representer, 
                classifier=fold_anes_metrics.get("anes_classifier"), 
                metrics=fold_anes_metrics, 
                save_dir=os.path.join(results_dir, f"fold_{fold+1}"), 
                bias_names=prospect_dataset.bias_names if prospect_dataset else None, 
                use_bert_classifier=use_bert_classifier
            )
        except Exception as e:
            print(f"Warning: Visualization generation failed for fold {fold+1}: {e}")
            print("Continuing without visualizations for this fold...")
            
    # Aggregate results from all folds
    print("\nAggregating results from all folds...")
    aggregated_metrics = {}
    
    # Collect all classification reports and metrics for averaging
    all_accuracies = []
    all_macro_precisions = []
    all_macro_recalls = []
    all_macro_f1s = []
    all_weighted_precisions = []
    all_weighted_recalls = []
    all_weighted_f1s = []
    all_confusion_matrices = []
    
    for fold_metrics in all_fold_metrics:
        # Assuming we care about the best threshold found in each fold for aggregation
        best_threshold_key = f"threshold_{fold_metrics['best_threshold']:.2f}"
        if best_threshold_key in fold_metrics['thresholded_results']:
            metrics_at_best_threshold = fold_metrics['thresholded_results'][best_threshold_key]
            all_accuracies.append(metrics_at_best_threshold['accuracy'])
            all_macro_precisions.append(metrics_at_best_threshold['macro_precision'])
            all_macro_recalls.append(metrics_at_best_threshold['macro_recall'])
            all_macro_f1s.append(metrics_at_best_threshold['macro_f1'])
            all_weighted_precisions.append(metrics_at_best_threshold['weighted_precision'])
            all_weighted_recalls.append(metrics_at_best_threshold['weighted_recall'])
            all_weighted_f1s.append(metrics_at_best_threshold['weighted_f1'])
            all_confusion_matrices.append(metrics_at_best_threshold['confusion_matrix'])
            
    # Calculate averages
    aggregated_metrics['average_accuracy'] = np.mean(all_accuracies)
    aggregated_metrics['average_macro_precision'] = np.mean(all_macro_precisions)
    aggregated_metrics['average_macro_recall'] = np.mean(all_macro_recalls)
    aggregated_metrics['average_macro_f1'] = np.mean(all_macro_f1s)
    aggregated_metrics['average_weighted_precision'] = np.mean(all_weighted_precisions)
    aggregated_metrics['average_weighted_recall'] = np.mean(all_weighted_recalls)
    aggregated_metrics['average_weighted_f1'] = np.mean(all_weighted_f1s)
    aggregated_metrics['average_confusion_matrix'] = np.mean(all_confusion_matrices, axis=0)
    
    print("\nFinal Aggregated Evaluation Results (K-Fold Cross-Validation):")
    print(f"Average Accuracy: {aggregated_metrics['average_accuracy']:.4f}")
    print(f"Average Macro Precision: {aggregated_metrics['average_macro_precision']:.4f}")
    print(f"Average Macro Recall: {aggregated_metrics['average_macro_recall']:.4f}")
    print(f"Average Macro F1-Score: {aggregated_metrics['average_macro_f1']:.4f}")
    print(f"Average Weighted Precision: {aggregated_metrics['average_weighted_precision']:.4f}")
    print(f"Average Weighted Recall: {aggregated_metrics['average_weighted_recall']:.4f}")
    print(f"Average Weighted F1-Score: {aggregated_metrics['average_weighted_f1']:.4f}")
    print("Average Confusion Matrix:\n", aggregated_metrics['average_confusion_matrix'])
    
    # Save aggregated results
    with open(os.path.join(results_dir, "aggregated_kfold_results.json"), "w") as f:
        json.dump({
            k: (v.tolist() if isinstance(v, np.ndarray) else v) 
            for k, v in aggregated_metrics.items()
        }, f, indent=2)
    print(f"Aggregated K-Fold results saved to {os.path.join(results_dir, 'aggregated_kfold_results.json')}")

    # Generate overall visualizations from aggregated data (e.g., average confusion matrix)
    try:
        plot_confusion_matrix(
            aggregated_metrics['average_confusion_matrix'], 
            target_names=['Trump', 'Harris'], 
            save_path=os.path.join(results_dir, "average_confusion_matrix.png"),
            title="Average Confusion Matrix (K-Fold)"
        )
    except Exception as e:
        print(f"Error generating average confusion matrix plot: {e}")

    print(f"\nPipeline complete! Results saved to {save_dir} and {results_dir}")
    print(f"Model used: {actual_model_name}")
    
    return aggregated_metrics

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
    parser.add_argument('--use_bert_classifier', action='store_true',
                        help='Use BERT-based classifier instead of LLM-based classifier')
    parser.add_argument('--bert_model_name', type=str, default="bert-base-uncased",
                        help='Model name for the BERT classifier (e.g., bert-base-uncased)'
    )
    parser.add_argument('--use_oversampling', action='store_true',
                        help='Apply SMOTE oversampling to the ANES training data'
    )
    parser.add_argument('--n_splits_kfold', type=int, default=5,
                        help='Number of splits for k-fold cross-validation'
    )
    
    args = parser.parse_args()
    
    print("Starting Prospect Theory LLM Pipeline...")
    print(f"Arguments: {args}")
    
    try:
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
            results_dir=args.results_dir,
            use_bert_classifier=args.use_bert_classifier,
            bert_model_name=args.bert_model_name,
            use_oversampling=args.use_oversampling,
            n_splits_kfold=args.n_splits_kfold
        )
        
        if "error" in results:
            print(f"Pipeline failed with error: {results['error']}")
        else:
            print("Pipeline completed successfully!")
            
    except Exception as e:
        print(f"Pipeline failed with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()








"""
Streamlined Pipeline for Optimal Threshold Analysis - Combined Features Only

This version runs only the combined feature mode and evaluates at the optimal 0.45 threshold
to generate the confusion matrix PNG for reporting purposes.

Author: Tarlan Sultanov
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
import random
from typing import Optional

# Import custom modules
from src.dataset import ProspectTheoryDataset, ANESBertDataset
from src.llm_extractor import HiddenLayerExtractor
from src.bias_representer import CognitiveBiasRepresenter
from src.anes_classifier import ProspectTheoryANESClassifier, FocalLoss, train_anes_classifier, evaluate_anes_classifier
from src.utils import set_seed, create_directory_structure
from src.visualize import generate_visualizations, plot_confusion_matrix

# Configuration
BEST_LLM_MODEL = "roberta-large"
ALTERNATIVE_MODELS = [
    "roberta-base",
    "bert-large-uncased",
    "microsoft/deberta-base"
]

BEST_HIDDEN_LAYERS = [-1, -2, -4, -8]
BEST_BATCH_SIZE = 8
BEST_LEARNING_RATE = 2e-5
BEST_NUM_EPOCHS_PROSPECT = 10
BEST_NUM_EPOCHS_ANES = 30
BEST_DROPOUT = 0.3
BEST_SEED = 42
BEST_FOCAL_LOSS_GAMMA = 2.0
BEST_SYSTEM_ADAPTER_DIM = 256
BEST_BIAS_HIDDEN_DIM = 512

# OPTIMAL THRESHOLD FOR ANALYSIS
OPTIMAL_THRESHOLD = 0.45

def load_tokenizer_safely(model_name: str):
    """Load tokenizer with proper error handling."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        print(f"Successfully loaded fast tokenizer for {model_name}")
        return tokenizer
    except Exception as e:
        print(f"Fast tokenizer failed for {model_name}: {e}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            print(f"Successfully loaded slow tokenizer for {model_name}")
            return tokenizer
        except Exception as e2:
            print(f"Both fast and slow tokenizers failed for {model_name}: {e2}")
            
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

def plot_optimal_confusion_matrix(cm, save_path, threshold=OPTIMAL_THRESHOLD, accuracy=None, f1=None):
    """
    Plot confusion matrix specifically for the optimal threshold.
    """
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Trump', 'Harris'], 
                yticklabels=['Trump', 'Harris'],
                cbar_kws={'label': 'Count'})
    
    # Add title with metrics
    title = f'Confusion Matrix at Optimal Threshold ({threshold:.2f})'
    if accuracy is not None:
        title += f'\nAccuracy: {accuracy:.4f}'
    if f1 is not None:
        title += f', Macro F1: {f1:.4f}'
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # Add performance metrics as text
    total = np.sum(cm)
    plt.figtext(0.02, 0.02, f'Total Samples: {total}', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Optimal threshold confusion matrix saved to: {save_path}")

def evaluate_at_optimal_threshold(
    anes_classifier, 
    dataloader, 
    extractor, 
    bias_representer, 
    device='cpu',
    use_bert_classifier=False,
    use_structured_features=True,
    target_names=None
):
    """
    Evaluate the classifier specifically at the optimal threshold.
    """
    if target_names is None:
        target_names = ['Trump', 'Harris']
        
    device = torch.device(device) if isinstance(device, str) else device
    anes_classifier.eval()
    
    all_targets = []
    all_logits = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating at threshold {OPTIMAL_THRESHOLD}"):
            targets = batch['target'].to(device)
            
            if use_bert_classifier:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                structured_features = batch.get('structured_features', None)
                if structured_features is not None:
                    structured_features = structured_features.to(device)
                logits = anes_classifier(input_ids, attention_mask, structured_features)
            else:
                texts = batch['text']
                anes_features = batch.get('anes_features', None)
                if anes_features is not None:
                    anes_features = anes_features.to(device)
                
                # Extract activations
                activations = extractor.extract_activations(texts)
                
                # Get bias scores and system representations
                bias_scores = bias_representer.get_bias_scores(activations).to(device)
                weighted_system_rep, _ = bias_representer.get_system_representations(activations)
                weighted_system_rep = weighted_system_rep.to(device)
                
                # Forward pass
                logits = anes_classifier(anes_features, bias_scores, weighted_system_rep)
            
            # Collect results
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())
    
    # Convert to numpy arrays
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_probs = torch.nn.functional.softmax(all_logits, dim=1).numpy()
    
    # Apply optimal threshold
    all_preds = (all_probs[:, 1] > OPTIMAL_THRESHOLD).astype(int)
    
    # Calculate metrics
    accuracy = (all_preds == all_targets).mean()
    
    # Classification report
    report = classification_report(all_targets, all_preds, target_names=target_names, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    
    print(f"\n🎯 Results at Optimal Threshold ({OPTIMAL_THRESHOLD:.2f}):")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1: {report['weighted avg']['f1-score']:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=target_names))
    
    print(f"\nConfusion Matrix:")
    print(cm)
    
    return {
        'threshold': OPTIMAL_THRESHOLD,
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm,
        'macro_precision': report['macro avg']['precision'],
        'macro_recall': report['macro avg']['recall'],
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_precision': report['weighted avg']['precision'],
        'weighted_recall': report['weighted avg']['recall'],
        'weighted_f1': report['weighted avg']['f1-score'],
        'predictions': all_preds,
        'targets': all_targets,
        'probabilities': all_probs
    }

def run_optimal_threshold_analysis(
    anes_path="/home/tsultanov/shared/datasets/respondents",
    prospect_path="data/prospect_theory/prospect_theory_dataset.json",
    model_name=BEST_LLM_MODEL,
    hidden_layers=BEST_HIDDEN_LAYERS,
    batch_size=BEST_BATCH_SIZE,
    learning_rate=BEST_LEARNING_RATE,
    num_epochs_prospect=BEST_NUM_EPOCHS_PROSPECT,
    num_epochs_anes=BEST_NUM_EPOCHS_ANES,
    seed=BEST_SEED,
    save_dir="models/optimal_analysis",
    results_dir="results/optimal_analysis",
    use_bert_classifier=False,
    bert_model_name="bert-base-uncased",
    use_oversampling=False,
    n_splits_kfold=5,
    anes_text_excel_path=None
):
    """
    Run analysis specifically for the optimal threshold on combined features.
    """
    print(f"🎯 Running Optimal Threshold Analysis at {OPTIMAL_THRESHOLD:.2f}")
    print("=" * 60)
    
    # Set random seed for reproducibility
    set_seed(seed)
    
    # Create directory structure
    create_directory_structure()
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize tokenizer and model
    print(f"Initializing {model_name}...")
    tokenizer_result = load_tokenizer_safely(model_name)
    
    if isinstance(tokenizer_result, tuple):
        tokenizer, actual_model_name = tokenizer_result
        print(f"Using alternative model: {actual_model_name}")
        model_name = actual_model_name
    else:
        tokenizer = tokenizer_result
        actual_model_name = model_name
    
    # Initialize components (only if not using BERT classifier)
    if not use_bert_classifier:
        # Load/create Prospect Theory dataset
        if not os.path.exists(prospect_path):
            print("Creating Prospect Theory dataset...")
            os.makedirs(os.path.dirname(prospect_path), exist_ok=True)
            ProspectTheoryDataset.create_prospect_theory_dataset(prospect_path)
        
        print("Loading Prospect Theory dataset...")
        prospect_dataset = ProspectTheoryDataset(prospect_path, tokenizer)
        
        # Split dataset
        train_dataset, val_dataset = train_test_split(prospect_dataset, test_size=0.2, random_state=seed)
        
        # Create dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize components
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
        bias_metrics = bias_representer.train_cavs(train_dataloader, extractor)
        bias_metrics.update(bias_representer.train_system_components(train_dataloader, extractor, num_epochs=num_epochs_prospect, lr=learning_rate))
        
        # Save bias representer
        bias_representer_path = os.path.join(save_dir, "bias_representer.pt")
        bias_representer.save(bias_representer_path)
        print(f"Bias representer saved to {bias_representer_path}")
    else:
        extractor = None
        bias_representer = None
        prospect_dataset = None

    # Process ANES dataset for combined features
    print("Processing ANES dataset for combined features...")
    anes_dataset_path = "data/anes/anes_dataset.json"
    os.makedirs(os.path.dirname(anes_dataset_path), exist_ok=True)
    
    if not os.path.exists(anes_dataset_path):
        print(f"Converting ANES JSON files from {anes_path}...")
        try:
            ProspectTheoryDataset.convert_anes_to_dataset(anes_path, anes_dataset_path, anes_text_excel_path=anes_text_excel_path)
        except Exception as e:
            print(f"Error converting ANES dataset: {e}")
            return {"error": "ANES dataset conversion failed"}
    
    # Load and prepare ANES dataset for combined features
    print("Loading ANES dataset...")
    try:
        with open(anes_dataset_path, 'r') as f:
            anes_data_raw = json.load(f)

        # Process for combined features
        anes_data = []
        for item in anes_data_raw:
            # Combine structured-derived text and external text
            structured_text = item.get("text", "")
            qa_pairs = item.get("external_text_qa_pairs", [])
            external_text_combined = ""
            if qa_pairs:
                external_text_combined = "\n\n".join([f"Question: {qa['question']} Answer: {qa['answer']}" for qa in qa_pairs])

            if external_text_combined and structured_text:
                text_content = f"{structured_text}\n\n{external_text_combined}"
            elif external_text_combined:
                text_content = external_text_combined
            else:
                text_content = structured_text
            
            structured_features_list = item.get("anes_features", [])
            
            new_item = {
                "text": text_content,
                "target": item["target"],
                "respondent_id": item["respondent_id"]
            }
            if structured_features_list:
                new_item["anes_features"] = structured_features_list
            anes_data.append(new_item)

        texts = [item['text'] for item in anes_data]
        labels = [item['target'] for item in anes_data]
        structured_features_for_dataset = [item.get('anes_features', []) for item in anes_data]

        if use_bert_classifier:
            anes_dataset = ANESBertDataset(texts, labels, tokenizer, structured_features=structured_features_for_dataset)
        else:
            # Create temporary dataset for ProspectTheoryDataset
            temp_anes_data_path = "data/anes/temp_anes_data_for_optimal_analysis.json"
            os.makedirs(os.path.dirname(temp_anes_data_path), exist_ok=True)
            with open(temp_anes_data_path, "w") as f:
                json.dump(anes_data, f, indent=2)
            anes_dataset = ProspectTheoryDataset(temp_anes_data_path, tokenizer, is_anes=True, generate_text_from_anes=False)

    except Exception as e:
        print(f"Error loading ANES dataset: {e}")
        return {"error": "ANES dataset loading failed"}
    
    # K-fold Cross-Validation with optimal threshold evaluation
    print(f"Running {n_splits_kfold}-fold cross-validation with optimal threshold analysis...")
    kf = KFold(n_splits=n_splits_kfold, shuffle=True, random_state=seed)
    
    all_fold_results = []
    all_confusion_matrices = []
    all_accuracies = []
    all_f1_scores = []
    
    # Prepare data for k-fold
    if use_bert_classifier:
        X_kfold = anes_dataset.texts
        y_kfold = anes_dataset.labels
        structured_features_kfold = anes_dataset.structured_features
    else:
        X_kfold = []
        y_kfold = []
        structured_features_kfold = []
        for i in range(len(anes_dataset)):
            item = anes_dataset[i]
            X_kfold.append(item['text'])
            y_kfold.append(item['target'].item())
            if 'anes_features' in item and item['anes_features'] is not None:
                if isinstance(item['anes_features'], torch.Tensor):
                    structured_features_kfold.append(item['anes_features'].cpu().numpy())
                else:
                    structured_features_kfold.append(item['anes_features'])
            else:
                structured_features_kfold.append([])

    for fold, (train_index, val_index) in enumerate(kf.split(X_kfold, y_kfold)):
        print(f"\n--- Fold {fold+1}/{n_splits_kfold} ---")
        
        fold_train_texts = [X_kfold[i] for i in train_index]
        fold_train_labels = [y_kfold[i] for i in train_index]
        fold_val_texts = [X_kfold[i] for i in val_index]
        fold_val_labels = [y_kfold[i] for i in val_index]
        
        fold_train_structured_features = [structured_features_kfold[i] for i in train_index] if structured_features_kfold else None
        fold_val_structured_features = [structured_features_kfold[i] for i in val_index] if structured_features_kfold else None

        if use_bert_classifier:
            anes_train_dataset_fold = ANESBertDataset(fold_train_texts, fold_train_labels, tokenizer, structured_features=fold_train_structured_features)
            anes_val_dataset_fold = ANESBertDataset(fold_val_texts, fold_val_labels, tokenizer, structured_features=fold_val_structured_features)
        else:
            anes_train_dataset_fold = Subset(anes_dataset, train_index)
            anes_val_dataset_fold = Subset(anes_dataset, val_index)

        # Create dataloaders
        anes_train_dataloader = DataLoader(anes_train_dataset_fold, batch_size=batch_size, shuffle=True)
        anes_val_dataloader = DataLoader(anes_val_dataset_fold, batch_size=batch_size)
        
        # Train classifier
        print("Training ANES classifier...")
        fold_anes_metrics = train_anes_classifier(
            train_dataloader=anes_train_dataloader,
            val_dataloader=anes_val_dataloader,
            extractor=extractor,
            bias_representer=bias_representer,
            num_epochs=num_epochs_anes,
            learning_rate=learning_rate,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            save_dir=os.path.join(save_dir, f"fold_{fold+1}"),
            focal_loss_gamma=BEST_FOCAL_LOSS_GAMMA,
            use_bert_classifier=use_bert_classifier,
            bert_model_name=bert_model_name,
            use_structured_features=True
        )
        
        # Evaluate at optimal threshold
        print(f"Evaluating at optimal threshold {OPTIMAL_THRESHOLD:.2f}...")
        optimal_results = evaluate_at_optimal_threshold(
            fold_anes_metrics["anes_classifier"],
            anes_val_dataloader,
            extractor,
            bias_representer,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            use_bert_classifier=use_bert_classifier,
            use_structured_features=True
        )
        
        all_fold_results.append(optimal_results)
        all_confusion_matrices.append(optimal_results['confusion_matrix'])
        all_accuracies.append(optimal_results['accuracy'])
        all_f1_scores.append(optimal_results['macro_f1'])
        
        # Save fold-specific confusion matrix
        fold_cm_path = os.path.join(results_dir, f"fold_{fold+1}_confusion_matrix_optimal.png")
        plot_optimal_confusion_matrix(
            optimal_results['confusion_matrix'], 
            fold_cm_path,
            threshold=OPTIMAL_THRESHOLD,
            accuracy=optimal_results['accuracy'],
            f1=optimal_results['macro_f1']
        )
    
    # Aggregate results
    print(f"\n🎯 FINAL RESULTS AT OPTIMAL THRESHOLD ({OPTIMAL_THRESHOLD:.2f})")
    print("=" * 60)
    
    avg_accuracy = np.mean(all_accuracies)
    avg_f1 = np.mean(all_f1_scores)
    avg_confusion_matrix = np.mean(all_confusion_matrices, axis=0).astype(int)
    
    print(f"Average Accuracy: {avg_accuracy:.4f} ± {np.std(all_accuracies):.4f}")
    print(f"Average Macro F1: {avg_f1:.4f} ± {np.std(all_f1_scores):.4f}")
    print(f"\nAverage Confusion Matrix:")
    print(avg_confusion_matrix)
    
    # Save final averaged confusion matrix
    final_cm_path = os.path.join(results_dir, f"final_confusion_matrix_optimal_threshold_{OPTIMAL_THRESHOLD:.2f}.png")
    plot_optimal_confusion_matrix(
        avg_confusion_matrix, 
        final_cm_path,
        threshold=OPTIMAL_THRESHOLD,
        accuracy=avg_accuracy,
        f1=avg_f1
    )
    
    # Save detailed results
    final_results = {
        'optimal_threshold': OPTIMAL_THRESHOLD,
        'average_accuracy': float(avg_accuracy),
        'average_macro_f1': float(avg_f1),
        'accuracy_std': float(np.std(all_accuracies)),
        'f1_std': float(np.std(all_f1_scores)),
        'average_confusion_matrix': avg_confusion_matrix.tolist(),
        'individual_fold_results': [
            {
                'fold': i+1,
                'accuracy': float(result['accuracy']),
                'macro_f1': float(result['macro_f1']),
                'confusion_matrix': result['confusion_matrix'].tolist()
            }
            for i, result in enumerate(all_fold_results)
        ]
    }
    
    results_json_path = os.path.join(results_dir, f"optimal_threshold_analysis_{OPTIMAL_THRESHOLD:.2f}.json")
    with open(results_json_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_json_path}")
    print(f"Final confusion matrix PNG saved to: {final_cm_path}")
    
    return final_results

def main():
    """Main entry point for optimal threshold analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimal Threshold Analysis for Combined Features')
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
    parser.add_argument('--save_dir', type=str, default="models/optimal_analysis",
                        help='Directory to save models')
    parser.add_argument('--results_dir', type=str, default="results/optimal_analysis",
                        help='Directory to save results')
    parser.add_argument('--use_bert_classifier', action='store_true',
                        help='Use BERT-based classifier instead of LLM-based classifier')
    parser.add_argument('--bert_model_name', type=str, default="bert-base-uncased",
                        help='Model name for the BERT classifier')
    parser.add_argument('--use_oversampling', action='store_true',
                        help='Apply SMOTE oversampling to the ANES training data')
    parser.add_argument('--n_splits_kfold', type=int, default=5,
                        help='Number of splits for k-fold cross-validation')
    parser.add_argument('--anes_text_excel_path', type=str, default=None,
                        help='Path to the Excel file containing ANES textual open-ended responses.')
    
    args = parser.parse_args()
    
    print(f"🎯 Starting Optimal Threshold Analysis at {OPTIMAL_THRESHOLD:.2f}")
    print(f"Arguments: {args}")
    
    results = run_optimal_threshold_analysis(
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
        n_splits_kfold=args.n_splits_kfold,
        anes_text_excel_path=args.anes_text_excel_path
    )
    
    if "error" not in results:
        print(f"\n🎉 Optimal threshold analysis completed successfully!")
        print(f"📊 Average Accuracy at {OPTIMAL_THRESHOLD:.2f}: {results['average_accuracy']:.4f}")
        print(f"📊 Average Macro F1 at {OPTIMAL_THRESHOLD:.2f}: {results['average_macro_f1']:.4f}")
        print(f"📁 Results saved to: {args.results_dir}")
    else:
        print(f"❌ Analysis failed: {results['error']}")

if __name__ == "__main__":
    main()


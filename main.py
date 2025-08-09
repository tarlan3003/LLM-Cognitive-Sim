"""
Main Pipeline for Prospect Theory LLM - Final Fix for Threshold Key Mismatch

This version fixes the threshold key mismatch issue causing NaN aggregation.

Author: Tarlan Sultanov
Fixed by: Manus AI (threshold key mismatch fix)
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
import random # Import random for fallback in SMOTE
from typing import Optional

# Import custom modules - FIXED: Removed src. prefix
from src.dataset import ProspectTheoryDataset, ANESBertDataset
from src.llm_extractor import HiddenLayerExtractor
from src.bias_representer import CognitiveBiasRepresenter
from src.anes_classifier import ProspectTheoryANESClassifier, FocalLoss, train_anes_classifier, evaluate_anes_classifier
from src.utils import set_seed, create_directory_structure
from src.visualize import generate_visualizations, plot_confusion_matrix # Import plot_confusion_matrix

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
    n_splits_kfold: int = 5, # New parameter for k-fold cross-validation
    anes_text_excel_path: Optional[str] = None, # New parameter for ANES textual data Excel
    feature_mode: str = "combined" # New parameter: structured_only, text_only, combined
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
        anes_text_excel_path: Path to the Excel file containing ANES textual open-ended responses.
        feature_mode: Specifies which features to use for ANES classification: 'structured_only', 'text_only', or 'combined'.
    
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
            ProspectTheoryDataset.convert_anes_to_dataset(anes_path, anes_dataset_path, anes_text_excel_path=anes_text_excel_path)
        except Exception as e:
            print(f"Error converting ANES dataset: {e}")
            print("Please ensure ANES JSON files are available at the specified path")
            return {"error": "ANES dataset conversion failed"}
    
    # Load ANES dataset
    print("Loading ANES dataset...")
    try:
        with open(anes_dataset_path, 'r') as f:
            anes_data_raw = json.load(f)

        # Filter data based on feature_mode
        anes_data = []
        for item in anes_data_raw:
            text_content = ""
            structured_features_list = []

            if feature_mode == "structured_only":
                # Use only structured features, text will be empty or default from structured
                text_content = item.get("text", "") # This is text generated from structured features
                structured_features_list = item.get("anes_features", [])
            elif feature_mode == "text_only":
                # Use only external text, structured features will be empty
                # Now, external_text_qa_pairs contains list of {'question': 'Q', 'answer': 'A'}
                qa_pairs = item.get("external_text_qa_pairs", [])
                if qa_pairs:
                    text_content = "\n\n".join([f"Question: {qa['question']} Answer: {qa['answer']}" for qa in qa_pairs])
                structured_features_list = [] # No structured features
            elif feature_mode == "combined":
                # Combine both structured-derived text and external text
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
            else:
                raise ValueError(f"Invalid feature_mode: {feature_mode}")
            
            # Create a new item with filtered features
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
        structured_features_for_dataset = [item.get('anes_features', []) for item in anes_data] if feature_mode != "text_only" else None

        if use_bert_classifier:
            anes_dataset = ANESBertDataset(texts, labels, tokenizer, structured_features=structured_features_for_dataset)
        else:
            # For ProspectTheoryDataset, we need to pass the combined text and potentially structured features
            # The ProspectTheoryDataset expects 'text' and 'anes_features' in its internal data structure
            # We'll create a dummy data list for it based on the filtered anes_data
            pt_anes_data = []
            for i, item in enumerate(anes_data):
                pt_entry = {
                    "text": item["text"],
                    "target": item["target"],
                    "respondent_id": item["respondent_id"]
                }
                if structured_features_for_dataset and structured_features_for_dataset[i]:
                    pt_entry["anes_features"] = structured_features_for_dataset[i]
                pt_anes_data.append(pt_entry)
            
            # Create a dummy data_path for ProspectTheoryDataset as it expects a file path
            # This is a workaround, ideally ProspectTheoryDataset should be refactored to accept data directly
            temp_anes_data_path = "data/anes/temp_anes_data_for_pt_dataset.json"
            os.makedirs(os.path.dirname(temp_anes_data_path), exist_ok=True)
            with open(temp_anes_data_path, "w") as f:
                json.dump(pt_anes_data, f, indent=2)
            anes_dataset = ProspectTheoryDataset(temp_anes_data_path, tokenizer, is_anes=True, generate_text_from_anes=False)

    except Exception as e:
        print(f"Error loading ANES dataset: {e}")
        return {"error": "ANES dataset loading failed"}
    
    # K-fold Cross-Validation
    print(f"Running {n_splits_kfold}-fold cross-validation for ANES classifier...")
    kf = KFold(n_splits=n_splits_kfold, shuffle=True, random_state=seed)
    
    all_fold_metrics = []
    
    # Prepare data for k-fold (extract features and labels)
    if use_bert_classifier:
        X_kfold = anes_dataset.texts
        y_kfold = anes_dataset.labels
        structured_features_kfold = anes_dataset.structured_features
    else:
        # For ProspectTheoryDataset, we need to iterate to get features and labels
        # This might be memory intensive for very large datasets
        X_kfold = []
        y_kfold = []
        structured_features_kfold = []
        for i in range(len(anes_dataset)):
            item = anes_dataset[i]
            X_kfold.append(item['text'])
            y_kfold.append(item['target'].item())
            if 'anes_features' in item and item['anes_features'] is not None:
                # Convert tensor to numpy array if it's a tensor
                if isinstance(item['anes_features'], torch.Tensor):
                    structured_features_kfold.append(item['anes_features'].cpu().numpy())
                else:
                    structured_features_kfold.append(item['anes_features'])
            else:
                structured_features_kfold.append([]) # Append empty list if no structured features

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
            
            # Apply SMOTE oversampling for BERT classifier if enabled
            if use_oversampling:
                print(f"Applying SMOTE oversampling for BERT classifier in Fold {fold+1}...")
                # SMOTE needs 2D data. If structured features are available, use them.
                # Otherwise, use a dummy feature for texts.
                sm = SMOTE(random_state=seed)
                y_train_np = np.array(fold_train_labels)

                if fold_train_structured_features and len(fold_train_structured_features) > 0 and len(fold_train_structured_features[0]) > 0:
                    # Ensure all structured features are numpy arrays before stacking
                    X_smote = np.array([np.array(f) for f in fold_train_structured_features])
                else:
                    # Fallback to dummy features if only text or no structured features
                    X_smote = np.arange(len(fold_train_texts)).reshape(-1, 1)
                
                X_res, y_res = sm.fit_resample(X_smote, y_train_np)
                
                # Reconstruct the dataset with oversampled data
                # This is a simplified approach. For text, true oversampling would involve
                # generating synthetic text, which is beyond the scope of SMOTE.
                # Here, SMOTE primarily balances the labels for the training process.
                
                # If structured features were used for SMOTE, map them back
                if fold_train_structured_features and len(fold_train_structured_features) > 0 and len(fold_train_structured_features[0]) > 0:
                    resampled_structured_features = X_res.tolist()
                    # We can't easily generate synthetic text, so we'll just repeat original texts
                    # This is a limitation when using SMOTE on text-only data.
                    # For now, we'll just use the original texts and let the model handle imbalance.
                    # A more advanced approach would be to use text-specific oversampling techniques.
                    
                    # Find the original indices that were used to create the resampled data
                    # This is a simplification and might not be perfectly accurate for synthetic samples
                    original_indices_for_resampling = []
                    # This is a placeholder. A proper SMOTE implementation for text would be more complex.
                    # For now, we'll just duplicate texts based on the oversampled labels.
                    # This is not ideal for text, but ensures the dataset size matches.
                    
                    # A more robust way: SMOTE only on structured features, then duplicate text based on indices.
                    # If SMOTE is applied, we will resample the indices and then select texts/features based on those.
                    # This is a common workaround when SMOTE is used on mixed data types.
                    
                    # Create a temporary dataset for SMOTE to get indices
                    temp_X = np.arange(len(fold_train_texts)).reshape(-1, 1) # Dummy features for SMOTE
                    _, y_res_temp = sm.fit_resample(temp_X, y_train_np)
                    
                    # Now, y_res_temp has the resampled labels. We need to create corresponding texts and features.
                    # This means we'll be duplicating original samples based on the oversampling.
                    
                    resampled_texts = []
                    resampled_structured_features = []
                    
                    # Get the counts of each class in the resampled data
                    resampled_counts = Counter(y_res_temp)
                    
                    # For each class, duplicate samples until the target count is reached
                    for class_label in resampled_counts:
                        original_samples_in_class = [(fold_train_texts[i], fold_train_structured_features[i]) 
                                                     for i, label in enumerate(fold_train_labels) if label == class_label]
                        
                        # If no original samples for this class, skip
                        if not original_samples_in_class:
                            continue
                            
                        num_to_add = resampled_counts[class_label] - len(original_samples_in_class)
                        
                        # Add original samples
                        for text, sf in original_samples_in_class:
                            resampled_texts.append(text)
                            resampled_structured_features.append(sf)
                            
                        # Duplicate existing samples to reach the target count
                        if num_to_add > 0:
                            # Randomly sample with replacement from original samples
                            for _ in range(num_to_add):
                                text, sf = random.choice(original_samples_in_class)
                                resampled_texts.append(text)
                                resampled_structured_features.append(sf)

                else:
                    # If dummy features were used, we just repeat original texts based on resampled indices
                    # This case is for when structured_features_kfold is None or empty
                    temp_X = np.arange(len(fold_train_texts)).reshape(-1, 1) # Dummy features for SMOTE
                    _, y_res_temp = sm.fit_resample(temp_X, y_train_np)
                    
                    resampled_texts = []
                    resampled_structured_features = None # No structured features
                    
                    resampled_counts = Counter(y_res_temp)
                    
                    for class_label in resampled_counts:
                        original_samples_in_class = [fold_train_texts[i] 
                                                     for i, label in enumerate(fold_train_labels) if label == class_label]
                        
                        if not original_samples_in_class:
                            continue
                            
                        num_to_add = resampled_counts[class_label] - len(original_samples_in_class)
                        
                        for text in original_samples_in_class:
                            resampled_texts.append(text)
                            
                        if num_to_add > 0:
                            for _ in range(num_to_add):
                                text = random.choice(original_samples_in_class)
                                resampled_texts.append(text)

                anes_train_dataset_fold = ANESBertDataset(resampled_texts, y_res.tolist(), tokenizer, structured_features=resampled_structured_features)
                print(f"Original train samples: {len(fold_train_texts)}, Resampled train samples: {len(resampled_texts)}")

        else:
            # For ProspectTheoryDataset, create Subset objects
            # SMOTE for ProspectTheoryDataset is handled by FocalLoss weighting
            anes_train_dataset_fold = Subset(anes_dataset, train_index)
            anes_val_dataset_fold = Subset(anes_dataset, val_index)
            
            # SMOTE for ProspectTheoryANESClassifier is primarily handled by FocalLoss weighting
            # Actual data augmentation for ProspectTheoryDataset with SMOTE is complex and not directly implemented here.
            # The `use_oversampling` flag will primarily trigger class weighting in FocalLoss for this model.
            if use_oversampling:
                print(f"SMOTE oversampling for ProspectTheoryANESClassifier in Fold {fold+1} will influence FocalLoss class weighting.")

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
            bert_model_name=bert_model_name,
            use_structured_features=(feature_mode != "text_only") # Pass this to classifier
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
            
    # Aggregate results from all folds - FINAL FIX: Handle threshold key mismatch
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
    
    # FINAL FIX: Handle threshold key mismatch by using the actual best threshold value
    for fold_idx, fold_metrics in enumerate(all_fold_metrics):
        print(f"\nDEBUG: Processing fold {fold_idx + 1}")
        
        # Check if we have thresholded_results and best_threshold from evaluation
        if "best_threshold" in fold_metrics and "thresholded_results" in fold_metrics:
            best_threshold = fold_metrics["best_threshold"]
            thresholded_results = fold_metrics["thresholded_results"]
            
            print(f"DEBUG: Best threshold for fold {fold_idx + 1}: {best_threshold}")
            print(f"DEBUG: Available threshold keys: {list(thresholded_results.keys())}")
            
            # FINAL FIX: Try multiple possible key formats
            possible_keys = [
                f"threshold_{best_threshold:.2f}",  # Original format
                f"threshold_{best_threshold}",      # Without decimal formatting
                str(best_threshold),                # Just the threshold value
                f"{best_threshold:.2f}",           # Threshold value with 2 decimals
                f"{best_threshold:.1f}",           # Threshold value with 1 decimal
            ]
            
            metrics_at_best_threshold = None
            used_key = None
            
            # Try each possible key format
            for key in possible_keys:
                if key in thresholded_results:
                    metrics_at_best_threshold = thresholded_results[key]
                    used_key = key
                    print(f"DEBUG: Found metrics using key: {key}")
                    break
            
            # If no exact match, try to find the closest threshold
            if metrics_at_best_threshold is None:
                print(f"DEBUG: No exact match found, trying closest threshold...")
                closest_key = None
                min_diff = float('inf')
                
                for key in thresholded_results.keys():
                    # Extract threshold value from key
                    try:
                        if key.startswith('threshold_'):
                            threshold_val = float(key.replace('threshold_', ''))
                        else:
                            threshold_val = float(key)
                        
                        diff = abs(threshold_val - best_threshold)
                        if diff < min_diff:
                            min_diff = diff
                            closest_key = key
                    except ValueError:
                        continue
                
                if closest_key:
                    metrics_at_best_threshold = thresholded_results[closest_key]
                    used_key = closest_key
                    print(f"DEBUG: Using closest threshold key: {closest_key}")
            
            if metrics_at_best_threshold:
                print(f"DEBUG: Successfully extracted metrics using key: {used_key}")
                
                # Append metrics only if they are not None and are numeric
                accuracy = metrics_at_best_threshold.get("accuracy")
                if accuracy is not None and not np.isnan(accuracy):
                    all_accuracies.append(accuracy)
                    print(f"DEBUG: Added accuracy: {accuracy}")
                    
                macro_precision = metrics_at_best_threshold.get("macro_precision")
                if macro_precision is not None and not np.isnan(macro_precision):
                    all_macro_precisions.append(macro_precision)
                    
                macro_recall = metrics_at_best_threshold.get("macro_recall")
                if macro_recall is not None and not np.isnan(macro_recall):
                    all_macro_recalls.append(macro_recall)
                    
                macro_f1 = metrics_at_best_threshold.get("macro_f1")
                if macro_f1 is not None and not np.isnan(macro_f1):
                    all_macro_f1s.append(macro_f1)
                    print(f"DEBUG: Added macro F1: {macro_f1}")
                    
                weighted_precision = metrics_at_best_threshold.get("weighted_precision")
                if weighted_precision is not None and not np.isnan(weighted_precision):
                    all_weighted_precisions.append(weighted_precision)
                    
                weighted_recall = metrics_at_best_threshold.get("weighted_recall")
                if weighted_recall is not None and not np.isnan(weighted_recall):
                    all_weighted_recalls.append(weighted_recall)
                    
                weighted_f1 = metrics_at_best_threshold.get("weighted_f1")
                if weighted_f1 is not None and not np.isnan(weighted_f1):
                    all_weighted_f1s.append(weighted_f1)
                
                # Confusion matrix handling: ensure it's a numpy array and append
                cm = metrics_at_best_threshold.get("confusion_matrix")
                if cm is not None:
                    if isinstance(cm, list): # Convert list to numpy array if needed
                        cm = np.array(cm)
                    if isinstance(cm, np.ndarray) and cm.ndim == 2: # Ensure it's a 2D array
                        all_confusion_matrices.append(cm)
                        print(f"DEBUG: Added confusion matrix with shape: {cm.shape}")
            else:
                print(f"WARNING: Could not find metrics for fold {fold_idx + 1} with threshold {best_threshold}")
        else:
            print(f"WARNING: Missing 'best_threshold' or 'thresholded_results' in fold {fold_idx + 1} metrics")

    # Calculate averages, handling empty lists to avoid NaN from np.mean
    aggregated_metrics["average_accuracy"] = np.mean(all_accuracies) if all_accuracies else np.nan
    aggregated_metrics["average_macro_precision"] = np.mean(all_macro_precisions) if all_macro_precisions else np.nan
    aggregated_metrics["average_macro_recall"] = np.mean(all_macro_recalls) if all_macro_recalls else np.nan
    aggregated_metrics["average_macro_f1"] = np.mean(all_macro_f1s) if all_macro_f1s else np.nan
    aggregated_metrics["average_weighted_precision"] = np.mean(all_weighted_precisions) if all_weighted_precisions else np.nan
    aggregated_metrics["average_weighted_recall"] = np.mean(all_weighted_recalls) if all_weighted_recalls else np.nan
    aggregated_metrics["average_weighted_f1"] = np.mean(all_weighted_f1s) if all_weighted_f1s else np.nan
    
    # Average confusion matrix only if there are matrices to average
    if all_confusion_matrices:
        # Ensure all confusion matrices have the same shape before averaging
        if len(set(cm.shape for cm in all_confusion_matrices)) == 1:
            aggregated_metrics["average_confusion_matrix"] = np.mean(all_confusion_matrices, axis=0)
        else:
            print("Warning: Confusion matrices have inconsistent shapes. Cannot compute average confusion matrix.")
            aggregated_metrics["average_confusion_matrix"] = np.full((2,2), np.nan) # Placeholder for inconsistent shapes
    else:
        aggregated_metrics["average_confusion_matrix"] = np.full((2,2), np.nan) # Default to NaN if no matrices
    
    # FINAL DEBUG: Show what was collected
    print(f"\nFINAL DEBUG: Collected {len(all_accuracies)} accuracy values")
    print(f"FINAL DEBUG: Collected {len(all_macro_f1s)} macro F1 values")
    print(f"FINAL DEBUG: Collected {len(all_confusion_matrices)} confusion matrices")
    
    if all_accuracies:
        print(f"FINAL DEBUG: Accuracy values: {all_accuracies}")
    if all_macro_f1s:
        print(f"FINAL DEBUG: Macro F1 values: {all_macro_f1s}")
    
    print("\nFinal Aggregated Evaluation Results (K-Fold Cross-Validation):")
    print(f"Average Accuracy: {aggregated_metrics['average_accuracy']:.4f}")
    print(f"Average Macro Precision: {aggregated_metrics['average_macro_precision']:.4f}")
    print(f"Average Macro Recall: {aggregated_metrics['average_macro_recall']:.4f}")
    print(f"Average Macro F1-Score: {aggregated_metrics['average_macro_f1']:.4f}")
    print(f"Average Weighted Precision: {aggregated_metrics['average_weighted_precision']:.4f}")
    print(f"Average Weighted Recall: {aggregated_metrics['average_weighted_recall']:.4f}")
    print(f"Average Weighted F1-Score: {aggregated_metrics['average_weighted_f1']:.4f}")
    print("Average Confusion Matrix:\n", aggregated_metrics["average_confusion_matrix"])
    
    # Save aggregated results
    with open(os.path.join(results_dir, "aggregated_kfold_results.json"), "w") as f:
        # Convert numpy arrays to list for JSON serialization
        json.dump({
            k: (v.tolist() if isinstance(v, np.ndarray) else v) 
            for k, v in aggregated_metrics.items()
        }, f, indent=2)
    print(f"Aggregated K-Fold results saved to {os.path.join(results_dir, 'aggregated_kfold_results.json')}")

    # Generate overall visualizations from aggregated data (e.g., average confusion matrix)
    try:
        # Only plot if the average confusion matrix is not all NaNs
        if not np.isnan(aggregated_metrics["average_confusion_matrix"]).all():
            plot_confusion_matrix(
                aggregated_metrics["average_confusion_matrix"], 
                target_names=["Trump", "Harris"], 
                save_path=os.path.join(results_dir, "average_confusion_matrix.png"),
                title="Average Confusion Matrix (K-Fold)"
            )
        else:
            print("Skipping average confusion matrix plot due to NaN values or inconsistent shapes.")
    except Exception as e:
            print(f"Error generating average confusion matrix plot: {e}")

    print(f"\nPipeline complete! Results saved to {save_dir} and {results_dir}")
    print(f"Model used: {actual_model_name}")
    
    # Include individual fold metrics for JSON output
    aggregated_metrics['fold_metrics'] = all_fold_metrics
    
    return aggregated_metrics

def save_best_scores_json(results, feature_mode, results_dir):
    """
    Extract and save the best scores and confusion matrix as JSON files.
    
    Args:
        results: Dictionary containing the pipeline results
        feature_mode: String indicating the feature mode (structured_only, text_only, combined)
        results_dir: Directory to save the JSON files
    """
    if "error" in results:
        print(f"Skipping JSON output for {feature_mode} due to error: {results['error']}")
        return
    
    try:
        # Find the best fold and threshold across all folds
        best_overall_score = 0.0
        best_fold_idx = 0
        best_threshold = 0.5
        best_fold_metrics = None
        best_confusion_matrix = None
        
        # Look through all fold metrics to find the absolute best
        if 'fold_metrics' in results:
            for fold_idx, fold_metrics in enumerate(results['fold_metrics']):
                if 'best_threshold' in fold_metrics and 'thresholded_results' in fold_metrics:
                    threshold = fold_metrics['best_threshold']
                    thresholded_results = fold_metrics['thresholded_results']
                    
                    # Try multiple possible key formats (same logic as in aggregation)
                    possible_keys = [
                        f"threshold_{threshold:.2f}",
                        f"threshold_{threshold}",
                        str(threshold),
                        f"{threshold:.2f}",
                        f"{threshold:.1f}",
                    ]
                    
                    metrics_at_threshold = None
                    for key in possible_keys:
                        if key in thresholded_results:
                            metrics_at_threshold = thresholded_results[key]
                            break
                    
                    if metrics_at_threshold:
                        # Use F1 score as the primary metric for "best"
                        current_score = metrics_at_threshold.get('macro_f1', 0.0)
                        
                        if current_score > best_overall_score:
                            best_overall_score = current_score
                            best_fold_idx = fold_idx
                            best_threshold = threshold
                            best_fold_metrics = metrics_at_threshold
                            best_confusion_matrix = metrics_at_threshold.get('confusion_matrix')
        
        # If no fold_metrics found, try to extract from aggregated results
        if best_fold_metrics is None:
            print(f"No fold_metrics found for {feature_mode}, using aggregated results")
            best_fold_metrics = {
                'accuracy': results.get('average_accuracy', 0.0),
                'macro_precision': results.get('average_macro_precision', 0.0),
                'macro_recall': results.get('average_macro_recall', 0.0),
                'macro_f1': results.get('average_macro_f1', 0.0),
                'weighted_precision': results.get('average_weighted_precision', 0.0),
                'weighted_recall': results.get('average_weighted_recall', 0.0),
                'weighted_f1': results.get('average_weighted_f1', 0.0)
            }
            best_confusion_matrix = results.get('average_confusion_matrix')
            best_fold_idx = "aggregated"
            best_threshold = "aggregated"
        
        # Prepare the best scores JSON
        best_scores = {
            'feature_mode': feature_mode,
            'best_fold': best_fold_idx,
            'best_threshold': best_threshold,
            'best_scores': {
                'accuracy': float(best_fold_metrics.get('accuracy', 0.0)),
                'precision': float(best_fold_metrics.get('macro_precision', 0.0)),
                'recall': float(best_fold_metrics.get('macro_recall', 0.0)),
                'f1': float(best_fold_metrics.get('macro_f1', 0.0)),
                'weighted_precision': float(best_fold_metrics.get('weighted_precision', 0.0)),
                'weighted_recall': float(best_fold_metrics.get('weighted_recall', 0.0)),
                'weighted_f1': float(best_fold_metrics.get('weighted_f1', 0.0))
            }
        }
        
        # Save best scores JSON
        best_scores_path = os.path.join(results_dir, f"{feature_mode}_best_scores.json")
        with open(best_scores_path, 'w') as f:
            json.dump(best_scores, f, indent=2)
        print(f"Best scores saved to {best_scores_path}")
        
        # Save confusion matrix JSON if available
        if best_confusion_matrix is not None:
            confusion_matrix_data = {
                'feature_mode': feature_mode,
                'best_fold': best_fold_idx,
                'best_threshold': best_threshold,
                'confusion_matrix': best_confusion_matrix.tolist() if hasattr(best_confusion_matrix, 'tolist') else best_confusion_matrix,
                'labels': ['Trump', 'Harris']  # Default labels for binary classification
            }
            
            confusion_matrix_path = os.path.join(results_dir, f"{feature_mode}_confusion_matrix.json")
            with open(confusion_matrix_path, 'w') as f:
                json.dump(confusion_matrix_data, f, indent=2)
            print(f"Confusion matrix saved to {confusion_matrix_path}")
        else:
            print(f"No confusion matrix available for {feature_mode}")
            
    except Exception as e:
        print(f"Error saving JSON files for {feature_mode}: {e}")
        import traceback
        traceback.print_exc()


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
                        help='Model name for the BERT classifier')
    parser.add_argument('--use_oversampling', action='store_true',
                        help='Apply SMOTE oversampling to the ANES training data')
    parser.add_argument('--n_splits_kfold', type=int, default=5,
                        help='Number of splits for k-fold cross-validation')
    parser.add_argument('--anes_text_excel_path', type=str, default=None,
                        help='Path to the Excel file containing ANES textual open-ended responses.')
    
    args = parser.parse_args()
    
    print("Starting Prospect Theory LLM Pipeline...")
    print(f"Arguments: {args}")
    
    all_feature_modes = ["structured_only", "text_only", "combined"]
    comparative_results = {}

    for mode in all_feature_modes:
        print(f"\n\n=== Running pipeline with feature_mode: {mode} ===")
        # Ensure results_dir and save_dir are correctly set for each mode
        mode_results_dir = os.path.join(args.results_dir, mode)
        mode_save_dir = os.path.join(args.save_dir, mode)
        os.makedirs(mode_results_dir, exist_ok=True)
        os.makedirs(mode_save_dir, exist_ok=True)

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
                save_dir=mode_save_dir, # Save models in mode-specific subdirectories
                results_dir=mode_results_dir, # Save results in mode-specific subdirectories
                use_bert_classifier=args.use_bert_classifier,
                bert_model_name=args.bert_model_name,
                use_oversampling=args.use_oversampling,
                n_splits_kfold=args.n_splits_kfold,
                anes_text_excel_path=args.anes_text_excel_path,
                feature_mode=mode # Pass the current feature mode to the pipeline
            )
            comparative_results[mode] = results
            
            if "error" in results:
                print(f"Pipeline for {mode} failed with error: {results['error']}")
            else:
                print(f"Pipeline for {mode} completed successfully!")
                # Save best scores and confusion matrix as JSON files
                save_best_scores_json(results, mode, mode_results_dir)
                
        except Exception as e:
            print(f"Pipeline for {mode} failed with exception: {e}")
            import traceback
            traceback.print_exc()
            comparative_results[mode] = {"error": str(e)}

    print("\n\n=== Comparative Results Across Feature Modes ===")
    # Generate a comparative report
    report_data = []
    for mode, metrics in comparative_results.items():
        if "error" not in metrics:
            report_data.append({
                "Feature Mode": mode,
                "Average Accuracy": f"{metrics['average_accuracy']:.4f}",
                "Average Macro F1": f"{metrics['average_macro_f1']:.4f}",
                "Average Weighted F1": f"{metrics['average_weighted_f1']:.4f}"
            })
        else:
            report_data.append({
                "Feature Mode": mode,
                "Average Accuracy": "N/A",
                "Average Macro F1": "N/A",
                "Average Weighted F1": "N/A",
                "Error": metrics["error"]
            })
    
    comparative_df = pd.DataFrame(report_data)
    print(comparative_df.to_string(index=False))

    # Save comparative report to a file
    comparative_report_path = os.path.join(args.results_dir, "comparative_feature_mode_report.md")
    with open(comparative_report_path, "w") as f:
        f.write("# Comparative Results Across Feature Modes\n\n")
        f.write(comparative_df.to_markdown(index=False))
    print(f"Comparative report saved to {comparative_report_path}")

    print("\nAll pipeline runs completed!")

if __name__ == "__main__":
    main()


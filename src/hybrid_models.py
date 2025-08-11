"""
Hybrid Model Implementation for Prospect Theory Analysis

This script implements a hybrid approach combining deep learning and classical ML
for the Prospect Theory analysis of ANES data, providing the best of both worlds.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union
import time
import argparse
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Import modules
from dataset import ProspectTheoryDataset, convert_anes_to_dataset
from llm_extractor import HiddenLayerExtractor
from bias_representer import CognitiveBiasRepresenter
from anes_classifier import ProspectTheoryANESClassifier, train_anes_classifier, evaluate_anes_classifier
from utils import create_directory_structure, set_seed, check_compatibility
from visualize import generate_all_visualizations

# Default paths
DEFAULT_ANES_PATH = "/home/tsultanov/shared/datasets/respondents"
DEFAULT_OUTPUT_DIR = "results"
DEFAULT_MODEL_DIR = "models"
DEFAULT_DATA_DIR = "data"

# Default parameters
DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_EPOCHS_PROSPECT = 5
DEFAULT_NUM_EPOCHS_ANES = 20
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_SEED = 42

# Hybrid approaches to test
HYBRID_APPROACHES = [
    {
        "name": "ensemble_voting",
        "description": "Ensemble voting of deep learning and classical models"
    },
    {
        "name": "stacked_generalization",
        "description": "Stacked generalization with deep learning features and classical meta-learner"
    },
    {
        "name": "feature_augmentation",
        "description": "Classical model with deep learning features"
    },
    {
        "name": "decision_tree_refinement",
        "description": "Deep learning predictions refined by decision trees"
    }
]


class HybridModel:
    """
    Hybrid model combining deep learning and classical ML approaches.
    """
    
    def __init__(
        self,
        approach: str,
        deep_model: ProspectTheoryANESClassifier,
        classical_model_type: str = "random_forest",
        num_classes: int = 2,
        device: str = "cpu"
    ):
        """
        Initialize the hybrid model.
        
        Args:
            approach: Hybrid approach to use
            deep_model: Deep learning model
            classical_model_type: Type of classical model to use
            num_classes: Number of classes
            device: Device to run on
        """
        self.approach = approach
        self.deep_model = deep_model
        self.classical_model_type = classical_model_type
        self.num_classes = num_classes
        self.device = device
        
        # Initialize classical model
        if classical_model_type == "random_forest":
            self.classical_model = RandomForestClassifier(n_estimators=200, random_state=42)
        elif classical_model_type == "gradient_boosting":
            self.classical_model = GradientBoostingClassifier(n_estimators=200, random_state=42)
        elif classical_model_type == "logistic_regression":
            self.classical_model = LogisticRegression(max_iter=1000, random_state=42)
        else:
            raise ValueError(f"Unknown classical model type: {classical_model_type}")
        
        # Initialize meta-learner for stacked generalization
        if approach == "stacked_generalization":
            self.meta_learner = LogisticRegression(max_iter=1000, random_state=42)
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
        # Initialize voting weights
        self.voting_weights = [0.5, 0.5]  # [deep_weight, classical_weight]
    
    def fit(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        extractor: HiddenLayerExtractor,
        bias_representer: CognitiveBiasRepresenter,
        num_epochs: int = 20,
        lr: float = 3e-4
    ):
        """
        Train the hybrid model.
        
        Args:
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            extractor: LLM hidden layer extractor
            bias_representer: Cognitive bias representer
            num_epochs: Number of epochs for deep learning training
            lr: Learning rate for deep learning training
        """
        # Train deep learning model
        print("Training deep learning model...")
        train_anes_classifier(
            self.deep_model,
            train_dataloader,
            extractor,
            bias_representer,
            num_epochs=num_epochs,
            lr=lr,
            device=self.device
        )
        
        # Extract features and predictions for classical model
        print("Extracting features and predictions for classical model...")
        X_train, y_train, deep_preds_train = self._extract_features_and_predictions(
            train_dataloader, extractor, bias_representer
        )
        
        X_val, y_val, deep_preds_val = self._extract_features_and_predictions(
            val_dataloader, extractor, bias_representer
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train classical model based on approach
        print(f"Training classical model with approach: {self.approach}...")
        
        if self.approach == "ensemble_voting":
            # Train classical model on original features
            self.classical_model.fit(X_train_scaled, y_train)
            
            # Optimize voting weights
            classical_preds_val = self.classical_model.predict_proba(X_val_scaled)
            
            # Grid search for optimal weights
            best_accuracy = 0.0
            best_weights = [0.5, 0.5]
            
            for w1 in np.linspace(0.0, 1.0, 11):
                w2 = 1.0 - w1
                weights = [w1, w2]
                
                # Weighted voting
                ensemble_preds = w1 * deep_preds_val + w2 * classical_preds_val
                ensemble_classes = np.argmax(ensemble_preds, axis=1)
                
                # Calculate accuracy
                accuracy = np.mean(ensemble_classes == y_val)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_weights = weights
            
            self.voting_weights = best_weights
            print(f"Optimized voting weights: {self.voting_weights}")
        
        elif self.approach == "stacked_generalization":
            # Create meta-features
            meta_features_train = np.hstack([deep_preds_train, X_train_scaled])
            
            # Train meta-learner
            self.meta_learner.fit(meta_features_train, y_train)
        
        elif self.approach == "feature_augmentation":
            # Extract deep learning embeddings
            deep_embeddings_train = self._extract_deep_embeddings(
                train_dataloader, extractor, bias_representer
            )
            
            # Combine with original features
            augmented_features_train = np.hstack([X_train_scaled, deep_embeddings_train])
            
            # Train classical model on augmented features
            self.classical_model.fit(augmented_features_train, y_train)
        
        elif self.approach == "decision_tree_refinement":
            # Train classical model on deep learning errors
            errors_mask = np.argmax(deep_preds_train, axis=1) != y_train
            
            if np.sum(errors_mask) > 10:  # Ensure enough error samples
                X_errors = X_train_scaled[errors_mask]
                y_errors = y_train[errors_mask]
                
                # Train classical model on errors
                self.classical_model.fit(X_errors, y_errors)
            else:
                # Not enough errors, train on all data
                self.classical_model.fit(X_train_scaled, y_train)
        
        else:
            raise ValueError(f"Unknown hybrid approach: {self.approach}")
    
    def predict(
        self,
        dataloader: torch.utils.data.DataLoader,
        extractor: HiddenLayerExtractor,
        bias_representer: CognitiveBiasRepresenter
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with the hybrid model.
        
        Args:
            dataloader: Data loader
            extractor: LLM hidden layer extractor
            bias_representer: Cognitive bias representer
            
        Returns:
            Tuple of predicted probabilities and classes
        """
        # Extract features and predictions
        X, y, deep_preds = self._extract_features_and_predictions(
            dataloader, extractor, bias_representer
        )
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions based on approach
        if self.approach == "ensemble_voting":
            # Get classical predictions
            classical_preds = self.classical_model.predict_proba(X_scaled)
            
            # Weighted voting
            w1, w2 = self.voting_weights
            ensemble_preds = w1 * deep_preds + w2 * classical_preds
            ensemble_classes = np.argmax(ensemble_preds, axis=1)
            
            return ensemble_preds, ensemble_classes
        
        elif self.approach == "stacked_generalization":
            # Create meta-features
            meta_features = np.hstack([deep_preds, X_scaled])
            
            # Meta-learner predictions
            meta_preds = self.meta_learner.predict_proba(meta_features)
            meta_classes = self.meta_learner.predict(meta_features)
            
            return meta_preds, meta_classes
        
        elif self.approach == "feature_augmentation":
            # Extract deep learning embeddings
            deep_embeddings = self._extract_deep_embeddings(
                dataloader, extractor, bias_representer
            )
            
            # Combine with original features
            augmented_features = np.hstack([X_scaled, deep_embeddings])
            
            # Classical model predictions on augmented features
            classical_preds = self.classical_model.predict_proba(augmented_features)
            classical_classes = self.classical_model.predict(augmented_features)
            
            return classical_preds, classical_classes
        
        elif self.approach == "decision_tree_refinement":
            # Get deep learning predictions
            deep_classes = np.argmax(deep_preds, axis=1)
            
            # For samples with high uncertainty, use classical model
            uncertainty = 1.0 - np.max(deep_preds, axis=1)
            uncertain_mask = uncertainty > 0.3  # Threshold for uncertainty
            
            # Initialize refined predictions with deep learning predictions
            refined_classes = deep_classes.copy()
            
            if np.sum(uncertain_mask) > 0:
                # Get classical predictions for uncertain samples
                classical_classes = self.classical_model.predict(X_scaled[uncertain_mask])
                
                # Replace uncertain predictions with classical predictions
                refined_classes[uncertain_mask] = classical_classes
            
            # Convert to one-hot for consistency
            refined_preds = np.zeros((len(refined_classes), self.num_classes))
            for i, c in enumerate(refined_classes):
                refined_preds[i, c] = 1.0
            
            return refined_preds, refined_classes
        
        else:
            raise ValueError(f"Unknown hybrid approach: {self.approach}")
    
    def _extract_features_and_predictions(
        self,
        dataloader: torch.utils.data.DataLoader,
        extractor: HiddenLayerExtractor,
        bias_representer: CognitiveBiasRepresenter
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract features and deep learning predictions from a dataloader.
        
        Args:
            dataloader: Data loader
            extractor: LLM hidden layer extractor
            bias_representer: Cognitive bias representer
            
        Returns:
            Tuple of features, targets, and deep learning predictions
        """
        all_features = []
        all_targets = []
        all_deep_preds = []
        
        with torch.no_grad():
            for batch in dataloader:
                texts = batch['text']
                anes_features = batch['anes_features'].to(self.device)
                targets = batch['target'].cpu().numpy()
                
                # Extract activations
                activations = extractor.extract_activations(texts)
                
                # Get bias scores and system representations
                bias_scores = bias_representer.get_bias_scores(activations).to(self.device)
                weighted_system_rep, _ = bias_representer.get_system_representations(activations)
                weighted_system_rep = weighted_system_rep.to(self.device)
                
                # Get deep learning predictions
                deep_outputs = self.deep_model(anes_features, bias_scores, weighted_system_rep)
                deep_probs = torch.softmax(deep_outputs, dim=1).cpu().numpy()
                
                # Collect features and targets
                all_features.append(anes_features.cpu().numpy())
                all_targets.append(targets)
                all_deep_preds.append(deep_probs)
        
        # Concatenate batches
        X = np.vstack(all_features)
        y = np.concatenate(all_targets)
        deep_preds = np.vstack(all_deep_preds)
        
        return X, y, deep_preds
    
    def _extract_deep_embeddings(
        self,
        dataloader: torch.utils.data.DataLoader,
        extractor: HiddenLayerExtractor,
        bias_representer: CognitiveBiasRepresenter
    ) -> np.ndarray:
        """
        Extract deep learning embeddings from a dataloader.
        
        Args:
            dataloader: Data loader
            extractor: LLM hidden layer extractor
            bias_representer: Cognitive bias representer
            
        Returns:
            Deep learning embeddings
        """
        all_embeddings = []
        
        with torch.no_grad():
            for batch in dataloader:
                texts = batch['text']
                anes_features = batch['anes_features'].to(self.device)
                
                # Extract activations
                activations = extractor.extract_activations(texts)
                
                # Get bias scores and system representations
                bias_scores = bias_representer.get_bias_scores(activations).to(self.device)
                weighted_system_rep, _ = bias_representer.get_system_representations(activations)
                weighted_system_rep = weighted_system_rep.to(self.device)
                
                # Get embeddings from the penultimate layer
                embeddings = self.deep_model.get_embeddings(anes_features, bias_scores, weighted_system_rep)
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        # Concatenate batches
        embeddings = np.vstack(all_embeddings)
        
        return embeddings


def compare_hybrid_approaches(
    anes_json_path: str,
    prospect_theory_path: Optional[str] = None,
    llm_model_name: str = "roberta-large",
    llm_layers: List[int] = [-1, -2, -4, -8],
    output_dir: str = DEFAULT_OUTPUT_DIR,
    model_dir: str = DEFAULT_MODEL_DIR,
    data_dir: str = DEFAULT_DATA_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_epochs_prospect: int = DEFAULT_NUM_EPOCHS_PROSPECT,
    num_epochs_anes: int = DEFAULT_NUM_EPOCHS_ANES,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    seed: int = DEFAULT_SEED,
    cross_validation_folds: int = 5,
    target_variable: str = "V241049",  # WHO WOULD R VOTE FOR: HARRIS VS TRUMP
    target_classes: List[str] = ["Donald Trump", "Kamala Harris"],
    device: Optional[str] = None
) -> Dict:
    """
    Compare hybrid approaches for the Prospect Theory analysis of ANES data.
    
    Args:
        anes_json_path: Path to ANES JSON files
        prospect_theory_path: Path to Prospect Theory dataset
        llm_model_name: Name of the LLM model to use
        llm_layers: Layers to extract from the LLM
        output_dir: Directory to save outputs
        model_dir: Directory to save models
        data_dir: Directory to save processed data
        batch_size: Batch size for training
        num_epochs_prospect: Number of epochs for Prospect Theory training
        num_epochs_anes: Number of epochs for ANES classifier training
        learning_rate: Learning rate for training
        seed: Random seed
        cross_validation_folds: Number of cross-validation folds
        target_variable: Target variable in ANES data
        target_classes: Target classes in ANES data
        device: Device to run on (cuda or cpu)
        
    Returns:
        Dictionary of comparison results
    """
    # Set random seed
    set_seed(seed)
    
    # Create directory structure
    create_directory_structure(data_dir, model_dir, output_dir)
    
    # Set device
    if device is None:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    print(f"Using LLM model: {llm_model_name}")
    
    # Process ANES data
    anes_output_path = os.path.join(data_dir, "anes", "anes_dataset.json")
    if not os.path.exists(anes_output_path):
        print(f"Converting ANES data from {anes_json_path} to {anes_output_path}")
        convert_anes_to_dataset(
            json_folder=anes_json_path,
            output_path=anes_output_path,
            target_variable=target_variable,
            include_classes=target_classes
        )
    
    # Create or load Prospect Theory dataset
    if prospect_theory_path is None:
        prospect_theory_path = os.path.join(data_dir, "prospect_theory", "prospect_theory_dataset.json")
    
    if not os.path.exists(prospect_theory_path):
        print(f"Creating Prospect Theory dataset at {prospect_theory_path}")
        os.makedirs(os.path.dirname(prospect_theory_path), exist_ok=True)
        ProspectTheoryDataset.create_prospect_theory_dataset(
            output_path=prospect_theory_path,
            num_examples=500
        )
    
    # Initialize tokenizer and extractor
    print(f"Initializing LLM extractor with {llm_model_name}")
    from transformers import AutoTokenizer
    import torch
    
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    extractor = HiddenLayerExtractor(llm_model_name, llm_layers, device=device)
    
    # Load datasets
    print("Loading datasets")
    prospect_dataset = ProspectTheoryDataset(prospect_theory_path, tokenizer)
    anes_dataset = ProspectTheoryDataset(anes_output_path, tokenizer, is_anes=True)
    
    print(f"Prospect Theory dataset: {len(prospect_dataset)} examples")
    print(f"ANES dataset: {len(anes_dataset)} examples")
    print(f"Bias types: {prospect_dataset.bias_names}")
    
    # Create dataloaders
    prospect_dataloader = torch.utils.data.DataLoader(
        prospect_dataset, batch_size=batch_size, shuffle=True
    )
    
    # Initialize bias representer
    print("Initializing cognitive bias representer")
    bias_representer = CognitiveBiasRepresenter(
        extractor.get_hidden_size(),
        prospect_dataset.bias_names,
        device=device
    )
    
    # Train CAVs
    print("Training Concept Activation Vectors (CAVs)")
    bias_representer.train_cavs(prospect_dataloader, extractor)
    
    # Train System 1/2 components
    print("Training System 1/2 components")
    bias_representer.train_system_components(
        prospect_dataloader, extractor, num_epochs=num_epochs_prospect
    )
    
    # Results dictionary
    comparison_results = {
        "approaches": {},
        "best_approach": None,
        "best_accuracy": 0.0,
        "environment": check_compatibility()
    }
    
    # Create comparison directory
    comparison_dir = os.path.join(output_dir, "hybrid_comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Classical model types to test
    classical_model_types = ["random_forest", "gradient_boosting", "logistic_regression"]
    
    # Compare each hybrid approach
    for approach_info in HYBRID_APPROACHES:
        approach_name = approach_info["name"]
        print(f"\n{'='*80}\nComparing hybrid approach: {approach_name}\n{'='*80}")
        
        # Approach results dictionary
        approach_results = {
            "classical_models": {},
            "best_classical_model": None,
            "best_accuracy": 0.0,
            "training_time": 0.0,
            "inference_time": 0.0,
            "description": approach_info["description"]
        }
        
        # Test each classical model type
        for classical_model_type in classical_model_types:
            print(f"\nTesting classical model type: {classical_model_type}")
            
            # Cross-validation
            cv_results = []
            kf = KFold(n_splits=cross_validation_folds, shuffle=True, random_state=seed)
            
            # Convert dataset to list for splitting
            anes_data_list = [anes_dataset[i] for i in range(len(anes_dataset))]
            
            fold_accuracies = []
            fold_reports = []
            fold_training_times = []
            fold_inference_times = []
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(anes_data_list)):
                print(f"\nFold {fold+1}/{cross_validation_folds}")
                
                # Create train and validation datasets
                train_data = [anes_data_list[i] for i in train_idx]
                val_data = [anes_data_list[i] for i in val_idx]
                
                # Create custom datasets
                class CustomDataset(torch.utils.data.Dataset):
                    def __init__(self, data):
                        self.data = data
                    
                    def __len__(self):
                        return len(self.data)
                    
                    def __getitem__(self, idx):
                        return self.data[idx]
                
                train_dataset = CustomDataset(train_data)
                val_dataset = CustomDataset(val_data)
                
                # Create dataloaders
                train_dataloader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True
                )
                val_dataloader = torch.utils.data.DataLoader(
                    val_dataset, batch_size=batch_size, shuffle=False
                )
                
                # Initialize ANES classifier
                anes_feature_dim = len(anes_dataset[0]['anes_features'])
                anes_classifier = ProspectTheoryANESClassifier(
                    anes_feature_dim,
                    extractor.get_hidden_size(),
                    len(prospect_dataset.bias_names),
                    combined_hidden_dim=512,
                    num_classes=len(target_classes)
                ).to(device)
                
                # Initialize hybrid model
                hybrid_model = HybridModel(
                    approach=approach_name,
                    deep_model=anes_classifier,
                    classical_model_type=classical_model_type,
                    num_classes=len(target_classes),
                    device=device
                )
                
                # Train hybrid model
                start_time = time.time()
                hybrid_model.fit(
                    train_dataloader,
                    val_dataloader,
                    extractor,
                    bias_representer,
                    num_epochs=num_epochs_anes,
                    lr=learning_rate
                )
                training_time = time.time() - start_time
                
                # Evaluate hybrid model
                start_time = time.time()
                pred_probs, pred_classes = hybrid_model.predict(
                    val_dataloader, extractor, bias_representer
                )
                inference_time = (time.time() - start_time) / len(val_dataset)
                
                # Get true labels
                true_labels = np.array([val_dataset[i]['target'].item() for i in range(len(val_dataset))])
                
                # Calculate metrics
                accuracy = np.mean(pred_classes == true_labels)
                report = classification_report(true_labels, pred_classes, target_names=target_classes, output_dict=True)
                
                # Save results
                fold_accuracies.append(accuracy)
                fold_reports.append(report)
                fold_training_times.append(training_time)
                fold_inference_times.append(inference_time)
                
                print(f"Fold {fold+1} - Accuracy: {accuracy:.4f}")
            
            # Calculate average results
            avg_accuracy = np.mean(fold_accuracies)
            std_accuracy = np.std(fold_accuracies)
            avg_training_time = np.mean(fold_training_times)
            avg_inference_time = np.mean(fold_inference_times)
            
            # Save results
            approach_results["classical_models"][classical_model_type] = {
                "accuracy": avg_accuracy,
                "std_accuracy": std_accuracy,
                "fold_accuracies": fold_accuracies,
                "fold_reports": fold_reports,
                "training_time": avg_training_time,
                "inference_time": avg_inference_time
            }
            
            # Update best classical model
            if avg_accuracy > approach_results["best_accuracy"]:
                approach_results["best_accuracy"] = avg_accuracy
                approach_results["best_classical_model"] = classical_model_type
                approach_results["training_time"] = avg_training_time
                approach_results["inference_time"] = avg_inference_time
            
            print(f"Classical model {classical_model_type} - Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
        
        # Save approach results
        comparison_results["approaches"][approach_name] = approach_results
        
        # Update best approach
        if approach_results["best_accuracy"] > comparison_results["best_accuracy"]:
            comparison_results["best_accuracy"] = approach_results["best_accuracy"]
            comparison_results["best_approach"] = approach_name
    
    # Save comparison results
    comparison_results_path = os.path.join(comparison_dir, "hybrid_comparison_results.json")
    with open(comparison_results_path, "w") as f:
        json.dump(comparison_results, f, indent=2)
    
    # Generate comparison visualizations
    generate_hybrid_comparison_visualizations(comparison_results, comparison_dir)
    
    # Generate comparison report
    generate_hybrid_comparison_report(comparison_results, comparison_dir)
    
    print("\nHybrid approach comparison complete!")
    print(f"Best approach: {comparison_results['best_approach']}")
    print(f"Best accuracy: {comparison_results['best_accuracy']:.4f}")
    
    return comparison_results


def generate_hybrid_comparison_visualizations(comparison_results: Dict, output_dir: str):
    """
    Generate visualizations comparing different hybrid approaches.
    
    Args:
        comparison_results: Dictionary of comparison results
        output_dir: Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for plotting
    approaches = []
    classical_models = []
    accuracies = []
    std_accuracies = []
    training_times = []
    inference_times = []
    descriptions = []
    
    for approach_name, approach_results in comparison_results["approaches"].items():
        for classical_model_type, model_results in approach_results["classical_models"].items():
            approaches.append(approach_name)
            classical_models.append(classical_model_type)
            accuracies.append(model_results["accuracy"])
            std_accuracies.append(model_results["std_accuracy"])
            training_times.append(model_results["training_time"])
            inference_times.append(model_results["inference_time"])
            descriptions.append(approach_results["description"])
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame({
        "Approach": approaches,
        "Classical Model": classical_models,
        "Accuracy": accuracies,
        "Std Accuracy": std_accuracies,
        "Training Time (s)": training_times,
        "Inference Time (s)": inference_times,
        "Description": descriptions
    })
    
    # Save DataFrame
    df.to_csv(os.path.join(output_dir, "hybrid_comparison.csv"), index=False)
    
    # Plot approach comparison
    plt.figure(figsize=(14, 10))
    sns.barplot(x="Approach", y="Accuracy", data=df)
    plt.title("Hybrid Approach Comparison")
    plt.xlabel("Approach")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "approach_comparison.png"))
    
    # Plot classical model comparison
    plt.figure(figsize=(14, 10))
    sns.barplot(x="Classical Model", y="Accuracy", data=df)
    plt.title("Classical Model Comparison in Hybrid Approaches")
    plt.xlabel("Classical Model")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "classical_model_comparison.png"))
    
    # Plot approach vs. classical model
    plt.figure(figsize=(16, 12))
    sns.catplot(
        x="Approach", y="Accuracy", hue="Classical Model",
        data=df, kind="bar", height=8, aspect=2
    )
    plt.title("Approach vs. Classical Model Comparison")
    plt.xlabel("Approach")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "approach_vs_classical_model.png"))
    
    # Plot best classical model for each approach
    best_by_approach = df.loc[df.groupby("Approach")["Accuracy"].idxmax()]
    plt.figure(figsize=(14, 10))
    sns.barplot(x="Approach", y="Accuracy", hue="Classical Model", data=best_by_approach)
    plt.title("Best Classical Model for Each Approach")
    plt.xlabel("Approach")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "best_classical_model_by_approach.png"))
    
    # Plot accuracy vs. training time
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x="Training Time (s)", y="Accuracy", hue="Approach", style="Classical Model", s=100, data=df)
    plt.title("Accuracy vs. Training Time")
    plt.xlabel("Training Time (s)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_vs_training_time.png"))
    
    # Plot accuracy vs. inference time
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x="Inference Time (s)", y="Accuracy", hue="Approach", style="Classical Model", s=100, data=df)
    plt.title("Accuracy vs. Inference Time")
    plt.xlabel("Inference Time (s)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_vs_inference_time.png"))


def generate_hybrid_comparison_report(comparison_results: Dict, output_dir: str):
    """
    Generate a comprehensive comparison report of hybrid approach results.
    
    Args:
        comparison_results: Dictionary of comparison results
        output_dir: Directory to save report
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Start report
    report = "# Hybrid Approaches for Prospect Theory Analysis\n\n"
    
    # Add environment information
    report += "## Environment Information\n\n"
    for key, value in comparison_results["environment"].items():
        report += f"- **{key}**: {value}\n"
    report += "\n"
    
    # Add overall results
    report += "## Overall Results\n\n"
    report += f"- **Best Approach**: {comparison_results['best_approach']}\n"
    report += f"- **Best Accuracy**: {comparison_results['best_accuracy']:.4f}\n\n"
    
    # Add approach comparison
    report += "## Approach Comparison\n\n"
    report += "| Approach | Description | Best Classical Model | Accuracy | Training Time (s) | Inference Time (s) |\n"
    report += "|----------|-------------|---------------------|----------|-------------------|-------------------|\n"
    
    for approach_name, approach_results in comparison_results["approaches"].items():
        report += f"| {approach_name} | {approach_results['description']} | {approach_results['best_classical_model']} | {approach_results['best_accuracy']:.4f} | {approach_results['training_time']:.2f} | {approach_results['inference_time']:.4f} |\n"
    
    report += "\n"
    
    # Add detailed results for each approach
    report += "## Detailed Results\n\n"
    
    for approach_name, approach_results in comparison_results["approaches"].items():
        report += f"### {approach_name}\n\n"
        report += f"**Description**: {approach_results['description']}\n\n"
        
        report += "| Classical Model | Accuracy | Std Accuracy | Training Time (s) | Inference Time (s) |\n"
        report += "|----------------|----------|--------------|-------------------|-------------------|\n"
        
        for classical_model_type, model_results in approach_results["classical_models"].items():
            report += f"| {classical_model_type} | {model_results['accuracy']:.4f} | {model_results['std_accuracy']:.4f} | {model_results['training_time']:.2f} | {model_results['inference_time']:.4f} |\n"
        
        report += "\n"
    
    # Add visualizations
    report += "## Visualizations\n\n"
    report += "### Approach Comparison\n\n"
    report += "![Approach Comparison](approach_comparison.png)\n\n"
    report += "### Classical Model Comparison\n\n"
    report += "![Classical Model Comparison](classical_model_comparison.png)\n\n"
    report += "### Approach vs. Classical Model\n\n"
    report += "![Approach vs. Classical Model](approach_vs_classical_model.png)\n\n"
    report += "### Best Classical Model for Each Approach\n\n"
    report += "![Best Classical Model for Each Approach](best_classical_model_by_approach.png)\n\n"
    report += "### Accuracy vs. Training Time\n\n"
    report += "![Accuracy vs. Training Time](accuracy_vs_training_time.png)\n\n"
    report += "### Accuracy vs. Inference Time\n\n"
    report += "![Accuracy vs. Inference Time](accuracy_vs_inference_time.png)\n\n"
    
    # Add conclusion
    report += "## Conclusion\n\n"
    report += f"Based on the comparison results, the best hybrid approach for the Prospect Theory analysis is **{comparison_results['best_approach']}**, "
    report += f"achieving an accuracy of **{comparison_results['best_accuracy']:.4f}**.\n\n"
    
    # Add recommendations
    report += "## Recommendations\n\n"
    report += "1. **Approach Selection**: Use the best performing hybrid approach for the final pipeline.\n"
    report += "2. **Classical Model**: Select the optimal classical model for the chosen approach.\n"
    report += "3. **Ensemble Tuning**: Fine-tune the ensemble weights for better performance.\n"
    report += "4. **Feature Engineering**: Experiment with different feature engineering techniques for the classical component.\n"
    report += "5. **Interpretability**: Leverage the interpretability of the hybrid approach to gain insights into the importance of different features.\n\n"
    
    # Save report
    report_path = os.path.join(output_dir, "hybrid_comparison_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    
    print(f"Hybrid comparison report saved to {report_path}")


def main():
    """Main function to parse arguments and run the comparison."""
    parser = argparse.ArgumentParser(description="Hybrid Approaches for Prospect Theory Analysis")
    
    parser.add_argument("--anes_path", type=str, default=DEFAULT_ANES_PATH,
                        help="Path to ANES JSON files")
    parser.add_argument("--prospect_path", type=str, default=None,
                        help="Path to Prospect Theory dataset")
    parser.add_argument("--llm_model", type=str, default="roberta-large",
                        help="Name of the LLM model to use")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save outputs")
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR,
                        help="Directory to save models")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Directory to save processed data")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Batch size for training")
    parser.add_argument("--num_epochs_prospect", type=int, default=DEFAULT_NUM_EPOCHS_PROSPECT,
                        help="Number of epochs for Prospect Theory training")
    parser.add_argument("--num_epochs_anes", type=int, default=DEFAULT_NUM_EPOCHS_ANES,
                        help="Number of epochs for ANES classifier training")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_LEARNING_RATE,
                        help="Learning rate for training")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed")
    parser.add_argument("--cv_folds", type=int, default=5,
                        help="Number of cross-validation folds")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run on (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Run the comparison
    compare_hybrid_approaches(
        anes_json_path=args.anes_path,
        prospect_theory_path=args.prospect_path,
        llm_model_name=args.llm_model,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_epochs_prospect=args.num_epochs_prospect,
        num_epochs_anes=args.num_epochs_anes,
        learning_rate=args.learning_rate,
        seed=args.seed,
        cross_validation_folds=args.cv_folds,
        device=args.device
    )


if __name__ == "__main__":
    main()

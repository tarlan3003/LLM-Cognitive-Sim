"""
Classical Machine Learning Models for Prospect Theory Analysis

This script implements and compares classical machine learning models
for the Prospect Theory analysis of ANES data, providing an alternative
to deep learning approaches.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union
import time
import argparse
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

# Import modules
from dataset import ProspectTheoryDataset, convert_anes_to_dataset
from llm_extractor import HiddenLayerExtractor
from bias_representer import CognitiveBiasRepresenter
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
DEFAULT_SEED = 42

# Models to compare
CLASSICAL_MODELS = {
    "logistic_regression": {
        "model": LogisticRegression,
        "params": {
            "C": [0.1, 1.0, 10.0],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"],
            "max_iter": [1000]
        }
    },
    "random_forest": {
        "model": RandomForestClassifier,
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        }
    },
    "gradient_boosting": {
        "model": GradientBoostingClassifier,
        "params": {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5]
        }
    },
    "svm": {
        "model": SVC,
        "params": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["linear", "rbf"],
            "probability": [True]
        }
    },
    "naive_bayes": {
        "model": GaussianNB,
        "params": {}
    }
}

# Feature engineering strategies
FEATURE_ENGINEERING_STRATEGIES = [
    {"name": "raw", "pca": None, "select_k": None},
    {"name": "pca_50", "pca": 50, "select_k": None},
    {"name": "select_k_50", "pca": None, "select_k": 50},
    {"name": "pca_20_select_k_50", "pca": 20, "select_k": 50}
]


def compare_classical_models(
    anes_json_path: str,
    prospect_theory_path: Optional[str] = None,
    llm_model_name: str = "roberta-large",
    llm_layers: List[int] = [-1, -2, -4, -8],
    output_dir: str = DEFAULT_OUTPUT_DIR,
    model_dir: str = DEFAULT_MODEL_DIR,
    data_dir: str = DEFAULT_DATA_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_epochs_prospect: int = DEFAULT_NUM_EPOCHS_PROSPECT,
    seed: int = DEFAULT_SEED,
    cross_validation_folds: int = 5,
    target_variable: str = "V241049",  # WHO WOULD R VOTE FOR: HARRIS VS TRUMP
    target_classes: List[str] = ["Donald Trump", "Kamala Harris"],
    device: Optional[str] = None
) -> Dict:
    """
    Compare classical machine learning models for the Prospect Theory analysis of ANES data.
    
    Args:
        anes_json_path: Path to ANES JSON files
        prospect_theory_path: Path to Prospect Theory dataset
        llm_model_name: Name of the LLM model to use for feature extraction
        llm_layers: Layers to extract from the LLM
        output_dir: Directory to save outputs
        model_dir: Directory to save models
        data_dir: Directory to save processed data
        batch_size: Batch size for training
        num_epochs_prospect: Number of epochs for Prospect Theory training
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
    
    # Extract features for classical models
    print("Extracting features for classical models")
    X_anes_features = []
    X_bias_scores = []
    X_system_weights = []
    y_targets = []
    
    for i in tqdm(range(len(anes_dataset))):
        sample = anes_dataset[i]
        text = sample['text']
        anes_features = sample['anes_features'].numpy()
        target = sample['target'].item()
        
        # Extract activations
        activations = extractor.extract_activations([text])
        
        # Get bias scores and system representations
        bias_scores = bias_representer.get_bias_scores(activations).cpu().numpy()[0]
        weighted_system_rep, system_weights = bias_representer.get_system_representations(activations)
        system_weights = system_weights.cpu().numpy()[0]
        
        X_anes_features.append(anes_features)
        X_bias_scores.append(bias_scores)
        X_system_weights.append(system_weights)
        y_targets.append(target)
    
    X_anes_features = np.array(X_anes_features)
    X_bias_scores = np.array(X_bias_scores)
    X_system_weights = np.array(X_system_weights)
    y_targets = np.array(y_targets)
    
    # Create feature combinations
    feature_combinations = {
        "anes_only": X_anes_features,
        "bias_only": X_bias_scores,
        "system_only": X_system_weights,
        "anes_bias": np.hstack([X_anes_features, X_bias_scores]),
        "anes_system": np.hstack([X_anes_features, X_system_weights]),
        "bias_system": np.hstack([X_bias_scores, X_system_weights]),
        "all_features": np.hstack([X_anes_features, X_bias_scores, X_system_weights])
    }
    
    # Results dictionary
    comparison_results = {
        "models": {},
        "best_model": None,
        "best_feature_set": None,
        "best_engineering": None,
        "best_accuracy": 0.0,
        "environment": check_compatibility()
    }
    
    # Create comparison directory
    comparison_dir = os.path.join(output_dir, "classical_comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Compare each model
    for model_name, model_info in CLASSICAL_MODELS.items():
        print(f"\n{'='*80}\nComparing model: {model_name}\n{'='*80}")
        
        model_class = model_info["model"]
        param_grid = model_info["params"]
        
        # Model results dictionary
        model_results = {
            "feature_sets": {},
            "best_feature_set": None,
            "best_engineering": None,
            "best_accuracy": 0.0,
            "training_time": 0.0,
            "inference_time": 0.0
        }
        
        # Test each feature set
        for feature_set_name, X in feature_combinations.items():
            print(f"\nTesting feature set: {feature_set_name}")
            
            # Feature set results dictionary
            feature_set_results = {
                "engineering_strategies": {},
                "best_engineering": None,
                "best_accuracy": 0.0
            }
            
            # Test each feature engineering strategy
            for strategy in FEATURE_ENGINEERING_STRATEGIES:
                print(f"Testing feature engineering strategy: {strategy['name']}")
                
                # Create pipeline steps
                steps = [("scaler", StandardScaler())]
                
                if strategy["pca"] is not None:
                    steps.append(("pca", PCA(n_components=min(strategy["pca"], X.shape[1]))))
                
                if strategy["select_k"] is not None:
                    steps.append(("select_k", SelectKBest(f_classif, k=min(strategy["select_k"], X.shape[1]))))
                
                steps.append(("model", model_class()))
                
                # Create pipeline
                pipeline = Pipeline(steps)
                
                # Create parameter grid
                pipeline_param_grid = {f"model__{param}": values for param, values in param_grid.items()}
                
                # Cross-validation
                cv_results = []
                kf = KFold(n_splits=cross_validation_folds, shuffle=True, random_state=seed)
                
                fold_accuracies = []
                fold_reports = []
                fold_best_params = []
                fold_feature_importances = []
                
                for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                    print(f"\nFold {fold+1}/{cross_validation_folds}")
                    
                    # Split data
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y_targets[train_idx], y_targets[val_idx]
                    
                    # Grid search
                    start_time = time.time()
                    grid_search = GridSearchCV(
                        pipeline,
                        pipeline_param_grid,
                        cv=3,
                        scoring="accuracy",
                        n_jobs=-1
                    )
                    grid_search.fit(X_train, y_train)
                    training_time = time.time() - start_time
                    
                    # Best model
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                    
                    # Evaluate
                    start_time = time.time()
                    y_pred = best_model.predict(X_val)
                    inference_time = (time.time() - start_time) / len(X_val)
                    
                    # Calculate metrics
                    accuracy = np.mean(y_pred == y_val)
                    report = classification_report(y_val, y_pred, target_names=target_classes, output_dict=True)
                    
                    # Get feature importances if available
                    feature_importances = None
                    if hasattr(best_model.named_steps["model"], "coef_"):
                        feature_importances = best_model.named_steps["model"].coef_[0]
                    elif hasattr(best_model.named_steps["model"], "feature_importances_"):
                        feature_importances = best_model.named_steps["model"].feature_importances_
                    
                    # Save results
                    fold_accuracies.append(accuracy)
                    fold_reports.append(report)
                    fold_best_params.append(best_params)
                    if feature_importances is not None:
                        fold_feature_importances.append(feature_importances)
                    
                    print(f"Fold {fold+1} - Accuracy: {accuracy:.4f}")
                
                # Calculate average results
                avg_accuracy = np.mean(fold_accuracies)
                std_accuracy = np.std(fold_accuracies)
                
                # Save results
                feature_set_results["engineering_strategies"][strategy["name"]] = {
                    "accuracy": avg_accuracy,
                    "std_accuracy": std_accuracy,
                    "fold_accuracies": fold_accuracies,
                    "fold_reports": fold_reports,
                    "fold_best_params": fold_best_params,
                    "fold_feature_importances": fold_feature_importances if fold_feature_importances else None,
                    "training_time": training_time,
                    "inference_time": inference_time
                }
                
                # Update best engineering strategy
                if avg_accuracy > feature_set_results["best_accuracy"]:
                    feature_set_results["best_accuracy"] = avg_accuracy
                    feature_set_results["best_engineering"] = strategy["name"]
                
                print(f"Strategy {strategy['name']} - Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
            
            # Save feature set results
            model_results["feature_sets"][feature_set_name] = feature_set_results
            
            # Update best feature set
            if feature_set_results["best_accuracy"] > model_results["best_accuracy"]:
                model_results["best_accuracy"] = feature_set_results["best_accuracy"]
                model_results["best_feature_set"] = feature_set_name
                model_results["best_engineering"] = feature_set_results["best_engineering"]
                model_results["training_time"] = feature_set_results["engineering_strategies"][feature_set_results["best_engineering"]]["training_time"]
                model_results["inference_time"] = feature_set_results["engineering_strategies"][feature_set_results["best_engineering"]]["inference_time"]
        
        # Save model results
        comparison_results["models"][model_name] = model_results
        
        # Update best model
        if model_results["best_accuracy"] > comparison_results["best_accuracy"]:
            comparison_results["best_accuracy"] = model_results["best_accuracy"]
            comparison_results["best_model"] = model_name
            comparison_results["best_feature_set"] = model_results["best_feature_set"]
            comparison_results["best_engineering"] = model_results["best_engineering"]
    
    # Save comparison results
    comparison_results_path = os.path.join(comparison_dir, "classical_comparison_results.json")
    with open(comparison_results_path, "w") as f:
        json.dump(comparison_results, f, indent=2)
    
    # Generate comparison visualizations
    generate_classical_comparison_visualizations(comparison_results, comparison_dir)
    
    # Generate comparison report
    generate_classical_comparison_report(comparison_results, comparison_dir)
    
    print("\nClassical model comparison complete!")
    print(f"Best model: {comparison_results['best_model']}")
    print(f"Best feature set: {comparison_results['best_feature_set']}")
    print(f"Best engineering strategy: {comparison_results['best_engineering']}")
    print(f"Best accuracy: {comparison_results['best_accuracy']:.4f}")
    
    return comparison_results


def generate_classical_comparison_visualizations(comparison_results: Dict, output_dir: str):
    """
    Generate visualizations comparing different classical models and feature sets.
    
    Args:
        comparison_results: Dictionary of comparison results
        output_dir: Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for plotting
    models = []
    feature_sets = []
    engineering_strategies = []
    accuracies = []
    std_accuracies = []
    training_times = []
    inference_times = []
    
    for model_name, model_results in comparison_results["models"].items():
        for feature_set_name, feature_set_results in model_results["feature_sets"].items():
            for strategy_name, strategy_results in feature_set_results["engineering_strategies"].items():
                models.append(model_name)
                feature_sets.append(feature_set_name)
                engineering_strategies.append(strategy_name)
                accuracies.append(strategy_results["accuracy"])
                std_accuracies.append(strategy_results["std_accuracy"])
                training_times.append(strategy_results["training_time"])
                inference_times.append(strategy_results["inference_time"])
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame({
        "Model": models,
        "Feature Set": feature_sets,
        "Engineering Strategy": engineering_strategies,
        "Accuracy": accuracies,
        "Std Accuracy": std_accuracies,
        "Training Time (s)": training_times,
        "Inference Time (s)": inference_times
    })
    
    # Save DataFrame
    df.to_csv(os.path.join(output_dir, "classical_comparison.csv"), index=False)
    
    # Plot model comparison
    plt.figure(figsize=(14, 10))
    sns.barplot(x="Model", y="Accuracy", data=df)
    plt.title("Classical Model Comparison")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"))
    
    # Plot feature set comparison
    plt.figure(figsize=(14, 10))
    sns.barplot(x="Feature Set", y="Accuracy", data=df)
    plt.title("Feature Set Comparison")
    plt.xlabel("Feature Set")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_set_comparison.png"))
    
    # Plot engineering strategy comparison
    plt.figure(figsize=(14, 10))
    sns.barplot(x="Engineering Strategy", y="Accuracy", data=df)
    plt.title("Engineering Strategy Comparison")
    plt.xlabel("Engineering Strategy")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "engineering_strategy_comparison.png"))
    
    # Plot model vs. feature set
    plt.figure(figsize=(16, 12))
    sns.catplot(
        x="Model", y="Accuracy", hue="Feature Set",
        data=df, kind="bar", height=8, aspect=2
    )
    plt.title("Model vs. Feature Set Comparison")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_vs_feature_set.png"))
    
    # Plot best model for each feature set
    best_by_feature_set = df.loc[df.groupby("Feature Set")["Accuracy"].idxmax()]
    plt.figure(figsize=(14, 10))
    sns.barplot(x="Feature Set", y="Accuracy", hue="Model", data=best_by_feature_set)
    plt.title("Best Model for Each Feature Set")
    plt.xlabel("Feature Set")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "best_model_by_feature_set.png"))
    
    # Plot best feature set for each model
    best_by_model = df.loc[df.groupby("Model")["Accuracy"].idxmax()]
    plt.figure(figsize=(14, 10))
    sns.barplot(x="Model", y="Accuracy", hue="Feature Set", data=best_by_model)
    plt.title("Best Feature Set for Each Model")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "best_feature_set_by_model.png"))
    
    # Plot accuracy vs. training time
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x="Training Time (s)", y="Accuracy", hue="Model", style="Feature Set", s=100, data=df)
    plt.title("Accuracy vs. Training Time")
    plt.xlabel("Training Time (s)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_vs_training_time.png"))
    
    # Plot accuracy vs. inference time
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x="Inference Time (s)", y="Accuracy", hue="Model", style="Feature Set", s=100, data=df)
    plt.title("Accuracy vs. Inference Time")
    plt.xlabel("Inference Time (s)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_vs_inference_time.png"))


def generate_classical_comparison_report(comparison_results: Dict, output_dir: str):
    """
    Generate a comprehensive comparison report of classical model results.
    
    Args:
        comparison_results: Dictionary of comparison results
        output_dir: Directory to save report
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Start report
    report = "# Classical Machine Learning Models for Prospect Theory Analysis\n\n"
    
    # Add environment information
    report += "## Environment Information\n\n"
    for key, value in comparison_results["environment"].items():
        report += f"- **{key}**: {value}\n"
    report += "\n"
    
    # Add overall results
    report += "## Overall Results\n\n"
    report += f"- **Best Model**: {comparison_results['best_model']}\n"
    report += f"- **Best Feature Set**: {comparison_results['best_feature_set']}\n"
    report += f"- **Best Engineering Strategy**: {comparison_results['best_engineering']}\n"
    report += f"- **Best Accuracy**: {comparison_results['best_accuracy']:.4f}\n\n"
    
    # Add model comparison
    report += "## Model Comparison\n\n"
    report += "| Model | Best Feature Set | Best Engineering | Accuracy | Training Time (s) | Inference Time (s) |\n"
    report += "|-------|-----------------|-----------------|----------|-------------------|-------------------|\n"
    
    for model_name, model_results in comparison_results["models"].items():
        report += f"| {model_name} | {model_results['best_feature_set']} | {model_results['best_engineering']} | {model_results['best_accuracy']:.4f} | {model_results['training_time']:.2f} | {model_results['inference_time']:.4f} |\n"
    
    report += "\n"
    
    # Add feature set comparison
    report += "## Feature Set Comparison\n\n"
    
    # Calculate average accuracy for each feature set
    feature_set_accuracies = {}
    for model_name, model_results in comparison_results["models"].items():
        for feature_set_name, feature_set_results in model_results["feature_sets"].items():
            if feature_set_name not in feature_set_accuracies:
                feature_set_accuracies[feature_set_name] = []
            feature_set_accuracies[feature_set_name].append(feature_set_results["best_accuracy"])
    
    report += "| Feature Set | Average Accuracy | Best Model | Best Accuracy |\n"
    report += "|------------|------------------|------------|---------------|\n"
    
    for feature_set_name, accuracies in feature_set_accuracies.items():
        avg_accuracy = np.mean(accuracies)
        
        # Find best model for this feature set
        best_model = None
        best_accuracy = 0.0
        for model_name, model_results in comparison_results["models"].items():
            if feature_set_name in model_results["feature_sets"]:
                accuracy = model_results["feature_sets"][feature_set_name]["best_accuracy"]
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model_name
        
        report += f"| {feature_set_name} | {avg_accuracy:.4f} | {best_model} | {best_accuracy:.4f} |\n"
    
    report += "\n"
    
    # Add engineering strategy comparison
    report += "## Engineering Strategy Comparison\n\n"
    
    # Calculate average accuracy for each engineering strategy
    strategy_accuracies = {}
    for model_name, model_results in comparison_results["models"].items():
        for feature_set_name, feature_set_results in model_results["feature_sets"].items():
            for strategy_name, strategy_results in feature_set_results["engineering_strategies"].items():
                if strategy_name not in strategy_accuracies:
                    strategy_accuracies[strategy_name] = []
                strategy_accuracies[strategy_name].append(strategy_results["accuracy"])
    
    report += "| Engineering Strategy | Average Accuracy |\n"
    report += "|---------------------|------------------|\n"
    
    for strategy_name, accuracies in strategy_accuracies.items():
        avg_accuracy = np.mean(accuracies)
        report += f"| {strategy_name} | {avg_accuracy:.4f} |\n"
    
    report += "\n"
    
    # Add detailed results for each model
    report += "## Detailed Results\n\n"
    
    for model_name, model_results in comparison_results["models"].items():
        report += f"### {model_name}\n\n"
        
        report += "| Feature Set | Best Engineering | Accuracy | Std Accuracy |\n"
        report += "|------------|-----------------|----------|-------------|\n"
        
        for feature_set_name, feature_set_results in model_results["feature_sets"].items():
            best_engineering = feature_set_results["best_engineering"]
            best_results = feature_set_results["engineering_strategies"][best_engineering]
            
            report += f"| {feature_set_name} | {best_engineering} | {best_results['accuracy']:.4f} | {best_results['std_accuracy']:.4f} |\n"
        
        report += "\n"
    
    # Add visualizations
    report += "## Visualizations\n\n"
    report += "### Model Comparison\n\n"
    report += "![Model Comparison](model_comparison.png)\n\n"
    report += "### Feature Set Comparison\n\n"
    report += "![Feature Set Comparison](feature_set_comparison.png)\n\n"
    report += "### Engineering Strategy Comparison\n\n"
    report += "![Engineering Strategy Comparison](engineering_strategy_comparison.png)\n\n"
    report += "### Model vs. Feature Set\n\n"
    report += "![Model vs. Feature Set](model_vs_feature_set.png)\n\n"
    report += "### Best Model for Each Feature Set\n\n"
    report += "![Best Model for Each Feature Set](best_model_by_feature_set.png)\n\n"
    report += "### Best Feature Set for Each Model\n\n"
    report += "![Best Feature Set for Each Model](best_feature_set_by_model.png)\n\n"
    report += "### Accuracy vs. Training Time\n\n"
    report += "![Accuracy vs. Training Time](accuracy_vs_training_time.png)\n\n"
    report += "### Accuracy vs. Inference Time\n\n"
    report += "![Accuracy vs. Inference Time](accuracy_vs_inference_time.png)\n\n"
    
    # Add conclusion
    report += "## Conclusion\n\n"
    report += f"Based on the comparison results, the best classical model for the Prospect Theory analysis is **{comparison_results['best_model']}** "
    report += f"with feature set **{comparison_results['best_feature_set']}** and engineering strategy **{comparison_results['best_engineering']}**, "
    report += f"achieving an accuracy of **{comparison_results['best_accuracy']:.4f}**.\n\n"
    
    # Add recommendations
    report += "## Recommendations\n\n"
    report += "1. **Model Selection**: Use the best performing model for the final pipeline.\n"
    report += "2. **Feature Set**: Use the optimal feature set for best results.\n"
    report += "3. **Engineering Strategy**: Apply the best engineering strategy to the features.\n"
    report += "4. **Hybrid Approach**: Consider combining classical models with deep learning for even better results.\n"
    report += "5. **Interpretability**: Leverage the interpretability of classical models to gain insights into the importance of different features.\n\n"
    
    # Save report
    report_path = os.path.join(output_dir, "classical_comparison_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    
    print(f"Classical comparison report saved to {report_path}")


def main():
    """Main function to parse arguments and run the comparison."""
    parser = argparse.ArgumentParser(description="Classical Machine Learning Models for Prospect Theory Analysis")
    
    parser.add_argument("--anes_path", type=str, default=DEFAULT_ANES_PATH,
                        help="Path to ANES JSON files")
    parser.add_argument("--prospect_path", type=str, default=None,
                        help="Path to Prospect Theory dataset")
    parser.add_argument("--llm_model", type=str, default="roberta-large",
                        help="Name of the LLM model to use for feature extraction")
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
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Random seed")
    parser.add_argument("--cv_folds", type=int, default=5,
                        help="Number of cross-validation folds")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run on (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Run the comparison
    compare_classical_models(
        anes_json_path=args.anes_path,
        prospect_theory_path=args.prospect_path,
        llm_model_name=args.llm_model,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_epochs_prospect=args.num_epochs_prospect,
        seed=args.seed,
        cross_validation_folds=args.cv_folds,
        device=args.device
    )


if __name__ == "__main__":
    main()

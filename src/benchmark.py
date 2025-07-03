"""
Model Benchmarking and Comparison for Prospect Theory LLM Pipeline

This script benchmarks multiple state-of-the-art language models and approaches
for the Prospect Theory analysis of ANES data, comparing performance and interpretability.
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

# Models to benchmark
MODELS_TO_BENCHMARK = [
    # Base models
    "roberta-base",
    "bert-base-uncased",
    # Large models
    "roberta-large",
    "bert-large-uncased",
    # Advanced models
    "microsoft/deberta-v3-base",
    "microsoft/deberta-v3-large",
    # Domain-specific models
    "cardiffnlp/twitter-roberta-base-sentiment",
    "nlptown/bert-base-multilingual-uncased-sentiment"
]

# Layer strategies to test
LAYER_STRATEGIES = [
    [-1],                  # Last layer only
    [-1, -2],              # Last two layers
    [-1, -2, -4, -8],      # Multiple layers at different depths
    [-4],                  # Middle layer only
    [-12, -8, -4, -1]      # Evenly spaced layers
]


def benchmark_models(
    anes_json_path: str,
    prospect_theory_path: Optional[str] = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    model_dir: str = DEFAULT_MODEL_DIR,
    data_dir: str = DEFAULT_DATA_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_epochs_prospect: int = DEFAULT_NUM_EPOCHS_PROSPECT,
    num_epochs_anes: int = DEFAULT_NUM_EPOCHS_ANES,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    seed: int = DEFAULT_SEED,
    models_to_test: List[str] = None,
    layer_strategies: List[List[int]] = None,
    cross_validation_folds: int = 5,
    target_variable: str = "V241049",  # WHO WOULD R VOTE FOR: HARRIS VS TRUMP
    target_classes: List[str] = ["Donald Trump", "Kamala Harris"],
    device: Optional[str] = None
) -> Dict:
    """
    Benchmark multiple models and layer strategies for the Prospect Theory LLM pipeline.
    
    Args:
        anes_json_path: Path to ANES JSON files
        prospect_theory_path: Path to Prospect Theory dataset
        output_dir: Directory to save outputs
        model_dir: Directory to save models
        data_dir: Directory to save processed data
        batch_size: Batch size for training
        num_epochs_prospect: Number of epochs for Prospect Theory training
        num_epochs_anes: Number of epochs for ANES classifier training
        learning_rate: Learning rate for training
        seed: Random seed
        models_to_test: List of model names to benchmark
        layer_strategies: List of layer strategies to test
        cross_validation_folds: Number of cross-validation folds
        target_variable: Target variable in ANES data
        target_classes: Target classes in ANES data
        device: Device to run on (cuda or cpu)
        
    Returns:
        Dictionary of benchmark results
    """
    # Set default values if not provided
    if models_to_test is None:
        models_to_test = MODELS_TO_BENCHMARK
    
    if layer_strategies is None:
        layer_strategies = LAYER_STRATEGIES
    
    # Set random seed
    set_seed(seed)
    
    # Create directory structure
    create_directory_structure(data_dir, model_dir, output_dir)
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
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
            num_examples=500  # Increased for better training
        )
    
    # Results dictionary
    benchmark_results = {
        "models": {},
        "best_model": None,
        "best_layers": None,
        "best_accuracy": 0.0,
        "environment": check_compatibility()
    }
    
    # Create comparison directory
    comparison_dir = os.path.join(output_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Benchmark each model
    for model_name in models_to_test:
        print(f"\n{'='*80}\nBenchmarking model: {model_name}\n{'='*80}")
        
        try:
            # Initialize tokenizer
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load datasets
            prospect_dataset = ProspectTheoryDataset(prospect_theory_path, tokenizer)
            anes_dataset = ProspectTheoryDataset(anes_output_path, tokenizer, is_anes=True)
            
            # Model results dictionary
            model_results = {
                "layer_strategies": {},
                "best_layers": None,
                "best_accuracy": 0.0,
                "training_time": 0.0,
                "inference_time": 0.0
            }
            
            # Test each layer strategy
            for layers in layer_strategies:
                layer_str = "_".join([str(abs(l)) for l in layers])
                print(f"\nTesting layer strategy: {layers}")
                
                # Create model directory
                model_specific_dir = os.path.join(model_dir, f"{model_name.replace('/', '_')}_{layer_str}")
                os.makedirs(model_specific_dir, exist_ok=True)
                
                # Create output directory
                output_specific_dir = os.path.join(output_dir, f"{model_name.replace('/', '_')}_{layer_str}")
                os.makedirs(output_specific_dir, exist_ok=True)
                
                try:
                    # Initialize extractor
                    start_time = time.time()
                    extractor = HiddenLayerExtractor(model_name, layers, device=device)
                    
                    # Create dataloaders
                    prospect_dataloader = torch.utils.data.DataLoader(
                        prospect_dataset, batch_size=batch_size, shuffle=True
                    )
                    
                    # Initialize bias representer
                    bias_representer = CognitiveBiasRepresenter(
                        extractor.get_hidden_size(),
                        prospect_dataset.bias_names,
                        device=device
                    )
                    
                    # Train CAVs
                    bias_representer.train_cavs(prospect_dataloader, extractor)
                    
                    # Train System 1/2 components
                    bias_representer.train_system_components(
                        prospect_dataloader, extractor, num_epochs=num_epochs_prospect
                    )
                    
                    # Cross-validation
                    cv_results = []
                    kf = KFold(n_splits=cross_validation_folds, shuffle=True, random_state=seed)
                    
                    # Convert dataset to list for splitting
                    anes_data_list = [anes_dataset[i] for i in range(len(anes_dataset))]
                    
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
                        
                        # Train ANES classifier
                        train_anes_classifier(
                            anes_classifier,
                            train_dataloader,
                            extractor,
                            bias_representer,
                            num_epochs=num_epochs_anes,
                            lr=learning_rate,
                            device=device
                        )
                        
                        # Evaluate ANES classifier
                        eval_results = evaluate_anes_classifier(
                            anes_classifier,
                            val_dataloader,
                            extractor,
                            bias_representer,
                            device=device,
                            target_names=target_classes
                        )
                        
                        cv_results.append(eval_results)
                    
                    # Calculate average results across folds
                    avg_results = {
                        "accuracy": np.mean([r["best_accuracy"] for r in cv_results]),
                        "std_accuracy": np.std([r["best_accuracy"] for r in cv_results]),
                        "avg_system_weights": np.mean([r["avg_system_weights"] for r in cv_results], axis=0),
                        "thresholded_results": {}
                    }
                    
                    # Calculate average thresholded results
                    for threshold in cv_results[0]["thresholded_results"]:
                        avg_results["thresholded_results"][threshold] = {
                            "accuracy": np.mean([r["thresholded_results"][threshold]["accuracy"] for r in cv_results]),
                            "std_accuracy": np.std([r["thresholded_results"][threshold]["accuracy"] for r in cv_results])
                        }
                    
                    # Save best model from last fold
                    bias_representer.save(os.path.join(model_specific_dir, "bias_representer.pt"))
                    anes_classifier.save(os.path.join(model_specific_dir, "anes_classifier.pt"))
                    
                    # Record training time
                    training_time = time.time() - start_time
                    
                    # Measure inference time
                    start_time = time.time()
                    with torch.no_grad():
                        for batch in val_dataloader:
                            texts = batch['text']
                            anes_features = batch['anes_features'].to(device)
                            
                            # Extract activations
                            activations = extractor.extract_activations(texts)
                            
                            # Get bias scores and system representations
                            bias_scores = bias_representer.get_bias_scores(activations).to(device)
                            weighted_system_rep, _ = bias_representer.get_system_representations(activations)
                            weighted_system_rep = weighted_system_rep.to(device)
                            
                            # Forward pass
                            _ = anes_classifier(anes_features, bias_scores, weighted_system_rep)
                    
                    inference_time = (time.time() - start_time) / len(val_dataloader)
                    
                    # Record results
                    model_results["layer_strategies"][str(layers)] = {
                        "accuracy": avg_results["accuracy"],
                        "std_accuracy": avg_results["std_accuracy"],
                        "thresholded_results": avg_results["thresholded_results"],
                        "avg_system_weights": avg_results["avg_system_weights"].tolist(),
                        "training_time": training_time,
                        "inference_time": inference_time,
                        "model_dir": model_specific_dir,
                        "output_dir": output_specific_dir
                    }
                    
                    # Update best layers
                    if avg_results["accuracy"] > model_results["best_accuracy"]:
                        model_results["best_accuracy"] = avg_results["accuracy"]
                        model_results["best_layers"] = layers
                        model_results["training_time"] = training_time
                        model_results["inference_time"] = inference_time
                    
                    # Generate visualizations for the best fold
                    best_fold_idx = np.argmax([r["best_accuracy"] for r in cv_results])
                    best_fold_results = cv_results[best_fold_idx]
                    
                    # Combine all feature names for visualization
                    feature_names = [f"anes_{i}" for i in range(anes_feature_dim)] + \
                                    prospect_dataset.bias_names + \
                                    [f"llm_{i}" for i in range(extractor.get_hidden_size())]
                    
                    generate_all_visualizations(
                        best_fold_results,
                        prospect_dataset.bias_names,
                        target_classes,
                        feature_names,
                        anes_classifier,
                        output_dir=output_specific_dir
                    )
                    
                    print(f"Layer strategy {layers} - Accuracy: {avg_results['accuracy']:.4f} ± {avg_results['std_accuracy']:.4f}")
                
                except Exception as e:
                    print(f"Error testing layer strategy {layers}: {e}")
                    model_results["layer_strategies"][str(layers)] = {
                        "error": str(e)
                    }
            
            # Record model results
            benchmark_results["models"][model_name] = model_results
            
            # Update best model
            if model_results["best_accuracy"] > benchmark_results["best_accuracy"]:
                benchmark_results["best_accuracy"] = model_results["best_accuracy"]
                benchmark_results["best_model"] = model_name
                benchmark_results["best_layers"] = model_results["best_layers"]
        
        except Exception as e:
            print(f"Error benchmarking model {model_name}: {e}")
            benchmark_results["models"][model_name] = {
                "error": str(e)
            }
    
    # Save benchmark results
    benchmark_results_path = os.path.join(comparison_dir, "benchmark_results.json")
    with open(benchmark_results_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    
    # Generate comparison visualizations
    generate_comparison_visualizations(benchmark_results, comparison_dir)
    
    # Generate comparison report
    generate_comparison_report(benchmark_results, comparison_dir)
    
    print("\nBenchmark complete!")
    print(f"Best model: {benchmark_results['best_model']}")
    print(f"Best layers: {benchmark_results['best_layers']}")
    print(f"Best accuracy: {benchmark_results['best_accuracy']:.4f}")
    
    return benchmark_results


def generate_comparison_visualizations(benchmark_results: Dict, output_dir: str):
    """
    Generate visualizations comparing different models and layer strategies.
    
    Args:
        benchmark_results: Dictionary of benchmark results
        output_dir: Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data for plotting
    models = []
    accuracies = []
    std_accuracies = []
    training_times = []
    inference_times = []
    layer_strategies = []
    
    for model_name, model_results in benchmark_results["models"].items():
        if "error" in model_results:
            continue
        
        for layers, layer_results in model_results["layer_strategies"].items():
            if "error" in layer_results:
                continue
            
            models.append(model_name)
            accuracies.append(layer_results["accuracy"])
            std_accuracies.append(layer_results["std_accuracy"])
            training_times.append(layer_results["training_time"])
            inference_times.append(layer_results["inference_time"])
            layer_strategies.append(layers)
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame({
        "Model": models,
        "Layer Strategy": layer_strategies,
        "Accuracy": accuracies,
        "Std Accuracy": std_accuracies,
        "Training Time (s)": training_times,
        "Inference Time (s)": inference_times
    })
    
    # Save DataFrame
    df.to_csv(os.path.join(output_dir, "benchmark_comparison.csv"), index=False)
    
    # Plot accuracy comparison
    plt.figure(figsize=(14, 10))
    sns.barplot(x="Model", y="Accuracy", hue="Layer Strategy", data=df)
    plt.title("Model Accuracy Comparison")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"))
    
    # Plot training time comparison
    plt.figure(figsize=(14, 10))
    sns.barplot(x="Model", y="Training Time (s)", hue="Layer Strategy", data=df)
    plt.title("Model Training Time Comparison")
    plt.xlabel("Model")
    plt.ylabel("Training Time (s)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_time_comparison.png"))
    
    # Plot inference time comparison
    plt.figure(figsize=(14, 10))
    sns.barplot(x="Model", y="Inference Time (s)", hue="Layer Strategy", data=df)
    plt.title("Model Inference Time Comparison")
    plt.xlabel("Model")
    plt.ylabel("Inference Time (s)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "inference_time_comparison.png"))
    
    # Plot accuracy vs. training time
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x="Training Time (s)", y="Accuracy", hue="Model", style="Layer Strategy", s=100, data=df)
    plt.title("Accuracy vs. Training Time")
    plt.xlabel("Training Time (s)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_vs_training_time.png"))
    
    # Plot accuracy vs. inference time
    plt.figure(figsize=(12, 10))
    sns.scatterplot(x="Inference Time (s)", y="Accuracy", hue="Model", style="Layer Strategy", s=100, data=df)
    plt.title("Accuracy vs. Inference Time")
    plt.xlabel("Inference Time (s)")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_vs_inference_time.png"))
    
    # Plot model ranking
    model_avg_acc = df.groupby("Model")["Accuracy"].mean().sort_values(ascending=False)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=model_avg_acc.index, y=model_avg_acc.values)
    plt.title("Model Ranking by Average Accuracy")
    plt.xlabel("Model")
    plt.ylabel("Average Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_ranking.png"))
    
    # Plot layer strategy ranking
    layer_avg_acc = df.groupby("Layer Strategy")["Accuracy"].mean().sort_values(ascending=False)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=layer_avg_acc.index, y=layer_avg_acc.values)
    plt.title("Layer Strategy Ranking by Average Accuracy")
    plt.xlabel("Layer Strategy")
    plt.ylabel("Average Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "layer_strategy_ranking.png"))


def generate_comparison_report(benchmark_results: Dict, output_dir: str):
    """
    Generate a comprehensive comparison report of benchmark results.
    
    Args:
        benchmark_results: Dictionary of benchmark results
        output_dir: Directory to save report
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Start report
    report = "# Prospect Theory LLM Benchmark Report\n\n"
    
    # Add environment information
    report += "## Environment Information\n\n"
    for key, value in benchmark_results["environment"].items():
        report += f"- **{key}**: {value}\n"
    report += "\n"
    
    # Add overall results
    report += "## Overall Results\n\n"
    report += f"- **Best Model**: {benchmark_results['best_model']}\n"
    report += f"- **Best Layer Strategy**: {benchmark_results['best_layers']}\n"
    report += f"- **Best Accuracy**: {benchmark_results['best_accuracy']:.4f}\n\n"
    
    # Add model comparison
    report += "## Model Comparison\n\n"
    report += "| Model | Best Layers | Accuracy | Training Time (s) | Inference Time (s) |\n"
    report += "|-------|------------|----------|-------------------|-------------------|\n"
    
    for model_name, model_results in benchmark_results["models"].items():
        if "error" in model_results:
            report += f"| {model_name} | Error | Error | Error | Error |\n"
        else:
            report += f"| {model_name} | {model_results['best_layers']} | {model_results['best_accuracy']:.4f} | {model_results['training_time']:.2f} | {model_results['inference_time']:.4f} |\n"
    
    report += "\n"
    
    # Add detailed results for each model
    report += "## Detailed Results\n\n"
    
    for model_name, model_results in benchmark_results["models"].items():
        report += f"### {model_name}\n\n"
        
        if "error" in model_results:
            report += f"Error: {model_results['error']}\n\n"
            continue
        
        report += "| Layer Strategy | Accuracy | Std Accuracy | Training Time (s) | Inference Time (s) |\n"
        report += "|----------------|----------|--------------|-------------------|-------------------|\n"
        
        for layers, layer_results in model_results["layer_strategies"].items():
            if "error" in layer_results:
                report += f"| {layers} | Error | Error | Error | Error |\n"
            else:
                report += f"| {layers} | {layer_results['accuracy']:.4f} | {layer_results['std_accuracy']:.4f} | {layer_results['training_time']:.2f} | {layer_results['inference_time']:.4f} |\n"
        
        report += "\n"
    
    # Add visualizations
    report += "## Visualizations\n\n"
    report += "### Accuracy Comparison\n\n"
    report += "![Accuracy Comparison](accuracy_comparison.png)\n\n"
    report += "### Training Time Comparison\n\n"
    report += "![Training Time Comparison](training_time_comparison.png)\n\n"
    report += "### Inference Time Comparison\n\n"
    report += "![Inference Time Comparison](inference_time_comparison.png)\n\n"
    report += "### Accuracy vs. Training Time\n\n"
    report += "![Accuracy vs. Training Time](accuracy_vs_training_time.png)\n\n"
    report += "### Accuracy vs. Inference Time\n\n"
    report += "![Accuracy vs. Inference Time](accuracy_vs_inference_time.png)\n\n"
    report += "### Model Ranking\n\n"
    report += "![Model Ranking](model_ranking.png)\n\n"
    report += "### Layer Strategy Ranking\n\n"
    report += "![Layer Strategy Ranking](layer_strategy_ranking.png)\n\n"
    
    # Add conclusion
    report += "## Conclusion\n\n"
    report += f"Based on the benchmark results, the best model for the Prospect Theory LLM pipeline is **{benchmark_results['best_model']}** "
    report += f"with layer strategy **{benchmark_results['best_layers']}**, achieving an accuracy of **{benchmark_results['best_accuracy']:.4f}**.\n\n"
    
    # Add recommendations
    report += "## Recommendations\n\n"
    report += "1. **Model Selection**: Use the best performing model for the final pipeline.\n"
    report += "2. **Layer Strategy**: Extract features from the optimal layer combination for best results.\n"
    report += "3. **Training Time**: Consider the trade-off between accuracy and training time when selecting a model.\n"
    report += "4. **Inference Time**: For real-time applications, consider models with lower inference time.\n"
    report += "5. **Further Exploration**: Experiment with ensemble methods combining multiple models or layer strategies.\n\n"
    
    # Save report
    report_path = os.path.join(output_dir, "benchmark_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    
    print(f"Comparison report saved to {report_path}")


def main():
    """Main function to parse arguments and run the benchmark."""
    parser = argparse.ArgumentParser(description="Prospect Theory LLM Benchmark")
    
    parser.add_argument("--anes_path", type=str, default=DEFAULT_ANES_PATH,
                        help="Path to ANES JSON files")
    parser.add_argument("--prospect_path", type=str, default=None,
                        help="Path to Prospect Theory dataset")
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
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Models to benchmark")
    parser.add_argument("--cv_folds", type=int, default=5,
                        help="Number of cross-validation folds")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run on (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Run the benchmark
    benchmark_models(
        anes_json_path=args.anes_path,
        prospect_theory_path=args.prospect_path,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_epochs_prospect=args.num_epochs_prospect,
        num_epochs_anes=args.num_epochs_anes,
        learning_rate=args.learning_rate,
        seed=args.seed,
        models_to_test=args.models,
        cross_validation_folds=args.cv_folds,
        device=args.device
    )


if __name__ == "__main__":
    main()

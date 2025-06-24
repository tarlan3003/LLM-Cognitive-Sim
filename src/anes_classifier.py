"""
ANES Classifier Module for Prospect Theory Pipeline

This module provides the classifier for ANES data using features derived from
LLM hidden layer representations and cognitive bias scores.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union


class FocalLoss(nn.Module):
    """
    Focal Loss implementation, identical to the one in the original notebook.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        alpha: 1D tensor of shape [num_classes] or None
        gamma: focusing parameter
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: Tensor[B, C]
        targets: Tensor[B] with class indices 0 ≤ targets[i] < C
        """
        # move class weights if provided
        if self.alpha is not None:
            self.alpha = self.alpha.to(logits.device)

        # standard CE with no reduction → [B]
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)             # [B], pt = probability of the true class
        loss = (1 - pt) ** self.gamma * ce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # [B]


class ProspectTheoryANESClassifier(nn.Module):
    """
    Classifier for ANES data using Prospect Theory features.
    
    This classifier combines traditional ANES features with cognitive bias scores
    and System 1/2 representations to predict voting preferences.
    """
    
    def __init__(
        self, 
        anes_feature_dim: int, 
        llm_hidden_dim: int, 
        num_biases: int, 
        combined_hidden_dim: int = 256, 
        num_classes: int = 2
    ):
        """
        Initialize the ANES classifier.
        
        Args:
            anes_feature_dim: Dimension of ANES features
            llm_hidden_dim: Dimension of LLM hidden states
            num_biases: Number of bias types
            combined_hidden_dim: Hidden dimension for combined features
            num_classes: Number of target classes
        """
        super().__init__()
        # Input: anes_features + bias_scores + weighted_system_llm_rep
        total_input_dim = anes_feature_dim + num_biases + llm_hidden_dim 
        
        self.combiner = nn.Sequential(
            nn.Linear(total_input_dim, combined_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(combined_hidden_dim, combined_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Linear(combined_hidden_dim // 2, num_classes)
        
        print(f"Initialized ProspectTheoryANESClassifier: {total_input_dim} input features, {num_classes} classes")

    def forward(
        self, 
        anes_features: torch.Tensor, 
        bias_scores: torch.Tensor, 
        weighted_system_rep: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            anes_features: ANES features [batch_size, anes_feature_dim]
            bias_scores: Bias scores [batch_size, num_biases]
            weighted_system_rep: Weighted system representation [batch_size, llm_hidden_dim]
            
        Returns:
            Logits [batch_size, num_classes]
        """
        # Concatenate all features
        combined_features = torch.cat([anes_features, bias_scores, weighted_system_rep], dim=1)
        
        hidden = self.combiner(combined_features)
        logits = self.classifier(hidden)
        return logits
    
    def save(self, path: str) -> None:
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model to
        """
        torch.save(self.state_dict(), path)
        print(f"Saved ProspectTheoryANESClassifier to {path}")
    
    @classmethod
    def load(
        cls, 
        path: str, 
        anes_feature_dim: int, 
        llm_hidden_dim: int, 
        num_biases: int, 
        device: str = 'cpu'
    ) -> 'ProspectTheoryANESClassifier':
        """
        Load the model from a file.
        
        Args:
            path: Path to load the model from
            anes_feature_dim: Dimension of ANES features
            llm_hidden_dim: Dimension of LLM hidden states
            num_biases: Number of bias types
            device: Device to load the model on
            
        Returns:
            Loaded ProspectTheoryANESClassifier
        """
        model = cls(anes_feature_dim, llm_hidden_dim, num_biases)
        model.load_state_dict(torch.load(path, map_location=device))
        model = model.to(device)
        print(f"Loaded ProspectTheoryANESClassifier from {path}")
        return model


def train_anes_classifier(
    anes_classifier: ProspectTheoryANESClassifier, 
    dataloader, 
    llm_extractor, 
    bias_representer, 
    num_epochs: int = 10, 
    lr: float = 1e-3, 
    device: str = 'cpu'
) -> Dict:
    """
    Train the ANES classifier.
    
    Args:
        anes_classifier: ProspectTheoryANESClassifier instance
        dataloader: DataLoader for training data
        llm_extractor: HiddenLayerExtractor instance
        bias_representer: CognitiveBiasRepresenter instance
        num_epochs: Number of epochs to train for
        lr: Learning rate
        device: Device to run the model on
        
    Returns:
        Dictionary of training metrics
    """
    optimizer = torch.optim.Adam(anes_classifier.parameters(), lr=lr)
    
    # Use Focal Loss for better handling of class imbalance, as in original notebook
    loss_fn = FocalLoss(gamma=2.0)
    
    # Training loop
    metrics = {'epoch_losses': [], 'epoch_accuracies': []}
    
    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            texts = batch['text']
            anes_features = batch['anes_features'].to(device)
            targets = batch['target'].to(device)
            
            # Extract activations
            activations = llm_extractor.extract_activations(texts)
            
            # Get bias scores and system representations
            bias_scores = bias_representer.get_bias_scores(activations).to(device)
            weighted_system_rep, _ = bias_representer.get_system_representations(activations)
            weighted_system_rep = weighted_system_rep.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = anes_classifier(anes_features, bias_scores, weighted_system_rep)
            loss = loss_fn(logits, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * len(targets)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == targets).sum().item()
            total += len(targets)
        
        # Epoch metrics
        epoch_loss = total_loss / total
        epoch_acc = correct / total
        metrics['epoch_losses'].append(epoch_loss)
        metrics['epoch_accuracies'].append(epoch_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}")
    
    return metrics


def evaluate_anes_classifier(
    anes_classifier: ProspectTheoryANESClassifier, 
    dataloader, 
    llm_extractor, 
    bias_representer, 
    device: str = 'cpu',
    target_names: List[str] = None,
    thresholds: List[float] = None
) -> Dict:
    """
    Evaluate the ANES classifier.
    
    Args:
        anes_classifier: ProspectTheoryANESClassifier instance
        dataloader: DataLoader for evaluation data
        llm_extractor: HiddenLayerExtractor instance
        bias_representer: CognitiveBiasRepresenter instance
        device: Device to run the model on
        target_names: Names of target classes
        thresholds: List of thresholds to evaluate (as in original notebook)
        
    Returns:
        Dictionary of evaluation metrics
    """
    if target_names is None:
        target_names = ['Trump', 'Harris']  # Default for binary classification
        
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]  # Default thresholds from original notebook
        
    anes_classifier.eval()
    
    all_preds = []
    all_targets = []
    all_logits = []
    all_bias_scores = []
    all_system_weights = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            texts = batch['text']
            anes_features = batch['anes_features'].to(device)
            targets = batch['target'].to(device)
            
            # Extract activations
            activations = llm_extractor.extract_activations(texts)
            
            # Get bias scores and system representations
            bias_scores = bias_representer.get_bias_scores(activations).to(device)
            weighted_system_rep, system_weights = bias_representer.get_system_representations(activations)
            weighted_system_rep = weighted_system_rep.to(device)
            
            # Forward pass
            logits = anes_classifier(anes_features, bias_scores, weighted_system_rep)
            
            # Collect results
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())
            all_bias_scores.append(bias_scores.cpu().numpy())
            all_system_weights.append(system_weights.cpu().numpy())
    
    # Convert to numpy arrays
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_probs = F.softmax(all_logits, dim=1).numpy()
    all_bias_scores = np.concatenate(all_bias_scores, axis=0)
    all_system_weights = np.concatenate(all_system_weights, axis=0)
    
    # Evaluate with different thresholds as in original notebook
    results = {}
    
    print("\nEvaluating with different thresholds:")
    for threshold in thresholds:
        # Apply threshold to probabilities
        all_preds = (all_probs[:, 1] > threshold).astype(int)
        
        # Calculate metrics
        accuracy = (all_preds == all_targets).mean()
        
        # Classification report
        report = classification_report(all_targets, all_preds, target_names=target_names, output_dict=True)
        
        # Print report
        print(f"\n✅ Classification Report (Thresholded @ {threshold:.2f}):")
        print(classification_report(all_targets, all_preds, target_names=target_names))
        
        # Store results
        results[f"threshold_{threshold}"] = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': confusion_matrix(all_targets, all_preds)
        }
    
    # Calculate bias scores and system weights by class
    bias_by_class = {}
    system_by_class = {}
    
    for i, class_name in enumerate(target_names):
        mask = all_targets == i
        if mask.sum() > 0:
            bias_by_class[class_name] = all_bias_scores[mask].mean(axis=0)
            system_by_class[class_name] = all_system_weights[mask].mean(axis=0)
    
    # Return all metrics
    return {
        'thresholded_results': results,
        'avg_bias_scores': all_bias_scores.mean(axis=0),
        'avg_system_weights': all_system_weights.mean(axis=0),
        'bias_by_class': bias_by_class,
        'system_by_class': system_by_class
    }


if __name__ == "__main__":
    # Example usage with RoBERTa model from original notebook
    import os
    import torch.utils.data
    from dataset import ProspectTheoryDataset, convert_anes_to_dataset
    from llm_extractor import HiddenLayerExtractor
    from bias_representer import CognitiveBiasRepresenter
    from transformers import RobertaTokenizer
    
    # Create directories
    os.makedirs("data/prospect_theory", exist_ok=True)
    os.makedirs("data/anes", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Convert ANES data from original JSON files
    json_folder = "/home/ubuntu/upload"  # Path to original JSON files
    anes_output_path = "data/anes/anes_dataset.json"
    
    # Convert ANES data if needed
    if not os.path.exists(anes_output_path):
        convert_anes_to_dataset(
            json_folder=json_folder,
            output_path=anes_output_path,
            target_variable="V241049",  # WHO WOULD R VOTE FOR: HARRIS VS TRUMP
            include_classes=["Donald Trump", "Kamala Harris"]
        )
    
    # Load tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    
    # Load dataset
    anes_dataset = ProspectTheoryDataset(anes_output_path, tokenizer, is_anes=True)
    
    # Create dummy prospect theory dataset for testing
    prospect_output_path = "data/prospect_theory/dummy.json"
    if not os.path.exists(prospect_output_path):
        ProspectTheoryDataset.create_prospect_theory_dataset(
            prospect_output_path, num_examples=100
        )
    
    # Load prospect theory dataset
    prospect_dataset = ProspectTheoryDataset(prospect_output_path, tokenizer)
    
    # Create dataloaders
    prospect_dataloader = torch.utils.data.DataLoader(prospect_dataset, batch_size=16, shuffle=True)
    anes_dataloader = torch.utils.data.DataLoader(anes_dataset, batch_size=16, shuffle=True)
    
    # Initialize extractor and representer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = HiddenLayerExtractor("roberta-base", [-1], device=device)
    representer = CognitiveBiasRepresenter(extractor.get_hidden_size(), prospect_dataset.bias_names, device=device)
    
    # Train CAVs and System 1/2 components
    representer.train_cavs(prospect_dataloader, extractor)
    representer.train_system_components(prospect_dataloader, extractor, num_epochs=2)
    
    # Initialize and train ANES classifier
    anes_feature_dim = 5  # Number of features in extract_legitimate_features
    classifier = ProspectTheoryANESClassifier(
        anes_feature_dim, extractor.get_hidden_size(), len(prospect_dataset.bias_names)
    ).to(device)
    
    # Train classifier
    train_anes_classifier(classifier, anes_dataloader, extractor, representer, num_epochs=2, device=device)
    
    # Evaluate classifier
    target_names = ["Donald Trump", "Kamala Harris"]
    metrics = evaluate_anes_classifier(
        classifier, anes_dataloader, extractor, representer, device=device, target_names=target_names
    )
    
    # Save models
    representer.save("models/bias_representer.pt")
    classifier.save("models/anes_classifier.pt")

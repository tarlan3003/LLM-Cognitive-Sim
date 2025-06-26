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
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union


class FocalLoss(nn.Module):
    """
    Focal Loss implementation, identical to the one in the original notebook.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction=\'mean\'):
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
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction=\'none\')
        pt = torch.exp(-ce)             # [B], pt = probability of the true class
        loss = (1 - pt) ** self.gamma * ce

        if self.reduction == \'mean\':
            return loss.mean()
        elif self.reduction == \'sum\':
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
        num_classes: int = 2,
        dropout_rate: float = 0.5  # Increased dropout for better regularization
    ):
        """
        Initialize the ANES classifier.
        
        Args:
            anes_feature_dim: Dimension of ANES features
            llm_hidden_dim: Dimension of LLM hidden states
            num_biases: Number of bias types
            combined_hidden_dim: Hidden dimension for combined features
            num_classes: Number of target classes
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        # Input: anes_features + bias_scores + weighted_system_llm_rep
        self.total_input_dim = anes_feature_dim + num_biases + llm_hidden_dim 
        
        # Deeper network with batch normalization and residual connections
        self.anes_encoder = nn.Sequential(
            nn.Linear(anes_feature_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2)
        )
        
        self.bias_encoder = nn.Sequential(
            nn.Linear(num_biases, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2)
        )
        
        self.llm_encoder = nn.Sequential(
            nn.Linear(llm_hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2)
        )
        
        # Combined features dimension after separate encoders
        combined_encoded_dim = 64 + 32 + 128
        
        self.combiner = nn.Sequential(
            nn.Linear(combined_encoded_dim, combined_hidden_dim),
            nn.BatchNorm1d(combined_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(combined_hidden_dim, combined_hidden_dim // 2),
            nn.BatchNorm1d(combined_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        self.classifier = nn.Linear(combined_hidden_dim // 2, num_classes)
        
        # For dynamic input handling - initialize to None
        self.adjusted_combiner = None
        
        print(f"Initialized ProspectTheoryANESClassifier: {self.total_input_dim} input features, {num_classes} classes")

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
        # Debug shape information
        batch_size = anes_features.shape[0]
        anes_dim = anes_features.shape[1] if len(anes_features.shape) > 1 else 1
        bias_dim = bias_scores.shape[1] if len(bias_scores.shape) > 1 else 1
        system_dim = weighted_system_rep.shape[1] if len(weighted_system_rep.shape) > 1 else 1
        
        # Ensure anes_features has correct shape
        if len(anes_features.shape) == 1:
            anes_features = anes_features.unsqueeze(1)
            
        # Ensure bias_scores has correct shape
        if len(bias_scores.shape) == 1:
            bias_scores = bias_scores.unsqueeze(1)
            
        # Ensure weighted_system_rep has correct shape
        if len(weighted_system_rep.shape) == 1:
            weighted_system_rep = weighted_system_rep.unsqueeze(1)
        
        try:
            # Process each feature type separately
            anes_encoded = self.anes_encoder(anes_features)
            bias_encoded = self.bias_encoder(bias_scores)
            llm_encoded = self.llm_encoder(weighted_system_rep)
            
            # Concatenate encoded features
            combined_features = torch.cat([anes_encoded, bias_encoded, llm_encoded], dim=1)
            
            # Process through combiner
            hidden = self.combiner(combined_features)
            
        except RuntimeError as e:
            # If there\'s a dimension mismatch, fall back to dynamic handling
            print(f"WARNING: Error in forward pass: {e}")
            print(f"ANES features: {anes_features.shape}, Bias scores: {bias_scores.shape}, System rep: {weighted_system_rep.shape}")
            
            # Concatenate all features directly
            combined_features = torch.cat([anes_features, bias_scores, weighted_system_rep], dim=1)
            
            # Check if dimensions match
            actual_dim = combined_features.shape[1]
            if actual_dim != self.total_input_dim:
                print(f"WARNING: Input dimension mismatch. Expected {self.total_input_dim}, got {actual_dim}")
                
                # Fixed: Check if adjusted_combiner exists AND THEN check its in_features
                need_new_layer = True
                if self.adjusted_combiner is not None:
                    if hasattr(self.adjusted_combiner, \'in_features\') and self.adjusted_combiner.in_features == actual_dim:
                        need_new_layer = False
                
                if need_new_layer:
                    self.adjusted_combiner = nn.Linear(actual_dim, 256).to(combined_features.device)
                    print(f"Adjusted first layer to accept {actual_dim} input features")
                
                # Use the adjusted combiner
                hidden = self.adjusted_combiner(combined_features)
                hidden = F.relu(hidden)
                hidden = F.dropout(hidden, 0.5, self.training)
                hidden = self.combiner[4:](hidden)  # Skip first linear, batch norm, relu, dropout
            else:
                # Use a simplified network if dimensions match but separate encoders failed
                hidden = nn.Sequential(
                    nn.Linear(actual_dim, 256),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.5)
                ).to(combined_features.device)(combined_features)
        
        # Final classification
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
        device: str = \'cpu\'
    ) -> \'ProspectTheoryANESClassifier\':
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
    train_dataloader, 
    val_dataloader, # Added validation dataloader
    extractor, 
    bias_representer, 
    num_epochs: int = 20,  # Increased from 10 to 20
    lr: float = 3e-4,      # Adjusted learning rate
    device: str = \'cpu\',
    save_dir: str = \'models\',
    focal_loss_gamma: float = 2.0
) -> Dict:
    """
    Train the ANES classifier.
    
    Args:
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        extractor: HiddenLayerExtractor instance
        bias_representer: CognitiveBiasRepresenter instance
        num_epochs: Number of epochs to train for
        lr: Learning rate
        device: Device to run the model on
        save_dir: Directory to save the best model
        focal_loss_gamma: Gamma parameter for Focal Loss
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Initialize classifier
    # Determine anes_feature_dim dynamically from the first batch
    first_batch = next(iter(train_dataloader))
    anes_feature_dim = first_batch[\'anes_features\'].shape[1]
    num_classes = len(torch.unique(first_batch[\'target\']))
    
    anes_classifier = ProspectTheoryANESClassifier(
        anes_feature_dim=anes_feature_dim,
        llm_hidden_dim=extractor.get_hidden_size(),
        num_biases=len(bias_representer.bias_names),
        num_classes=num_classes
    ).to(device)

    # Calculate class weights for balanced training
    all_train_targets = []
    for batch in train_dataloader:
        all_train_targets.append(batch[\'target\'])
    
    if all_train_targets:
        all_train_targets = torch.cat(all_train_targets)
        class_counts = torch.bincount(all_train_targets)
        # Handle case where a class might have zero samples
        class_weights = 1.0 / (class_counts.float() + 1e-5) # Add small epsilon to avoid division by zero
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        print(f"Using class weights: {class_weights}")
    else:
        class_weights = None
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        anes_classifier.parameters(), 
        lr=lr, 
        weight_decay=1e-5 # Using a fixed weight decay for now
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode=\'min\', 
        factor=0.5, 
        patience=3
    )
    
    # Use Focal Loss with class weights
    loss_fn = FocalLoss(alpha=class_weights, gamma=focal_loss_gamma)
    
    # Training loop
    metrics = {\'train_losses\': [], \'train_accuracies\': [], \'val_losses\': [], \'val_accuracies\': []}
    best_val_acc = 0.0
    best_model_path = os.path.join(save_dir, \'anes_classifier_best.pt\')
    
    for epoch in range(num_epochs):
        anes_classifier.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0
        
        for batch in tqdm(train_dataloader, desc=f"ANES Train Epoch {epoch+1}/{num_epochs}"):
            texts = batch[\'text\']
            anes_features = batch[\'anes_features\'].to(device)
            targets = batch[\'target\'].to(device)
            
            # Extract activations
            activations = extractor.extract_activations(texts)
            
            # Get bias scores and system representations
            bias_scores = bias_representer.get_bias_scores(activations).to(device)
            weighted_system_rep, _ = bias_representer(activations) # Use forward pass of bias_representer
            weighted_system_rep = weighted_system_rep.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = anes_classifier(anes_features, bias_scores, weighted_system_rep)
            loss = loss_fn(logits, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(anes_classifier.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track metrics
            total_train_loss += loss.item() * len(targets)
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == targets).sum().item()
            total_train += len(targets)
        
        # Evaluate on validation set
        anes_classifier.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                texts = batch[\'text\']
                anes_features = batch[\'anes_features\'].to(device)
                targets = batch[\'target\'].to(device)
                
                activations = extractor.extract_activations(texts)
                bias_scores = bias_representer.get_bias_scores(activations).to(device)
                weighted_system_rep, _ = bias_representer(activations)
                weighted_system_rep = weighted_system_rep.to(device)
                
                logits = anes_classifier(anes_features, bias_scores, weighted_system_rep)
                loss = loss_fn(logits, targets)
                
                total_val_loss += loss.item() * len(targets)
                preds = torch.argmax(logits, dim=1)
                correct_val += (preds == targets).sum().item()
                total_val += len(targets)

        # Epoch metrics
        epoch_train_loss = total_train_loss / total_train
        epoch_train_acc = correct_train / total_train
        epoch_val_loss = total_val_loss / total_val
        epoch_val_acc = correct_val / total_val
        
        metrics[\'train_losses\'].append(epoch_train_loss)
        metrics[\'train_accuracies\'].append(epoch_train_acc)
        metrics[\'val_losses\'].append(epoch_val_loss)
        metrics[\'val_accuracies\'].append(epoch_val_acc)
        
        print(f"ANES Epoch {epoch+1}/{num_epochs}: Train Loss={epoch_train_loss:.4f}, Train Acc={epoch_train_acc:.4f} | Val Loss={epoch_val_loss:.4f}, Val Acc={epoch_val_acc:.4f}")
        
        # Update learning rate
        scheduler.step(epoch_val_loss)
        
        # Save best model based on validation accuracy
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            anes_classifier.save(best_model_path)
            print(f"New best validation accuracy: {best_val_acc:.4f}. Model saved to {best_model_path}")
    
    # Load the best model for final evaluation
    anes_classifier.load_state_dict(torch.load(best_model_path, map_location=device))
    anes_classifier.eval()
    print(f"Loaded best model from {best_model_path} for final evaluation.")

    # Final evaluation on validation set
    eval_results = evaluate_anes_classifier(
        anes_classifier=anes_classifier, 
        dataloader=val_dataloader, 
        llm_extractor=extractor, 
        bias_representer=bias_representer, 
        device=device,
        target_names=[\'Trump\', \'Harris\'] # Assuming 2 classes for now
    )
    
    return eval_results


def evaluate_anes_classifier(
    anes_classifier: ProspectTheoryANESClassifier, 
    dataloader, 
    llm_extractor, 
    bias_representer, 
    device: str = \'cpu\',
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
        target_names = [\'Trump\', \'Harris\']  # Default for binary classification
        
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
            texts = batch[\'text\']
            anes_features = batch[\'anes_features\'].to(device)
            targets = batch[\'target\'].to(device)
            
            # Extract activations
            activations = llm_extractor.extract_activations(texts)
            
            # Get bias scores and system representations
            bias_scores = bias_representer.get_bias_scores(activations).to(device)
            weighted_system_rep, system_weights = bias_representer(activations) # Use forward pass
            weighted_system_rep = weighted_system_rep.to(device)
            system_weights = system_weights.to(device)
            
            logits = anes_classifier(anes_features, bias_scores, weighted_system_rep)
            
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
            all_bias_scores.extend(bias_scores.cpu().numpy())
            all_system_weights.extend(system_weights.cpu().numpy())

    # Convert to numpy arrays
    all_targets = np.array(all_targets)
    all_logits = np.array(all_logits)
    all_bias_scores = np.array(all_bias_scores)
    all_system_weights = np.array(all_system_weights)
    
    # Calculate average system weights
    avg_system_weights = np.mean(all_system_weights, axis=0)

    # Evaluate for each threshold
    results = {"system_weights": avg_system_weights.tolist()}
    for threshold in thresholds:
        # Apply threshold to probabilities of class 1 (Harris)
        # Assuming logits are for [class 0, class 1] (Trump, Harris)
        probs = F.softmax(torch.tensor(all_logits), dim=1).numpy()
        # Predict 1 (Harris) if prob of Harris > threshold, else 0 (Trump)
        threshold_preds = (probs[:, 1] > threshold).astype(int)
        
        acc = accuracy_score(all_targets, threshold_preds)
        report = classification_report(all_targets, threshold_preds, target_names=target_names, output_dict=True, zero_division=0)
        
        results[f"threshold_{threshold}"] = {
            "accuracy": acc,
            "class_metrics": {
                target_names[0]: report[target_names[0]],
                target_names[1]: report[target_names[1]]
            },
            "macro_precision": report[\'macro avg\'][\'precision\'],
            "macro_recall": report[\'macro avg\'][\'recall\'],
            "macro_f1": report[\'macro avg\'][\'f1\'],
            "weighted_precision": report[\'weighted avg\'][\'precision\'],
            "weighted_recall": report[\'weighted avg\'][\'recall\'],
            "weighted_f1": report[\'weighted avg\'][\'f1\'],
            "confusion_matrix": confusion_matrix(all_targets, threshold_preds).tolist()
        }
        
        # Try to calculate AUC if there are at least two classes present in targets
        if len(np.unique(all_targets)) > 1:
            try:
                auc = roc_auc_score(all_targets, probs[:, 1])
                results[f"threshold_{threshold}"]["roc_auc"] = auc
            except ValueError:
                results[f"threshold_{threshold}"]["roc_auc"] = "N/A - only one class present in targets"
        else:
            results[f"threshold_{threshold}"]["roc_auc"] = "N/A - only one class present in targets"

    return results


if __name__ == "__main__":
    # Example usage
    from src.dataset import ProspectTheoryDataset
    from src.llm_extractor import HiddenLayerExtractor
    from src.bias_representer import CognitiveBiasRepresenter
    from transformers import AutoTokenizer
    import os
    
    # Setup dummy data and models for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dummy prospect theory dataset
    os.makedirs("data/prospect_theory", exist_ok=True)
    ProspectTheoryDataset.create_prospect_theory_dataset("data/prospect_theory/dummy_prospect.json", num_examples=50)
    
    # Create dummy ANES dataset (simplified for testing)
    # In a real scenario, this would come from processed ANES data
    anes_dummy_data = []
    for i in range(100):
        anes_dummy_data.append({
            "text": f"Respondent {i} features...",
            "anes_features": [random.random() for _ in range(10)], # 10 dummy features
            "target": random.randint(0, 1) # 0 or 1
        })
    os.makedirs("data/anes", exist_ok=True)
    with open("data/anes/dummy_anes.json", "w") as f:
        json.dump(anes_dummy_data, f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    
    # Load datasets
    prospect_dataset = ProspectTheoryDataset("data/prospect_theory/dummy_prospect.json", tokenizer)
    anes_dataset = ProspectTheoryDataset("data/anes/dummy_anes.json", tokenizer, is_anes=True, text_key="text")
    
    # Create dataloaders
    prospect_dataloader = DataLoader(prospect_dataset, batch_size=16, shuffle=True)
    anes_train_dataloader = DataLoader(anes_dataset, batch_size=16, shuffle=True)
    anes_val_dataloader = DataLoader(anes_dataset, batch_size=16, shuffle=False)
    
    # Initialize extractor and representer
    extractor = HiddenLayerExtractor("roberta-base", [-1], device=device)
    bias_representer = CognitiveBiasRepresenter(extractor.get_hidden_size(), prospect_dataset.bias_names, device=device)
    
    # Train bias representer components
    bias_representer.train_cavs(prospect_dataloader, extractor)
    bias_representer.train_system_components(prospect_dataloader, extractor, num_epochs=2)
    
    # Train ANES classifier
    print("\nStarting ANES classifier training...")
    anes_eval_results = train_anes_classifier(
        train_dataloader=anes_train_dataloader,
        val_dataloader=anes_val_dataloader,
        extractor=extractor,
        bias_representer=bias_representer,
        num_epochs=5,
        lr=1e-4,
        device=device,
        save_dir="models_test"
    )
    
    print("\nANES Classifier Evaluation Results:")
    for threshold_key, metrics in anes_eval_results.items():
        if threshold_key.startswith("threshold_"):
            print(f"\nResults for {threshold_key}:")
            print(f"Accuracy: {metrics[\'accuracy\']:.4f}")
            print(f"Confusion Matrix:\n{np.array(metrics[\'confusion_matrix\'])}")
            print(f"ROC AUC: {metrics.get(\'roc_auc\', \'N/A\'):.4f}")
            for class_name, class_metrics in metrics[\'class_metrics\'].items():
                print(f"  {class_name}: Precision={class_metrics[\'precision\']:.4f}, Recall={class_metrics[\'recall\']:.4f}, F1={class_metrics[\'f1\']:.4f}")
            print(f"  macro avg: Precision={metrics[\'macro_precision\']:.4f}, Recall={metrics[\'macro_recall\']:.4f}, F1={metrics[\'macro_f1\']:.4f}")
            print(f"  weighted avg: Precision={metrics[\'weighted_precision\']:.4f}, Recall={metrics[\'weighted_recall\']:.4f}, F1={metrics[\'weighted_f1\']:.4f}")
    print(f"\nAverage System Weights: {anes_eval_results[\'system_weights\']}")



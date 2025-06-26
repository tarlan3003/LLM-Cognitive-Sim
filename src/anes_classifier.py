"""
ANES Classifier Module for Prospect Theory LLM - Best Performing Version

This module implements the ANES classifier that predicts voting preferences
based on cognitive bias representations and ANES features.

Author: Tarlan Sultanov
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    This loss function gives more weight to hard examples and less weight to easy examples.
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Class weights
            gamma: Focusing parameter
            reduction: Reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Forward pass.
        
        Args:
            inputs: Predicted logits
            targets: Target labels
            
        Returns:
            Loss value
        """
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt) ** self.gamma * BCE_loss
        
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha = self.alpha[targets.data.view(-1)]
            else:
                alpha = torch.tensor([self.alpha, 1 - self.alpha]).to(inputs.device)[targets.data.view(-1)]
            F_loss = alpha * F_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

class ProspectTheoryANESClassifier(nn.Module):
    """
    Classifier for predicting voting preferences based on cognitive bias representations.
    
    This model combines:
    1. ANES features
    2. Cognitive bias scores
    3. System 1/2 thinking weights
    """
    
    def __init__(
        self, 
        anes_feature_dim: int, 
        bias_dim: int, 
        system_dim: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.3
    ):
        """
        Initialize the ANES classifier.
        
        Args:
            anes_feature_dim: Dimension of ANES features
            bias_dim: Dimension of bias scores
            system_dim: Dimension of system weights
            hidden_dim: Dimension of hidden layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.anes_feature_dim = anes_feature_dim
        self.bias_dim = bias_dim
        self.system_dim = system_dim
        self.hidden_dim = hidden_dim
        
        # ANES feature encoder
        self.anes_encoder = nn.Sequential(
            nn.Linear(anes_feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Bias score encoder
        self.bias_encoder = nn.Sequential(
            nn.Linear(bias_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # System weight encoder
        self.system_encoder = nn.Sequential(
            nn.Linear(system_dim, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Combined features dimension
        combined_dim = (hidden_dim // 2) + (hidden_dim // 2) + (hidden_dim // 4)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # For dynamic input handling
        self.adjusted_combiner = None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize model weights.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        anes_features: torch.Tensor, 
        bias_scores: torch.Tensor, 
        system_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            anes_features: ANES features
            bias_scores: Cognitive bias scores
            system_weights: System 1/2 thinking weights
            
        Returns:
            Logits for voting preference
        """
        # Encode features
        anes_encoded = self.anes_encoder(anes_features)
        bias_encoded = self.bias_encoder(bias_scores)
        system_encoded = self.system_encoder(system_weights)
        
        # Concatenate features
        combined_features = torch.cat([anes_encoded, bias_encoded, system_encoded], dim=1)
        
        # Handle dynamic input dimension changes
        actual_dim = combined_features.shape[1]
        expected_dim = (self.hidden_dim // 2) + (self.hidden_dim // 2) + (self.hidden_dim // 4)
        
        if actual_dim != expected_dim:
            # Print debug info for the first occurrence
            if not hasattr(self, 'dimension_mismatch_reported') or not self.dimension_mismatch_reported:
                print(f"Warning: Input dimension mismatch. Expected {expected_dim}, got {actual_dim}.")
                print(f"ANES encoded shape: {anes_encoded.shape}")
                print(f"Bias encoded shape: {bias_encoded.shape}")
                print(f"System encoded shape: {system_encoded.shape}")
                print(f"Combined shape: {combined_features.shape}")
                self.dimension_mismatch_reported = True
            
            # Create a new adjusted combiner if needed
            need_new_layer = True
            if self.adjusted_combiner is not None:
                if hasattr(self.adjusted_combiner, 'in_features') and self.adjusted_combiner.in_features == actual_dim:
                    need_new_layer = False
            
            if need_new_layer:
                self.adjusted_combiner = nn.Linear(actual_dim, self.hidden_dim).to(combined_features.device)
                print(f"Created new adjusted combiner: {actual_dim} -> {self.hidden_dim}")
            
            # Use the adjusted combiner
            combined = self.adjusted_combiner(combined_features)
            combined = F.relu(combined)
            combined = F.dropout(combined, p=0.3, training=self.training)
            logits = self.classifier[-1](combined)
        else:
            # Use the standard classifier
            logits = self.classifier(combined_features)
        
        return logits.squeeze(-1)

def train_anes_classifier(
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    extractor: 'HiddenLayerExtractor',
    bias_representer: 'CognitiveBiasRepresenter',
    num_epochs: int = 20,
    learning_rate: float = 2e-4,
    device: torch.device = None,
    save_dir: str = "models",
    focal_loss_gamma: float = 2.0
) -> Dict[str, float]:
    """
    Train the ANES classifier.
    
    Args:
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        extractor: Hidden layer extractor
        bias_representer: Cognitive bias representer
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        save_dir: Directory to save models
        focal_loss_gamma: Gamma parameter for Focal Loss
        
    Returns:
        Dictionary of evaluation metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get dimensions from first batch
    for batch in train_dataloader:
        anes_feature_dim = batch['anes_features'].shape[1]
        break
    
    # Get bias dimension from representer
    bias_dim = bias_representer.num_biases
    
    # Initialize classifier
    classifier = ProspectTheoryANESClassifier(
        anes_feature_dim=anes_feature_dim,
        bias_dim=bias_dim,
        system_dim=2,
        hidden_dim=256,
        dropout=0.3
    ).to(device)
    
    # Calculate class weights for balanced loss
    class_counts = torch.zeros(2)
    for batch in train_dataloader:
        targets = batch['target']
        for t in range(2):
            class_counts[t] += (targets == t).sum().item()
    
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    print(f"Using class weights: {class_weights}")
    
    # Define loss function
    loss_fn = FocalLoss(alpha=class_weights.to(device), gamma=focal_loss_gamma)
    
    # Define optimizer
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate)
    
    # Define learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)"):
            # Extract features
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            anes_features = batch['anes_features'].to(device)
            targets = batch['target'].to(device)
            
            # Get hidden representations
            hidden_reps = extractor(input_ids, attention_mask)
            
            # Get bias scores and system weights
            bias_scores, system_weights = bias_representer(hidden_reps)
            
            # Forward pass
            logits = classifier(anes_features, bias_scores, system_weights)
            
            # Calculate loss
            loss = loss_fn(logits, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item() * input_ids.size(0)
            preds = (torch.sigmoid(logits) > 0.5).long()
            train_correct += (preds == targets).sum().item()
            train_total += input_ids.size(0)
        
        # Calculate average loss and accuracy
        train_loss /= train_total
        train_accuracy = train_correct / train_total
        
        # Validation
        val_metrics = evaluate_anes_classifier(
            classifier,
            val_dataloader,
            extractor,
            bias_representer,
            device
        )
        
        # Update learning rate
        scheduler.step(val_metrics['loss'])
        
        # Print metrics
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}")
        print(f"  Val - Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_metrics['accuracy']
            best_val_loss = val_metrics['loss']
            best_model_state = classifier.state_dict()
            
            # Save model
            import os
            os.makedirs(save_dir, exist_ok=True)
            torch.save(classifier.state_dict(), os.path.join(save_dir, "anes_classifier.pt"))
            print(f"  Saved best model with accuracy: {best_val_accuracy:.4f}")
    
    # Load best model
    classifier.load_state_dict(best_model_state)
    
    # Final evaluation with different thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    final_metrics = {}
    
    for threshold in thresholds:
        threshold_metrics = evaluate_anes_classifier(
            classifier,
            val_dataloader,
            extractor,
            bias_representer,
            device,
            threshold=threshold
        )
        
        final_metrics[f"threshold_{threshold}"] = threshold_metrics
    
    # Calculate average system weights
    system_weights_sum = torch.zeros(2)
    total_samples = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            # Extract features
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Get hidden representations
            hidden_reps = extractor(input_ids, attention_mask)
            
            # Get system weights
            _, system_weights = bias_representer(hidden_reps)
            
            # Update sum
            system_weights_sum += system_weights.sum(dim=0).cpu()
            total_samples += input_ids.size(0)
    
    # Calculate average
    avg_system_weights = system_weights_sum / total_samples
    final_metrics['system_weights'] = avg_system_weights.tolist()
    
    return final_metrics

def evaluate_anes_classifier(
    classifier: ProspectTheoryANESClassifier,
    dataloader: torch.utils.data.DataLoader,
    extractor: 'HiddenLayerExtractor',
    bias_representer: 'CognitiveBiasRepresenter',
    device: torch.device = None,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Evaluate the ANES classifier.
    
    Args:
        classifier: ANES classifier
        dataloader: DataLoader for evaluation data
        extractor: Hidden layer extractor
        bias_representer: Cognitive bias representer
        device: Device to evaluate on
        threshold: Classification threshold
        
    Returns:
        Dictionary of evaluation metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    classifier.eval()
    
    # Evaluation metrics
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Extract features
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            anes_features = batch['anes_features'].to(device)
            targets = batch['target'].to(device)
            
            # Get hidden representations
            hidden_reps = extractor(input_ids, attention_mask)
            
            # Get bias scores and system weights
            bias_scores, system_weights = bias_representer(hidden_reps)
            
            # Forward pass
            logits = classifier(anes_features, bias_scores, system_weights)
            
            # Calculate loss
            loss = loss_fn(logits, targets.float())
            
            # Update metrics
            total_loss += loss.item() * input_ids.size(0)
            
            # Apply threshold
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).long()
            
            # Store predictions and targets
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    accuracy = accuracy_score(all_targets, all_preds)
    
    # Get classification report
    target_names = ["Donald Trump", "Kamala Harris"]
    report = classification_report(all_targets, all_preds, target_names=target_names, output_dict=True)
    
    # Extract metrics
    class_metrics = {}
    for i, name in enumerate(target_names):
        class_metrics[name] = {
            'precision': report[name]['precision'],
            'recall': report[name]['recall'],
            'f1': report[name]['f1-score']
        }
    
    # Calculate average loss
    avg_loss = total_loss / len(dataloader.dataset)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'class_metrics': class_metrics,
        'macro_precision': report['macro avg']['precision'],
        'macro_recall': report['macro avg']['recall'],
        'macro_f1': report['macro avg']['f1-score'],
        'weighted_precision': report['weighted avg']['precision'],
        'weighted_recall': report['weighted avg']['recall'],
        'weighted_f1': report['weighted avg']['f1-score']
    }

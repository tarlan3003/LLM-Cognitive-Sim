
"""
ANES Classifier Module for Prospect Theory Pipeline - Fixed Version

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
import os
from transformers import AutoModel, AutoTokenizer


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
        
        self.anes_feature_dim = anes_feature_dim
        self.llm_hidden_dim = llm_hidden_dim
        self.num_biases = num_biases
        
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
        batch_size = anes_features.shape[0]
        
        # Ensure all inputs have correct dimensions
        if len(anes_features.shape) == 1:
            anes_features = anes_features.unsqueeze(0)
        if len(bias_scores.shape) == 1:
            bias_scores = bias_scores.unsqueeze(0)
        if len(weighted_system_rep.shape) == 1:
            weighted_system_rep = weighted_system_rep.unsqueeze(0)
        
        # Validate input dimensions
        assert anes_features.shape[1] == self.anes_feature_dim, f"Expected ANES features dim {self.anes_feature_dim}, got {anes_features.shape[1]}"
        assert bias_scores.shape[1] == self.num_biases, f"Expected bias scores dim {self.num_biases}, got {bias_scores.shape[1]}"
        assert weighted_system_rep.shape[1] == self.llm_hidden_dim, f"Expected LLM hidden dim {self.llm_hidden_dim}, got {weighted_system_rep.shape[1]}"
        
        # Process each feature type separately
        anes_encoded = self.anes_encoder(anes_features)
        bias_encoded = self.bias_encoder(bias_scores)
        llm_encoded = self.llm_encoder(weighted_system_rep)
        
        # Concatenate encoded features
        combined_features = torch.cat([anes_encoded, bias_encoded, llm_encoded], dim=1)
        
        # Process through combiner
        hidden = self.combiner(combined_features)
        
        # Final classification
        logits = self.classifier(hidden)
        return logits
    
    def save(self, path: str) -> None:
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model to
        """
        save_dict = {
            'state_dict': self.state_dict(),
            'anes_feature_dim': self.anes_feature_dim,
            'llm_hidden_dim': self.llm_hidden_dim,
            'num_biases': self.num_biases
        }
        torch.save(save_dict, path)
        print(f"Saved ProspectTheoryANESClassifier to {path}")
    
    @classmethod
    def load(
        cls, 
        path: str, 
        device: str = 'cpu'
    ) -> 'ProspectTheoryANESClassifier':
        """
        Load the model from a file.
        
        Args:
            path: Path to load the model from
            device: Device to load the model on
            
        Returns:
            Loaded ProspectTheoryANESClassifier
        """
        save_dict = torch.load(path, map_location=device)
        model = cls(
            save_dict['anes_feature_dim'], 
            save_dict['llm_hidden_dim'], 
            save_dict['num_biases']
        )
        model.load_state_dict(save_dict['state_dict'])
        model = model.to(device)
        print(f"Loaded ProspectTheoryANESClassifier from {path}")
        return model


class BERTANESClassifier(nn.Module):
    """
    BERT-based classifier for ANES data.
    """
    def __init__(self, model_name: str, num_classes: int = 2, dropout_rate: float = 0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # Use pooled output for classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def train_anes_classifier(
    train_dataloader, 
    val_dataloader,
    extractor, 
    bias_representer, 
    num_epochs: int = 20,
    learning_rate: float = 3e-4,
    device: str = 'cpu',
    save_dir: str = 'models',
    focal_loss_gamma: float = 2.0,
    use_bert_classifier: bool = False, # New parameter to switch classifier type
    bert_model_name: str = "bert-base-uncased" # Model name for BERT classifier
) -> Dict:
    """
    Train the ANES classifier - FIXED: Updated function signature to match usage.
    
    Args:
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        extractor: HiddenLayerExtractor instance
        bias_representer: CognitiveBiasRepresenter instance
        num_epochs: Number of epochs to train for
        learning_rate: Learning rate
        device: Device to run the model on
        save_dir: Directory to save models
        focal_loss_gamma: Gamma parameter for focal loss
        use_bert_classifier: Whether to use the BERT-based classifier
        bert_model_name: Model name for the BERT classifier
        
    Returns:
        Dictionary of training metrics and trained classifier
    """
    device = torch.device(device) if isinstance(device, str) else device
    
    if use_bert_classifier:
        print(f"Initializing BERTANESClassifier with {bert_model_name}...")
        anes_classifier = BERTANESClassifier(bert_model_name).to(device)
    else:
        # Determine dimensions from first batch for ProspectTheoryANESClassifier
        sample_batch = next(iter(train_dataloader))
        sample_texts = sample_batch['text']
        sample_anes_features = sample_batch['anes_features']
        
        # Extract sample activations to determine dimensions
        sample_activations = extractor.extract_activations(sample_texts)
        sample_bias_scores = bias_representer.get_bias_scores(sample_activations)
        sample_weighted_rep, _ = bias_representer.get_system_representations(sample_activations)
        
        anes_feature_dim = sample_anes_features.shape[1]
        llm_hidden_dim = sample_weighted_rep.shape[1]
        num_biases = len(bias_representer.bias_names)
        
        print(f"Initializing ProspectTheoryANESClassifier with dims: ANES={anes_feature_dim}, LLM={llm_hidden_dim}, Biases={num_biases}")
        
        # Initialize classifier
        anes_classifier = ProspectTheoryANESClassifier(
            anes_feature_dim=anes_feature_dim,
            llm_hidden_dim=llm_hidden_dim,
            num_biases=num_biases
        ).to(device)
    
    # Calculate class weights for balanced training
    all_targets = []
    for batch in train_dataloader:
        all_targets.append(batch['target'])
    
    if all_targets:
        all_targets = torch.cat(all_targets)
        class_counts = torch.bincount(all_targets)
        class_weights = 1.0 / class_counts.float()
        class_weights = class_weights / class_weights.sum() * len(class_counts)
        class_weights = class_weights.to(device)
        print(f"Using class weights: {class_weights}")
    else:
        class_weights = None
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        anes_classifier.parameters(), 
        lr=learning_rate, 
        weight_decay=1e-5
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3
    )
    
    # Use Focal Loss with class weights
    loss_fn = FocalLoss(alpha=class_weights, gamma=focal_loss_gamma)
    
    # Training loop
    metrics = {'epoch_losses': [], 'epoch_accuracies': []}
    best_acc = 0.0
    best_state = None
    
    for epoch in range(num_epochs):
        anes_classifier.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            targets = batch['target'].to(device)
            
            optimizer.zero_grad()

            if use_bert_classifier:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                logits = anes_classifier(input_ids, attention_mask)
            else:
                texts = batch['text']
                anes_features = batch['anes_features'].to(device)
                
                # Extract activations
                activations = extractor.extract_activations(texts)
                
                # Get bias scores and system representations
                bias_scores = bias_representer.get_bias_scores(activations).to(device)
                weighted_system_rep, _ = bias_representer.get_system_representations(activations)
                weighted_system_rep = weighted_system_rep.to(device)
                
                # Forward pass
                logits = anes_classifier(anes_features, bias_scores, weighted_system_rep)
            
            loss = loss_fn(logits, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(anes_classifier.parameters(), max_norm=1.0)
            
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
        
        # Update learning rate
        scheduler.step(epoch_loss)
        
        # Save best model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_state = {k: v.cpu().clone() for k, v in anes_classifier.state_dict().items()}
            print(f"New best accuracy: {best_acc:.4f}")
    
    # Restore best model
    if best_state is not None:
        anes_classifier.load_state_dict(best_state)
        print(f"Restored best model with accuracy: {best_acc:.4f}")
    
    # Save the trained classifier
    os.makedirs(save_dir, exist_ok=True)
    classifier_path = os.path.join(save_dir, "anes_classifier.pt")
    anes_classifier.save(classifier_path)
    
    # Evaluate on validation set
    val_metrics = evaluate_anes_classifier(
        anes_classifier, val_dataloader, extractor, bias_representer, device, use_bert_classifier
    )
    
    # Combine training and validation metrics
    final_metrics = {
        'training_metrics': metrics,
        'anes_classifier': anes_classifier,
        'best_accuracy': best_acc
    }
    final_metrics.update(val_metrics)
    
    return final_metrics


def evaluate_anes_classifier(
    anes_classifier: Union[ProspectTheoryANESClassifier, BERTANESClassifier], 
    dataloader, 
    extractor, 
    bias_representer, 
    device: str = 'cpu',
    use_bert_classifier: bool = False,
    target_names: List[str] = None,
    thresholds: List[float] = None
) -> Dict:
    """
    Evaluate the ANES classifier.
    
    Args:
        anes_classifier: ProspectTheoryANESClassifier or BERTANESClassifier instance
        dataloader: DataLoader for evaluation data
        extractor: HiddenLayerExtractor instance
        bias_representer: CognitiveBiasRepresenter instance
        device: Device to run the model on
        use_bert_classifier: Whether the classifier is BERT-based
        target_names: Names of target classes
        thresholds: List of thresholds to evaluate
        
    Returns:
        Dictionary of evaluation metrics
    """
    if target_names is None:
        target_names = ['Trump', 'Harris']  # Default for binary classification
        
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]  # Default thresholds
        
    device = torch.device(device) if isinstance(device, str) else device
    anes_classifier.eval()
    
    all_preds = []
    all_targets = []
    all_logits = []
    all_bias_scores = []
    all_system_weights = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            targets = batch['target'].to(device)
            
            if use_bert_classifier:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                logits = anes_classifier(input_ids, attention_mask)
            else:
                texts = batch['text']
                anes_features = batch['anes_features'].to(device)
                
                # Extract activations
                activations = extractor.extract_activations(texts)
                
                # Get bias scores and system representations
                bias_scores = bias_representer.get_bias_scores(activations).to(device)
                weighted_system_rep, system_weights = bias_representer.get_system_representations(activations)
                weighted_system_rep = weighted_system_rep.to(device)
                
                # Collect bias scores and system weights only for non-BERT classifier
                all_bias_scores.append(bias_scores.cpu().numpy())
                all_system_weights.append(system_weights.cpu().numpy())

                # Forward pass
                logits = anes_classifier(anes_features, bias_scores, weighted_system_rep)
            
            # Collect results
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())
    
    # Convert to numpy arrays
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_probs = F.softmax(all_logits, dim=1).numpy()
    
    # Only concatenate if data was collected (i.e., not BERT classifier)
    if not use_bert_classifier:
        all_bias_scores = np.concatenate(all_bias_scores, axis=0)
        all_system_weights = np.concatenate(all_system_weights, axis=0)
    
    # Find best threshold using validation data
    best_threshold = 0.5
    best_accuracy = 0.0
    
    for threshold in np.linspace(0.3, 0.7, 9):  # More fine-grained threshold search
        preds = (all_probs[:, 1] > threshold).astype(int)
        accuracy = (preds == all_targets).mean()
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    print(f"\nBest threshold: {best_threshold:.2f} with accuracy: {best_accuracy:.4f}")
    
    # Evaluate with different thresholds
    results = {}
    
    print("\nEvaluating with different thresholds:")
    for threshold in thresholds:
        # Apply threshold to probabilities
        all_preds = (all_probs[:, 1] > threshold).astype(int)
        
        # Calculate metrics
        accuracy = (all_preds == all_targets).mean()
        
        # Classification report
        try:
            report = classification_report(all_targets, all_preds, target_names=target_names, output_dict=True)
            
            # Print report
            print(f"\n✅ Classification Report (Threshold @ {threshold:.2f}):")
            print(classification_report(all_targets, all_preds, target_names=target_names))
            
            # Store results
            results[f"threshold_{threshold}"] = {
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': confusion_matrix(all_targets, all_preds),
                'macro_precision': report['macro avg']['precision'],
                'macro_recall': report['macro avg']['recall'],
                'macro_f1': report['macro avg']['f1-score'],
                'weighted_precision': report['weighted avg']['precision'],
                'weighted_recall': report['weighted avg']['recall'],
                'weighted_f1': report['weighted avg']['f1-score'],
                'class_metrics': {name: report[name] for name in target_names if name in report}
            }
        except Exception as e:
            print(f"Error computing metrics for threshold {threshold}: {e}")
            results[f"threshold_{threshold}"] = {'accuracy': accuracy, 'error': str(e)}
    
    # Calculate bias scores and system weights by class (only for non-BERT classifier)
    bias_by_class = {}
    system_by_class = {}
    if not use_bert_classifier:
        for i, class_name in enumerate(target_names):
            mask = all_targets == i
            if mask.sum() > 0:
                bias_by_class[class_name] = all_bias_scores[mask].mean(axis=0)
                system_by_class[class_name] = all_system_weights[mask].mean(axis=0)
    
    # Return all metrics
    return {
        'thresholded_results': results,
        'avg_bias_scores': all_bias_scores.mean(axis=0) if not use_bert_classifier else None,
        'avg_system_weights': all_system_weights.mean(axis=0) if not use_bert_classifier else None,
        'bias_by_class': bias_by_class if not use_bert_classifier else None,
        'system_by_class': system_by_class if not use_bert_classifier else None,
        'best_threshold': best_threshold,
        'best_accuracy': best_accuracy,
        'system_weights': all_system_weights.mean(axis=0) if not use_bert_classifier else None  # For compatibility with main script
    }


if __name__ == "__main__":
    # Example usage
    print("ANES Classifier module loaded successfully!")
    print("Key components:")
    print("- FocalLoss: Advanced loss function for imbalanced classification")
    print("- ProspectTheoryANESClassifier: Main classifier combining multiple feature types")
    print("- BERTANESClassifier: BERT-based classifier for direct text classification")
    print("- train_anes_classifier: Training function with proper error handling")
    print("- evaluate_anes_classifier: Comprehensive evaluation with multiple metrics")




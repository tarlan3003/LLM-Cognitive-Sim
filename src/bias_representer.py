"""
Cognitive Bias Representation Module - Best Performing Version

This module implements the cognitive bias representation model that detects
specific biases from Prospect Theory in text using Concept Activation Vectors
and System 1/2 thinking classification.

Author: Tarlan Sultanov
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional

class CognitiveBiasRepresenter(nn.Module):
    """
    Model for representing cognitive biases from Prospect Theory.
    
    This model has two main components:
    1. Concept Activation Vectors (CAVs) for specific bias detection
    2. System 1/2 thinking classification with adapter-based approach
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        num_biases: int, 
        system_adapter_dim: int = 128,
        dropout: float = 0.3
    ):
        """
        Initialize the cognitive bias representer.
        
        Args:
            input_dim: Dimension of input representations
            hidden_dim: Dimension of hidden layers
            num_biases: Number of bias types to detect
            system_adapter_dim: Dimension of system adapter
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_biases = num_biases
        self.system_adapter_dim = system_adapter_dim
        
        # Bias detection components
        self.bias_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.bias_classifier = nn.Linear(hidden_dim // 2, num_biases)
        
        # System 1/2 thinking components
        self.system_adapter_1 = nn.Sequential(
            nn.Linear(input_dim, system_adapter_dim),
            nn.LayerNorm(system_adapter_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.system_adapter_2 = nn.Sequential(
            nn.Linear(input_dim, system_adapter_dim),
            nn.LayerNorm(system_adapter_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.system_classifier = nn.Linear(system_adapter_dim * 2, 2)
        
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
    
    def forward(self, hidden_reps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            hidden_reps: Hidden representations from LLM
            
        Returns:
            Tuple of (bias_scores, system_weights)
        """
        # Bias detection
        bias_features = self.bias_encoder(hidden_reps)
        bias_scores = torch.sigmoid(self.bias_classifier(bias_features))
        
        # System 1/2 thinking
        system_1_features = self.system_adapter_1(hidden_reps)
        system_2_features = self.system_adapter_2(hidden_reps)
        
        # Concatenate system features
        system_features = torch.cat([system_1_features, system_2_features], dim=1)
        system_logits = self.system_classifier(system_features)
        system_weights = torch.softmax(system_logits, dim=1)
        
        return bias_scores, system_weights
    
    def train(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        extractor: 'HiddenLayerExtractor',
        num_epochs: int = 5,
        learning_rate: float = 3e-4,
        device: torch.device = None
    ) -> Dict[str, float]:
        """
        Train the cognitive bias representer.
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            extractor: Hidden layer extractor
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            device: Device to train on
            
        Returns:
            Dictionary of evaluation metrics
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.to(device)
        
        # Define loss functions
        bias_criterion = nn.BCELoss()
        system_criterion = nn.CrossEntropyLoss()
        
        # Define optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate)
        
        # Training loop
        best_val_loss = float('inf')
        best_metrics = {}
        
        for epoch in range(num_epochs):
            # Training
            self.train()
            train_bias_loss = 0.0
            train_system_loss = 0.0
            train_total_loss = 0.0
            
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)"):
                # Extract hidden representations
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                bias_labels = batch['bias_labels'].to(device)
                system_label = batch['system_label'].to(device)
                
                hidden_reps = extractor(input_ids, attention_mask)
                
                # Forward pass
                bias_scores, system_weights = self(hidden_reps)
                
                # Calculate losses
                bias_loss = bias_criterion(bias_scores, bias_labels)
                system_loss = system_criterion(system_weights, system_label)
                
                # Combined loss
                total_loss = bias_loss + system_loss
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                # Update metrics
                train_bias_loss += bias_loss.item()
                train_system_loss += system_loss.item()
                train_total_loss += total_loss.item()
            
            # Calculate average losses
            train_bias_loss /= len(train_dataloader)
            train_system_loss /= len(train_dataloader)
            train_total_loss /= len(train_dataloader)
            
            # Validation
            val_metrics = self.evaluate(val_dataloader, extractor, device)
            
            # Print metrics
            print(f"Epoch {epoch+1}/{num_epochs}:")
            print(f"  Train - Bias Loss: {train_bias_loss:.4f}, System Loss: {train_system_loss:.4f}, Total Loss: {train_total_loss:.4f}")
            print(f"  Val - Bias Loss: {val_metrics['bias_loss']:.4f}, System Loss: {val_metrics['system_loss']:.4f}, Total Loss: {val_metrics['total_loss']:.4f}")
            print(f"  Val - System Accuracy: {val_metrics['system_accuracy']:.4f}")
            
            # Save best model
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                best_metrics = val_metrics
        
        return best_metrics
    
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        extractor: 'HiddenLayerExtractor',
        device: torch.device = None
    ) -> Dict[str, float]:
        """
        Evaluate the cognitive bias representer.
        
        Args:
            dataloader: DataLoader for evaluation data
            extractor: Hidden layer extractor
            device: Device to evaluate on
            
        Returns:
            Dictionary of evaluation metrics
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.to(device)
        self.eval()
        
        # Define loss functions
        bias_criterion = nn.BCELoss()
        system_criterion = nn.CrossEntropyLoss()
        
        # Evaluation metrics
        bias_loss = 0.0
        system_loss = 0.0
        total_loss = 0.0
        system_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Extract hidden representations
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                bias_labels = batch['bias_labels'].to(device)
                system_label = batch['system_label'].to(device)
                
                hidden_reps = extractor(input_ids, attention_mask)
                
                # Forward pass
                bias_scores, system_weights = self(hidden_reps)
                
                # Calculate losses
                batch_bias_loss = bias_criterion(bias_scores, bias_labels)
                batch_system_loss = system_criterion(system_weights, system_label)
                
                # Combined loss
                batch_total_loss = batch_bias_loss + batch_system_loss
                
                # Update metrics
                bias_loss += batch_bias_loss.item() * input_ids.size(0)
                system_loss += batch_system_loss.item() * input_ids.size(0)
                total_loss += batch_total_loss.item() * input_ids.size(0)
                
                # Calculate system accuracy
                system_preds = torch.argmax(system_weights, dim=1)
                system_correct += (system_preds == system_label).sum().item()
                total_samples += input_ids.size(0)
        
        # Calculate average metrics
        bias_loss /= total_samples
        system_loss /= total_samples
        total_loss /= total_samples
        system_accuracy = system_correct / total_samples
        
        return {
            'bias_loss': bias_loss,
            'system_loss': system_loss,
            'total_loss': total_loss,
            'system_accuracy': system_accuracy
        }
    
    def get_bias_scores(
        self,
        hidden_reps: torch.Tensor
    ) -> torch.Tensor:
        """
        Get bias scores for input representations.
        
        Args:
            hidden_reps: Hidden representations from LLM
            
        Returns:
            Bias scores
        """
        self.eval()
        with torch.no_grad():
            bias_features = self.bias_encoder(hidden_reps)
            bias_scores = torch.sigmoid(self.bias_classifier(bias_features))
        
        return bias_scores
    
    def get_system_weights(
        self,
        hidden_reps: torch.Tensor
    ) -> torch.Tensor:
        """
        Get system weights for input representations.
        
        Args:
            hidden_reps: Hidden representations from LLM
            
        Returns:
            System weights
        """
        self.eval()
        with torch.no_grad():
            system_1_features = self.system_adapter_1(hidden_reps)
            system_2_features = self.system_adapter_2(hidden_reps)
            system_features = torch.cat([system_1_features, system_2_features], dim=1)
            system_logits = self.system_classifier(system_features)
            system_weights = torch.softmax(system_logits, dim=1)
        
        return system_weights

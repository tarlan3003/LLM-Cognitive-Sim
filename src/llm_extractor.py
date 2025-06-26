"""
LLM Hidden Layer Extraction Module - Best Performing Version

This module handles the extraction of hidden layer representations from
large language models for cognitive bias detection.

Author: Tarlan Sultanov
"""

import torch
import numpy as np
from transformers import AutoModel
from typing import List, Dict, Union, Tuple

class HiddenLayerExtractor:
    """
    Extracts hidden layer representations from large language models.
    
    This class handles:
    1. Loading pre-trained language models
    2. Extracting activations from specific layers
    3. Processing these activations for downstream tasks
    """
    
    def __init__(self, model_name: str, layers: List[int] = [-1]):
        """
        Initialize the hidden layer extractor.
        
        Args:
            model_name: Name of the pre-trained model to use
            layers: List of layers to extract (negative indices count from the end)
        """
        self.model_name = model_name
        self.layers = layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        print(f"Loading {model_name} on {self.device}...")
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Get model configuration
        self.config = self.model.config
        self.hidden_size = self.config.hidden_size
        
        # Register hooks for layer extraction
        self.hooks = []
        self.layer_activations = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """
        Register forward hooks to extract layer activations.
        """
        def get_activation(name):
            def hook(model, input, output):
                self.layer_activations[name] = output
            return hook
        
        # Remove any existing hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # Register new hooks
        if hasattr(self.model, 'encoder'):
            # RoBERTa, BERT, etc.
            for i, layer in enumerate(self.model.encoder.layer):
                if i in self.layers or i - len(self.model.encoder.layer) in self.layers:
                    hook = layer.register_forward_hook(get_activation(f'layer_{i}'))
                    self.hooks.append(hook)
        elif hasattr(self.model, 'h'):
            # GPT-2, etc.
            for i, layer in enumerate(self.model.h):
                if i in self.layers or i - len(self.model.h) in self.layers:
                    hook = layer.register_forward_hook(get_activation(f'layer_{i}'))
                    self.hooks.append(hook)
        else:
            raise ValueError(f"Unsupported model architecture: {self.model_name}")
    
    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Extract hidden layer representations.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Concatenated hidden layer representations
        """
        # Clear previous activations
        self.layer_activations = {}
        
        # Move inputs to device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Process activations
        all_activations = []
        
        # Add last hidden state if -1 in layers
        if -1 in self.layers:
            last_hidden = outputs.last_hidden_state
            all_activations.append(self._process_activation(last_hidden, attention_mask))
        
        # Add other layer activations
        for name, activation in self.layer_activations.items():
            # Skip if this is the last layer (already added)
            if name == f'layer_{len(self.model.encoder.layer) - 1}' and -1 in self.layers:
                continue
            
            processed = self._process_activation(activation, attention_mask)
            all_activations.append(processed)
        
        # Concatenate all activations
        if len(all_activations) == 1:
            return all_activations[0]
        else:
            return torch.cat(all_activations, dim=1)
    
    def _process_activation(self, activation: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Process layer activation.
        
        Args:
            activation: Layer activation tensor
            attention_mask: Attention mask
            
        Returns:
            Processed activation
        """
        # Mean pooling with attention mask
        mask_expanded = attention_mask.unsqueeze(-1).expand(activation.size()).float()
        sum_activations = torch.sum(activation * mask_expanded, 1)
        sum_mask = torch.sum(mask_expanded, 1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        
        return sum_activations / sum_mask
    
    def get_output_dim(self) -> int:
        """
        Get the output dimension of the extractor.
        
        Returns:
            Output dimension
        """
        return self.hidden_size * len(self.layers)
    
    def get_layer_names(self) -> List[str]:
        """
        Get the names of extracted layers.
        
        Returns:
            List of layer names
        """
        layer_names = []
        for layer_idx in self.layers:
            if layer_idx < 0:
                actual_idx = len(self.model.encoder.layer) + layer_idx
            else:
                actual_idx = layer_idx
            layer_names.append(f'layer_{actual_idx}')
        
        return layer_names

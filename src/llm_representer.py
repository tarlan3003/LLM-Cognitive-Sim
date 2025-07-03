"""
LLM Hidden Layer Extraction Module for Prospect Theory Pipeline

This module provides functionality to extract hidden layer representations
from pre-trained language models for use in the Prospect Theory pipeline.
"""
import torch
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Union, Optional

class HiddenLayerExtractor:
    """
    Extract hidden layer representations from pre-trained language models.
    """
    def __init__(self, model_name: str, target_layers_indices: List[int], device: str = 'cpu'):
        """
        Initialize the hidden layer extractor.
        
        Args:
            model_name: Name of the pre-trained model to use
            target_layers_indices: List of layer indices to extract from
            device: Device to run the model on
        """
        self.device = device
        print(f"Loading model {model_name}...")
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.target_layers_indices = target_layers_indices
        self.hooks = []
        self._activations = {}
        # Ensure model is in eval mode
        self.model.eval()
        print(f"Model loaded successfully on {self.device}")
    def _get_layer(self, layer_idx: int):
        """
        Get the specified layer from the model.
        
        Args:
            layer_idx: Index of the layer to get
        Returns:
            The specified layer module
        """
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
            layers = self.model.encoder.layer
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            layers = self.model.transformer.h
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'decoder') and hasattr(self.model.model.decoder, 'layers'):
            layers = self.model.model.decoder.layers
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
        else:
            raise NotImplementedError(f"Layer access not implemented for {self.model.config.model_type}")
        if layer_idx < 0:
            layer_idx = len(layers) + layer_idx
        if not (0 <= layer_idx < len(layers)):
            raise ValueError(f"Layer index {layer_idx} out of bounds for model with {len(layers)} layers.")
        return layers[layer_idx]
    def _hook_fn(self, layer_name: str):
        """Hook function to capture layer activations."""
        def fn(module, input, output):
            activation = output[0] if isinstance(output, tuple) else output
            self._activations[layer_name] = activation.detach()
        return fn
    def _register_hooks(self):
        """Register forward hooks for all target layers."""
        self._remove_hooks()
        for layer_idx in self.target_layers_indices:
            try:
                layer = self._get_layer(layer_idx)
                hook = layer.register_forward_hook(self._hook_fn(f"layer_{layer_idx}"))
                self.hooks.append(hook)
            except (NotImplementedError, ValueError) as e:
                print(f"Warning: Could not register hook for layer {layer_idx}: {e}")
    def _remove_hooks(self):
        """Remove all registered hooks and reset activations."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self._activations = {}
    def extract_activations(self, text_batch: Union[str, List[str]], pooling: str = 'mean') -> Dict[str, torch.Tensor]:
        """
        Extract activations for a batch of texts.
        
        Args:
            text_batch: Text or list of texts to extract activations for
            pooling: Pooling method to use ('mean', 'max', 'cls')
        Returns:
            Dictionary mapping layer names to activation tensors
        """
        self._register_hooks()
        # Handle single text vs list of texts
        if isinstance(text_batch, str):
            text_batch = [text_batch]
        inputs = self.tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            for layer_idx in self.target_layers_indices:
                actual_layer_idx = layer_idx + 1 if layer_idx >= 0 else layer_idx
                if abs(actual_layer_idx) < len(outputs.hidden_states):
                    self._activations[f"layer_{layer_idx}"] = outputs.hidden_states[actual_layer_idx]
                else:
                    print(f"Warning: Requested layer {layer_idx} (actual {actual_layer_idx}) not found in model's hidden states.")
        processed_activations = {}
        for layer_name, activation in self._activations.items():
            # activation shape: [batch_size, seq_len, hidden_size]
            if pooling == 'mean':
                pooled_activation = activation.mean(dim=1)
            elif pooling == 'max':
                pooled_activation = activation.max(dim=1)[0]
            elif pooling == 'cls':
                if activation.shape[1] > 0:
                    pooled_activation = activation[:, 0]
                else:
                    print(f"Warning: Cannot use 'cls' pooling for {layer_name} due to zero sequence length. Using mean pooling instead.")
                    pooled_activation = activation.mean(dim=1)
            else:
                raise ValueError(f"Unsupported pooling method: {pooling}")
            processed_activations[layer_name] = pooled_activation.cpu()
        self._remove_hooks()
        return processed_activations
    def get_hidden_size(self) -> int:
        """
        Get the hidden size of the model.
        
        Returns:
            Hidden size
        """
        return self.model.config.hidden_size

if __name__ == "__main__":
    # Example usage with RoBERTa model
    extractor = HiddenLayerExtractor("roberta-base", [-1, -2])
    text = "Political interest: Very much interested\nCampaign interest: Somewhat interested\nEconomic views: Liberal\nState: California\nMedia consumption: Daily\nQ: Who would this respondent vote for in a Harris vs Trump election?"
    activations = extractor.extract_activations(text)
    # Print activation shapes for each target layer
    for layer_name, activation in activations.items():
        print(f"{layer_name}: {activation.shape}")

"""
LLM Hidden Layer Extraction Module for Prospect Theory Pipeline

This module provides functionality to extract hidden layer representations
from pre-trained language models for use in the Prospect Theory pipeline.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, RobertaModel, RobertaTokenizer
from typing import Dict, List, Union, Optional


class HiddenLayerExtractor:
    """
    Extract hidden layer representations from pre-trained language models.
    """
    
    def __init__(
        self, 
        model_name: str, 
        target_layers_indices: List[int], 
        device: str = 'cpu'
    ):
        """
        Initialize the hidden layer extractor.
        
        Args:
            model_name: Name of the pre-trained model to use
            target_layers_indices: List of layer indices to extract from
            device: Device to run the model on
        """
        self.device = device
        print(f"Loading model {model_name}...")
        
        # Use RobertaModel for compatibility with the original notebook
        if "roberta" in model_name.lower():
            self.model = RobertaModel.from_pretrained(model_name).to(self.device)
            self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
        self.target_layers_indices = target_layers_indices
        self.hooks = []
        self._activations = {}

        # Ensure model is in eval mode for consistent activations
        self.model.eval()
        print(f"Model loaded successfully on {self.device}")

    def _get_layer(self, layer_idx: int):
        """
        Get the specified layer from the model.
        
        Args:
            layer_idx: Index of the layer to get
            
        Returns:
            The specified layer
            
        Raises:
            NotImplementedError: If the model architecture is not supported
        """
        # Handle negative indices
        if layer_idx < 0:
            if "roberta" in self.model.config.model_type:
                num_layers = len(self.model.encoder.layer)
                layer_idx = num_layers + layer_idx
                return self.model.encoder.layer[layer_idx]
            elif "opt" in self.model.config.model_type:
                num_layers = len(self.model.model.decoder.layers)
                layer_idx = num_layers + layer_idx
                return self.model.model.decoder.layers[layer_idx]
            elif "gpt2" in self.model.config.model_type:
                num_layers = len(self.model.transformer.h)
                layer_idx = num_layers + layer_idx
                return self.model.transformer.h[layer_idx]
            elif "llama" in self.model.config.model_type:
                num_layers = len(self.model.model.layers)
                layer_idx = num_layers + layer_idx
                return self.model.model.layers[layer_idx]
            else:
                raise NotImplementedError(f"Layer access not implemented for {self.model.config.model_type}")
        
        # Handle positive indices
        if "roberta" in self.model.config.model_type:
            return self.model.encoder.layer[layer_idx]
        elif "opt" in self.model.config.model_type:
            return self.model.model.decoder.layers[layer_idx]
        elif "gpt2" in self.model.config.model_type:
            return self.model.transformer.h[layer_idx]
        elif "llama" in self.model.config.model_type:
             return self.model.model.layers[layer_idx]
        else:
            raise NotImplementedError(f"Layer access not implemented for {self.model.config.model_type}")

    def _hook_fn(self, layer_name: str):
        """
        Hook function to capture layer activations.
        
        Args:
            layer_name: Name to use for the layer in the activations dictionary
            
        Returns:
            Hook function
        """
        def fn(module, input, output):
            # Output can be a tuple, hidden state is usually the first element
            activation = output[0] if isinstance(output, tuple) else output
            self._activations[layer_name] = activation.detach()
        return fn

    def _register_hooks(self):
        """Register hooks for all target layers."""
        self._remove_hooks()
        for layer_idx in self.target_layers_indices:
            layer = self._get_layer(layer_idx)
            hook = layer.register_forward_hook(self._hook_fn(f"layer_{layer_idx}"))
            self.hooks.append(hook)

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self._activations = {}

    def extract_activations(
        self, 
        text_batch: Union[str, List[str]], 
        pooling: str = 'mean'
    ) -> Dict[str, torch.Tensor]:
        """
        Extract activations for a batch of texts.
        
        Args:
            text_batch: Text or list of texts to extract activations for
            pooling: Pooling method to use ('mean', 'max', 'cls')
            
        Returns:
            Dictionary mapping layer names to activation tensors
        """
        self._register_hooks()
        
        # Handle both single text and list of texts
        if isinstance(text_batch, str):
            text_batch = [text_batch]
            
        inputs = self.tokenizer(
            text_batch, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            # For RoBERTa, use the model directly
            if "roberta" in self.model.config.model_type:
                self.model(**inputs)
            else:
                # For other models, use the model with inputs
                self.model(**inputs)
        
        # Process activations from all layers
        processed_activations = {}
        for layer_name, activation in self._activations.items():
            # Shape: [batch_size, seq_len, hidden_size]
            if pooling == 'mean':
                # Mean pool over sequence length
                pooled_activation = activation.mean(dim=1)
            elif pooling == 'max':
                # Max pool over sequence length
                pooled_activation = activation.max(dim=1)[0]
            elif pooling == 'cls':
                # Use [CLS] token representation (first token)
                pooled_activation = activation[:, 0]
            else:
                raise ValueError(f"Unsupported pooling method: {pooling}")
                
            processed_activations[layer_name] = pooled_activation.cpu()
            
        self._remove_hooks()  # Clean up hooks after extraction
        return processed_activations
    
    def get_hidden_size(self) -> int:
        """
        Get the hidden size of the model.
        
        Returns:
            Hidden size
        """
        return self.model.config.hidden_size


if __name__ == "__main__":
    # Example usage with RoBERTa model from original notebook
    extractor = HiddenLayerExtractor("roberta-base", [-1, -2])
    
    # Extract activations for a single text
    text = "Political interest: Very much interested\nCampaign interest: Somewhat interested\nEconomic views: Liberal\nState: California\nMedia consumption: Daily\nQ: Who would this respondent vote for in a Harris vs Trump election?"
    activations = extractor.extract_activations(text)
    
    # Print activation shapes
    for layer_name, activation in activations.items():
        print(f"{layer_name}: {activation.shape}")

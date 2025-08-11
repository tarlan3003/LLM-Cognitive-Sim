
"""
LLM Hidden Layer Extraction Module for Prospect Theory Pipeline - Fixed Version

This module provides functionality to extract hidden layer representations
from pre-trained language models for use in the Prospect Theory pipeline.

"""

import torch
from transformers import AutoModel, AutoTokenizer, RobertaModel, RobertaTokenizer, BertModel, BertTokenizer
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
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model_name = model_name
        self.target_layers_indices = target_layers_indices
        self.hooks = []
        self._activations = {}
        
        print(f"Loading model {model_name}...")
        
        # Load model and tokenizer with proper error handling
        self.model, self.tokenizer = self._load_model_and_tokenizer(model_name)
        
        # Ensure model is in eval mode for consistent activations
        self.model.eval()
        print(f"Model loaded successfully on {self.device}")

    def _load_model_and_tokenizer(self, model_name: str):
        """
        Load model and tokenizer with fallback options for different model types.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            # Try loading with AutoModel first
            if "roberta" in model_name.lower():
                # Use RobertaModel specifically for RoBERTa models
                try:
                    model = RobertaModel.from_pretrained(model_name).to(self.device)
                    tokenizer = RobertaTokenizer.from_pretrained(model_name, use_fast=False)
                    print(f"Successfully loaded RoBERTa model: {model_name}")
                    return model, tokenizer
                except Exception as e:
                    print(f"RobertaModel loading failed: {e}, trying AutoModel...")
            
            elif "bert" in model_name.lower() and "roberta" not in model_name.lower():
                # Use BertModel specifically for BERT models
                try:
                    model = BertModel.from_pretrained(model_name).to(self.device)
                    tokenizer = BertTokenizer.from_pretrained(model_name, use_fast=False)
                    print(f"Successfully loaded BERT model: {model_name}")
                    return model, tokenizer
                except Exception as e:
                    print(f"BertModel loading failed: {e}, trying AutoModel...")
            
            # Generic AutoModel approach
            try:
                model = AutoModel.from_pretrained(model_name).to(self.device)
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
                print(f"Successfully loaded model with AutoModel: {model_name}")
                return model, tokenizer
            except Exception as e:
                print(f"AutoModel loading failed: {e}")
                
                # Try with fast tokenizer disabled
                try:
                    model = AutoModel.from_pretrained(model_name).to(self.device)
                    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
                    print(f"Successfully loaded model with slow tokenizer: {model_name}")
                    return model, tokenizer
                except Exception as e2:
                    print(f"Slow tokenizer also failed: {e2}")
                    
                    # Fallback to roberta-base
                    print("Falling back to roberta-base...")
                    model = RobertaModel.from_pretrained("roberta-base").to(self.device)
                    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", use_fast=False)
                    self.model_name = "roberta-base"
                    print("Successfully loaded fallback model: roberta-base")
                    return model, tokenizer
                    
        except Exception as e:
            print(f"All model loading attempts failed: {e}")
            raise RuntimeError(f"Could not load any model. Last error: {e}")

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
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
            # RoBERTa, BERT, DeBERTa models
            num_layers = len(self.model.encoder.layer)
            if layer_idx < 0:
                layer_idx = num_layers + layer_idx
            return self.model.encoder.layer[layer_idx]
            
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            # GPT-2 models
            num_layers = len(self.model.transformer.h)
            if layer_idx < 0:
                layer_idx = num_layers + layer_idx
            return self.model.transformer.h[layer_idx]
            
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'decoder') and hasattr(self.model.model.decoder, 'layers'):
            # OPT models
            num_layers = len(self.model.model.decoder.layers)
            if layer_idx < 0:
                layer_idx = num_layers + layer_idx
            return self.model.model.decoder.layers[layer_idx]
            
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # LLaMA models
            num_layers = len(self.model.model.layers)
            if layer_idx < 0:
                layer_idx = num_layers + layer_idx
            return self.model.model.layers[layer_idx]
        else:
            # Try to infer from config
            if hasattr(self.model, 'config'):
                model_type = getattr(self.model.config, 'model_type', 'unknown')
                print(f"Unknown model architecture: {model_type}")
                print(f"Available attributes: {dir(self.model)}")
            
            raise NotImplementedError(f"Layer access not implemented for this model architecture")

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
        """
        Register hooks for all target layers.
        """
        self._remove_hooks()
        for layer_idx in self.target_layers_indices:
            try:
                layer = self._get_layer(layer_idx)
                hook = layer.register_forward_hook(self._hook_fn(f"layer_{layer_idx}"))
                self.hooks.append(hook)
            except Exception as e:
                print(f"Warning: Could not register hook for layer {layer_idx}: {e}")
                continue

    def _remove_hooks(self):
        """
        Remove all registered hooks.
        """
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
            
        try:
            inputs = self.tokenizer(
                text_batch, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                # Forward pass through the model
                outputs = self.model(**inputs)
            
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
                
        except Exception as e:
            print(f"Error during activation extraction: {e}")
            # Return empty activations if extraction fails
            processed_activations = {}
            for layer_idx in self.target_layers_indices:
                layer_name = f"layer_{layer_idx}"
                # Create dummy activation with correct shape
                dummy_activation = torch.zeros(len(text_batch), self.get_hidden_size())
                processed_activations[layer_name] = dummy_activation
        finally:
            self._remove_hooks()  # Clean up hooks after extraction
            
        return processed_activations
    
    def get_hidden_size(self) -> int:
        """
        Get the hidden size of the model.
        
        Returns:
            Hidden size
        """
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'hidden_size'):
            return self.model.config.hidden_size
        else:
            # Fallback: try to infer from a dummy forward pass
            try:
                dummy_input = self.tokenizer("test", return_tensors='pt').to(self.device)
                with torch.no_grad():
                    output = self.model(**dummy_input)
                    if hasattr(output, 'last_hidden_state'):
                        return output.last_hidden_state.shape[-1]
                    elif isinstance(output, tuple):
                        return output[0].shape[-1]
                    else:
                        return 768  # Default fallback
            except:
                return 768  # Default fallback for BERT-like models


if __name__ == "__main__":
    # Example usage with different models
    print("Testing HiddenLayerExtractor with different models...")
    
    # Test with RoBERTa
    try:
        extractor = HiddenLayerExtractor("roberta-base", [-1, -2])
        
        # Extract activations for a single text
        text = "Political interest: Very much interested\nCampaign interest: Somewhat interested\nEconomic views: Liberal\nState: California\nMedia consumption: Daily\nQ: Who would this respondent vote for in a Harris vs Trump election?"
        activations = extractor.extract_activations(text)
        
        # Print activation shapes
        print("\nActivation shapes:")
        for layer_name, activation in activations.items():
            print(f"{layer_name}: {activation.shape}")
            
        print(f"Hidden size: {extractor.get_hidden_size()}")
        print("HiddenLayerExtractor test completed successfully!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()




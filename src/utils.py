"""
Utility functions for the Prospect Theory LLM Pipeline.

This module provides utility functions for the Prospect Theory LLM Pipeline,
including directory creation, seed setting, and other helper functions.
"""

import os
import random
import numpy as np
import torch
from typing import Optional


def create_directory_structure(
    data_dir: str = "data",
    model_dir: str = "models",
    output_dir: str = "results"
):
    """
    Create the directory structure for the Prospect Theory LLM Pipeline.
    
    Args:
        data_dir: Directory for data
        model_dir: Directory for models
        output_dir: Directory for outputs
    """
    # Create data directories
    os.makedirs(os.path.join(data_dir, "prospect_theory"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "anes"), exist_ok=True)
    
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Created directory structure: {data_dir}, {model_dir}, {output_dir}")


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Set random seed to {seed}")


def get_pytorch_version() -> str:
    """
    Get the PyTorch version.
    
    Returns:
        PyTorch version string
    """
    return torch.__version__


def check_compatibility():
    """
    Check compatibility of the environment.
    
    Returns:
        Dictionary of compatibility information
    """
    import platform
    import sys
    
    return {
        "python_version": platform.python_version(),
        "pytorch_version": get_pytorch_version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine()
    }


def get_available_models():
    """
    Get a list of recommended models for the pipeline.
    
    Returns:
        Dictionary of recommended models with descriptions
    """
    return {
        "roberta-base": "Smaller model, faster training, less memory usage",
        "roberta-large": "Larger model, better performance, more memory usage",
        "microsoft/deberta-v3-base": "Modern architecture, excellent performance",
        "microsoft/deberta-v3-large": "State-of-the-art performance, high memory usage",
        "bert-base-uncased": "Classic model, widely used, moderate performance",
        "bert-large-uncased": "Larger classic model, better performance"
    }


def get_memory_usage():
    """
    Get current memory usage.
    
    Returns:
        Dictionary of memory usage information
    """
    import psutil
    
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        "rss": memory_info.rss / (1024 ** 2),  # RSS in MB
        "vms": memory_info.vms / (1024 ** 2),  # VMS in MB
        "percent": process.memory_percent()
    }


def estimate_model_memory(model_name: str) -> float:
    """
    Estimate memory usage for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Estimated memory usage in GB
    """
    # Rough estimates based on model size
    memory_estimates = {
        "roberta-base": 0.5,
        "roberta-large": 1.5,
        "microsoft/deberta-v3-base": 0.7,
        "microsoft/deberta-v3-large": 2.0,
        "bert-base-uncased": 0.4,
        "bert-large-uncased": 1.3
    }
    
    # Default estimate if model not in list
    return memory_estimates.get(model_name, 1.0)


def check_model_compatibility(model_name: str) -> bool:
    """
    Check if a model is compatible with the current environment.
    
    Args:
        model_name: Name of the model
        
    Returns:
        True if compatible, False otherwise
    """
    try:
        from transformers import AutoModel, AutoTokenizer
        
        # Try to load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Try to load model config (without downloading full model)
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        
        return True
    except Exception as e:
        print(f"Error checking model compatibility: {e}")
        return False


def get_recommended_batch_size(model_name: str) -> int:
    """
    Get recommended batch size for a model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Recommended batch size
    """
    # Base recommendations on model size and available memory
    if not torch.cuda.is_available():
        # CPU mode - use smaller batches
        if "large" in model_name:
            return 4
        else:
            return 8
    
    # GPU mode - check memory
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
    
    if "large" in model_name:
        if gpu_memory > 16:
            return 16
        elif gpu_memory > 8:
            return 8
        else:
            return 4
    else:
        if gpu_memory > 16:
            return 32
        elif gpu_memory > 8:
            return 16
        else:
            return 8


if __name__ == "__main__":
    # Example usage
    create_directory_structure()
    set_seed(42)
    
    # Print compatibility information
    compatibility = check_compatibility()
    print("\nCompatibility Information:")
    for key, value in compatibility.items():
        print(f"  {key}: {value}")
    
    # Print available models
    models = get_available_models()
    print("\nRecommended Models:")
    for model, description in models.items():
        print(f"  {model}: {description}")
        print(f"    - Estimated memory: {estimate_model_memory(model):.1f} GB")
        print(f"    - Recommended batch size: {get_recommended_batch_size(model)}")
        print(f"    - Compatible: {check_model_compatibility(model)}")

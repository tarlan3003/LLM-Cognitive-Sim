
"""
Cognitive Bias Representation Module for Prospect Theory Pipeline - Fixed Version with Dimension Handling

This module provides functionality to represent cognitive biases and System 1/2 thinking
patterns based on LLM hidden layer representations.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union


class CognitiveBiasRepresenter:
    """
    Represent cognitive biases and System 1/2 thinking patterns based on LLM activations.
    
    This class combines Concept Activation Vectors (CAVs) for specific biases with
    a Mixture of Experts (MoE) approach for System 1/2 thinking.
    """
    
    def __init__(
        self, 
        llm_hidden_size: int, 
        bias_names: List[str], 
        system_adapter_bottleneck: int = 128, 
        device: str = 'cpu',
        num_layers: int = 4  # Number of layers being extracted
    ):
        """
        Initialize the cognitive bias representer.
        
        Args:
            llm_hidden_size: Hidden size of the LLM (single layer)
            bias_names: List of bias names to represent
            system_adapter_bottleneck: Bottleneck size for System 1/2 adapters
            device: Device to run the model on
            num_layers: Number of layers being extracted (for dimension calculation)
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        self.bias_names = bias_names
        self.llm_hidden_size = llm_hidden_size
        self.num_layers = num_layers
        self.cav_classifiers = {}  # To be populated with trained LogisticRegression models for each bias
        
        # Calculate combined input size (multiple layers concatenated)
        self.combined_input_size = llm_hidden_size * num_layers
        
        print(f"Initializing with single layer size: {llm_hidden_size}")
        print(f"Number of layers: {num_layers}")
        print(f"Combined input size: {self.combined_input_size}")
        
        # Dimension reduction layer to handle variable input sizes
        self.input_projection = nn.Linear(self.combined_input_size, llm_hidden_size).to(self.device)
        
        # System 1/2 Adapters (MoE-like) - now using single layer size
        self.system1_adapter = nn.Sequential(
            nn.Linear(llm_hidden_size, system_adapter_bottleneck),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(system_adapter_bottleneck, llm_hidden_size)
        ).to(self.device)
        
        self.system2_adapter = nn.Sequential(
            nn.Linear(llm_hidden_size, system_adapter_bottleneck),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(system_adapter_bottleneck, llm_hidden_size)
        ).to(self.device)
        
        # Router for System 1/2 - using single layer size
        self.system_router = nn.Linear(llm_hidden_size, 2).to(self.device)
        
        print(f"Initialized CognitiveBiasRepresenter with {len(bias_names)} biases on {self.device}")

    def _project_to_single_layer_size(self, combined_activations: torch.Tensor) -> torch.Tensor:
        """
        Project combined activations to single layer size.
        
        Args:
            combined_activations: Concatenated activations from multiple layers
            
        Returns:
            Projected activations with single layer size
        """
        # Check if input size matches expected combined size
        if combined_activations.shape[1] == self.combined_input_size:
            # Use projection layer
            return self.input_projection(combined_activations)
        elif combined_activations.shape[1] == self.llm_hidden_size:
            # Already single layer size
            return combined_activations
        else:
            # Handle unexpected sizes
            print(f"Warning: Unexpected input size {combined_activations.shape[1]}, expected {self.combined_input_size} or {self.llm_hidden_size}")
            
            # Create a new projection layer for this size
            actual_size = combined_activations.shape[1]
            if not hasattr(self, f'_temp_projection_{actual_size}'):
                temp_projection = nn.Linear(actual_size, self.llm_hidden_size).to(self.device)
                setattr(self, f'_temp_projection_{actual_size}', temp_projection)
                print(f"Created temporary projection layer for size {actual_size}")
            
            temp_projection = getattr(self, f'_temp_projection_{actual_size}')
            return temp_projection(combined_activations)

    def train_cav(
        self, 
        bias_name: str, 
        positive_activations: np.ndarray, 
        negative_activations: np.ndarray
    ) -> LogisticRegression:
        """
        Train a CAV for a specific bias.
        
        Args:
            bias_name: Name of the bias to train for
            positive_activations: Activations for positive examples
            negative_activations: Activations for negative examples
            
        Returns:
            Trained classifier
        """
        X = np.concatenate([positive_activations, negative_activations])
        y = np.concatenate([np.ones(len(positive_activations)), np.zeros(len(negative_activations))])
        
        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        classifier = LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=1000)
        classifier.fit(X_train, y_train)
        
        # Evaluate
        train_acc = classifier.score(X_train, y_train)
        test_acc = classifier.score(X_test, y_test)
        
        print(f"Trained CAV for {bias_name}: Train acc={train_acc:.4f}, Test acc={test_acc:.4f}")
        
        self.cav_classifiers[bias_name] = classifier
        return classifier

    def train_cavs(self, dataloader, extractor) -> Dict:
        """
        Train Concept Activation Vectors for each bias.
        
        Args:
            dataloader: DataLoader containing training data
            extractor: HiddenLayerExtractor instance
            
        Returns:
            Dictionary of training metrics
        """
        print("Training CAVs for cognitive biases...")
        
        # Collect activations for each bias
        bias_activations = {bias: {'positive': [], 'negative': []} for bias in self.bias_names}
        
        for batch in tqdm(dataloader, desc="Collecting activations for CAV training"):
            texts = batch['text']
            bias_labels = batch['bias_labels']
            
            # Extract activations from LLM
            activations = extractor.extract_activations(texts)
            
            # Combine activations from all layers
            combined_activations = torch.cat(list(activations.values()), dim=1)
            
            print(f"Combined activations shape: {combined_activations.shape}")
            
            # Separate positive and negative examples for each bias
            for i, bias in enumerate(self.bias_names):
                positive_mask = bias_labels[:, i] == 1
                negative_mask = bias_labels[:, i] == 0
                
                if positive_mask.sum() > 0:
                    bias_activations[bias]['positive'].append(combined_activations[positive_mask])
                if negative_mask.sum() > 0:
                    bias_activations[bias]['negative'].append(combined_activations[negative_mask])
        
        # Train CAV for each bias
        trained_cavs = 0
        for bias in self.bias_names:
            if bias_activations[bias]['positive'] and bias_activations[bias]['negative']:
                positive_acts = torch.cat(bias_activations[bias]['positive']).numpy()
                negative_acts = torch.cat(bias_activations[bias]['negative']).numpy()
                
                if len(positive_acts) > 0 and len(negative_acts) > 0:
                    self.train_cav(bias, positive_acts, negative_acts)
                    trained_cavs += 1
                else:
                    print(f"Warning: Insufficient data for bias {bias}")
            else:
                print(f"Warning: No data found for bias {bias}")
        
        print(f"Successfully trained {trained_cavs}/{len(self.bias_names)} CAVs")
        return {'cavs_trained': trained_cavs, 'total_biases': len(self.bias_names)}

    def train_system_components(self, dataloader, extractor, num_epochs=10, lr=1e-3) -> Dict:
        """
        Train System 1/2 components.
        
        Args:
            dataloader: DataLoader containing training data
            extractor: HiddenLayerExtractor instance
            num_epochs: Number of training epochs
            lr: Learning rate
            
        Returns:
            Dictionary of training metrics
        """
        print("Training System 1/2 components...")
        
        # Optimizer for system components
        optimizer = torch.optim.Adam(
            list(self.input_projection.parameters()) +
            list(self.system1_adapter.parameters()) + 
            list(self.system2_adapter.parameters()) + 
            list(self.system_router.parameters()), 
            lr=lr
        )
        
        epoch_losses = []
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            for batch in tqdm(dataloader, desc=f"System training epoch {epoch+1}/{num_epochs}"):
                texts = batch['text']
                system_labels = batch['system_label'].to(self.device)
                
                # Extract activations
                activations = extractor.extract_activations(texts)
                combined_activations = torch.cat(list(activations.values()), dim=1).to(self.device)
                
                print(f"Batch {num_batches}: Combined activations shape: {combined_activations.shape}")
                
                # Project to single layer size
                projected_activations = self._project_to_single_layer_size(combined_activations)
                
                print(f"Batch {num_batches}: Projected activations shape: {projected_activations.shape}")
                
                # Forward pass through adapters
                system1_rep = self.system1_adapter(projected_activations)
                system2_rep = self.system2_adapter(projected_activations)
                
                # Router predictions
                router_logits = self.system_router(projected_activations)
                
                # Loss computation
                router_loss = F.cross_entropy(router_logits, system_labels)
                
                # Backward pass
                optimizer.zero_grad()
                router_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    list(self.input_projection.parameters()) +
                    list(self.system1_adapter.parameters()) + 
                    list(self.system2_adapter.parameters()) + 
                    list(self.system_router.parameters()), 
                    max_norm=1.0
                )
                
                optimizer.step()
                
                total_loss += router_loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            epoch_losses.append(avg_loss)
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        return {
            'system_training_losses': epoch_losses,
            'final_loss': epoch_losses[-1] if epoch_losses else 0,
            'epochs_completed': num_epochs
        }

    def get_bias_scores(self, activations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get bias scores for given activations.
        
        Args:
            activations: Dictionary of layer activations
            
        Returns:
            Tensor of bias scores [batch_size, num_biases]
        """
        # Combine activations from all layers
        combined_activations = torch.cat(list(activations.values()), dim=1)
        
        # Get bias scores using trained CAVs
        bias_scores = []
        
        for bias in self.bias_names:
            if bias in self.cav_classifiers:
                # Use CAV to get bias score
                classifier = self.cav_classifiers[bias]
                scores = classifier.predict_proba(combined_activations.cpu().numpy())[:, 1]  # Probability of positive class
                bias_scores.append(torch.tensor(scores, dtype=torch.float))
            else:
                # Default to zero if CAV not trained
                bias_scores.append(torch.zeros(combined_activations.shape[0], dtype=torch.float))
        
        return torch.stack(bias_scores, dim=1)

    def get_system_representations(self, activations: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get System 1/2 weighted representations.
        
        Args:
            activations: Dictionary of layer activations
            
        Returns:
            Tuple of (weighted_representation, system_weights)
        """
        # Combine activations from all layers
        combined_activations = torch.cat(list(activations.values()), dim=1).to(self.device)
        
        # Project to single layer size
        projected_activations = self._project_to_single_layer_size(combined_activations)
        
        # Get system representations
        system1_rep = self.system1_adapter(projected_activations)
        system2_rep = self.system2_adapter(projected_activations)
        
        # Get router weights
        router_logits = self.system_router(projected_activations)
        system_weights = F.softmax(router_logits, dim=1)
        
        # Weighted combination
        weighted_rep = (system_weights[:, 0:1] * system1_rep + 
                       system_weights[:, 1:2] * system2_rep)
        
        return weighted_rep, system_weights

    def save(self, path: str) -> None:
        """
        Save the bias representer to a file.
        
        Args:
            path: Path to save the model to
        """
        save_dict = {
            'bias_names': self.bias_names,
            'llm_hidden_size': self.llm_hidden_size,
            'num_layers': self.num_layers,
            'combined_input_size': self.combined_input_size,
            'input_projection': self.input_projection.state_dict(),
            'system1_adapter': self.system1_adapter.state_dict(),
            'system2_adapter': self.system2_adapter.state_dict(),
            'system_router': self.system_router.state_dict(),
            'cav_classifiers': {}
        }
        
        # Save CAV classifiers
        for bias_name, classifier in self.cav_classifiers.items():
            save_dict['cav_classifiers'][bias_name] = {
                'coef_': classifier.coef_,
                'intercept_': classifier.intercept_,
                'classes_': classifier.classes_
            }
        
        torch.save(save_dict, path)
        print(f"Saved CognitiveBiasRepresenter to {path}")

    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'CognitiveBiasRepresenter':
        """
        Load the bias representer from a file.
        
        Args:
            path: Path to load the model from
            device: Device to load the model on
            
        Returns:
            Loaded CognitiveBiasRepresenter
        """
        save_dict = torch.load(path, map_location=device)
        
        # Create instance
        representer = cls(
            llm_hidden_size=save_dict['llm_hidden_size'],
            bias_names=save_dict['bias_names'],
            num_layers=save_dict.get('num_layers', 4),
            device=device
        )
        
        # Load neural network components
        representer.input_projection.load_state_dict(save_dict['input_projection'])
        representer.system1_adapter.load_state_dict(save_dict['system1_adapter'])
        representer.system2_adapter.load_state_dict(save_dict['system2_adapter'])
        representer.system_router.load_state_dict(save_dict['system_router'])
        
        # Load CAV classifiers
        for bias_name, cav_data in save_dict['cav_classifiers'].items():
            classifier = LogisticRegression()
            classifier.coef_ = cav_data['coef_']
            classifier.intercept_ = cav_data['intercept_']
            classifier.classes_ = cav_data['classes_']
            representer.cav_classifiers[bias_name] = classifier
        
        print(f"Loaded CognitiveBiasRepresenter from {path}")
        return representer


if __name__ == "__main__":
    # Example usage
    import torch
    
    # Test dimension handling
    print("Testing CognitiveBiasRepresenter with dimension handling...")
    
    bias_names = ["anchoring", "framing", "availability", "confirmation_bias", "loss_aversion"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize representer
    representer = CognitiveBiasRepresenter(
        llm_hidden_size=1024,  # Single layer size
        bias_names=bias_names,
        num_layers=4,  # Number of layers being extracted
        device=device
    )
    
    # Test with different input sizes
    print("\nTesting dimension projection:")
    
    # Test with combined size (4 layers * 1024 = 4096)
    combined_input = torch.randn(8, 4096).to(device)
    projected = representer._project_to_single_layer_size(combined_input)
    print(f"Combined input {combined_input.shape} -> Projected {projected.shape}")
    
    # Test with single layer size
    single_input = torch.randn(8, 1024).to(device)
    projected_single = representer._project_to_single_layer_size(single_input)
    print(f"Single input {single_input.shape} -> Projected {projected_single.shape}")
    
    print("Dimension handling test completed successfully!")




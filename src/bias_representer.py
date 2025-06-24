"""
Cognitive Bias Representation Module for Prospect Theory Pipeline

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
        device: str = 'cpu'
    ):
        """
        Initialize the cognitive bias representer.
        
        Args:
            llm_hidden_size: Hidden size of the LLM
            bias_names: List of bias names to represent
            system_adapter_bottleneck: Bottleneck size for System 1/2 adapters
            device: Device to run the model on
        """
        self.device = device
        self.bias_names = bias_names
        self.cav_classifiers = {}  # To be populated with trained LogisticRegression models for each bias
        
        # System 1/2 Adapters (MoE-like)
        self.system1_adapter = nn.Sequential(
            nn.Linear(llm_hidden_size, system_adapter_bottleneck),
            nn.ReLU(),
            nn.Linear(system_adapter_bottleneck, llm_hidden_size)
        ).to(device)
        
        self.system2_adapter = nn.Sequential(
            nn.Linear(llm_hidden_size, system_adapter_bottleneck),
            nn.ReLU(),
            nn.Linear(system_adapter_bottleneck, llm_hidden_size)
        ).to(device)
        
        # Router for System 1/2
        self.system_router = nn.Linear(llm_hidden_size, 2).to(device)
        
        print(f"Initialized CognitiveBiasRepresenter with {len(bias_names)} biases on {device}")

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
        
        classifier = LogisticRegression(solver='liblinear', class_weight='balanced')
        classifier.fit(X_train, y_train)
        
        # Evaluate
        train_acc = classifier.score(X_train, y_train)
        test_acc = classifier.score(X_test, y_test)
        
        print(f"Trained CAV for {bias_name}: Train acc={train_acc:.4f}, Test acc={test_acc:.4f}")
        
        self.cav_classifiers[bias_name] = classifier
        return classifier

    def get_bias_scores(
        self, 
        activations: Union[Dict[str, torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """
        Get scores for each bias using trained CAVs.
        
        Args:
            activations: Activations to get scores for (dict of layer activations or tensor)
            
        Returns:
            Tensor of bias scores [batch_size, num_biases]
        """
        bias_scores = []
        
        # Convert to numpy for scikit-learn
        if isinstance(activations, dict):
            # If activations is a dict of layer activations, use the last layer
            layer_name = list(activations.keys())[-1]
            activations_np = activations[layer_name].cpu().numpy()
        else:
            # If activations is already a tensor
            activations_np = activations.cpu().numpy()
            
        for bias_name in self.bias_names:
            if bias_name in self.cav_classifiers:
                # Predict probability of positive class (bias present)
                score = self.cav_classifiers[bias_name].predict_proba(activations_np)[:, 1]
                bias_scores.append(torch.tensor(score, dtype=torch.float).unsqueeze(1))
            else:
                # Return zeros if CAV not trained for this bias
                bias_scores.append(torch.zeros(activations_np.shape[0], 1, dtype=torch.float))
                
        return torch.cat(bias_scores, dim=1)

    def get_system_representations(
        self, 
        activations: Union[Dict[str, torch.Tensor], torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get System 1/2 representations and router weights.
        
        Args:
            activations: Activations to get representations for (dict of layer activations or tensor)
            
        Returns:
            Tuple of (weighted_system_rep, system_weights)
        """
        # Handle dict of layer activations
        if isinstance(activations, dict):
            layer_name = list(activations.keys())[-1]
            activations_tensor = activations[layer_name].to(self.device)
        else:
            activations_tensor = activations.to(self.device)
            
        system_logits = self.system_router(activations_tensor)
        system_weights = F.softmax(system_logits, dim=-1)
        
        rep1 = self.system1_adapter(activations_tensor)
        rep2 = self.system2_adapter(activations_tensor)
        
        # Weighted average of system representations based on router
        # Shape: [batch_size, llm_hidden_size]
        weighted_system_rep = system_weights[:, 0].unsqueeze(1) * rep1 + system_weights[:, 1].unsqueeze(1) * rep2
        
        return weighted_system_rep, system_weights

    def train_system_components(
        self, 
        dataloader, 
        llm_extractor, 
        num_epochs: int = 5, 
        lr: float = 1e-4
    ) -> Dict:
        """
        Train System 1/2 adapters and router.
        
        Args:
            dataloader: DataLoader for training data
            llm_extractor: HiddenLayerExtractor instance
            num_epochs: Number of epochs to train for
            lr: Learning rate
            
        Returns:
            Dictionary of training metrics
        """
        # Optimizer for System 1/2 adapters and router
        system_params = list(self.system1_adapter.parameters()) + \
                        list(self.system2_adapter.parameters()) + \
                        list(self.system_router.parameters())
        optimizer = torch.optim.Adam(system_params, lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        
        # Training loop
        metrics = {'epoch_losses': [], 'epoch_accuracies': []}
        
        for epoch in range(num_epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                texts = batch['text']
                true_system_labels = batch['system_label'].to(self.device)
                
                # Extract activations
                activations = llm_extractor.extract_activations(texts)
                layer_name = list(activations.keys())[-1]
                activations_tensor = activations[layer_name].to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                system_logits = self.system_router(activations_tensor)
                loss = loss_fn(system_logits, true_system_labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Track metrics
                total_loss += loss.item() * len(true_system_labels)
                preds = torch.argmax(system_logits, dim=1)
                correct += (preds == true_system_labels).sum().item()
                total += len(true_system_labels)
            
            # Epoch metrics
            epoch_loss = total_loss / total
            epoch_acc = correct / total
            metrics['epoch_losses'].append(epoch_loss)
            metrics['epoch_accuracies'].append(epoch_acc)
            
            print(f"Epoch {epoch+1}/{num_epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}")
        
        return metrics

    def train_cavs(self, dataloader, llm_extractor) -> Dict[str, LogisticRegression]:
        """
        Train CAVs for each bias type.
        
        Args:
            dataloader: DataLoader for training data
            llm_extractor: HiddenLayerExtractor instance
            
        Returns:
            Dictionary mapping bias names to CAV classifiers
        """
        # Collect activations and labels for each bias
        bias_activations = {bias: {'positive': [], 'negative': []} for bias in self.bias_names}
        
        for batch in tqdm(dataloader, desc="Collecting activations for CAVs"):
            texts = batch['text']
            bias_labels = batch['bias_labels'].cpu().numpy()
            
            # Extract activations
            activations = llm_extractor.extract_activations(texts)
            layer_name = list(activations.keys())[-1]
            activations_np = activations[layer_name].cpu().numpy()
            
            # Collect activations for each bias
            for i, bias_name in enumerate(self.bias_names):
                for j in range(len(texts)):
                    if bias_labels[j, i] == 1:
                        bias_activations[bias_name]['positive'].append(activations_np[j])
                    else:
                        bias_activations[bias_name]['negative'].append(activations_np[j])
        
        # Train CAVs for each bias
        for bias_name in self.bias_names:
            positive = np.array(bias_activations[bias_name]['positive'])
            negative = np.array(bias_activations[bias_name]['negative'])
            
            if len(positive) == 0 or len(negative) == 0:
                print(f"Skipping CAV for {bias_name}: Not enough examples")
                continue
                
            print(f"Training CAV for {bias_name}: {len(positive)} positive, {len(negative)} negative examples")
            self.train_cav(bias_name, positive, negative)
            
        return self.cav_classifiers
    
    def save(self, path: str) -> None:
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model to
        """
        torch.save({
            'system1_adapter': self.system1_adapter.state_dict(),
            'system2_adapter': self.system2_adapter.state_dict(),
            'system_router': self.system_router.state_dict(),
            'bias_names': self.bias_names,
            'cav_classifiers': self.cav_classifiers
        }, path)
        print(f"Saved CognitiveBiasRepresenter to {path}")
    
    @classmethod
    def load(cls, path: str, llm_hidden_size: int, device: str = 'cpu') -> 'CognitiveBiasRepresenter':
        """
        Load the model from a file.
        
        Args:
            path: Path to load the model from
            llm_hidden_size: Hidden size of the LLM
            device: Device to load the model on
            
        Returns:
            Loaded CognitiveBiasRepresenter
        """
        checkpoint = torch.load(path, map_location=device)
        bias_names = checkpoint['bias_names']
        
        model = cls(llm_hidden_size, bias_names, device=device)
        model.system1_adapter.load_state_dict(checkpoint['system1_adapter'])
        model.system2_adapter.load_state_dict(checkpoint['system2_adapter'])
        model.system_router.load_state_dict(checkpoint['system_router'])
        model.cav_classifiers = checkpoint['cav_classifiers']
        
        print(f"Loaded CognitiveBiasRepresenter from {path}")
        return model


if __name__ == "__main__":
    # Example usage with RoBERTa model from original notebook
    import torch.utils.data
    from dataset import ProspectTheoryDataset
    from llm_extractor import HiddenLayerExtractor
    from transformers import RobertaTokenizer
    
    # Create dummy dataset
    os.makedirs("data/prospect_theory", exist_ok=True)
    ProspectTheoryDataset.create_prospect_theory_dataset(
        "data/prospect_theory/dummy.json", num_examples=100
    )
    
    # Load tokenizer and dataset
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    dataset = ProspectTheoryDataset("data/prospect_theory/dummy.json", tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Initialize extractor and representer
    extractor = HiddenLayerExtractor("roberta-base", [-1])
    representer = CognitiveBiasRepresenter(extractor.get_hidden_size(), dataset.bias_names)
    
    # Train CAVs
    representer.train_cavs(dataloader, extractor)
    
    # Train system components
    representer.train_system_components(dataloader, extractor, num_epochs=2)
    
    # Test on a single example
    text = "Political interest: Very much interested\nCampaign interest: Somewhat interested\nEconomic views: Liberal\nState: California\nMedia consumption: Daily\nQ: Who would this respondent vote for in a Harris vs Trump election?"
    activations = extractor.extract_activations(text)
    
    bias_scores = representer.get_bias_scores(activations)
    weighted_rep, system_weights = representer.get_system_representations(activations)
    
    print(f"Bias scores: {bias_scores}")
    print(f"System weights: {system_weights}")

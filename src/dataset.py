"""
Dataset handling for Prospect Theory LLM Pipeline

This module provides dataset classes for loading and processing data for the
Prospect Theory LLM Pipeline, including both Prospect Theory training data
and ANES classification data.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Union, Tuple
from collections import Counter


class ProspectTheoryDataset(Dataset):
    """
    Dataset for Prospect Theory training and ANES classification.
    
    This dataset handles both:
    1. Prospect Theory training data with bias and system labels
    2. ANES classification data with target labels
    """
    
    def __init__(
        self, 
        data_path: str, 
        tokenizer, 
        max_length: int = 512,
        is_anes: bool = False,
        text_key: str = "text",
        generate_text_from_anes: bool = False
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the dataset file (CSV or JSON)
            tokenizer: Tokenizer for the LLM
            max_length: Maximum sequence length for tokenization
            is_anes: Whether this is ANES data (vs. Prospect Theory training data)
            text_key: Key for text field in the data
            generate_text_from_anes: Whether to generate text from ANES features
        """
        self.data = self._load_data(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_anes = is_anes
        self.text_key = text_key
        self.generate_text_from_anes = generate_text_from_anes
        
        # Extract bias names if available
        self.bias_names = self._get_bias_names()
        
        print(f"Loaded dataset with {len(self.data)} examples")
        if self.bias_names:
            print(f"Found {len(self.bias_names)} bias types: {', '.join(self.bias_names)}")

    def _load_data(self, data_path: str) -> List[Dict]:
        """Load data from CSV or JSON file."""
        if data_path.endswith('.csv'):
            return pd.read_csv(data_path).to_dict(orient='records')
        elif data_path.endswith('.json'):
            with open(data_path, 'r') as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")

    def _get_bias_names(self) -> List[str]:
        """Extract all bias names from the dataset."""
        if not self.data:
            return []
        
        # Assuming bias_labels is a dictionary in each data item
        if 'bias_labels' in self.data[0] and isinstance(self.data[0]['bias_labels'], dict):
            return list(self.data[0]['bias_labels'].keys())
        return []
    
    def _generate_text_from_anes_features(self, features: Dict) -> str:
        """
        Generate text prompt from ANES features.
        
        This is a simple implementation that can be enhanced with more
        sophisticated text generation techniques.
        """
        text = ""
        
        # Add demographic information if available
        if 'political_interest' in features:
            text += f"Political interest: {features['political_interest']}\n"
        if 'campaign_interest' in features:
            text += f"Campaign interest: {features['campaign_interest']}\n"
        if 'economic_views' in features:
            text += f"Economic views: {features['economic_views']}\n"
        if 'state' in features:
            text += f"State: {features['state']}\n"
        if 'media_consumption' in features:
            text += f"Media consumption: {features['media_consumption']}\n"
            
        # Add question
        text += "Q: Who would this respondent vote for in a Harris vs Trump election?"
        
        return text

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        
        # Get or generate text
        if self.generate_text_from_anes and self.is_anes:
            text = self._generate_text_from_anes_features(item)
        else:
            text = item.get(self.text_key, "")
        
        # Tokenize text
        encoding = self.tokenizer(
            text, 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_length, 
            return_tensors='pt'
        )
        
        result = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'text': text  # Keep raw text for reference
        }
        
        # Add bias labels if available
        if 'bias_labels' in item and self.bias_names:
            bias_labels = torch.tensor(
                [item['bias_labels'].get(bias, 0) for bias in self.bias_names], 
                dtype=torch.float
            )
            result['bias_labels'] = bias_labels
        
        # Add system label if available
        if 'system_label' in item:
            result['system_label'] = torch.tensor(item['system_label'], dtype=torch.long)
        
        # Add ANES features if available
        if 'anes_features' in item:
            if isinstance(item['anes_features'], list):
                result['anes_features'] = torch.tensor(item['anes_features'], dtype=torch.float)
            else:
                # Handle case where anes_features is a dictionary
                anes_features = []
                for key in sorted(item['anes_features'].keys()):
                    value = item['anes_features'][key]
                    if isinstance(value, (int, float)):
                        anes_features.append(value)
                result['anes_features'] = torch.tensor(anes_features, dtype=torch.float)
        
        # Add target if available
        if 'target' in item:
            result['target'] = torch.tensor(item['target'], dtype=torch.long)
        
        return result


class ANESDataset(Dataset):
    """
    Dataset for ANES data using the same format as in the original notebook.
    """
    
    def __init__(self, texts, labels, tokenizer, max_len=256):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text inputs
            labels: List of labels
            tokenizer: Tokenizer for the LLM
            max_len: Maximum sequence length for tokenization
        """
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0).long(),
            "attention_mask": enc["attention_mask"].squeeze(0).float(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def extract_legitimate_features(responses):
    """
    Extract only legitimate, non-leaky features from the responses.
    
    This function is kept identical to the original implementation.
    """
    features = {}
    
    # Helper to extract response safely
    def extract_response(responses, code):
        for r in responses:
            if r["variable_code"] == code:
                ans = r.get("respondent_answer")
                if ans in ['Inapplicable', 'Refused', "Don't know", 'Error']:
                    return "NA"
                return str(ans)
        return "NA"
    
    # Demographic features
    features["political_interest"] = extract_response(responses, "V241004")  # Political interest
    features["campaign_interest"] = extract_response(responses, "V241005")   # Campaign interest
    
    # Economic views (if available)
    features["economic_views"] = extract_response(responses, "V241127")
    
    # State/region information
    features["state"] = extract_response(responses, "V241017")
    
    # Media consumption (example)
    features["media_consumption"] = extract_response(responses, "V241201")
    
    # Convert features to a single text representation
    input_text = (
        f"Political interest: {features['political_interest']}\n"
        f"Campaign interest: {features['campaign_interest']}\n"
        f"Economic views: {features['economic_views']}\n"
        f"State: {features['state']}\n"
        f"Media consumption: {features['media_consumption']}\n"
        f"Q: Who would this respondent vote for in a Harris vs Trump election?"
    )
    
    return input_text, features


def load_data(data_folder, variable_code, exclude_classes=None, include_classes=None):
    """
    Loads question-response pairs for a given ANES variable code.
    Uses only legitimate features that don't leak the outcome.
    
    This function is kept identical to the original implementation.
    """
    examples = []
    label_map = {}
    next_label_id = 0
    features_data = []

    excluded_count = 0
    included_count = 0
    missing_answer_count = 0
    not_included_count = 0
    matched_count = 0

    if exclude_classes is None:
        exclude_classes = ['Inapplicable', 'Refused', "Don't know", 'Error', "Don't know"]

    json_files = [f for f in os.listdir(data_folder) if f.endswith('.json')]
    print(f"Processing {len(json_files)} JSON files for variable {variable_code}")

    for i, fname in enumerate(json_files):
        if i % 500 == 0:
            print(f"Progress: {i}/{len(json_files)} files processed")

        try:
            with open(os.path.join(data_folder, fname)) as f:
                respondent = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            continue

        responses = respondent.get("responses", [])
        found = False
        for item in responses:
            if item.get("variable_code") != variable_code:
                continue

            question = item.get("full_question_text", "")
            possible_answers = [opt["text"] for opt in item.get("possible_answers", [])]
            respondent_answer = item.get("respondent_answer", None)

            if not respondent_answer:
                missing_answer_count += 1
                continue

            if respondent_answer in exclude_classes:
                excluded_count += 1
                continue

            if include_classes and respondent_answer not in include_classes:
                not_included_count += 1
                continue

            included_count += 1

            if respondent_answer not in label_map:
                label_map[respondent_answer] = next_label_id
                next_label_id += 1
            label = label_map[respondent_answer]

            # Extract legitimate features instead of leaky ones
            input_text, features = extract_legitimate_features(responses)
            
            examples.append((input_text, label))
            features_data.append(features)
            matched_count += 1
            found = True
            break  # Only use first match per respondent

    # Summary logging
    print(f"\n📊 Summary for variable {variable_code}:")
    print(f"  ➤ Total JSON files: {len(json_files)}")
    print(f"  ➤ Valid examples collected: {matched_count}")
    print(f"  ➤ Unique labels: {len(label_map)}")
    print(f"  ➤ Skipped due to missing answers: {missing_answer_count}")
    print(f"  ➤ Skipped due to exclusion list: {excluded_count}")
    print(f"  ➤ Skipped (not in include_classes): {not_included_count}")
    if include_classes:
        print(f"  ➤ Included only: {include_classes}")
    print(f"  ➤ Final label map: {label_map}")

    # Class distribution
    label_counts = Counter([label for _, label in examples])
    print("\n🔍 Class distribution (label IDs):", label_counts)
    for label, count in label_counts.items():
        for key, val in label_map.items():
            if val == label:
                print(f"  ➤ '{key}': {count} samples")

    return examples, label_map, features_data


def convert_anes_to_dataset(
    json_folder: str,
    output_path: str,
    target_variable: str = "V241049",
    include_classes: List[str] = None,
    feature_codes: List[str] = None
):
    """
    Convert ANES JSON files to a dataset for the pipeline.
    
    Args:
        json_folder: Folder containing ANES JSON files
        output_path: Path to save the dataset
        target_variable: Variable code for the target
        include_classes: List of classes to include
        feature_codes: List of feature codes to include
    """
    if include_classes is None:
        include_classes = ["Donald Trump", "Kamala Harris"]
        
    if feature_codes is None:
        # Default legitimate features that don't leak the outcome
        feature_codes = [
            "V241004",  # Political interest
            "V241005",  # Campaign interest
            "V241127",  # Economic views
            "V241017",  # State
            "V241201",  # Media consumption
        ]
    
    # Load data using the original function
    examples, label_map, features_data = load_data(
        json_folder, target_variable, include_classes=include_classes
    )
    
    # Convert to the format expected by ProspectTheoryDataset
    data = []
    for (text, label), features in zip(examples, features_data):
        data.append({
            'text': text,
            'anes_features': features,
            'target': label
        })
    
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Created ANES dataset with {len(data)} examples at {output_path}")
    print(f"Target distribution: {Counter([d['target'] for d in data])}")
    
    return data


def get_dataloaders(
    prospect_data_path: str,
    anes_data_path: str,
    tokenizer,
    batch_size: int = 16,
    prospect_val_split: float = 0.2,
    anes_val_split: float = 0.2
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Get dataloaders for Prospect Theory and ANES datasets.
    
    Args:
        prospect_data_path: Path to Prospect Theory dataset
        anes_data_path: Path to ANES dataset
        tokenizer: Tokenizer for the LLM
        batch_size: Batch size for dataloaders
        prospect_val_split: Validation split for Prospect Theory dataset
        anes_val_split: Validation split for ANES dataset
        
    Returns:
        Tuple of (prospect_train_loader, prospect_val_loader, anes_train_loader, anes_val_loader)
    """
    # Load Prospect Theory dataset
    prospect_dataset = ProspectTheoryDataset(prospect_data_path, tokenizer)
    
    # Split into train/val
    prospect_train_size = int((1 - prospect_val_split) * len(prospect_dataset))
    prospect_val_size = len(prospect_dataset) - prospect_train_size
    
    prospect_train_dataset, prospect_val_dataset = torch.utils.data.random_split(
        prospect_dataset, [prospect_train_size, prospect_val_size]
    )
    
    # Create dataloaders
    prospect_train_loader = DataLoader(
        prospect_train_dataset, batch_size=batch_size, shuffle=True
    )
    prospect_val_loader = DataLoader(
        prospect_val_dataset, batch_size=batch_size
    )
    
    # Load ANES dataset
    anes_dataset = ProspectTheoryDataset(anes_data_path, tokenizer, is_anes=True)
    
    # Split into train/val
    anes_train_size = int((1 - anes_val_split) * len(anes_dataset))
    anes_val_size = len(anes_dataset) - anes_train_size
    
    anes_train_dataset, anes_val_dataset = torch.utils.data.random_split(
        anes_dataset, [anes_train_size, anes_val_size]
    )
    
    # Create dataloaders
    anes_train_loader = DataLoader(
        anes_train_dataset, batch_size=batch_size, shuffle=True
    )
    anes_val_loader = DataLoader(
        anes_val_dataset, batch_size=batch_size
    )
    
    return prospect_train_loader, prospect_val_loader, anes_train_loader, anes_val_loader


if __name__ == "__main__":
    # Example usage with paths from the original notebook
    from transformers import AutoTokenizer
    
    # Create directories
    os.makedirs("data/prospect_theory", exist_ok=True)
    os.makedirs("data/anes", exist_ok=True)
    
    # Convert ANES data from original JSON files
    # Assuming the original JSON files are in the same location as in the notebook
    json_folder = "/home/ubuntu/upload"  # Update this to match your actual path
    output_path = "data/anes/anes_dataset.json"
    
    convert_anes_to_dataset(
        json_folder=json_folder,
        output_path=output_path,
        target_variable="V241049",  # WHO WOULD R VOTE FOR: HARRIS VS TRUMP
        include_classes=["Donald Trump", "Kamala Harris"]
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    
    # Load dataset
    anes_dataset = ProspectTheoryDataset(output_path, tokenizer, is_anes=True)
    
    # Print example
    print("Example item:")
    item = anes_dataset[0]
    for k, v in item.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: Tensor of shape {v.shape}")
        else:
            print(f"{k}: {v}")


"""
Dataset handling for Prospect Theory LLM Pipeline - Fixed Version

This module provides dataset classes for loading and processing data for the
Prospect Theory LLM Pipeline, including both Prospect Theory training data
and ANES classification data.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import random
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
        if data_path.endswith(".csv"):
            return pd.read_csv(data_path).to_dict(orient="records")
        elif data_path.endswith(".json"):
            with open(data_path, "r") as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")

    def _get_bias_names(self) -> List[str]:
        """Extract all bias names from the dataset."""
        if not self.data:
            return []
        
        # Assuming bias_labels is a dictionary in each data item
        if "bias_labels" in self.data[0] and isinstance(self.data[0]["bias_labels"], dict):
            return list(self.data[0]["bias_labels"].keys())
        return []
    
    def _generate_text_from_anes_features(self, features: Dict) -> str:
        """
        Generate text prompt from ANES features.
        
        This is a simple implementation that can be enhanced with more
        sophisticated text generation techniques.
        """
        text = ""
        
        # Add demographic information if available
        for key, value in features.items():
            if isinstance(value, (str, int, float)) and key != "target":
                text += f"{key}: {value}\n"
            
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
            padding="max_length", 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        
        result = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "text": text  # Keep raw text for reference
        }
        
        # Add bias labels if available
        if "bias_labels" in item and self.bias_names:
            bias_labels = torch.tensor(
                [item["bias_labels"].get(bias, 0) for bias in self.bias_names], 
                dtype=torch.float
            )
            result["bias_labels"] = bias_labels
        
        # Add system label if available
        if "system_label" in item:
            result["system_label"] = torch.tensor(item["system_label"], dtype=torch.long)
        
        # Add ANES features if available
        if "anes_features" in item:
            if isinstance(item["anes_features"], list):
                result["anes_features"] = torch.tensor(item["anes_features"], dtype=torch.float)
            else:
                # Handle case where anes_features is a dictionary
                anes_features = []
                for key in sorted(item["anes_features"].keys()):
                    value = item["anes_features"][key]
                    if isinstance(value, (int, float)):
                        anes_features.append(value)
                    elif isinstance(value, str) and value.isdigit():
                        anes_features.append(float(value))
                    elif isinstance(value, str):
                        # One-hot encode categorical features
                        if key + "_" + value not in item.get("_feature_mapping", {}):
                            # Skip if not in feature mapping
                            continue
                        feature_idx = item["_feature_mapping"][key + "_" + value]
                        anes_features.append(1.0 if feature_idx else 0.0)
                result["anes_features"] = torch.tensor(anes_features, dtype=torch.float)
        
        # Add target if available
        if "target" in item:
            result["target"] = torch.tensor(item["target"], dtype=torch.long)
        
        return result

    @staticmethod
    def create_prospect_theory_dataset(
        output_path: str,
        num_examples: int = 100,
        bias_names: List[str] = None,
        system_probs: List[float] = None
    ) -> List[Dict]:
        """
        Create a dummy Prospect Theory dataset for training.
        
        Args:
            output_path: Path to save the dataset
            num_examples: Number of examples to generate
            bias_names: List of bias names to include
            system_probs: Probabilities of System 1 vs System 2 thinking
            
        Returns:
            List of generated examples
        """
        if bias_names is None:
            bias_names = ["anchoring", "framing", "availability", "confirmation_bias", "loss_aversion"]
            
        if system_probs is None:
            system_probs = [0.7, 0.3]  # Default: 70% System 1, 30% System 2
            
        # Scenarios for different biases
        scenarios = {
            "anchoring": [
                "When asked about the price of a new car, the respondent was first shown a luxury model priced at $80,000.",
                "The survey first mentioned that the average American household spends $5,000 per year on healthcare.",
                "Before asking about inflation expectations, the interviewer mentioned that inflation was 2% last year."
            ],
            "framing": [
                "The policy was described as 'saving 200 lives' rather than 'letting 800 people die'.",
                "The tax cut was framed as 'giving money back to taxpayers' rather than 'reducing government revenue'.",
                "The candidate's position was described as 'supporting traditional values' rather than 'opposing progressive reforms'."
            ],
            "availability": [
                "The respondent had recently seen news coverage of a violent crime in their neighborhood.",
                "After a major hurricane, the respondent was asked about climate change concerns.",
                "Having just read about a vaccine side effect, the respondent was asked about vaccine safety."
            ],
            "confirmation_bias": [
                "The respondent, a lifelong Republican, was shown information about economic growth under Republican presidents.",
                "A strong environmentalist was presented with data supporting renewable energy benefits.",
                "A gun rights advocate was shown statistics about defensive gun use."
            ],
            "loss_aversion": [
                "The respondent was told they would lose existing benefits rather than gain new ones.",
                "The policy was described as increasing costs rather than reducing savings.",
                "The investment option was framed in terms of potential losses rather than potential gains."
            ]
        }
        
        # Generate examples
        data = []
        for i in range(num_examples):
            # Randomly select biases present in this example
            present_biases = random.sample(bias_names, k=random.randint(1, 3))
            
            # Generate text from scenarios
            text_parts = []
            for bias in present_biases:
                if bias in scenarios:
                    text_parts.append(random.choice(scenarios[bias]))
            
            # Combine text parts
            text = " ".join(text_parts)
            
            # Add context and question
            text += "\n\nQ: How would this framing affect the respondent's decision?"
            
            # Create bias labels
            bias_labels = {bias: 1 if bias in present_biases else 0 for bias in bias_names}
            
            # Determine system (1 = System 1, 0 = System 2)
            system_label = 0 if random.random() < system_probs[1] else 1
            
            # Create example
            example = {
                "text": text,
                "bias_labels": bias_labels,
                "system_label": system_label
            }
            
            data.append(example)
        
        # Save to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"Created Prospect Theory dataset with {len(data)} examples at {output_path}")
        
        return data

    @staticmethod
    def convert_anes_to_dataset(
        json_folder: str,
        output_path: str,
        target_variable: str = "V241049",  # WHO WOULD R VOTE FOR: HARRIS VS TRUMP
        include_classes: List[str] = None
    ) -> None:
        """
        Convert ANES JSON files to dataset format.
        
        Args:
            json_folder: Path to folder containing ANES JSON files
            output_path: Path to save the converted dataset
            target_variable: Variable code for the target classification
            include_classes: List of classes to include (others will be filtered out)
        """
        if include_classes is None:
            include_classes = ["Donald Trump", "Kamala Harris"]
        
        print(f"Converting ANES JSON files from {json_folder}...")
        
        dataset = []
        processed_files = 0
        skipped_files = 0
        
        # Process all JSON files in the folder
        for filename in os.listdir(json_folder):
            if not filename.endswith(".json"):
                continue
                
            filepath = os.path.join(json_folder, filename)
            
            try:
                with open(filepath, "r") as f:
                    respondent_data = json.load(f)
                
                # Extract features and target
                text_features, structured_features = extract_legitimate_features(respondent_data.get("responses", []))
                target_response = extract_target_response(respondent_data.get("responses", []), target_variable)
                
                # Skip if no valid target response
                if target_response not in include_classes:
                    skipped_files += 1
                    continue
                
                # Create target label
                target_label = include_classes.index(target_response)
                
                # Create dataset entry
                entry = {
                    "text": text_features,
                    "anes_features": list(structured_features.values()),
                    "target": target_label,
                    "respondent_id": respondent_data.get("respondent_id", filename.replace(".json", ""))
                }
                
                dataset.append(entry)
                processed_files += 1
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                skipped_files += 1
                continue
        
        # Save dataset
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(dataset, f, indent=2)
        
        print(f"Converted {processed_files} files to dataset. Skipped {skipped_files} files.")
        print(f"Dataset saved to {output_path}")


class ANESBertDataset(Dataset):
    """
    Dataset for ANES data specifically for BERT-based models.
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


def extract_target_response(responses: List[Dict], target_variable: str) -> str:
    """
    Extract the target response for classification.
    
    Args:
        responses: List of response dictionaries
        target_variable: Variable code for the target
        
    Returns:
        Target response string
    """
    for response in responses:
        if response.get("variable_code") == target_variable:
            return response.get("respondent_answer", "Unknown")
    return "Unknown"


def extract_legitimate_features(responses: List[Dict]) -> Tuple[str, Dict]:
    """
    Extract only legitimate, non-leaky features from the responses.
    
    Enhanced with more features and feature engineering.
    
    Args:
        responses: List of response dictionaries from ANES data
        
    Returns:
        Tuple of (text_representation, structured_features)
    """
    features = {}
    
    # Helper to extract response safely
    def extract_response(responses, code):
        for r in responses:
            if r["variable_code"] == code:
                ans = r.get("respondent_answer")
                if ans in ["Inapplicable", "Refused", "Don't know", "Error"]:
                    return "NA"
                return str(ans)
        return "NA"
    
    # Demographic features
    features["age"] = extract_response(responses, "V201507x")  # Age
    features["gender"] = extract_response(responses, "V201600")  # Gender
    features["education"] = extract_response(responses, "V201510")  # Education level
    features["income"] = extract_response(responses, "V201617x")  # Income
    features["race"] = extract_response(responses, "V201549x")  # Race/ethnicity
    
    # Political engagement
    features["political_interest"] = extract_response(responses, "V241004")  # Political interest
    features["campaign_interest"] = extract_response(responses, "V241005")   # Campaign interest
    features["voter_registration"] = extract_response(responses, "V241013")  # Voter registration
    features["voting_frequency"] = extract_response(responses, "V241031")    # How often votes
    
    # Economic views
    features["economic_views"] = extract_response(responses, "V241127")  # Economic views
    features["economy_better_worse"] = extract_response(responses, "V241111")  # Economy better/worse
    features["personal_finance"] = extract_response(responses, "V241112")  # Personal financial situation
    
    # Geographic information
    features["state"] = extract_response(responses, "V241017")  # State
    features["urban_rural"] = extract_response(responses, "V241018")  # Urban/rural
    
    # Media consumption
    features["media_consumption"] = extract_response(responses, "V241201")  # Media consumption
    features["social_media_use"] = extract_response(responses, "V241242")  # Social media use
    features["news_interest"] = extract_response(responses, "V241211")  # News interest
    
    # Policy views (non-partisan)
    features["immigration_importance"] = extract_response(responses, "V241310")  # Immigration importance
    features["healthcare_importance"] = extract_response(responses, "V241311")  # Healthcare importance
    features["economy_importance"] = extract_response(responses, "V241312")  # Economy importance
    features["covid_importance"] = extract_response(responses, "V241313")  # COVID importance
    
    # Feature engineering
    # Convert categorical features to numeric where possible
    try:
        if features["political_interest"] not in ["NA"]:
            features["political_interest_num"] = float(features["political_interest"])
    except:
        features["political_interest_num"] = 0.0
        
    try:
        if features["campaign_interest"] not in ["NA"]:
            features["campaign_interest_num"] = float(features["campaign_interest"])
    except:
        features["campaign_interest_num"] = 0.0
    
    # Create interaction features
    if features["political_interest_num"] > 0 and features["campaign_interest_num"] > 0:
        features["political_engagement"] = features["political_interest_num"] * features["campaign_interest_num"]
    else:
        features["political_engagement"] = 0.0
    
    # Convert features to a single text representation
    input_text = f"""Demographics:\nAge: {features['age']}\nGender: {features['gender']}\nEducation: {features['education']}\nIncome: {features['income']}\nRace: {features['race']}\n\nPolitical Engagement:\nPolitical interest: {features['political_interest']}\nCampaign interest: {features['campaign_interest']}\nVoter registration: {features['voter_registration']}\nVoting frequency: {features['voting_frequency']}\n\nEconomic Views:\nEconomic views: {features['economic_views']}\nEconomy better/worse: {features['economy_better_worse']}\nPersonal finance: {features['personal_finance']}\n\nGeographic:\nState: {features['state']}\nUrban/Rural: {features['urban_rural']}\n\nMedia:\nMedia consumption: {features['media_consumption']}\nSocial media use: {features['social_media_use']}\nNews interest: {features['news_interest']}\n\nPolicy Priorities:\nImmigration importance: {features['immigration_importance']}\nHealthcare importance: {features['healthcare_importance']}\nEconomy importance: {features['economy_importance']}\nCOVID importance: {features['covid_importance']}\n\nQuestion: Based on these characteristics and preferences, who would this respondent likely vote for in a presidential election?"""
    
    # Create structured features for numerical processing
    structured_features = {
        'political_interest_num': features['political_interest_num'],
        'campaign_interest_num': features['campaign_interest_num'],
        'political_engagement': features['political_engagement'],
        'age_numeric': 0.0,  # Would need proper age parsing
        'income_numeric': 0.0  # Would need proper income parsing
    }
    
    return input_text.strip(), structured_features




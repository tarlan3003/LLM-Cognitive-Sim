"""
Dataset handling for Prospect Theory LLM - Best Performing Version

This module handles data loading, preprocessing, and feature extraction
for both Prospect Theory and ANES datasets.

Author: Tarlan Sultanov
"""

import os
import json
import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

class ProspectTheoryDataset(Dataset):
    """
    Dataset class for Prospect Theory and ANES data.
    
    This class handles both:
    1. Prospect Theory dataset with cognitive bias annotations
    2. ANES dataset with voting preferences
    """
    
    def __init__(self, data_path, tokenizer, is_anes=False, max_length=128):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the dataset JSON file
            tokenizer: Tokenizer for encoding text
            is_anes: Whether this is an ANES dataset
            max_length: Maximum sequence length for tokenization
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.is_anes = is_anes
        self.max_length = max_length
        
        # Define bias types
        self.bias_types = [
            "loss_aversion", 
            "framing_effect", 
            "anchoring", 
            "availability", 
            "representativeness", 
            "status_quo_bias"
        ]
        
        # Load data
        self.data = self._load_data()
    
    def _load_data(self):
        """
        Load data from JSON file.
        
        Returns:
            List of data samples
        """
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        
        return data
    
    def __len__(self):
        """
        Get dataset length.
        
        Returns:
            Number of samples in the dataset
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a dataset item.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary with encoded inputs and labels
        """
        item = self.data[idx]
        
        if self.is_anes:
            # ANES dataset
            text = item['text']
            anes_features = torch.tensor(item['anes_features'], dtype=torch.float32)
            target = torch.tensor(item['target'], dtype=torch.long)
            
            # Tokenize text
            encoded = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Remove batch dimension
            input_ids = encoded['input_ids'].squeeze(0)
            attention_mask = encoded['attention_mask'].squeeze(0)
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'anes_features': anes_features,
                'target': target
            }
        else:
            # Prospect Theory dataset
            text = item['text']
            bias_labels = torch.tensor(item['bias_labels'], dtype=torch.float32)
            system_label = torch.tensor(item['system_label'], dtype=torch.long)
            
            # Tokenize text
            encoded = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Remove batch dimension
            input_ids = encoded['input_ids'].squeeze(0)
            attention_mask = encoded['attention_mask'].squeeze(0)
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'bias_labels': bias_labels,
                'system_label': system_label
            }
    
    @staticmethod
    def create_prospect_theory_dataset(output_path, num_samples=1000, seed=42):
        """
        Create a synthetic Prospect Theory dataset for training.
        
        Args:
            output_path: Path to save the dataset
            num_samples: Number of samples to generate
            seed: Random seed for reproducibility
            
        Returns:
            Path to the created dataset
        """
        random.seed(seed)
        np.random.seed(seed)
        
        bias_types = [
            "loss_aversion", 
            "framing_effect", 
            "anchoring", 
            "availability", 
            "representativeness", 
            "status_quo_bias"
        ]
        
        # Templates for different biases
        templates = {
            "loss_aversion": [
                "I would rather avoid losing $X than gain $Y.",
                "The potential loss of $X feels worse than the potential gain of $Y.",
                "I'm more concerned about losing my current benefits than gaining new ones.",
                "The risk of losing $X is too high, even if I might gain $Y.",
                "I prefer to keep what I have rather than risk it for more."
            ],
            "framing_effect": [
                "The program will save 200 lives out of 600 people.",
                "The program will result in 400 deaths out of 600 people.",
                "This policy has a 70% success rate.",
                "This policy has a 30% failure rate.",
                "The glass is half full with this approach."
            ],
            "anchoring": [
                "The initial price was $X, so $Y seems like a good deal.",
                "Compared to last year's budget of $X, this year's $Y is reasonable.",
                "The first offer was $X, so I think $Y is fair.",
                "Given that similar products cost around $X, this one at $Y is priced well.",
                "Starting from the baseline of X, a change to Y isn't dramatic."
            ],
            "availability": [
                "After seeing the news about X, I'm worried about Y happening to me.",
                "I remember a vivid example of X, so I think Y is common.",
                "Because I can easily recall X, I believe Y happens frequently.",
                "The recent X incident makes me think Y is a major risk.",
                "Stories about X make me overestimate the likelihood of Y."
            ],
            "representativeness": [
                "She's an environmentalist, so she probably votes for Democrats.",
                "He works in finance, so he's likely a Republican.",
                "As a teacher, she must support education funding increases.",
                "Being from Texas, he probably opposes gun control.",
                "Since she's religious, she must be socially conservative."
            ],
            "status_quo_bias": [
                "I prefer to keep our current healthcare system rather than try something new.",
                "Let's stick with what we know works instead of experimenting.",
                "The existing policy may have flaws, but at least we understand them.",
                "I'd rather maintain our current approach than risk change.",
                "The current system is familiar, so I'm hesitant to support alternatives."
            ]
        }
        
        # System 1 vs System 2 indicators
        system_indicators = {
            0: [  # System 1 (fast, intuitive)
                "immediately felt", "instinctively", "my gut tells me", 
                "without thinking", "intuitively", "first impression",
                "emotional response", "quick reaction", "just feels right"
            ],
            1: [  # System 2 (slow, deliberative)
                "after careful consideration", "analytically", "weighing the evidence",
                "thinking it through", "deliberating", "upon reflection",
                "logical analysis", "systematic evaluation", "reasoned judgment"
            ]
        }
        
        data = []
        
        for _ in range(num_samples):
            # Randomly select primary bias type (with higher probability)
            primary_bias = random.choice(bias_types)
            
            # Generate bias labels (multi-label, with primary bias having higher value)
            bias_labels = np.random.uniform(0.0, 0.3, len(bias_types))
            primary_idx = bias_types.index(primary_bias)
            bias_labels[primary_idx] = np.random.uniform(0.7, 1.0)
            
            # Randomly select system (0 = System 1, 1 = System 2)
            system = random.randint(0, 1)
            
            # Generate text with bias and system indicators
            template = random.choice(templates[primary_bias])
            system_indicator = random.choice(system_indicators[system])
            
            # Replace placeholders
            text = template.replace("$X", str(random.randint(100, 1000)))
            text = text.replace("$Y", str(random.randint(100, 1000)))
            text = text.replace("X", random.choice(["terrorism", "crime", "accidents", "disease", "natural disasters"]))
            text = text.replace("Y", random.choice(["terrorism", "crime", "accidents", "disease", "natural disasters"]))
            
            # Add system indicator
            text = f"I {system_indicator} that {text}"
            
            data.append({
                "text": text,
                "bias_labels": bias_labels.tolist(),
                "system_label": system
            })
        
        # Save dataset
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f)
        
        print(f"Created Prospect Theory dataset with {num_samples} samples at {output_path}")
        return output_path
    
    @staticmethod
    def convert_anes_to_dataset(anes_path, output_path):
        """
        Convert ANES JSON files to a dataset for the pipeline.
        
        Args:
            anes_path: Path to ANES JSON files
            output_path: Path to save the dataset
            
        Returns:
            Path to the created dataset
        """
        # Get all JSON files in the directory
        json_files = [f for f in os.listdir(anes_path) if f.endswith('.json')]
        
        data = []
        
        for json_file in tqdm(json_files, desc="Processing ANES files"):
            file_path = os.path.join(anes_path, json_file)
            
            try:
                with open(file_path, 'r') as f:
                    respondent = json.load(f)
                
                # Extract target variable (voting preference)
                # V241049: WHO WOULD R VOTE FOR: HARRIS VS TRUMP
                target = None
                if 'V241049' in respondent:
                    vote_preference = respondent['V241049']
                    if vote_preference == 1:  # Donald Trump
                        target = 0
                    elif vote_preference == 2:  # Kamala Harris
                        target = 1
                
                # Skip if target is missing
                if target is None:
                    continue
                
                # Extract legitimate features
                anes_features = extract_legitimate_features(respondent)
                
                # Skip if features are missing
                if anes_features is None:
                    continue
                
                # Generate text description
                text = generate_text_description(respondent, anes_features)
                
                data.append({
                    "text": text,
                    "anes_features": anes_features,
                    "target": target
                })
            
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
        
        # Save dataset
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f)
        
        print(f"Created ANES dataset with {len(data)} samples at {output_path}")
        return output_path

def extract_legitimate_features(respondent):
    """
    Extract legitimate features from ANES respondent data.
    
    This function carefully selects features that don't leak the target variable.
    
    Args:
        respondent: ANES respondent data
        
    Returns:
        List of features
    """
    features = []
    
    try:
        # Demographic features
        # Age (V201507x)
        if 'V201507x' in respondent:
            age = respondent['V201507x']
            if age > 0:
                features.append(age / 100.0)  # Normalize age
            else:
                features.append(0.5)  # Default value
        else:
            features.append(0.5)  # Default value
        
        # Gender (V201600)
        if 'V201600' in respondent:
            gender = respondent['V201600']
            features.append(1.0 if gender == 1 else 0.0)  # 1 = Male, 2 = Female
        else:
            features.append(0.5)  # Default value
        
        # Race (V201549x)
        race_features = [0.0] * 5  # White, Black, Hispanic, Asian, Other
        if 'V201549x' in respondent:
            race = respondent['V201549x']
            if race == 1:
                race_features[0] = 1.0  # White
            elif race == 2:
                race_features[1] = 1.0  # Black
            elif race == 3:
                race_features[2] = 1.0  # Hispanic
            elif race == 4:
                race_features[3] = 1.0  # Asian
            else:
                race_features[4] = 1.0  # Other
        else:
            race_features[4] = 1.0  # Default to Other
        features.extend(race_features)
        
        # Education (V201510)
        education_features = [0.0] * 5  # <HS, HS, Some college, Bachelor's, Graduate
        if 'V201510' in respondent:
            education = respondent['V201510']
            if education <= 8:
                education_features[0] = 1.0  # Less than high school
            elif education <= 10:
                education_features[1] = 1.0  # High school
            elif education <= 12:
                education_features[2] = 1.0  # Some college
            elif education == 13:
                education_features[3] = 1.0  # Bachelor's
            elif education >= 14:
                education_features[4] = 1.0  # Graduate
        else:
            education_features[2] = 1.0  # Default to Some college
        features.extend(education_features)
        
        # Income (V202469x)
        if 'V202469x' in respondent:
            income = respondent['V202469x']
            if income > 0:
                features.append(min(income / 30.0, 1.0))  # Normalize income
            else:
                features.append(0.5)  # Default value
        else:
            features.append(0.5)  # Default value
        
        # Marital status (V201508)
        marital_features = [0.0] * 5  # Married, Widowed, Divorced, Separated, Never married
        if 'V201508' in respondent:
            marital = respondent['V201508']
            if marital == 1:
                marital_features[0] = 1.0  # Married
            elif marital == 2:
                marital_features[1] = 1.0  # Widowed
            elif marital == 3:
                marital_features[2] = 1.0  # Divorced
            elif marital == 4:
                marital_features[3] = 1.0  # Separated
            elif marital == 5:
                marital_features[4] = 1.0  # Never married
        else:
            marital_features[0] = 1.0  # Default to Married
        features.extend(marital_features)
        
        # Region (V203003)
        region_features = [0.0] * 4  # Northeast, Midwest, South, West
        if 'V203003' in respondent:
            region = respondent['V203003']
            if region == 1:
                region_features[0] = 1.0  # Northeast
            elif region == 2:
                region_features[1] = 1.0  # Midwest
            elif region == 3:
                region_features[2] = 1.0  # South
            elif region == 4:
                region_features[3] = 1.0  # West
        else:
            region_features[2] = 1.0  # Default to South
        features.extend(region_features)
        
        # Urban/rural (V203004)
        urban_features = [0.0] * 3  # Urban, Suburban, Rural
        if 'V203004' in respondent:
            urban = respondent['V203004']
            if urban == 1:
                urban_features[0] = 1.0  # Urban
            elif urban == 2:
                urban_features[1] = 1.0  # Suburban
            elif urban == 3:
                urban_features[2] = 1.0  # Rural
        else:
            urban_features[1] = 1.0  # Default to Suburban
        features.extend(urban_features)
        
        # Economic features
        
        # Economic condition (V201339)
        if 'V201339' in respondent:
            econ = respondent['V201339']
            if econ > 0 and econ <= 5:
                features.append((6 - econ) / 5.0)  # Normalize and invert (higher = better)
            else:
                features.append(0.5)  # Default value
        else:
            features.append(0.5)  # Default value
        
        # Personal financial situation (V201340)
        if 'V201340' in respondent:
            finance = respondent['V201340']
            if finance > 0 and finance <= 5:
                features.append((6 - finance) / 5.0)  # Normalize and invert (higher = better)
            else:
                features.append(0.5)  # Default value
        else:
            features.append(0.5)  # Default value
        
        # Employment status (V201562x)
        employment_features = [0.0] * 4  # Working, Temporarily laid off, Unemployed, Not in labor force
        if 'V201562x' in respondent:
            employment = respondent['V201562x']
            if employment == 1:
                employment_features[0] = 1.0  # Working
            elif employment == 2:
                employment_features[1] = 1.0  # Temporarily laid off
            elif employment == 3:
                employment_features[2] = 1.0  # Unemployed
            elif employment >= 4:
                employment_features[3] = 1.0  # Not in labor force
        else:
            employment_features[0] = 1.0  # Default to Working
        features.extend(employment_features)
        
        # Policy views (avoiding direct political leakage)
        
        # Healthcare (V201333)
        if 'V201333' in respondent:
            healthcare = respondent['V201333']
            if healthcare > 0 and healthcare <= 7:
                features.append(healthcare / 7.0)  # Normalize
            else:
                features.append(0.5)  # Default value
        else:
            features.append(0.5)  # Default value
        
        # Immigration (V201334)
        if 'V201334' in respondent:
            immigration = respondent['V201334']
            if immigration > 0 and immigration <= 7:
                features.append(immigration / 7.0)  # Normalize
            else:
                features.append(0.5)  # Default value
        else:
            features.append(0.5)  # Default value
        
        # Gun policy (V201335)
        if 'V201335' in respondent:
            guns = respondent['V201335']
            if guns > 0 and guns <= 7:
                features.append(guns / 7.0)  # Normalize
            else:
                features.append(0.5)  # Default value
        else:
            features.append(0.5)  # Default value
        
        # Abortion (V201336)
        if 'V201336' in respondent:
            abortion = respondent['V201336']
            if abortion > 0 and abortion <= 4:
                features.append(abortion / 4.0)  # Normalize
            else:
                features.append(0.5)  # Default value
        else:
            features.append(0.5)  # Default value
        
        # Climate change (V201337)
        if 'V201337' in respondent:
            climate = respondent['V201337']
            if climate > 0 and climate <= 4:
                features.append(climate / 4.0)  # Normalize
            else:
                features.append(0.5)  # Default value
        else:
            features.append(0.5)  # Default value
        
        # COVID-19 concerns (V201624)
        if 'V201624' in respondent:
            covid = respondent['V201624']
            if covid > 0 and covid <= 5:
                features.append(covid / 5.0)  # Normalize
            else:
                features.append(0.5)  # Default value
        else:
            features.append(0.5)  # Default value
        
        # Media consumption
        
        # TV news (V201631x)
        if 'V201631x' in respondent:
            tv_news = respondent['V201631x']
            if tv_news >= 0:
                features.append(min(tv_news / 7.0, 1.0))  # Normalize
            else:
                features.append(0.5)  # Default value
        else:
            features.append(0.5)  # Default value
        
        # Internet news (V201633x)
        if 'V201633x' in respondent:
            internet_news = respondent['V201633x']
            if internet_news >= 0:
                features.append(min(internet_news / 7.0, 1.0))  # Normalize
            else:
                features.append(0.5)  # Default value
        else:
            features.append(0.5)  # Default value
        
        # Social media use (V201637x)
        if 'V201637x' in respondent:
            social_media = respondent['V201637x']
            if social_media >= 0:
                features.append(min(social_media / 7.0, 1.0))  # Normalize
            else:
                features.append(0.5)  # Default value
        else:
            features.append(0.5)  # Default value
        
        # Psychological traits
        
        # Social trust (V201233)
        if 'V201233' in respondent:
            trust = respondent['V201233']
            if trust > 0 and trust <= 5:
                features.append(trust / 5.0)  # Normalize
            else:
                features.append(0.5)  # Default value
        else:
            features.append(0.5)  # Default value
        
        # Authoritarianism (V202263)
        if 'V202263' in respondent:
            auth = respondent['V202263']
            if auth > 0 and auth <= 5:
                features.append(auth / 5.0)  # Normalize
            else:
                features.append(0.5)  # Default value
        else:
            features.append(0.5)  # Default value
        
        # Need for cognition (V202264)
        if 'V202264' in respondent:
            cognition = respondent['V202264']
            if cognition > 0 and cognition <= 5:
                features.append(cognition / 5.0)  # Normalize
            else:
                features.append(0.5)  # Default value
        else:
            features.append(0.5)  # Default value
        
        # Risk aversion (V202265)
        if 'V202265' in respondent:
            risk = respondent['V202265']
            if risk > 0 and risk <= 5:
                features.append(risk / 5.0)  # Normalize
            else:
                features.append(0.5)  # Default value
        else:
            features.append(0.5)  # Default value
        
        # Altruism (V202266)
        if 'V202266' in respondent:
            altruism = respondent['V202266']
            if altruism > 0 and altruism <= 5:
                features.append(altruism / 5.0)  # Normalize
            else:
                features.append(0.5)  # Default value
        else:
            features.append(0.5)  # Default value
        
        # Feature engineering: Interaction terms
        
        # Age * Education interaction
        features.append(features[0] * features[7])  # Age * Bachelor's or higher
        
        # Income * Region interaction
        features.append(features[11] * features[17])  # Income * South
        
        # Urban * Policy views interaction
        features.append(features[20] * features[28])  # Urban * Climate change
        
        # Risk aversion * Economic condition interaction
        features.append(features[34] * features[24])  # Risk aversion * Economic condition
        
        # Ensure all features are valid
        for i, f in enumerate(features):
            if not isinstance(f, (int, float)) or np.isnan(f) or np.isinf(f):
                features[i] = 0.5  # Replace invalid values
        
        return features
    
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def generate_text_description(respondent, features):
    """
    Generate a text description of the respondent based on features.
    
    Args:
        respondent: ANES respondent data
        features: Extracted features
        
    Returns:
        Text description
    """
    # Age
    age_value = int(features[0] * 100)
    if age_value < 30:
        age_desc = "young adult"
    elif age_value < 50:
        age_desc = "middle-aged adult"
    else:
        age_desc = "older adult"
    
    # Gender
    gender_desc = "male" if features[1] > 0.5 else "female"
    
    # Race
    race_idx = np.argmax(features[2:7])
    race_descs = ["white", "Black", "Hispanic", "Asian", "of other racial background"]
    race_desc = race_descs[race_idx]
    
    # Education
    edu_idx = np.argmax(features[7:12])
    edu_descs = ["with less than high school education", 
                "with high school education", 
                "with some college education", 
                "with a bachelor's degree", 
                "with graduate education"]
    edu_desc = edu_descs[edu_idx]
    
    # Income
    income_value = features[12] * 30
    if income_value < 5:
        income_desc = "low-income"
    elif income_value < 15:
        income_desc = "middle-income"
    else:
        income_desc = "high-income"
    
    # Region
    region_idx = np.argmax(features[18:22])
    region_descs = ["Northeast", "Midwest", "South", "West"]
    region_desc = region_descs[region_idx]
    
    # Urban/rural
    urban_idx = np.argmax(features[22:25])
    urban_descs = ["urban", "suburban", "rural"]
    urban_desc = urban_descs[urban_idx]
    
    # Economic views
    econ_value = features[25]
    if econ_value < 0.4:
        econ_desc = "pessimistic about the economy"
    elif econ_value < 0.7:
        econ_desc = "neutral about the economy"
    else:
        econ_desc = "optimistic about the economy"
    
    # Policy views
    policy_values = features[28:31]
    policy_idx = np.argmax(policy_values)
    policy_strength = policy_values[policy_idx]
    
    if policy_idx == 0:
        if policy_strength > 0.7:
            policy_desc = "strongly concerned about climate change"
        else:
            policy_desc = "somewhat concerned about climate change"
    elif policy_idx == 1:
        if policy_strength > 0.7:
            policy_desc = "strongly concerned about COVID-19"
        else:
            policy_desc = "somewhat concerned about COVID-19"
    else:
        if policy_strength > 0.7:
            policy_desc = "highly engaged with news media"
        else:
            policy_desc = "moderately engaged with news media"
    
    # Psychological traits
    psych_values = features[32:36]
    psych_idx = np.argmax(psych_values)
    psych_strength = psych_values[psych_idx]
    
    if psych_idx == 0:
        if psych_strength > 0.7:
            psych_desc = "high social trust"
        else:
            psych_desc = "moderate social trust"
    elif psych_idx == 1:
        if psych_strength > 0.7:
            psych_desc = "strong authoritarian tendencies"
        else:
            psych_desc = "moderate authoritarian tendencies"
    elif psych_idx == 2:
        if psych_strength > 0.7:
            psych_desc = "high need for cognition"
        else:
            psych_desc = "moderate need for cognition"
    else:
        if psych_strength > 0.7:
            psych_desc = "high risk aversion"
        else:
            psych_desc = "moderate risk aversion"
    
    # Combine into a description
    description = f"This respondent is a {age_desc} {gender_desc} who is {race_desc} {edu_desc}. "
    description += f"They are {income_desc} and live in a {urban_desc} area in the {region_desc}. "
    description += f"They are {econ_desc} and {policy_desc}. "
    description += f"Psychologically, they show {psych_desc}."
    
    return description

# LLM-Cognitive-Sim: Cognitive Bias Extraction for Political Behavior Prediction

## Overview

This project implements an interpretable AI framework that extracts cognitive biases from Large Language Models to predict political behavior. The approach adapts Testing with Concept Activation Vectors (TCAV) to natural language processing for psychological construct extraction, grounded in Prospect Theory.

**Key Results:**
- 80.26% accuracy at 0.45 threshold for political behavior prediction
- Extracts interpretable cognitive bias patterns from voter responses
- Validates dual-process theory in political decision-making (3:1 deliberative:intuitive ratio)

## Architecture

The system consists of three main components:
1. **RoBERTa-Large** for text representation extraction
2. **Cognitive Bias Representer** using Concept Activation Vectors (CAV)
3. **ANES Classifier** for multi-modal political behavior prediction

## Repository Structure

```
LLM-Cognitive-Sim/
├── main.py                          # Main execution script
├── data preprocessing/
│   └── json_extraction.ipynb        # ANES data preprocessing
├── src/
│   ├── dataset.py                   # Dataset classes
│   ├── bias_representer.py          # Cognitive bias extraction
│   └── anes_classifier.py           # Political behavior classifier
|   |__ .....
├── baseline models/
│   └── random_forest_baseline.ipynb # Traditional ML baseline
├── prospect_theory_test/
│   └── apple_silicon_cav_thinking_fast_slow.ipynb # Validation experiments
└── README.md
```

## Data Source

This project uses the **ANES 2024 Time Series Study** dataset:
- **Source**: [American National Election Studies (ANES)](https://electionstudies.org/data-center/2024-time-series-study/)
- **Dataset**: ANES 2024 Time Series Study
- **Format**: JSON files with structured survey responses and open-ended text

**Note**: ANES data requires registration and agreement to terms of use. 

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/LLM-Cognitive-Sim.git
cd LLM-Cognitive-Sim

# Install dependencies
pip install -r requirements.txt

```

## How to Run

### 1. Data Preprocessing
```bash
# Open and run the data preprocessing notebook
jupyter notebook "data preprocessing/json_extraction.ipynb"
```

### 2. Train Models
```bash
# Run main training pipeline
python main.py
```

### 3. Run Baselines
```bash
# Compare with traditional ML baseline
jupyter notebook "baseline models/random_forest_baseline.ipynb"
```

### 4. Validation Experiments
```bash
# Run Prospect Theory validation
jupyter notebook "prospect_theory_test/apple_silicon_cav_thinking_fast_slow.ipynb"
```

## Key Features

- **Interpretable AI**: Extracts meaningful cognitive bias patterns
- **Multi-modal**: Combines structured demographics with text analysis
- **Theoretically Grounded**: Based on Prospect Theory and dual-process theory
- **Cross-validated**: 5-fold stratified cross-validation for robust evaluation
- **Comparative Analysis**: Includes traditional ML and deep learning baselines

## Results

The framework achieves:
- **Combined Model**: 80.26% accuracy (structured + text + cognitive biases)
- **Structured-only**: ~75% accuracy (demographics and political attitudes)
- **Text-only**: ~55% accuracy (open-ended responses only)

**Cognitive Bias Findings:**
- Universal bias patterns across Trump and Harris voters
- Strong negative correlation between anchoring and confirmation bias (-0.79)
- Dual-process validation: 75% deliberative vs 25% intuitive processing

## Requirements

- Python 3.9+
- PyTorch 1.12+
- Transformers 4.20+



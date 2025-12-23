# Activation-Guided Local Editing for Jailbreaking Attacks

## Project Overview

This project implements the AGILE framework, an activation-guided local editing method for jailbreak attacks. The framework analyzes the internal activation patterns of large language models when refusing harmful requests and trains classifiers to guide the generation of adversarial examples.

## Project Structure

```
AGILE/
├── agile/
│   └── model/
│       ├── AGILEManager.py      # Core manager for AGILE framework
│       ├── ClassifierManager.py # Training and management of classifiers
│       ├── LLMManager.py        # Loading and interaction with large language models
│       └── utils.py             # Utility functions
│   └── demo/
│       └── AGILE_harmbench_5_injected_llama-3-8b-instruct_p-5_sampled_llama-3-8b-instruct_judged.json
                                 # A demo of the generated adversarial examples
├── scripts/
│   └── train_refuse-guided_classifier.py  # Training script for classifier
└── requirements.txt             # Project dependencies
```

## Core Components

### 1. AGILEManager.py
- **Generator class**: Implements contextual scaffolding and adaptive rephrasing
- **Editor class**: Implements synonym substitution and token injection
- Responsible for the core logic of generating adversarial examples

### 2. ClassifierManager.py
- **ClassifierManager class**: Manages training and evaluation of multi-layer classifiers
- **LinearLayerClassifier**: Linear layer classifier
- **MLPLayerClassifier**: Multi-layer perceptron classifier
- **MLPClassifierTorch**: PyTorch implementation of MLP classifier

### 3. LLMManager.py
- **EmbeddingManager**: Manages model activations and embeddings
- **ModelExtraction**: Extracts internal model activations
- **ModelGeneration**: Model generation functionality
- **Sampler**: Batch sampler
- **GPTJudge**: GPT evaluator

### 4. train_refuse-guided_classifier.py
- Main script for training refusal-guided classifiers
- Analyzes activation patterns when models refuse responses
- Builds training datasets and trains classifiers


## Important Notes

⚠️ **Important Disclaimer**: This code is for academic research purposes only, intended for evaluating the safety and robustness of large language models. Please do not use for any malicious purposes.

⚠️ **Code Status**: Current code contains core algorithm implementations but lacks complete runtime environment configuration. This is an attachment version prepared for paper submission. A complete version will be released later. For an example of the generated adversarial examples, please refer to the `demo` folder.

## Dependencies

Main dependencies include:
- PyTorch
- Transformers
- scikit-learn
- spaCy
- sentence-transformers
- Other dependencies see `requirements.txt`

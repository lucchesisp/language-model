# Language Model - Mathematical Foundations

## Introduction

This repository contains educational materials exploring the mathematical foundations of language models. Through interactive Jupyter notebooks, we dive deep into the core concepts that power modern language models, from probability theory to neural network training.

The notebooks provide both theoretical explanations and practical implementations using NumPy, with visualizations to illustrate key concepts. Each section includes technical descriptions for those familiar with machine learning, as well as intuitive explanations for beginners.

## Key Concepts

### 1. Probability and Language Models
- Understanding language models as probability distributions over token sequences
- Conditional probability: predicting the next token given context
- Computing `P(x_t | x_<t)` - the probability of a token at time t given all previous tokens

### 2. Softmax Function
- Converting raw neural network scores (logits) into valid probability distributions
- Mathematical properties: handling negative values, normalization, and amplification
- Formula: `Softmax(z_i) = exp(z_i) / Σ exp(z_j)`
- Why exponential? Ensures positivity, amplifies differences, and provides mathematical convenience

### 3. Temperature Scaling
- Controlling model confidence and output diversity
- Low temperature (T < 1.0): more deterministic, focused predictions
- High temperature (T > 1.0): more random, creative outputs
- Applications: code generation (low T) vs creative writing (high T)

### 4. Cross-Entropy Loss
- Measuring prediction error during training
- Formula: `Loss = -log(P(correct_token))`
- Understanding the penalty structure: confident mistakes vs correct predictions
- The relationship between probability and loss through logarithms

### 5. Complete Forward Pass
- End-to-end pipeline: Tokens → Neural Model → Logits → Softmax → Probabilities → Loss
- How these components work together in practice
- Simulation of the complete prediction and evaluation process

## Notebooks

### Completed

- [Math Foundations](notebooks/math_foundations.ipynb) - Core mathematical concepts for language models including probability, softmax, temperature, cross-entropy loss, and forward pass simulation

### Work in Progress

- [Tokenization](notebooks/tokenization.ipynb) - Breaking text into tokens, subword tokenization (BPE, WordPiece), vocabulary building
- [Attention Mechanism](notebooks/attention_mechanism.ipynb) - Self-attention, multi-head attention, query-key-value paradigm
- [Complete LLM Architecture](notebooks/llm_architecture.ipynb) - Transformer blocks, positional encodings, layer normalization, residual connections
- [Training](notebooks/training.ipynb) - Backpropagation, gradient descent, optimization algorithms, training loops
- [LoRA (Low-Rank Adaptation)](notebooks/lora.ipynb) - Efficient fine-tuning with low-rank matrices
- [QLoRA (Quantized LoRA)](notebooks/qlora.ipynb) - Combining quantization with LoRA for memory-efficient fine-tuning

## Setup

This project uses Poetry for dependency management. To set up the environment:

```bash
# Install dependencies
poetry install

# Activate the virtual environment
poetry shell

# Launch Jupyter
jupyter notebook
```

## Dependencies

- Python 3.14+
- NumPy
- Matplotlib
- Seaborn
- Jupyter

## Author

Gabriel Lucchesi

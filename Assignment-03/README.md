---

# Machine Translation with Attention using PyTorch

This project implements a sequence-to-sequence (seq2seq) model with an attention mechanism for machine translation tasks using PyTorch. The model translates text from one language to another, with a focus on the attention mechanism to improve translation quality.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Getting Started](#getting-started)
4. [Data Preparation](#data-preparation)
5. [Model Architecture](#model-architecture)
6. [Training and Evaluation](#training-and-evaluation)
7. [Hyperparameter Sweep](#hyperparameter-sweep)
8. [Usage](#usage)
9. [Requirements](#requirements)
10. [Credits](#credits)

## Overview

This project provides a comprehensive solution for machine translation tasks, offering functionalities for data preparation, model definition, training, evaluation, and hyperparameter tuning. The key components include:

- **Data Preparation**: Functions to preprocess input data, including reading CSV files, tokenizing text, and converting sequences to tensors.
- **Model Architecture**: Classes for defining the encoder and decoder models, along with the attention mechanism. The encoder processes input sequences and produces hidden states, while the decoder uses attention to generate output sequences.
- **Training and Evaluation**: Functions for training the model using teacher forcing, evaluating its performance, and logging metrics using wandb.
- **Hyperparameter Sweep**: Configuration for a hyperparameter sweep using wandb, enabling exploration of various hyperparameter combinations to optimize model performance.

## Features

- Implements a seq2seq model with attention mechanism for machine translation.
- Supports training, validation, and testing of translation models.
- Flexible hyperparameter tuning using wandb's hyperparameter sweep.
- Easily customizable for different datasets and languages.

## Getting Started

To get started with this project, follow these steps:

1. Clone the repository to your local machine.
2. Install the required libraries listed [NumPy, Pandas, OS, wandb, PyTorch, torch.nn, torch.autograd, torch.utils.data.DataLoader, torch.optim, torch.nn.functional, argparse] .
3. Prepare your data in CSV format, with separate columns for input and target sequences.
4. Update the file paths in the code to point to your data files.
5. Run the `train_sweep()` function to start the training process.
6. Monitor the training progress and results using the wandb dashboard.

## Data Preparation

The data preparation process involves the following steps:

1. Reading CSV files containing input and target sequences.
2. Tokenizing text to create vocabulary dictionaries.
3. Converting text sequences to tensors for model input.

## Model Architecture

The model architecture consists of the following components:

- **Encoder**: Processes input sequences and produces hidden states.
- **Decoder**: Uses an attention mechanism to generate output sequences based on encoder hidden states.
- **Attention Mechanism**: Helps the decoder focus on relevant parts of the input sequence during translation.

## Training and Evaluation

During training and evaluation, the following steps are performed:

1. Forward pass: The input sequence is fed through the encoder, and the decoder generates output sequences.
2. Calculation of loss: The negative log-likelihood loss is computed between the predicted and target sequences.
3. Backpropagation: Gradients are computed and used to update model parameters.
4. Logging: Metrics such as training loss, validation accuracy, and validation loss are logged using wandb.

## Hyperparameter Sweep

Hyperparameter tuning is facilitated using wandb's hyperparameter sweep functionality. Various hyperparameters, including embedding size, hidden size, learning rate, and dropout rate, can be tuned to optimize model performance.

## Usage

To use this project:

1. Install the required libraries listed as in 'getting started' section.
2. Prepare your data in CSV format, following the guidelines provided in the [Data Preparation](#data-preparation) section.
3. Update the file paths in the code to point to your data files.
4. Run the `train_sweep()` function to start the training process.
5. Monitor the training progress and results using the wandb dashboard.

## Requirements

- Python 3.x
- PyTorch
- wandb
- numpy
- pandas

## Credits

This project was developed by Kartik Raman. For questions or inquiries, please contact [ramankartik12@gmail.com].

---

# Training Entry Point (`train.py`)

## Introduction

The `train.py` file is the primary entry point for training LLaVA. It is responsible for parsing training arguments, loading the pretrained model, preparing datasets, configuring the tokenizer, initializing the trainer, and launching the complete training process.

Rather than implementing optimization algorithms itself, this file orchestrates the entire training workflow by connecting all major components of the repository.

---

# Position in the Training Pipeline

```
Training Command
        │
        ▼
train.py
        │
        ├── Parse Arguments
        ├── Load Model
        ├── Load Tokenizer
        ├── Prepare Dataset
        ├── Initialize Trainer
        └── Start Training
```

This file serves as the controller for the training process.

---

# Responsibilities

The major responsibilities of `train.py` include:

- Parsing training configurations
- Loading the pretrained LLaVA model
- Initializing the tokenizer
- Preparing training datasets
- Configuring multimodal settings
- Creating the trainer
- Saving checkpoints

---

# Training Workflow

The complete workflow is:

```
Training Script
       │
       ▼
Parse Arguments
       │
       ▼
Load Model
       │
       ▼
Load Tokenizer
       │
       ▼
Initialize Vision Tower
       │
       ▼
Load Dataset
       │
       ▼
Data Collator
       │
       ▼
LLaVATrainer
       │
       ▼
Training Loop
       │
       ▼
Save Model
```

---

# Argument Parsing

Training begins by reading command-line arguments.

These arguments define:

- Model path
- Dataset path
- Batch size
- Learning rate
- Number of epochs
- Output directory
- Precision (FP16/BF16)
- Distributed training settings

This design allows experiments to be configured without modifying the source code.

---

# Loading the Model

The pretrained LLaVA model is loaded first.

This step initializes:

```
Tokenizer

↓

LLaMA

↓

Vision Tower

↓

Multimodal Projector
```

The model is then moved to the appropriate training device.

---

# Configuring the Vision Tower

Before training begins, the Vision Tower is initialized.

Typical tasks include:

- Loading pretrained CLIP weights
- Selecting feature extraction layer
- Configuring image resolution
- Freezing or enabling trainable parameters

Depending on the training stage, some components remain frozen while others are updated.

---

# Dataset Preparation

The dataset contains paired image-text examples.

Each training sample consists of:

```
Image

+

Conversation

↓

Training Example
```

The dataset loader converts these samples into tensors suitable for model training.

---

# Data Collator

Individual samples are combined into mini-batches by the data collator.

Responsibilities include:

- Padding sequences
- Stacking image tensors
- Creating attention masks
- Preparing labels

Output example:

```
Batch

├── Images
├── Input IDs
├── Labels
└── Attention Masks
```

---

# Initializing the Trainer

Instead of implementing the optimization loop manually, `train.py` creates an instance of:

```python
LLaVATrainer
```

The trainer manages:

- Forward pass
- Backpropagation
- Gradient accumulation
- Checkpointing
- Logging
- Evaluation

---

# Saving Checkpoints

During training, checkpoints are periodically saved.

A checkpoint typically contains:

- Model weights
- Optimizer state
- Scheduler state
- Training progress

This allows interrupted training to resume without starting from scratch.

---

# Interaction with Other Files

```
train.py
     │
     ▼
builder.py
     │
     ▼
llava_arch.py
     │
     ▼
LLaVATrainer
     │
     ▼
Dataset
```

`train.py` coordinates the entire training pipeline but delegates the actual optimization process to the trainer.

---

# Summary

The `train.py` file serves as the main controller for LLaVA training. It parses configuration arguments, loads the model and tokenizer, prepares datasets, initializes the trainer, and launches the complete training workflow.

By separating orchestration from optimization, the implementation remains modular, extensible, and easy to maintain.

---

# Next

The next document explores `llava_trainer.py`, which extends Hugging Face's `Trainer` and implements the training loop, optimization process, checkpoint management, and distributed training support used by LLaVA.
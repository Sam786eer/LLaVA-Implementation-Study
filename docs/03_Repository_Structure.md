# Repository Structure

## Introduction

The official LLaVA repository is organized into multiple modules, each responsible for a different part of the system such as model construction, training, inference, serving, and evaluation.

Understanding the repository structure before reading the source code makes it much easier to follow the implementation.

The focus of this implementation study is the core multimodal pipeline rather than deployment or web interfaces.

---

# Repository Overview

```
LLaVA/
│
├── llava/
│   ├── model/
│   ├── train/
│   ├── eval/
│   ├── serve/
│   ├── constants.py
│   ├── conversation.py
│   ├── mm_utils.py
│   └── utils.py
│
├── scripts/
├── playground/
├── docs/
└── pyproject.toml
```

---

# Core Directory

```
llava/
```

This directory contains almost the entire implementation of LLaVA.

Everything related to the multimodal model is implemented here.

---

# model/

```
llava/model/
```

This is the most important directory in the repository.

It contains:

- Model loading
- Vision encoder
- Multimodal projector
- Language model integration
- Multimodal fusion logic

The files studied in this repository mainly belong to this directory.

---

## builder.py

Responsible for loading pretrained models.

Main responsibilities include:

- Loading tokenizer
- Loading LLaVA model
- Loading CLIP Vision Tower
- Loading Image Processor
- Loading Multimodal Projector

This is the entry point for inference.

---

## llava_arch.py

Implements the multimodal architecture.

Important responsibilities:

- Initialize vision modules
- Encode images
- Project image features
- Replace image tokens
- Prepare embeddings for LLaMA

This file contains the core multimodal fusion logic.

---

## multimodal_encoder/

Contains implementations of different vision encoders.

Example:

```
clip_encoder.py
```

This file loads the CLIP Vision Transformer and extracts visual features from images.

---

## multimodal_projector/

Contains the feature alignment module.

The projector converts CLIP embeddings into LLaMA embeddings.

Supported projector types include:

- Linear
- MLP
- Identity

---

## language_model/

Contains the LLaVA language model implementation.

Important file:

```
llava_llama.py
```

Responsibilities include:

- Loading LLaMA
- Forward pass
- Generation
- Connecting with llava_arch.py

---

# train/

Contains everything required for training.

Important files include:

- train.py
- llava_trainer.py

Responsibilities:

- Dataset loading
- Data collator
- Optimizer
- Scheduler
- Checkpoint saving
- Distributed training

---

# eval/

Contains evaluation scripts.

Example:

```
run_llava.py
```

This file performs inference by:

- Loading the model
- Processing images
- Building prompts
- Running generation
- Producing responses

---

# serve/

Contains deployment utilities.

Examples include:

- Web server
- Model worker
- Gradio interface

These files are useful for deployment but are outside the scope of this implementation study.

---

# conversation.py

Responsible for prompt construction.

It defines:

- Conversation templates
- Roles
- Prompt formatting
- Separator styles

Every user query passes through this file before tokenization.

---

# mm_utils.py

Contains multimodal utility functions.

Examples:

- Image preprocessing
- Image token insertion
- Tokenizer helpers
- Any-resolution image processing

---

# File Dependency

```
run_llava.py
        │
        ▼
builder.py
        │
        ▼
llava_llama.py
        │
        ▼
llava_arch.py
        │
        ▼
clip_encoder.py
        │
        ▼
multimodal_projector.py
```

This dependency chain represents the complete inference pipeline inside the official implementation.

---

# Files Covered in This Study

The implementation study focuses on the following files:

- builder.py
- llava_arch.py
- clip_encoder.py
- multimodal_projector/builder.py
- llava_llama.py
- conversation.py
- mm_utils.py
- train.py
- llava_trainer.py
- run_llava.py

These files together explain the complete architecture, training pipeline, and inference workflow of LLaVA.

---

# Next

The next document begins the actual source code walkthrough with `builder.py`, which is responsible for loading the tokenizer, language model, vision encoder, image processor, and multimodal projector.
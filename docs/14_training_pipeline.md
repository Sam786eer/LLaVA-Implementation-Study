# End-to-End Training Pipeline

## Introduction

Training LLaVA is a multi-stage process that transforms paired image-text data into a multimodal language model capable of understanding both visual and textual information. This document provides a complete overview of the training pipeline and explains how different modules interact during model optimization.

Unlike previous documents that focused on individual source files, this chapter presents the complete data flow from the training dataset to the optimized model.

---

# Complete Training Pipeline

```
                    Training Dataset
                           │
                           ▼
                 Image-Text Conversations
                           │
                           ▼
                    Dataset Loader
                           │
                           ▼
                     Data Collator
                           │
                           ▼
                     Mini Batch
                           │
                           ▼
                    train.py
                           │
                           ▼
                  Load Pretrained Model
                           │
                           ▼
                  Vision Encoder (CLIP)
                           │
                           ▼
                 Multimodal Projector
                           │
                           ▼
                  Prepare Multimodal Inputs
                           │
                           ▼
                  LLaMA Forward Pass
                           │
                           ▼
                    Compute Loss
                           │
                           ▼
                    Backpropagation
                           │
                           ▼
                   Optimizer Update
                           │
                           ▼
                  Save Model Checkpoint
```

---

# Step 1: Dataset Preparation

Training begins with a dataset containing paired images and conversations.

Each sample typically consists of:

```
Image

+

Instruction

+

Expected Response
```

Example:

```
Image:
(cat.jpg)

Instruction:
"What animal is shown?"

Expected Response:
"A cat is sitting on a sofa."
```

---

# Step 2: Dataset Loading

The dataset loader reads each training sample and converts it into a format suitable for training.

Responsibilities include:

- Loading images
- Reading conversation data
- Tokenizing text
- Preparing labels

The output is a structured training example.

---

# Step 3: Batch Construction

Multiple training examples are grouped into mini-batches.

Each batch contains:

```
Images

Input IDs

Labels

Attention Masks
```

Padding is applied so that all sequences within the batch have the same length.

---

# Step 4: Model Initialization

Before training begins, the following components are loaded:

```
Tokenizer

↓

Vision Encoder

↓

Multimodal Projector

↓

LLaMA
```

These components together form the complete LLaVA model.

---

# Step 5: Image Encoding

Each image is passed through the pretrained CLIP Vision Encoder.

```
Image

↓

Vision Transformer

↓

Visual Features
```

The encoder extracts semantic representations rather than classification outputs.

---

# Step 6: Feature Projection

Visual features are transformed into the LLaMA embedding space.

```
CLIP Features

↓

MM Projector

↓

Image Embeddings
```

These embeddings now have the same dimensionality as text embeddings.

---

# Step 7: Multimodal Input Construction

The `<image>` placeholder token in the prompt is replaced with projected image embeddings.

```
Prompt

↓

<image>

↓

Image Embeddings

↓

Merged Embedding Sequence
```

The final sequence contains both visual and textual information.

---

# Step 8: Forward Pass

The merged embeddings are passed through the LLaMA decoder.

```
Merged Embeddings

↓

Transformer Layers

↓

Predicted Tokens
```

The model predicts the next token for every position in the sequence.

---

# Step 9: Loss Computation

The predicted tokens are compared with the expected responses.

```
Predictions

+

Ground Truth

↓

Cross-Entropy Loss
```

The loss measures how accurately the model predicts the target sequence.

---

# Step 10: Backpropagation

The computed loss is propagated backward through the network.

```
Loss

↓

Backward Pass

↓

Gradients
```

Gradients indicate how model parameters should be updated.

---

# Step 11: Parameter Update

The optimizer updates the trainable parameters.

```
Gradients

↓

Optimizer

↓

Updated Parameters
```

Depending on the training stage, different components may be frozen or trainable.

---

# Step 12: Checkpoint Saving

At regular intervals, the training state is saved.

Each checkpoint typically includes:

- Model weights
- Optimizer state
- Scheduler state
- Current training step

This allows interrupted training to resume without losing progress.

---

# File Interaction

```
train.py
      │
      ▼
builder.py
      │
      ▼
clip_encoder.py
      │
      ▼
multimodal_projector.py
      │
      ▼
llava_arch.py
      │
      ▼
llava_llama.py
      │
      ▼
llava_trainer.py
```

Together, these files implement the complete training pipeline.

---

# Summary

The LLaVA training pipeline combines image processing, multimodal feature alignment, language modeling, and optimization into a unified workflow. Starting from paired image-text data, the model learns to generate accurate textual responses by minimizing prediction error through repeated forward and backward passes.

Understanding this end-to-end process provides a comprehensive view of how the individual modules documented in earlier chapters work together during training.

---

# Next

The final document presents the complete inference pipeline, showing how a trained LLaVA model processes a user image and prompt to generate a response.
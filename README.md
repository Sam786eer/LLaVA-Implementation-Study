# LLaVA: An Implementation Study

> A comprehensive implementation study of the official **LLaVA (Large Language and Vision Assistant)** codebase, focusing on architecture, tensor flow, multimodal fusion, training, inference, and implementation details.

---

## Overview

LLaVA is a Vision-Language Model (VLM) that combines a pretrained **CLIP Vision Encoder** with a **LLaMA Large Language Model** through a **Multimodal Projector**, enabling image understanding and natural language generation.

This repository documents my study of the official LLaVA implementation by analyzing the source code, understanding the architecture, tracing tensor flow, and connecting the research paper with the implementation.

---

## Objectives

- Understand the complete LLaVA architecture.
- Study the interaction between CLIP and LLaMA.
- Analyze multimodal feature alignment.
- Understand the training pipeline.
- Understand the inference pipeline.
- Explain important implementation files.

---

# Overall Architecture

```text
                Image
                  │
                  ▼
        Image Processor
                  │
                  ▼
       CLIP Vision Encoder
                  │
          Visual Features
                  │
                  ▼
      Multimodal Projector
                  │
                  ▼
     LLaMA Embedding Space
                  │
                  ▼
prepare_inputs_labels_for_multimodal()
                  │
                  ▼
       Replace <image> Token
                  │
                  ▼
           LLaMA Decoder
                  │
                  ▼
        Generated Response
```

---

# Implementation Pipeline

```text
Image
 │
 ▼
Image Processor
 │
 ▼
CLIP Vision Tower
 │
 ▼
Vision Features
 │
 ▼
MM Projector
 │
 ▼
Image Embeddings
 │
 ▼
Insert into Prompt
 │
 ▼
LLaMA
 │
 ▼
Autoregressive Generation
 │
 ▼
Final Output
```

---

# Repository Structure

```
LLaVA-Implementation-Study
│
├── README.md
├── LICENSE
├── docs/
│
├── diagrams/
│
└── references/
```

---

# Documentation

| Document | Description |
|-----------|-------------|
| Vision Language Models | Fundamentals of Vision-Language Models |
| LLaVA Architecture | Complete architecture walkthrough |
| Repository Structure | Organization of the official repository |
| Model Loading | Model initialization and loading |
| Vision Encoder | CLIP Vision Tower |
| Multimodal Projector | Feature alignment |
| Language Model | LLaMA integration |
| Conversation | Prompt construction |
| Image Processing | Image preprocessing utilities |
| Training Pipeline | Training workflow |
| Custom Trainer | Optimizer, sampler and checkpointing |
| Inference Pipeline | End-to-end inference |

---

# Source Files Covered

- `builder.py`
- `llava_arch.py`
- `clip_encoder.py`
- `builder.py` (Multimodal Projector)
- `llava_llama.py`
- `conversation.py`
- `mm_utils.py`
- `train.py`
- `llava_trainer.py`
- `run_llava.py`

---

# Key Topics

- Vision-Language Models
- CLIP Vision Encoder
- LLaMA
- Multimodal Projector
- Tensor Flow
- Image Embeddings
- Prompt Construction
- Training Pipeline
- Inference Pipeline
- Hugging Face Transformers

---

# References

- LLaVA: Visual Instruction Tuning
- Official LLaVA GitHub Repository
- CLIP: Learning Transferable Visual Models
- LLaMA
- Hugging Face Transformers Documentation

---

# Disclaimer

This repository is an educational implementation study of the official LLaVA project.

It does **not** contain or redistribute the original LLaVA source code. All credit for the original implementation and research belongs to the LLaVA authors.

The explanations, diagrams, and documentation in this repository are written by me for learning and educational purposes.
# Model Loading (`builder.py`)

## Introduction

The `builder.py` file is the entry point for loading a pretrained LLaVA model.

Whenever inference or evaluation begins, the model is first loaded through this file. It is responsible for constructing every major component required by LLaVA, including the tokenizer, language model, vision encoder, multimodal projector, and image processor.

Instead of manually creating each module, the rest of the codebase simply calls:

```python
load_pretrained_model(...)
```

This function returns a fully initialized LLaVA model ready for inference.

---

# Purpose of builder.py

The primary responsibility of this file is to:

- Load the tokenizer
- Load the pretrained LLaVA model
- Load the Vision Tower
- Load the Image Processor
- Configure the context length
- Return all required objects

The rest of the implementation does not need to know how these components are created.

---

# High-Level Workflow

```
Model Path
     │
     ▼
Load Tokenizer
     │
     ▼
Load LLaVA Model
     │
     ▼
Load Vision Tower
     │
     ▼
Load Image Processor
     │
     ▼
Configure Model
     │
     ▼
Return Components
```

---

# Returned Objects

The function returns four important objects.

```python
tokenizer
model
image_processor
context_len
```

Each object serves a different purpose.

---

## Tokenizer

Responsible for converting text into token IDs.

Example:

```
"What is in this image?"

↓

[1, 345, 89, ...]
```

The tokenizer also recognizes special multimodal tokens such as:

```
<image>

<im_start>

<im_end>
```

These tokens indicate where visual information should be inserted.

---

## Model

The returned model is an instance of LLaVA.

Internally it consists of:

```
CLIP Vision Tower

↓

MM Projector

↓

LLaMA
```

During inference, calling

```python
model.generate(...)
```

automatically performs the complete multimodal pipeline.

---

## Image Processor

Images cannot be directly passed into CLIP.

The image processor performs preprocessing such as:

- Resize
- Normalize
- Convert to tensor
- Batch images

The output is compatible with the CLIP Vision Encoder.

---

## Context Length

Large Language Models have a maximum context length.

The builder determines the supported context length and returns it for later use during generation.

---

# Different Loading Modes

The builder supports loading models in different ways.

Examples include:

- Full pretrained checkpoints
- LoRA adapters
- Quantized models (4-bit / 8-bit)
- Base language models

The loading logic automatically selects the correct configuration depending on the supplied model path.

---

# Why Centralize Model Loading?

Instead of scattering loading logic across multiple files, LLaVA keeps everything inside one function.

Advantages include:

- Cleaner implementation
- Easier maintenance
- Consistent initialization
- Simpler inference scripts

Files such as `run_llava.py` only need a single function call to initialize the complete model.

---

# Role in the Repository

```
run_llava.py
      │
      ▼
load_pretrained_model()
      │
      ▼
builder.py
      │
      ├── Tokenizer
      ├── LLaVA Model
      ├── Vision Tower
      ├── Image Processor
      └── Context Length
```

This makes `builder.py` the entry point for almost every inference workflow.

---

# Key Takeaways

- `builder.py` is responsible for initializing the complete LLaVA model.
- It hides the complexity of model construction behind a single function.
- It loads the tokenizer, vision encoder, language model, image processor, and configuration.
- It returns everything required to begin multimodal inference.

---

# Next

The next document explores `llava_arch.py`, which contains the core multimodal architecture and the logic that connects visual embeddings with the language model.
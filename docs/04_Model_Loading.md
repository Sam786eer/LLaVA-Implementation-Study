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

# Core Function

The central function of this file is:

```python
def load_pretrained_model(
    model_path,
    model_base,
    model_name,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    **kwargs
):
```

This function constructs and returns every component required for inference.

---

# What Happens Inside?

The loading process follows these steps:

```text
Read Model Path
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
Return Components
```

---

# Simplified Implementation

The overall logic can be summarized as:

```python
tokenizer = AutoTokenizer.from_pretrained(model_path)

model = LlavaLlamaForCausalLM.from_pretrained(model_path)

vision_tower = model.get_vision_tower()

vision_tower.load_model()

image_processor = vision_tower.image_processor

return tokenizer, model, image_processor, context_len
```

> **Note:** This is a simplified version for learning purposes. The official implementation also handles LoRA adapters, quantized models (4-bit/8-bit), device mapping, and configuration options.

---

# Function Responsibilities

| Function | Purpose |
|----------|---------|
| `load_pretrained_model()` | Loads the complete LLaVA model |
| `AutoTokenizer.from_pretrained()` | Loads the tokenizer |
| `from_pretrained()` | Loads pretrained model weights |
| `get_vision_tower()` | Returns the CLIP Vision Encoder |
| `load_model()` | Loads the vision model weights |

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
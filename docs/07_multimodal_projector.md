# Multimodal Projector (`multimodal_projector/builder.py`)

## Introduction

The CLIP Vision Encoder and the LLaMA Language Model operate in different embedding spaces. Although the Vision Encoder extracts rich visual representations, these feature vectors cannot be directly consumed by LLaMA because their dimensions and feature distributions differ.

The Multimodal Projector bridges this gap by transforming CLIP image features into embeddings compatible with the LLaMA language model.

Without the projector, LLaVA would not be able to combine visual and textual information.

---

# Position in the Pipeline

```
Image
 │
 ▼
CLIP Vision Encoder
 │
 ▼
Visual Features
 │
 ▼
Multimodal Projector
 │
 ▼
Image Embeddings
 │
 ▼
LLaMA
```

The projector performs the feature alignment stage.

---

# Why Is a Projector Needed?

Suppose the Vision Encoder produces the following tensor:

```
(B, 576, 1024)
```

LLaMA expects embeddings of dimension:

```
4096
```

These tensors cannot be used together directly.

The projector learns a mapping from the CLIP embedding space to the LLaMA embedding space.

---

# High-Level Workflow

```
Visual Features
        │
        ▼
 Select Projector Type
        │
        ▼
 Project Features
        │
        ▼
 Return Image Embeddings
```

---

# Supported Projector Types

The official implementation supports multiple projector architectures.

## 1. Linear Projector

```
Linear Layer

1024

↓

4096
```

A single fully connected layer performs the transformation.

Advantages:

- Simple
- Fast
- Few trainable parameters

---

## 2. MLP Projector

```
1024

↓

Linear

↓

GELU

↓

Linear

↓

4096
```

This is the default configuration used in LLaVA.

Compared to a single linear layer, the MLP can learn more expressive nonlinear mappings.

---

## 3. Identity Projector

```
Input

↓

Identity

↓

Output
```

No transformation is performed.

This option is useful only when the feature dimensions are already compatible.

---

# Builder Pattern

The file does not directly construct a specific projector.

Instead, it reads the configuration and creates the requested projector dynamically.

```
Configuration
       │
       ▼
builder.py
       │
       ▼
Linear

or

MLP

or

Identity
```

This design makes it easy to experiment with different projector architectures.

---

# Tensor Transformation

Example:

Before projection:

```
(B, 576, 1024)
```

After projection:

```
(B, 576, 4096)
```

Notice that:

- Batch size remains unchanged.
- Number of visual tokens remains unchanged.
- Only the feature dimension changes.

This transformed tensor is now compatible with LLaMA.

---

# Why Not Fine-Tune CLIP?

Instead of modifying the pretrained Vision Encoder, LLaVA trains only the lightweight projector during the first stage of training.

Benefits include:

- Lower computational cost
- Faster convergence
- Preserves CLIP's pretrained visual knowledge
- Requires significantly fewer trainable parameters

---

# Interaction with Other Files

```
clip_encoder.py
        │
        ▼
multimodal_projector/
        │
        ▼
llava_arch.py
        │
        ▼
llava_llama.py
```

The projector serves as the bridge between image features and the language model.

---

# Summary

The Multimodal Projector is a lightweight neural network that aligns CLIP visual features with the embedding space of LLaMA.

The official implementation supports multiple projector types, including Linear, MLP, and Identity, with the MLP projector being the default configuration.

Its output is a sequence of image embeddings that can be directly inserted into the language model.

---

# Next

The next document explores `llava_llama.py`, where the projected image embeddings are combined with textual embeddings and processed by the LLaMA language model.
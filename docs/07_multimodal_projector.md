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

The default LLaVA implementation uses a small Multi-Layer Perceptron (MLP) for feature alignment.

```mermaid
flowchart LR
    A[1024-D CLIP Feature]
    --> B[Linear]
    --> C[GELU]
    --> D[Linear]
    --> E[4096-D LLaMA Feature]
```

### Simplified Implementation

```python
projector = nn.Sequential(
    nn.Linear(1024, 4096),
    nn.GELU(),
    nn.Linear(4096, 4096)
)
```

Compared to a single linear layer, the MLP learns a richer nonlinear transformation between the CLIP and LLaMA embedding spaces.

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

---

# Key Functions

The projector module is responsible for constructing the feature alignment network used between the Vision Encoder and the Language Model.

| Function | Purpose |
|----------|---------|
| `build_vision_projector()` | Creates the projector specified in the configuration |
| `forward()` | Projects CLIP features into the LLaMA embedding space |

Unlike the Vision Tower, the projector is lightweight and usually contains only a few trainable layers.

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

---

# Simplified Construction Logic

The builder dynamically selects the projector architecture.

```python
if projector_type == "linear":
    projector = nn.Linear(mm_hidden_size, hidden_size)

elif projector_type == "mlp":
    projector = nn.Sequential(
        nn.Linear(mm_hidden_size, hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, hidden_size)
    )

elif projector_type == "identity":
    projector = IdentityMap()
```

### Explanation

- **Linear** uses a single fully connected layer.
- **MLP** adds a non-linear GELU activation for greater expressive power.
- **Identity** performs no transformation and is mainly used for experimentation.

> **Note:** The official implementation supports additional configuration options. The snippet above illustrates the core logic.

---

# Tensor Transformation

During inference, only the embedding dimension changes.

```mermaid
flowchart LR
    A["(B,576,1024)"]
    --> B[MLP Projector]
    --> C["(B,576,4096)"]
```

Where:

- **B** = Batch Size
- **576** = Number of image patches
- **1024** = CLIP embedding dimension
- **4096** = LLaMA embedding dimension

The projector preserves both the batch size and the number of visual tokens while aligning the feature dimension.

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

---

# Code Flow

The projector executes immediately after visual feature extraction.

```
Image

↓

CLIP Vision Encoder

↓

Patch Features

↓

MM Projector.forward()

↓

Projected Embeddings

↓

prepare_inputs_labels_for_multimodal()
```

These projected embeddings are now compatible with the language model and can replace the `<image>` placeholder token in the prompt.

---

---

# Projector Comparison

| Projector | Layers | Non-Linearity | Typical Usage |
|-----------|--------|---------------|---------------|
| Linear | 1 | No | Simple baseline |
| MLP | 2 | GELU | Default LLaVA configuration |
| Identity | 0 | No | Compatible embedding spaces only |

The MLP projector provides a balance between computational efficiency and expressive feature alignment, making it the preferred choice in the official LLaVA implementation.

---

# Summary

The Multimodal Projector is a lightweight neural network that aligns CLIP visual features with the embedding space of LLaMA.

The official implementation supports multiple projector types, including Linear, MLP, and Identity, with the MLP projector being the default configuration.

Its output is a sequence of image embeddings that can be directly inserted into the language model.

---

# Next

The next document explores `llava_llama.py`, where the projected image embeddings are combined with textual embeddings and processed by the LLaMA language model.
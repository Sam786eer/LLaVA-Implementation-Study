# LLaVA Architecture

## Introduction

LLaVA (Large Language and Vision Assistant) is a Vision-Language Model (VLM) that combines a pretrained vision encoder with a pretrained Large Language Model (LLM). Rather than training an entirely new multimodal model from scratch, LLaVA connects these two pretrained models using a lightweight Multimodal Projector.

The official implementation consists of three major components:

1. CLIP Vision Encoder
2. Multimodal Projector
3. LLaMA Language Model

Together, these components allow LLaVA to understand images and answer natural language questions about them.

---

# Overall Architecture

```
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
        Image Embeddings (LLaMA Space)
                      │
                      ▼
prepare_inputs_labels_for_multimodal()
                      │
                      ▼
Replace <image> Token Embeddings
                      │
                      ▼
            LLaMA Decoder
                      │
                      ▼
             Generated Response
```

---

# Component 1 — Image Processor

The Image Processor prepares the input image before it is passed to the vision encoder.

Typical preprocessing steps include:

- Resize image
- Normalize pixel values
- Convert image into tensors
- Batch multiple images

The output is a tensor suitable for the CLIP Vision Encoder.

---

# Component 2 — CLIP Vision Encoder

The Vision Encoder extracts semantic information from the image.

Instead of producing labels, CLIP generates a sequence of feature vectors representing different image regions.

Output example:

```
Image

↓

(1, 576, 1024)
```

where

- Batch Size = 1
- 576 visual tokens
- 1024-dimensional feature vector

These features contain rich semantic information but cannot yet be understood by the language model.

---

# Component 3 — Multimodal Projector

The CLIP feature dimension does not match the embedding dimension expected by LLaMA.

Therefore, LLaVA introduces a Multimodal Projector.

Its responsibilities are:

- Transform CLIP features
- Match LLaMA embedding size
- Preserve semantic information

Example:

```
(1,576,1024)

↓

MLP Projector

↓

(1,576,4096)
```

Now every visual token has the same embedding dimension as LLaMA.

---

# Component 4 — Multimodal Fusion

After projection, image embeddings replace the placeholder image token inside the prompt.

Example prompt:

```
<image>

Describe the image.
```

becomes internally

```
[Visual Embeddings]

Describe the image.
```

This replacement allows LLaMA to process visual information exactly like text embeddings.

---

# Component 5 — LLaMA

LLaMA processes:

- User prompt
- Image embeddings
- Previous generated tokens

The decoder then predicts one token at a time until the response is complete.

---

# Complete Data Flow

```
Image
 │
 ▼
Image Processor
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
Insert into Prompt
 │
 ▼
LLaMA
 │
 ▼
Generated Response
```

---

# Why This Architecture Works

The architecture is effective because it reuses powerful pretrained models.

- CLIP already understands images.
- LLaMA already understands language.
- The Multimodal Projector learns how to align visual features with the language model.

This design significantly reduces training cost while achieving strong multimodal performance.

---

# Implementation Mapping

| Component | Official Implementation |
|------------|------------------------|
| Model Loader | `builder.py` |
| Vision Encoder | `clip_encoder.py` |
| Projector | `multimodal_projector/builder.py` |
| Fusion Logic | `llava_arch.py` |
| Language Model | `llava_llama.py` |
| Inference | `run_llava.py` |

---

# Summary

LLaVA consists of three core modules:

- CLIP Vision Encoder
- Multimodal Projector
- LLaMA Language Model

The Image Processor prepares the image, CLIP extracts visual features, the Projector aligns them with LLaMA's embedding space, and LLaMA generates the final response.

The following documents explore how each component is implemented in the official codebase.
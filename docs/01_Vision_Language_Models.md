# Vision-Language Models (VLMs)

## Introduction

Large Language Models (LLMs) have demonstrated remarkable capabilities in understanding and generating natural language. However, they are inherently limited to processing text and cannot directly interpret visual information such as images.

Vision-Language Models (VLMs) extend LLMs by enabling them to jointly understand images and text. They combine a visual encoder with a language model, allowing the model to answer questions about images, describe scenes, perform reasoning, and follow multimodal instructions.

LLaVA is one of the most influential open-source Vision-Language Models and serves as the focus of this implementation study.

---

# Why Do We Need Vision-Language Models?

Traditional Large Language Models only process text.

For example, consider the following image.

```
🖼️ Image of a traffic intersection
```

A normal LLM cannot answer questions like:

- What objects are present?
- How many vehicles are there?
- What color is the traffic light?
- Is it safe to cross the road?

because the model has no mechanism to understand pixels.

A Vision-Language Model solves this limitation by converting an image into a representation that the language model can understand.

---

# High-Level Architecture

```
                Image
                  │
                  ▼
          Vision Encoder
                  │
          Visual Features
                  │
                  ▼
      Feature Alignment Module
                  │
                  ▼
        Large Language Model
                  │
                  ▼
         Natural Language Output
```

The vision encoder extracts meaningful visual features.

These features are transformed into the embedding space of the language model before being processed together with the text prompt.

---

# Components of a Vision-Language Model

## 1. Vision Encoder

The vision encoder extracts semantic information from an image.

Its responsibilities include:

- Object recognition
- Scene understanding
- Spatial relationships
- High-level visual representations

Popular vision encoders include:

- CLIP Vision Transformer (ViT)
- EVA CLIP
- SigLIP

LLaVA uses the CLIP Vision Transformer.

---

## 2. Feature Alignment Module

The visual features produced by the vision encoder cannot be directly processed by the language model because they exist in different embedding spaces.

A feature alignment module bridges this gap.

Typical approaches include:

- Linear Projection
- Multi-Layer Perceptron (MLP)
- Q-Former (used in BLIP-2)

LLaVA uses a Multimodal Projector implemented as a small MLP.

---

## 3. Large Language Model

The language model receives both textual embeddings and projected visual embeddings.

Its responsibilities include:

- Understanding user instructions
- Reasoning over visual information
- Generating natural language responses

LLaVA uses the LLaMA language model.

---

# Information Flow

```
Image
 │
 ▼
Vision Encoder
 │
 ▼
Visual Features
 │
 ▼
Multimodal Projector
 │
 ▼
Language Model
 │
 ▼
Generated Response
```

This is the fundamental pipeline followed by most modern Vision-Language Models.

---

# Why LLaVA?

LLaVA became popular because of its simple yet effective design.

Instead of introducing complex cross-modal attention mechanisms, LLaVA directly connects a pretrained CLIP Vision Encoder to a pretrained LLaMA model through a lightweight Multimodal Projector.

This design enables strong multimodal performance while keeping the architecture relatively simple.

---

# Key Takeaways

- Vision-Language Models combine image understanding with language generation.
- A Vision Encoder converts images into visual features.
- A Feature Alignment Module maps visual features into the language model's embedding space.
- The Language Model jointly processes visual and textual information.
- LLaVA follows a simple architecture consisting of CLIP, a Multimodal Projector, and LLaMA.

---

# Next

The next document explains the complete architecture of LLaVA and how these components interact inside the official implementation.
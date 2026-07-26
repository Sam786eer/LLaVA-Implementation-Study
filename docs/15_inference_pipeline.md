# End-to-End Inference Pipeline

## Introduction

After training is complete, LLaVA can perform multimodal inference by combining visual and textual information to generate natural language responses. Unlike the training pipeline, inference does not update model parameters. Instead, it uses the pretrained model to process user inputs and predict the most appropriate response.

This document presents the complete inference workflow, illustrating how every major component of the LLaVA implementation interacts during prediction.

---

# Complete Inference Pipeline

```
                    User Input
                 (Image + Prompt)
                         │
                         ▼
               Conversation Template
                         │
                         ▼
                  Prompt Formatting
                         │
                         ▼
                  Image Processing
                         │
                         ▼
                  CLIP Vision Encoder
                         │
                         ▼
                Visual Feature Extraction
                         │
                         ▼
                Multimodal Projector
                         │
                         ▼
              Projected Image Embeddings
                         │
                         ▼
      prepare_inputs_labels_for_multimodal()
                         │
                         ▼
             Merge Image & Text Embeddings
                         │
                         ▼
                   LLaMA Decoder
                         │
                         ▼
                Next Token Prediction
                         │
                         ▼
              Repeat Until Completion
                         │
                         ▼
                  Decode Token IDs
                         │
                         ▼
               Final Textual Response
```

---

# Step 1: User Input

Inference begins when a user provides:

- An image
- A text prompt

Example:

```
Image:
street.jpg

Prompt:
"What objects are visible in this image?"
```

---

# Step 2: Conversation Formatting

The prompt is converted into the conversation format expected by LLaVA.

Example:

```
USER:

<image>

What objects are visible in this image?

ASSISTANT:
```

This ensures compatibility with the prompt format used during training.

---

# Step 3: Image Preprocessing

The image is prepared using the CLIP Image Processor.

Operations include:

- Resize
- Center Crop
- Normalize
- Convert to Tensor

Output:

```
(B, 3, 336, 336)
```

---

# Step 4: Vision Encoding

The processed image is passed through the CLIP Vision Transformer.

```
Image Tensor

↓

Vision Transformer

↓

Patch Embeddings
```

Typical output:

```
(B, 576, 1024)
```

Each embedding represents one image patch.

---

# Step 5: Feature Projection

The Multimodal Projector maps CLIP features into the LLaMA embedding space.

```
(B, 576, 1024)

↓

MLP Projector

↓

(B, 576, 4096)
```

Now the visual embeddings have the same dimensionality as textual embeddings.

---

# Step 6: Multimodal Input Construction

The `<image>` placeholder token is replaced with projected image embeddings.

Before:

```
<image>

Describe this scene.
```

After:

```
Image Embeddings

Describe this scene.
```

The model now receives a unified embedding sequence containing both visual and textual information.

---

# Step 7: Language Model Inference

The merged embeddings are passed through the LLaMA decoder.

```
Merged Embeddings

↓

Transformer Layers

↓

Next Token Prediction
```

The decoder predicts one token at a time based on all previously generated tokens.

---

# Step 8: Autoregressive Generation

Generation proceeds iteratively.

```
Prompt

↓

Predict Token 1

↓

Predict Token 2

↓

Predict Token 3

↓

...

↓

End-of-Sequence Token
```

This continues until an end-of-sequence token is generated or the maximum generation length is reached.

---

# Step 9: Decoding

The generated token IDs are converted back into human-readable text.

```
Token IDs

↓

Tokenizer

↓

Natural Language Response
```

The decoded text is returned to the user.

---

# File Interaction

The inference pipeline connects nearly every major module in the repository.

```
run_llava.py
      │
      ▼
builder.py
      │
      ▼
conversation.py
      │
      ▼
mm_utils.py
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
Generated Response
```

Each module performs a specific task while contributing to the overall inference workflow.

---

# Difference Between Training and Inference

| Training | Inference |
|----------|-----------|
| Uses image-text dataset | Uses user input |
| Computes loss | No loss computation |
| Updates model parameters | Parameters remain fixed |
| Performs backpropagation | No backpropagation |
| Saves checkpoints | Generates responses |
| Optimizes the model | Uses the trained model |

---

# Summary

The inference pipeline demonstrates how LLaVA processes multimodal inputs to generate natural language responses. Starting from a user-provided image and prompt, the system preprocesses the image, extracts visual features, aligns them with textual embeddings, performs autoregressive decoding using LLaMA, and returns a coherent response.

Together with the training pipeline, this completes the end-to-end understanding of the official LLaVA implementation.

---

# Conclusion

This implementation study explored the complete architecture and workflow of the official LLaVA repository. Beginning with model loading and architectural components, it progressed through multimodal fusion, training, inference, and utility modules, concluding with comprehensive end-to-end pipeline analyses.

By understanding how these modules interact, readers gain practical insight into how modern Vision-Language Models integrate visual perception with large language models to perform multimodal reasoning and generation.
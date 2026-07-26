# Running LLaVA (`run_llava.py`)

## Introduction

The `run_llava.py` file is the primary inference script in the LLaVA repository. It demonstrates how a pretrained LLaVA model is loaded, how images and prompts are processed, and how the model generates responses.

Unlike the training pipeline, which focuses on optimizing model parameters, this script performs inference using a pretrained checkpoint.

For most users, `run_llava.py` serves as the main entry point for interacting with the model.

---

# Position in the Inference Pipeline

```
User Prompt
      │
      ▼
Load Image
      │
      ▼
Process Image
      │
      ▼
Load Model
      │
      ▼
Generate Response
      │
      ▼
Display Output
```

---

# Responsibilities

The main responsibilities of this file include:

- Parsing command-line arguments
- Loading the pretrained model
- Loading the tokenizer
- Loading the image processor
- Reading user images
- Formatting prompts
- Running inference
- Decoding generated tokens

---

# Overall Workflow

```
Start

 │

 ▼

Parse Arguments

 │

 ▼

Load Pretrained Model

 │

 ▼

Load Image

 │

 ▼

Preprocess Image

 │

 ▼

Build Prompt

 │

 ▼

Tokenize Input

 │

 ▼

Generate Response

 │

 ▼

Decode Tokens

 │

 ▼

Print Answer
```

---

# Step 1: Parse Arguments

The script begins by reading user-provided arguments.

Typical arguments include:

- Model path
- Image path
- User prompt
- Conversation template
- Temperature
- Maximum generation length

This allows the same script to work with different checkpoints and images.

---

# Step 2: Load the Model

The script calls:

```python
load_pretrained_model(...)
```

This initializes:

```
Tokenizer

↓

Vision Tower

↓

Multimodal Projector

↓

LLaMA
```

Once loaded, the model is ready for inference.

---

# Step 3: Load the Image

The input image is read from disk.

```
Image File

↓

PIL Image

↓

RGB Format
```

The image is then passed to the preprocessing pipeline.

---

# Step 4: Process the Image

The image processor converts the raw image into tensors.

```
Image

↓

Resize

↓

Normalize

↓

Tensor

↓

Batch
```

The resulting tensor is compatible with the Vision Encoder.

---

# Step 5: Build the Prompt

The user prompt is formatted using the conversation template.

Example:

```
USER:

<image>

Describe this image.

ASSISTANT:
```

This formatted prompt matches the structure used during training.

---

# Step 6: Tokenization

The prompt is converted into token IDs.

The special `<image>` token is preserved.

```
Prompt

↓

Tokenizer

↓

Input IDs
```

The image placeholder will later be replaced with image embeddings inside the model.

---

# Step 7: Generate Response

The script calls:

```python
model.generate(...)
```

Internally, the model performs:

```
Image

+

Prompt

↓

Vision Encoder

↓

MM Projector

↓

LLaMA

↓

Generated Tokens
```

Generation continues until an end-of-sequence token or the maximum token limit is reached.

---

# Step 8: Decode Output

The generated token IDs are converted back into natural language.

```
Token IDs

↓

Tokenizer Decode

↓

Response Text
```

The decoded response is printed to the terminal or returned to the caller.

---

# Generation Parameters

The script exposes several generation parameters, including:

- `temperature`
- `top_p`
- `num_beams`
- `max_new_tokens`

These parameters allow users to control response diversity and generation length without modifying the model.

---

# Interaction with Other Files

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
llava_llama.py
      │
      ▼
Generated Response
```

The inference script coordinates all major components but delegates the actual computation to the underlying model.

---

# Summary

The `run_llava.py` file is the primary inference entry point for LLaVA. It loads the pretrained model, preprocesses images, formats prompts, performs multimodal inference, and converts generated token IDs into human-readable responses.

By combining the repository's major components into a single workflow, it provides a simple interface for evaluating LLaVA on image-text tasks.

---

# Next

The next document presents the complete end-to-end **training pipeline**, illustrating how data flows from the training dataset to the optimized multimodal model.
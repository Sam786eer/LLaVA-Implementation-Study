# Multimodal Utilities (`mm_utils.py`)

## Introduction

The `mm_utils.py` file contains a collection of utility functions that simplify multimodal processing in LLaVA. Rather than implementing common operations repeatedly across different modules, these helper functions centralize tasks such as image preprocessing, image token handling, prompt processing, and model utilities.

Although this file does not implement the model itself, it plays a critical role in connecting the vision and language components during inference.

---

# Position in the Pipeline

```
User Input
      │
      ▼
conversation.py
      │
      ▼
mm_utils.py
      │
      ├── Process Images
      ├── Insert Image Tokens
      ├── Tokenize Prompt
      └── Prepare Inputs
      │
      ▼
LLaVA Model
```

---

# Responsibilities

The major responsibilities of this file include:

- Image preprocessing
- Image token insertion
- Prompt tokenization
- Utility functions for inference
- Model name extraction
- Stopping criteria for generation

---

# Image Processing

Images received from the user must be converted into tensors before they can be passed to the Vision Encoder.

Typical workflow:

```
Image
   │
   ▼
Resize
   │
   ▼
Normalize
   │
   ▼
Convert to Tensor
   │
   ▼
Batch Images
```

The resulting tensor is compatible with the CLIP Vision Encoder.

---

# process_images()

One of the most commonly used helper functions is:

```python
process_images()
```

Responsibilities:

- Accept one or more images
- Apply the CLIP Image Processor
- Convert images into tensors
- Return batched image tensors

Output example:

```
(B, 3, 336, 336)
```

where:

- **B** = Batch Size
- **3** = RGB Channels
- **336 × 336** = Image Resolution

---

# tokenizer_image_token()

Text prompts containing images require special placeholder tokens.

Example prompt:

```
USER:

<image>

Describe this picture.
```

The tokenizer converts this into token IDs while preserving the image placeholder.

Example:

```
Text

↓

Special Image Token

↓

Token IDs
```

Later, this placeholder is replaced by image embeddings inside `llava_arch.py`.

---

# get_model_name_from_path()

This helper extracts the model name from a checkpoint path.

Example:

```
checkpoints/llava-v1.5-7b

↓

llava-v1.5-7b
```

This simplifies configuration and logging across the repository.

---

# KeywordsStoppingCriteria

During text generation, the model should stop when a predefined keyword or separator is generated.

Example:

```
Generated Tokens

↓

Detect Stop Keyword

↓

Terminate Generation
```

This prevents unnecessary output beyond the expected assistant response.

---

# Why Centralize Utilities?

Without `mm_utils.py`, identical preprocessing logic would need to be duplicated across multiple files.

Centralizing these operations provides several benefits:

- Cleaner code
- Better maintainability
- Consistent preprocessing
- Easier debugging
- Reusable helper functions

---

# Interaction with Other Files

```
conversation.py
        │
        ▼
mm_utils.py
        │
        ├── Image Processing
        ├── Prompt Utilities
        ├── Token Utilities
        └── Stopping Criteria
        │
        ▼
run_llava.py
        │
        ▼
llava_arch.py
```

Nearly every inference-related module relies on helper functions defined in this file.

---

# Summary

The `mm_utils.py` file provides reusable helper functions that support multimodal inference in LLaVA. It handles image preprocessing, image token insertion, prompt tokenization, model utilities, and generation stopping logic.

Although it does not implement any neural network components, it serves as an essential bridge between user input and the multimodal architecture.

---

# Next

The next document explores `train.py`, the main training entry point responsible for dataset loading, model initialization, distributed training configuration, and launching the LLaVA training pipeline.
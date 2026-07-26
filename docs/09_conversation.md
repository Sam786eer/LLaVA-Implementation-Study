# Conversation Templates (`conversation.py`)

## Introduction

The `conversation.py` file is responsible for managing the interaction between the user and the LLaVA model. Instead of sending raw user input directly to the language model, this file formats conversations into structured prompts that follow predefined templates.

These templates ensure that the model receives input in the same format used during training, leading to more consistent and accurate responses.

---

# Position in the Pipeline

```
User Input
      │
      ▼
Conversation Template
      │
      ▼
Formatted Prompt
      │
      ▼
Tokenizer
      │
      ▼
LLaVA Model
```

This file handles prompt construction before tokenization.

---

# Why Are Conversation Templates Needed?

Large Language Models are trained on conversations with a specific structure.

For example, instead of passing:

```
Describe this image.
```

LLaVA formats the prompt as:

```
USER:
<image>

Describe this image.

ASSISTANT:
```

This consistent structure helps the model understand speaker roles and generate appropriate responses.

---

# Main Components

The file primarily defines:

- Conversation class
- Message history
- Separator styles
- Conversation templates

These components work together to build the final prompt.

---

# Conversation Class

The `Conversation` class stores:

- System prompt
- User messages
- Assistant messages
- Separator style
- Roles

Example:

```
System

↓

User

↓

Assistant

↓

User

↓

Assistant
```

Each message is appended in order to preserve the conversation history.

---

# Roles

Each message is associated with a role.

Typical roles include:

```
USER

ASSISTANT
```

Some templates also include:

```
SYSTEM
```

These role identifiers are inserted into the prompt before tokenization.

---

# Separator Styles

Different language models expect different conversation formats.

To support multiple models, LLaVA defines several separator styles.

Examples include:

- Single Separator
- Two Separators
- LLaMA 2 Style
- MPT Style
- Plain Style

Each style determines how messages are separated in the final prompt.

---

# Prompt Construction

The conversation object converts message history into a single prompt.

Example:

```
System Prompt

↓

User Message

↓

Assistant Response

↓

User Message

↓

Assistant:
```

The final prompt ends with the assistant role so the language model knows where to begin generating the next response.

---

# Handling Images

When an image is included, the prompt contains a special placeholder token.

Example:

```
USER:

<image>

What is shown in this picture?

ASSISTANT:
```

Later, during multimodal processing, the `<image>` token is replaced with image embeddings.

---

# Benefits

Using conversation templates provides:

- Consistent prompt formatting
- Support for multiple language models
- Easy extension to new chat formats
- Separation between prompt construction and model execution

---

# Interaction with Other Files

```
conversation.py
        │
        ▼
mm_utils.py
        │
        ▼
Tokenizer
        │
        ▼
llava_llama.py
```

The conversation module prepares prompts, while later components tokenize and process them.

---

# Summary

The `conversation.py` file standardizes user-model interactions by defining conversation templates, roles, separator styles, and prompt construction logic.

Its output is a formatted prompt that is passed to the tokenizer before multimodal processing begins.

---

# Next

The next document explores `mm_utils.py`, which provides helper utilities for image processing, token handling, and multimodal input preparation used throughout the LLaVA codebase.
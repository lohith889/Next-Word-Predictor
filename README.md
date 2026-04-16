# Next-Word-Predictor

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat&logo=tensorflow" alt="TensorFlow">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat" alt="License">
</p>

A deep learning-based next word prediction system using LSTM (Long Short-Term Memory) neural networks trained on the BookCorpus dataset.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Files](#files)
- [License](#license)

---

## Overview

This project implements a **Next Word Prediction model** using an LSTM-based neural network. Given a sequence of words, the model learns to predict the most probable next word based on context, enabling applications like:

- Smart keyboard suggestions
- Autocomplete systems
- Text generation assistants

---

## Architecture

```
+----------------------------------------------------------+
|                     MODEL ARCHITECTURE                    |
+----------------------------------------------------------+

    INPUT SEQUENCE
    ["the", "quick", "brown"]
           |
           v
    +------------------+
    |   Embedding      |  (Maps words to dense vectors)
    |   Layer          |  (Vocabulary -> 128-dim embedding)
    +------------------+
           |
           v
    +------------------+
    |     LSTM         |  (Processes sequential data)
    |   Layer 1        |  (256 units, returns sequences)
    +------------------+
           |
           v
    +------------------+
    |     LSTM         |  (Second LSTM layer)
    |   Layer 2        |  (256 units)
    +------------------+
           |
           v
    +------------------+
    |    Dense         |  (Fully connected layer)
    |    (ReLU)        |  (Output: 128)
    +------------------+
           |
           v
    +------------------+
    |    Dense         |  (Output layer)
    |   (Softmax)      |  (Vocabulary size - probability)
    +------------------+
           |
           v
      PREDICTED
       NEXT WORD
```

---

## How It Works

### Data Pipeline

```
+------------------------------------------------------------------+
|                        DATA FLOW DIAGRAM                        |
+------------------------------------------------------------------+

    +-----------+      +------------+      +------------------+
    | BookCorpus|      |   Token    |      |   Sequential     |
    |  Dataset  | ---> |   izer     | ---> |   Generation    |
    | (74M+)    |      | (Word IDs) |      |   (N-grams)     |
    +-----------+      +------------+      +------------------+
                                                       |
                                                       v
                                          +------------------------+
                                          |   Pad Sequences       |
                                          |   (Fixed Length: 40)  |
                                          +------------------------+
                                                       |
                                                       v
                                          +------------------------+
                                          |   One-Hot Encoding    |
                                          |   (Target Words)      |
                                          +------------------------+
                                                       |
                                                       v
                                          +------------------------+
                                          |   Model Training      |
                                          |   (LSTM Neural Net)   |
                                          +------------------------+
```

### Prediction Process

```
+------------------------------------------------------------------+
|                     PREDICTION WORKFLOW                        |
+------------------------------------------------------------------+

    User Input:  "the quick brown"
           |
           v
    +------------------+      +------------------+
    |   Tokenization  | ---> |   Word IDs       |
    |   "the" -> 1     |      |   [1, 45, 92]    |
    |   "quick" -> 45  |      +------------------+
    |   "brown" -> 92  |
    +------------------+
           |
           v
    +------------------+      +------------------+
    |   Padding        | ---> |   [0, 0, ...,    |
    |   (40 words)    |      |    1, 45, 92]    |
    +------------------+      +------------------+
           |
           v
    +------------------+      +------------------+
    |    LSTM Model    | ---> |   Probabilities  |
    |   (next_word.h5)|      |   [0.1, 0.05,    |
    |                  |      |    0.03, ...]    |
    +------------------+      +------------------+
           |
           v
    +------------------+      +------------------+
    |   ArgMax /       | ---> |   "fox"          |
    |   Sampling      |      |   (Predicted)    |
    +------------------+      +------------------+
```

---

## Project Structure

```
Next-Word-Predictor/
|
|-- nwp.ipynb              # Main Jupyter Notebook
|   |-- Data Loading       # BookCorpus dataset
|   |-- Preprocessing      # Tokenization & sequencing
|   |-- Model Training     # LSTM model definition & training
|   |-- Prediction         # Next word prediction logic
|
|-- next_word.h5           # Trained model weights (HDF5)
|-- tokenized_text.pkl     # Preprocessed tokenizer
|-- README.md             # This file
```

---

## Installation

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- NumPy
- pickle

### Install Dependencies

```bash
pip install tensorflow numpy
```

### Dataset

The model uses the **BookCorpus** dataset (74M+ text samples) loaded via Hugging Face:

```python
from datasets import load_dataset
ds = load_dataset("rojagtap/bookcorpus")
```

---

## Usage

### Load and Use the Model

```python
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = tf.keras.models.load_model('next_word.h5')

# Load tokenizer
with open('tokenized_text.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Predict next word
def predict_next_word(text, max_length=40):
    token_list = tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_length, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return ""

# Example
text = "the quick brown"
next_word = predict_next_word(text)
print(f"Input: '{text}' -> Predicted next word: '{next_word}'")
```

### Example Output

```
Input: 'the quick brown' -> Predicted next word: 'fox'
Input: 'she was very'    -> Predicted next word: 'happy'
Input: 'in the'          -> Predicted next word: 'morning'
```

---

## Model Performance

| Metric | Value |
|--------|-------|
| Dataset | BookCorpus (74M+ samples) |
| Vocabulary Size | ~30,000 words |
| Sequence Length | 40 words |
| Embedding Dimension | 128 |
| LSTM Units (Layer 1) | 256 |
| LSTM Units (Layer 2) | 256 |
| Training Epochs | 20+ |
| Model Format | HDF5 (.h5) |

---

## Files

| File | Description |
|------|-------------|
| `nwp.ipynb` | Complete training and prediction pipeline |
| `next_word.h5` | Trained LSTM model weights |
| `tokenized_text.pkl` | Saved tokenizer for text processing |
| `README.md` | Project documentation |

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- [BookCorpus Dataset](https://huggingface.co/datasets/rojagtap/bookcorpus) for training data
- [TensorFlow](https://www.tensorflow.org/) for deep learning framework
- [Hugging Face](https://huggingface.co/) for dataset access
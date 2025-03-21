# NeuralNetwork2: A Customizable Neural Network with Attention Mechanism

## Overview
`neural_network2.py` implements a fully customizable neural network that supports dense layers, attention mechanisms, dropout, and token embedding. The model is built using NumPy and supports text tokenization using the Hugging Face `transformers` library. This neural network can be used for various tasks such as classification and sequence-based learning.

## Features
- Fully connected feedforward layers
- Configurable activation functions (`relu`, `sigmoid`, `softmax`, `leaky_relu`)
- Optional attention mechanism for sequence processing
- Dropout support to prevent overfitting
- Hugging Face tokenizer integration for text preprocessing
- Save and load functionality using `pickle`

## When to Use
Use this neural network if:
- You need a flexible, lightweight, and customizable neural network implementation.
- You want to integrate text-based embeddings with self-attention.
- You prefer to work with a NumPy-based implementation rather than deep learning frameworks like TensorFlow or PyTorch.

## Installation
Before using this script, install the required dependencies:
```bash
pip install numpy transformers
```

## Usage
### Initializing the Model
```python
from neural_network2 import NeuralNetwork

layer_sizes = [128, 64, 32, 10]  # Example: Input layer (128), two hidden layers, output layer (10 classes)
activation_functions = ['relu', 'relu', 'softmax']  # Activation functions per layer

model = NeuralNetwork(
    layer_sizes=layer_sizes,
    activation_functions=activation_functions,
    learning_rate=0.01,
    epochs=50,
    batch_size=32,
    dropout_rate=0.2,
    use_embedding=True,  # Enable embedding if working with text
    vocab_size=5000,  # Vocabulary size for tokenization
    embed_dim=128,  # Embedding vector dimension
    max_seq_length=512,  # Maximum sequence length
    tokenizer_name='bert-base-uncased'  # Hugging Face tokenizer
)
```

### Training the Model
```python
# Example dummy dataset
import numpy as np

X_train = np.random.randint(0, 5000, (1000, 128))  # Simulated tokenized input
Y_train = np.eye(10)[np.random.choice(10, 1000)]  # One-hot encoded labels

model.train(X_train, Y_train)
```

### Making Predictions
```python
X_test = np.random.randint(0, 5000, (10, 128))  # Simulated test input
y_pred = model.predict(X_test)
print(y_pred)
```

### Saving and Loading the Model
```python
model.save("model.pkl")

# Load model later
loaded_model = NeuralNetwork.load("model.pkl", tokenizer_name='bert-base-uncased')
```

## Explanation of Key Components
- **Softmax Function:** Implements softmax activation for multi-class classification.
- **Forward Propagation:** Computes activations through dense layers and optional self-attention.
- **Backward Propagation:** Updates weights using gradient descent.
- **Embedding Layer:** Converts tokenized input into vector representations.
- **Attention Mechanism:** Computes weighted attention scores to enhance sequential data processing.

## Limitations
- Lacks GPU acceleration (relies solely on NumPy).
- No built-in support for convolutional layers or recurrent layers.

## Conclusion
This neural network implementation provides a simple yet effective approach for working with dense networks and attention mechanisms. It is ideal for educational purposes, quick prototyping, and lightweight deep learning applications.


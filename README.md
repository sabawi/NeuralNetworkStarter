# Simple Neural Network in Python

This README provides a comprehensive overview of the attached Python code, which implements a basic feedforward neural network. It details the network's capabilities, how to use it, its features, limitations, and advantages, along with illustrative examples to get you excited about experimenting with it!

## What is it?

This code defines a `NeuralNetwork` class that implements a fundamental feedforward neural network. It's designed to be a flexible and educational tool for understanding the core concepts of neural networks. While not as sophisticated as deep learning libraries like TensorFlow or PyTorch, it provides a hands-on way to build and train simple neural network models for various tasks.

## What does it do?

The `NeuralNetwork` class allows you to:

* **Define network architecture:** Specify the number and size of layers in your network.
* **Choose activation functions:** Select from sigmoid, tanh, ReLU, leaky ReLU, and softmax activation functions for different layers.
* **Train your model:** Train the network using backpropagation on provided training data and labels.
* **Make predictions:** Use the trained model to predict outputs for new input data.
* **Evaluate performance:** Assess the model's performance using metrics like accuracy, precision, recall, F1-score for classification, and MSE, MAE, R² for regression.
* **Save and load models:** Persist trained models to disk and load them for later use.
* **Implement dropout:** Use dropout regularization to prevent overfitting.
* **Utilize embeddings for text:** Incorporate a basic embedding layer to process text data.
* **Perform simple text generation:** With the embedding layer, the model can be used for rudimentary text generation from a prompt.

## How to use it?

Here's a step-by-step guide on how to use the `NeuralNetwork` class:

### 1. Import the necessary modules:

```python
import numpy as np
from your_module import NeuralNetwork # Assuming you saved the code in 'your_module.py'
```

### 2. Initialize the Neural Network:

You need to define the layer_sizes for your network. For example, a network with an input layer of size 10, one hidden layer of size 5, and an output layer of size 2 would be defined as [10, 5, 2].

```python
# Example for a simple classification task
layer_sizes = [10, 5, 2]
model = NeuralNetwork(layer_sizes=layer_sizes, learning_rate=0.01, epochs=200, batch_size=32, activation_functions=['relu', 'softmax'])

# Example for a regression task
layer_sizes_regression = [5, 3, 1]
model_regression = NeuralNetwork(layer_sizes=layer_sizes_regression, learning_rate=0.005, epochs=150, batch_size=16, activation_functions=['relu', 'linear']) # Assuming you'd add 'linear' activation

# Example for text processing (requires vocab_size, embed_dim, and tokenizer_name)
layer_sizes_text = [768, 128, 3] # Example: embedding -> hidden -> output
vocab_size = 10000
embed_dim = 768
tokenizer_name = 'bert-base-uncased' # Or any other Hugging Face tokenizer
model_text = NeuralNetwork(layer_sizes=layer_sizes_text, learning_rate=0.0001, epochs=10, batch_size=16,
                           use_embedding=True, vocab_size=vocab_size, embed_dim=embed_dim, tokenizer_name=tokenizer_name,
                           activation_functions=['relu', 'softmax'])
```

### 3. Prepare your data:

Your training data (training_data) should be a NumPy array or a list of samples, and your labels (labels) should be a corresponding NumPy array or list. For text data, training_data should be a list of strings.

```python
# Example for classification
X_train = np.random.rand(100, 10) # 100 samples, 10 features
y_train = np.random.randint(0, 2, 100) # Binary classification labels (0 or 1)

# Example for regression
X_train_reg = np.random.rand(80, 5)
y_train_reg = np.random.rand(80, 1)

# Example for text classification
text_data = ["This is a positive review.", "This movie was terrible.", "I enjoyed the acting.", "The plot was confusing."]
text_labels = np.array([0, 1, 0, 1]) # 0 for positive, 1 for negative
```

### 4. Train the model:

Call the train method with your training data and labels.

```python
# For classification
model.train(X_train, y_train)

# For regression
model_regression.train(X_train_reg, y_train_reg)

# For text classification
model_text.train(text_data, text_labels)
```

### 5. Make predictions:

Use the predict method to get predictions on new data.

```python
# For classification
X_new = np.random.rand(5, 10)
predictions = model.predict(X_new)
print("Classification Predictions:", predictions)

# For regression
X_new_reg = np.random.rand(3, 5)
predictions_reg = model_regression.predict(X_new_reg)
print("Regression Predictions:", predictions_reg)

# For text classification
new_texts = ["This is an amazing film!", "I did not like it at all."]
text_predictions = model_text.predict(new_texts)
print("Text Classification Predictions:", text_predictions)
```

### 6. Evaluate the model:

If you have test data and labels, you can evaluate the model's performance.

```python
# For classification
X_test = np.random.rand(50, 10)
y_test = np.random.randint(0, 2, 50)
metrics = model.evaluate(X_test, y_test)
print("Classification Metrics:", metrics)

# For regression
X_test_reg = np.random.rand(20, 5)
y_test_reg = np.random.rand(20, 1)
metrics_reg = model_regression.evaluate(X_test_reg, y_test_reg)
print("Regression Metrics:", metrics_reg)

# For text classification
test_texts = ["A truly wonderful experience.", "The worst movie I've ever seen."]
test_labels_text = np.array([0, 1])
metrics_text = model_text.evaluate(test_texts, test_labels_text)
print("Text Classification Metrics:", metrics_text)
```

### 7. Save and load the model:

You can save your trained model and load it later.

```python
# Save the model
model.save("my_classification_model.pkl")

# Load the model
loaded_model = NeuralNetwork(layer_sizes=[1]) # You need to provide an initial layer size (will be overwritten on load)
loaded_model.load("my_classification_model.pkl")

# Now you can use the loaded_model for predictions or further training
```

### 8. Text Generation (Exciting Feature!):

If you initialized the model with use_embedding=True and a tokenizer_name, you can enable text generation and try it out!

```python
# Assuming model_text is initialized for text processing
model_text.enable_text_generation()
prompt = "The quick brown fox"
generated_text = model_text.generate(prompt, max_length=50, temperature=0.8)
print(f"Generated text from prompt 'The quick brown fox':\n{generated_text}")

prompt_2 = "Once upon a time in a faraway land"
generated_text_2 = model_text.generate(prompt_2, max_length=30, temperature=1.2)
print(f"\nGenerated text from prompt 'Once upon a time in a faraway land':\n{generated_text_2}")
```

## Types and Features of Neural Networks Implemented

This implementation includes the following types and features:

- **Feedforward Neural Network (FFNN)**: The core architecture is a simple FFNN where information flows in one direction, from the input layer through hidden layers to the output layer.
- **Dense Layers**: Each layer is a fully connected or dense layer, where every neuron in one layer is connected to every neuron in the subsequent layer.
- **Activation Functions**:
  - **Sigmoid**: Used for binary classification output layers or in hidden layers.
  - **Tanh (Hyperbolic Tangent)**: Another common activation function for hidden layers, similar to sigmoid but with an output range of -1 to 1.
  - **ReLU (Rectified Linear Unit)**: A popular activation function for hidden layers due to its simplicity and efficiency.
  - **Leaky ReLU**: A variation of ReLU that allows a small, non-zero gradient when the input is negative, addressing the "dying ReLU" problem.
  - **Softmax**: Typically used in the output layer for multi-class classification, converting raw scores into probability distributions over the classes.
- **Dropout**: A regularization technique where randomly selected neurons are "dropped out" during training, preventing overfitting by reducing the interdependence of neurons.
- **Embedding Layer**: A basic embedding layer is implemented for handling text data. It maps discrete tokens (words or subwords) to dense vector representations. This allows the network to understand semantic relationships between words.
- **Simple Text Generation**: By leveraging the embedding layer and a tokenizer from the transformers library, the model can perform basic auto-regressive text generation. It predicts the next token based on the preceding sequence.
- **Customizable Learning Rate, Epochs, and Batch Size**: These hyperparameters can be adjusted to control the training process.
- **Xavier/Glorot and He Initialization**: The weights are initialized using these methods, which help in faster convergence during training.

## Limitations

This implementation has several limitations:

- **Simplicity**: It's a basic implementation and does not include more advanced neural network architectures like Convolutional Neural Networks (CNNs) for image processing or Recurrent Neural Networks (RNNs) and Transformers for sequential data.
- **Scalability**: Training on very large datasets might be slow due to the lack of optimizations found in dedicated deep learning libraries.
- **Limited Embedding Capabilities**: The embedding layer is a simple matrix lookup. It doesn't have the complexity of pre-trained embeddings like Word2Vec, GloVe, or those from Transformer models, although it can utilize tokenizers from the transformers library.
- **Basic Text Generation**: The text generation is rudimentary and relies on sampling the next token based on probabilities. It might not produce coherent or contextually rich text for complex tasks.
- **No GPU Acceleration**: The code relies on NumPy and does not utilize GPU acceleration, which is crucial for training large neural networks efficiently.
- **Lack of Advanced Optimization Techniques**: It uses basic gradient descent with a fixed learning rate (though a basic learning rate schedule can be implemented externally). More advanced optimizers like Adam or RMSprop are not included.
- **Error Handling**: While some basic error handling is present, it might not be as robust as in production-ready libraries.

## Advantages

Despite its limitations, this implementation offers several advantages:

- **Educational Value**: It provides a clear and understandable implementation of the fundamental principles of neural networks, making it an excellent tool for learning.
- **Flexibility**: It can be adapted for various tasks like binary and multi-class classification, as well as regression.
- **Embedding Layer for Text**: The inclusion of an embedding layer allows for basic processing of text data, which is a key aspect of many modern applications.
- **Transparency**: The code is relatively straightforward, allowing users to easily inspect and modify the network's behavior.
- **No External Dependencies (Core)**: The core neural network implementation relies only on NumPy, making it easy to set up and run in environments where installing large libraries might be challenging. The text processing feature depends on the transformers library.
- **Saving and Loading**: The ability to save and load trained models allows for persistence and reuse.

## Get Excited! Examples of Key Features

Here are some exciting examples showcasing the key features of this neural network:

### 1. Simple Binary Classification:

```python
import numpy as np
from your_module import NeuralNetwork

# Create some synthetic binary classification data
X = np.random.rand(100, 5)
y = np.random.randint(0, 2, 100).reshape(-1, 1) # Reshape labels for consistency

# Initialize a simple network
model = NeuralNetwork(layer_sizes=[5, 3, 1], learning_rate=0.05, epochs=100, activation_functions=['relu', 'sigmoid'])

# Train the model
print("Training the binary classifier...")
model.train(X, y, verbose=True)

# Make predictions
X_test = np.random.rand(10, 5)
predictions = model.predict(X_test)
print("\nBinary Classification Predictions (probabilities):\n", predictions)

# Evaluate (simple accuracy for demonstration)
y_test = np.random.randint(0, 2, 10).reshape(-1, 1)
predicted_classes = np.round(predictions)
accuracy = np.mean(predicted_classes == y_test)
print("\nBinary Classification Accuracy:", accuracy)
```

### 2. Basic Text Sentiment Analysis:

```python
import numpy as np
from your_module import NeuralNetwork

# Sample text data and labels (0: positive, 1: negative)
text_data = ["I love this!", "This is terrible.", "Great movie!", "Absolutely awful."]
text_labels = np.array([0, 1, 0, 1])

# Initialize a text processing model
vocab_size = 5000
embed_dim = 100
tokenizer_name = 'bert-base-uncased' # You need to install transformers: pip install transformers
layer_sizes_text = [embed_dim, 10, 2] # Embedding dim -> hidden -> output (2 classes)
model_sentiment = NeuralNetwork(layer_sizes=layer_sizes_text, learning_rate=0.001, epochs=5, batch_size=2,
                                use_embedding=True, vocab_size=vocab_size, embed_dim=embed_dim, tokenizer_name=tokenizer_name,
                                activation_functions=['relu', 'softmax'])

# Train the sentiment analysis model
print("\nTraining the text sentiment analysis model...")
model_sentiment.train(text_data, text_labels, verbose=True)

# Predict sentiment for new sentences
new_sentences = ["This was fantastic!", "I really hated it."]
sentiment_predictions = model_sentiment.predict(new_sentences)
print("\nSentiment Analysis Predictions (probabilities for [positive, negative]):\n", sentiment_predictions)

# Get class labels
predicted_classes = np.argmax(sentiment_predictions, axis=1)
print("\nPredicted Sentiment Classes (0: positive, 1: negative):\n", predicted_classes)
```

### 3. Unleashing the Text Generator!

```python
import numpy as np
from your_module import NeuralNetwork

# Initialize a text generation model (ensure vocab_size and embed_dim are set appropriately)
vocab_size = 10000
embed_dim = 768
tokenizer_name = 'gpt2' # Let's try a generative model tokenizer (install transformers)
layer_sizes_generation = [embed_dim, 128, vocab_size] # Embedding -> hidden -> output (predict next token)
model_generator = NeuralNetwork(layer_sizes=layer_sizes_generation, learning_rate=0.0001, epochs=3, batch_size=4,
                                use_embedding=True, vocab_size=vocab_size, embed_dim=embed_dim, tokenizer_name=tokenizer_name,
                                activation_functions=['relu', 'softmax'])

# Enable text generation
model_generator.enable_text_generation()

# Try generating some text (note: you'd ideally train this on a large corpus for meaningful output)
prompt = "The cat sat on the"
print(f"\nGenerating text from prompt: 'The cat sat on the'")
generated_text = model_generator.generate(prompt, max_length=20, temperature=0.9)
print(f"Generated text: {generated_text}")

prompt_2 = "Artificial intelligence will"
print(f"\nGenerating text from prompt: 'Artificial intelligence will'")
generated_text_2 = model_generator.generate(prompt_2, max_length=15, temperature=1.0)
print(f"Generated text: {generated_text_2}")
```

These examples should give you a taste of what you can do with this simple neural network implementation. Feel free to experiment with different architectures, hyperparameters, and datasets! Happy learning!

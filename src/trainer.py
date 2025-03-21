import numpy as np
from neural_network2 import NeuralNetwork

# Training data for XOR gate
training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([[0], [1], [1], [0]])

# Define layer sizes and activation functions
layer_sizes = [2, 4, 1]  # Input layer: 2 neurons, Hidden layer: 4 neurons, Output layer: 1 neuron
activation_functions = ['leaky_relu', 'sigmoid']  # Use Leaky ReLU in the hidden layer

# Initialize and train the Neural Network
# Initialize the network
nn = NeuralNetwork(
    layer_sizes,
    activation_functions,
    learning_rate=0.2,  # Higher learning rate
    epochs=5000
)

# Train the Neural Network
nn.train(training_data, labels)

# Test the Neural Network
print("Neural Network Predictions After Training:")
for inputs in training_data:
    print(f"Input: {inputs} -> Output: {nn.predict(inputs)}")
    
# Save the weights of the Neural Network
nn.save("xor_weights.pkl")

# Load the weights of the Neural Network
model = NeuralNetwork.load("xor_weights.pkl")  

print("\nNeural Network Predictions After Loading Weights:")
for inputs in training_data:
    print(f"Input: {inputs} -> Output: {model.predict(inputs)}")

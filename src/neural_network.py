import numpy as np
import pickle

class NeuralNetwork:
    def __init__(self, layer_sizes, activation_functions=None, learning_rate=0.01, epochs=100):
        """
        Initialize the Neural Network.
        :param layer_sizes: List of integers specifying the number of neurons in each layer (input, hidden, output).
        :param activation_functions: List of activation function names for each layer (e.g., 'sigmoid', 'relu').
        :param learning_rate: Learning rate for weight updates.
        :param epochs: Number of training iterations.
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1  # Number of weight matrices = number of layers - 1
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Default activation functions if not provided
        if activation_functions is None:
            activation_functions = ['sigmoid'] * self.num_layers
        elif len(activation_functions) != self.num_layers:
            raise ValueError("Number of activation functions must match the number of layers.")
        self.activation_functions = activation_functions

        # Initialize weights and biases
        self.weights = []
        self.biases = []
        for i in range(self.num_layers):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i + 1]
            # He initialization for ReLU
            if self.activation_functions[i] == 'relu':
                self.weights.append(np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size))
            else:
                self.weights.append(np.random.randn(input_size, output_size) * 0.01)  # Default for other activations
            self.biases.append(np.random.randn(output_size))

    def activation(self, x, func_name):
        """Apply the specified activation function."""
        if func_name == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif func_name == 'relu':
            return np.maximum(0, x)
        elif func_name == 'softmax':
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Prevent overflow
            return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        elif func_name == 'leaky_relu':
            return np.maximum(0.1 * x, x)  # Slope of 0.1 for x < 0
        else:
            raise ValueError(f"Unsupported activation function: {func_name}")

    def activation_derivative(self, x, func_name):
        """Compute the derivative of the specified activation function."""
        if func_name == 'sigmoid':
            sigmoid = 1 / (1 + np.exp(-x))
            return sigmoid * (1 - sigmoid)
        elif func_name == 'relu':
            return (x > 0).astype(float)
        elif func_name == 'softmax':
            raise NotImplementedError("Softmax derivative is handled during backpropagation.")
        elif func_name == 'leaky_relu':
            return np.where(x > 0, 1.0, 0.1)
        else:
            raise ValueError(f"Unsupported activation function: {func_name}")

    def forward_propagation(self, inputs):
        """Perform forward propagation through all layers."""
        self.activations = [inputs]  # Store activations for each layer
        self.weighted_inputs = []   # Store weighted inputs (before activation)

        current_input = inputs
        for i in range(self.num_layers):
            weighted_input = np.dot(current_input, self.weights[i]) + self.biases[i]
            self.weighted_inputs.append(weighted_input)

            # Apply activation function
            activation_func = self.activation_functions[i]
            if i == self.num_layers - 1 and activation_func == 'softmax':  # Special case for softmax
                activation_output = self.activation(weighted_input, 'softmax')
            else:
                activation_output = self.activation(weighted_input, activation_func)

            self.activations.append(activation_output)
            current_input = activation_output

        return self.activations[-1]  # Return final output

    def backward_propagation(self, inputs, targets, output):
        """Perform backpropagation to update weights and biases."""
        # Initialize gradients
        batch_size = inputs.shape[0]
        d_weights = [np.zeros_like(w) for w in self.weights]
        d_biases = [np.zeros_like(b) for b in self.biases]

        # Compute output error
        if self.activation_functions[-1] == 'softmax':
            # For softmax + cross-entropy loss, the gradient simplifies to (output - target)
            output_error = output - targets
        else:
            output_error = (output - targets) * self.activation_derivative(self.weighted_inputs[-1], self.activation_functions[-1])

        # Backpropagate errors
        deltas = [output_error]
        for i in range(self.num_layers - 1, 0, -1):
            delta = np.dot(deltas[-1], self.weights[i].T) * self.activation_derivative(self.weighted_inputs[i - 1], self.activation_functions[i - 1])
            deltas.append(delta)

        deltas.reverse()  # Reverse to match layer order

        # Update weights and biases
        for i in range(self.num_layers):
            d_weights[i] = np.dot(self.activations[i].T, deltas[i]) / batch_size
            d_biases[i] = np.mean(deltas[i], axis=0)

            self.weights[i] -= self.learning_rate * d_weights[i]
            self.biases[i] -= self.learning_rate * d_biases[i]

    def train(self, training_data, labels):
        """
        Train the Neural Network using the training data.
        :param training_data: List of input feature vectors.
        :param labels: Corresponding target labels.
        """
        for epoch in range(self.epochs):
            for inputs, label in zip(training_data, labels):
                print(f"Epoch {epoch + 1}/{self.epochs} - Training on: {inputs} -> {label}")

                # Convert inputs and labels to numpy arrays
                inputs = np.array(inputs, ndmin=2)  # Ensure inputs is a 2D array
                label = np.array(label, ndmin=2)   # Ensure label is a 2D array

                # Forward propagation
                output = self.forward_propagation(inputs)

                # Backward propagation
                self.backward_propagation(inputs, label, output)

    def predict(self, inputs):
        """Predict the output for given inputs."""
        inputs = np.array(inputs, ndmin=2)  # Ensure inputs is a 2D array
        return self.forward_propagation(inputs)

    def save_weights(self, filepath):
        """
        Save the weights, biases, and architecture of the neural network.
        :param filepath: Path to the file where the model will be saved.
        """
        data = {
            'weights': self.weights,
            'biases': self.biases,
            'layer_sizes': self.layer_sizes,
            'activation_functions': self.activation_functions
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath):
        """
        Load a saved model from a file.
        :param filepath: Path to the saved model file.
        :return: A NeuralNetwork instance with loaded weights and architecture.
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Extract architecture and hyperparameters
        layer_sizes = data['layer_sizes']
        activation_functions = data['activation_functions']
        
        # Initialize the model with the saved architecture
        model = cls(
            layer_sizes=layer_sizes,
            activation_functions=activation_functions
        )
        
        # Load weights and biases
        model.weights = data['weights']
        model.biases = data['biases']
        
        print(f"Model loaded from {filepath}")
        return model
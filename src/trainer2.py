import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from neural_network2 import NeuralNetwork

# Generate synthetic data
n_samples = 1000  # Number of samples
noise = 0.1  # Noise level (higher = more overlap between classes)

# Generate features (X) and labels (y)
X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)

# Reshape labels to match the neural network's expected output format (2D array)
y = y.reshape(-1, 1)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Plot the data
plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train.ravel(), cmap='viridis', edgecolors='k')
plt.title("Training Data (Nonlinearly Separable)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Define layer sizes and activation functions
layer_sizes = [2, 10, 10, 1]  # Input layer (2 features), two hidden layers (10 neurons each), output layer
activation_functions = ['relu', 'relu', 'sigmoid']  # ReLU for hidden layers, Sigmoid for binary classification

# Initialize and train the Neural Network
nn = NeuralNetwork(
    layer_sizes=layer_sizes,
    activation_functions=activation_functions,
    learning_rate=0.1,
    epochs=5000
)

# Train on the larger dataset
nn.train(X_train, y_train)

# Test the Neural Network
def plot_decision_boundary(X, y, model):
    # Create a grid of points to evaluate the model
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Predict for each point in the grid
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='viridis', edgecolors='k')
    plt.title("Decision Boundary")
    plt.show()

# Plot the decision boundary on the test set
plot_decision_boundary(X_test, y_test, nn)

# save the weights of the Neural Network
nn.save("classifier_weights.pkl")

#load the weights of the Neural Network
model = NeuralNetwork.load("classifier_weights.pkl")

print("\nNeural Network Predictions After Loading Weights:")
print(model.predict(X_test))


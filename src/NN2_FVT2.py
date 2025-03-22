import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from neural_network2 import NeuralNetwork


def test_neural_network():
    # Test Convolutional Layer
    conv_nn = NeuralNetwork(
        layer_sizes=[(1, 28, 28), 16, 10],  # Input: 1x28x28, Conv output: 16x26x26, Dense: 10
        layer_types=['conv', 'dense'],
        conv_params=[{'filters': 16, 'kernel_size': 3, 'stride': 1}, None],
        activation_functions=['relu', 'softmax'],
        learning_rate=0.01
    )
    x_conv = np.random.randn(2, 1, 28, 28)  # Example input
    y_conv = np.zeros((2, 10))  # Fix targets
    y_conv[0, 0] = 1  # Class 0
    y_conv[1, 1] = 1  # Class 1
    output = conv_nn.forward_propagation(x_conv)
    conv_nn.backward_propagation(x_conv, y_conv, output)
    print("Convolutional layer test passed")

    # Test Recurrent Layer
    rnn_nn = NeuralNetwork(
        layer_sizes=[5, 10, 2],  # Input: 5 features, Hidden: 10, Output: 2
        layer_types=['recurrent', 'dense'],
        recurrent_params=[{'hidden_size': 10}, None],
        activation_functions=['tanh', 'softmax'],
        learning_rate=0.01
    )
    x_rnn = np.random.randn(2, 3, 5)  # Batch of 2 sequences, length 3, 5 features
    y_rnn = np.array([[1, 0], [0, 1]])
    output = rnn_nn.forward_propagation(x_rnn)
    assert output.shape == (2, 2), "RNN output shape mismatch"
    rnn_nn.backward_propagation(x_rnn, y_rnn, output)
    print("Recurrent layer test passed")

    # Test Multi-Head Attention
    attn_nn = NeuralNetwork(
        layer_sizes=[10],  # Dummy for embedding
        use_embedding=True,
        vocab_size=100,
        embed_dim=64,
        max_seq_length=10,
        activation_functions=['softmax'],
        layer_types=['dense']
    )
    x_attn = np.random.randint(0, 100, (2, 10))  # Batch of 2 sequences
    output = attn_nn.forward_propagation(x_attn)
    assert output.shape == (2, 10), "Attention output shape mismatch"
    y_attn = np.random.randn(2, 10)
    attn_nn.backward_propagation(x_attn, y_attn, output)
    print("Multi-head attention test passed")

if __name__ == "__main__":
    test_neural_network()
# Neural Network Function Verification Testing

## Overview
This repository contains function verification tests for the `NeuralNetwork` class, ensuring that its core functionalities work as expected. The test suite is implemented using Python's `unittest` framework and covers initialization, embedding layers, tokenization, forward propagation, dropout, attention mechanisms, training, reinforcement learning, and model serialization.

## File Structure
- `NN2_FVT.py`: Contains the test cases for verifying various components of the `NeuralNetwork` class.
- `neural_network2.py`: Assumed to be the implementation file for the `NeuralNetwork` class.

## Test Cases
The following functionalities are tested:

### 1. Initialization
- Verifies that the `NeuralNetwork` object is correctly initialized with appropriate layer sizes, activation functions, and biases.

### 2. Embedding Initialization
- Ensures that when embedding is used, the model initializes the embedding matrix correctly.

### 3. Tokenization
- Tests the tokenization function to verify that input texts are tokenized into valid sequences.

### 4. Forward Propagation
- Checks that the forward pass through the network works as expected.

### 5. Dropout Functionality
- Ensures that dropout is applied during training but not during evaluation.

### 6. Attention Mechanism
- Validates the self-attention implementation with masked sequences.

### 7. Training Loop
- Tests the model training process using sample text inputs and expected labels.

### 8. Reinforcement Learning (RL) Setup
- Evaluates the RL setup, ensuring that states, actions, and advantages are computed correctly.

### 9. Save & Load Model
- Ensures that the model can be saved and loaded while preserving its configuration.

## Running the Tests
To execute the tests, run the following command:

```bash
python -m unittest NN2_FVT.py
```

Ensure that all dependencies are installed before running the tests.

## Dependencies
Install required dependencies using:

```bash
pip install numpy transformers
```

## Notes
- The `NeuralNetwork` class is assumed to be implemented in `neural_network2.py`.
- Update paths accordingly if the implementation file is named differently.

## License
This project is open-source and can be modified or extended as needed.


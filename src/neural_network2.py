import numpy as np
import pickle
import os
from transformers import AutoTokenizer
import logging
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)

def softmax(x, axis=-1):
    x = np.clip(x, -500, 500)  # Prevent overflow
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01, epochs=100, batch_size=32, dropout_rate=0.0,
                activation_functions=None, use_embedding=False, vocab_size=None, embed_dim=None,
                max_seq_length=None, tokenizer_name=None, eos_token=None, pad_token=None):
        """Initialize the neural network
        
        Args:
            layer_sizes: List of layer sizes, including input and output
            learning_rate: Learning rate for weight updates
            epochs: Number of training epochs
            batch_size: Number of samples per batch
            dropout_rate: Dropout rate for regularization
            activation_functions: List of activation functions for each layer
            use_embedding: Whether to use word embeddings
            vocab_size: Size of vocabulary (if using embeddings)
            embed_dim: Embedding dimension (if using embeddings)
            max_seq_length: Maximum sequence length (if using embeddings)
            tokenizer_name: Name of pre-trained tokenizer (e.g., 'bert-base-uncased')
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.training = True
        self.training_losses = []
        self.validation_metrics = []
        self.label_encoder = None
        self.is_generative = False
        
        # Embedding settings
        self.use_embedding = use_embedding
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_length = max_seq_length
        
        # Initialize tokenizer if name provided
        if tokenizer_name:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                logging.info(f"Loaded tokenizer: {tokenizer_name}")
                
                # Set custom EOS token if provided
                if eos_token is not None:
                    # Add as special token if it's not already in the vocabulary
                    if eos_token not in self.tokenizer.get_vocab():
                        special_tokens_dict = {'eos_token': eos_token}
                        self.tokenizer.add_special_tokens(special_tokens_dict)
                        logging.info(f"Added custom EOS token: {eos_token}")
                    else:
                        self.tokenizer.eos_token = eos_token
                        logging.info(f"Set EOS token to existing token: {eos_token}")
                
                # Set pad token to eos token if not already set
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                # Set custom pad token if provided
                if pad_token is not None:
                    # Add as special token if it's not already in the vocabulary
                    if pad_token not in self.tokenizer.get_vocab():
                        special_tokens_dict = {'pad_token': pad_token}
                        self.tokenizer.add_special_tokens(special_tokens_dict)
                        logging.info(f"Added custom PAD token: {pad_token}")
                    else:
                        self.tokenizer.pad_token = pad_token
                        logging.info(f"Set PAD token to existing token: {pad_token}")
                    
            except Exception as e:
                logging.error(f"Failed to load tokenizer: {e}")
                self.tokenizer = None
        else:
            self.tokenizer = None
        
        # Set default activation functions if not provided
        if activation_functions is None:
            # Use relu for hidden layers and softmax for output layer
            activation_functions = ['relu'] * (len(layer_sizes) - 2) + ['softmax']
        
        # Ensure there's one activation function per layer transition
        assert len(activation_functions) == len(layer_sizes) - 1, \
            f"Expected {len(layer_sizes) - 1} activation functions, got {len(activation_functions)}"
        
        self.activation_functions = activation_functions
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # Determine if we have a language model architecture
        is_lm_architecture = use_embedding and vocab_size is not None and layer_sizes[0] == vocab_size
        
        for i in range(len(layer_sizes) - 1):
            # Handle special case for embedding layer in language models
            if i == 0 and use_embedding:
                if is_lm_architecture:
                    # For language model where first layer size is vocab_size
                    # Initialize weights for embedding layer
                    self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01)
                else:
                    # For classification model where first layer size is embed_dim
                    # We'll still create weights but they might not be used directly
                    self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01)
            else:
                # Standard weight initialization
                self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01)
            
            # Initialize biases to zeros
            self.biases.append(np.zeros(layer_sizes[i+1]))
        
        # Storage for layer outputs during forward pass
        self.layer_outputs = [None] * (len(layer_sizes) - 1)
    
    def _initialize_parameters(self):
        for i in range(self.num_layers):
            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i+1]
            
            # Xavier/Glorot initialization for better convergence
            scale = np.sqrt(2.0 / (input_size + output_size))
            if self.activation_functions[i] == 'relu':
                # He initialization for ReLU
                scale = np.sqrt(2.0 / input_size)
                
            self.weights.append(np.random.randn(input_size, output_size) * scale)
            self.biases.append(np.zeros(output_size))


    def _apply_activation(self, activation_name, x):
        """Apply activation function to input"""
        if activation_name == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif activation_name == 'relu':
            return np.maximum(0, x)
        elif activation_name == 'leaky_relu':
            return np.where(x > 0, x, 0.01 * x)  # Alpha = 0.01
        elif activation_name == 'tanh':
            return np.tanh(x)
        elif activation_name == 'softmax':
            exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        else:
            # Linear activation (no transformation)
            return x
    
    
    def _activation_derivative(self, activation_name, activated_x):
        """Calculate derivative of activation function"""
        if activation_name == 'sigmoid':
            return activated_x * (1 - activated_x)
        elif activation_name == 'relu':
            return np.where(activated_x > 0, 1, 0)
        elif activation_name == 'leaky_relu':
            return np.where(activated_x > 0, 1, 0.01)  # Alpha = 0.01
        elif activation_name == 'tanh':
            return 1 - activated_x ** 2
        elif activation_name == 'softmax':
            # For softmax with cross-entropy loss, this is handled separately
            return np.ones_like(activated_x)
        else:
            # Linear activation derivative is 1
            return np.ones_like(activated_x)


    def forward_propagation(self, inputs, attention_mask=None):
        """
        Perform forward propagation with support for different input types
        """
        # Convert inputs to numpy array if not already
        if not isinstance(inputs, np.ndarray):
            inputs = np.array(inputs)
        
        # Check if we're dealing with simple numerical inputs (XOR case)
        if not self.use_embedding and inputs.dtype.kind in ('i', 'f'):
            # Reset layer outputs
            self.layer_outputs = [None] * (len(self.layer_sizes) - 1)
            
            # First layer
            layer_output = np.dot(inputs, self.weights[0]) + self.biases[0]
            layer_output = self._apply_activation(self.activation_functions[0], layer_output)
            self.layer_outputs[0] = layer_output
            
            # Process through remaining layers
            for l in range(1, len(self.layer_sizes) - 1):
                layer_output = np.dot(self.layer_outputs[l-1], self.weights[l]) + self.biases[l]
                layer_output = self._apply_activation(self.activation_functions[l], layer_output)
                self.layer_outputs[l] = layer_output
            
            return self.layer_outputs[-1]        
        
        
        
        batch_size = inputs.shape[0]
        
        # Detect model type and input type
        is_sequence_data = len(inputs.shape) > 1 and self.use_embedding
        is_language_model = self.use_embedding and self.vocab_size is not None and self.layer_sizes[0] == self.vocab_size
        is_sentiment_model = self.use_embedding and self.layer_sizes[0] != self.vocab_size
        
        # For sentiment analysis model (layers don't start with vocab_size)
        if is_sentiment_model and is_sequence_data:
            # Reset layer outputs
            self.layer_outputs = [None] * (len(self.layer_sizes) - 1)
            
            # Create an embedding representation by average pooling the token embeddings
            pooled_representation = np.zeros((batch_size, self.layer_sizes[0]))
            token_counts = np.zeros(batch_size)
            
            # Process each token position
            for pos in range(inputs.shape[1]):
                for i in range(batch_size):
                    # Skip padded tokens
                    if attention_mask is None or attention_mask[i, pos] > 0:
                        token_id = inputs[i, pos]
                        if token_id < self.vocab_size:
                            # Use a simple deterministic embedding based on token ID
                            np.random.seed(int(token_id) % 10000)  # Make it deterministic
                            token_embedding = np.random.randn(self.layer_sizes[0]) * 0.1
                            pooled_representation[i] += token_embedding
                            token_counts[i] += 1
            
            # Average the token embeddings
            for i in range(batch_size):
                if token_counts[i] > 0:
                    pooled_representation[i] /= token_counts[i]
            
            # Use this as the first layer output
            self.layer_outputs[0] = self._apply_activation(
                self.activation_functions[0], pooled_representation
            )
            
            # Forward through the rest of the network normally
            for l in range(1, len(self.layer_sizes) - 1):
                layer_output = np.dot(self.layer_outputs[l-1], self.weights[l]) + self.biases[l]
                layer_output = self._apply_activation(self.activation_functions[l], layer_output)
                self.layer_outputs[l] = layer_output
            
            return self.layer_outputs[-1]
        
        # For language model with sequence data
        elif is_language_model and is_sequence_data:
            seq_length = inputs.shape[1]
            
            # Initialize storage for layer outputs
            self.layer_outputs = [None] * (len(self.layer_sizes) - 1)
            
            # Create output tensor of shape (batch_size, seq_length, vocab_size)
            outputs = np.zeros((batch_size, seq_length, self.layer_sizes[-1]))
            
            for pos in range(seq_length):
                # Skip padded positions
                if attention_mask is not None and np.sum(attention_mask[:, pos]) == 0:
                    continue
                    
                # Get token IDs for this position
                pos_inputs = inputs[:, pos]
                
                # Create one-hot encodings
                one_hot = np.zeros((batch_size, self.vocab_size))
                for i in range(batch_size):
                    if attention_mask is None or attention_mask[i, pos] > 0:
                        if pos_inputs[i] < self.vocab_size:
                            one_hot[i, int(pos_inputs[i])] = 1
                
                # Forward through the network
                # First layer (embedding)
                layer_output = np.dot(one_hot, self.weights[0]) + self.biases[0]
                layer_output = self._apply_activation(self.activation_functions[0], layer_output)
                
                # Process through remaining layers
                for l in range(1, len(self.layer_sizes) - 1):
                    layer_output = np.dot(layer_output, self.weights[l]) + self.biases[l]
                    layer_output = self._apply_activation(self.activation_functions[l], layer_output)
                
                # Store output for this position
                outputs[:, pos, :] = layer_output
            
            return outputs
        
        # For non-sequence data (standard classification)
        else:
            # Reset layer outputs
            self.layer_outputs = [None] * (len(self.layer_sizes) - 1)
            
            # First layer
            layer_output = np.dot(inputs, self.weights[0]) + self.biases[0]
            layer_output = self._apply_activation(self.activation_functions[0], layer_output)
            self.layer_outputs[0] = layer_output
            
            # Process through remaining layers
            for l in range(1, len(self.layer_sizes) - 1):
                layer_output = np.dot(self.layer_outputs[l-1], self.weights[l]) + self.biases[l]
                layer_output = self._apply_activation(self.activation_functions[l], layer_output)
                self.layer_outputs[l] = layer_output
            
            return self.layer_outputs[-1]
            
    def _process_single_position(self, pos_inputs, attention_mask, pos):
        """Process a single position in a sequence through the network"""
        batch_size = pos_inputs.shape[0]
        
        # Create one-hot encodings for token IDs
        one_hot = np.zeros((batch_size, self.vocab_size))
        for i in range(batch_size):
            if attention_mask is None or attention_mask[i, pos] > 0:
                if pos_inputs[i] < self.vocab_size:
                    one_hot[i, pos_inputs[i]] = 1
        
        # First layer (embedding)
        layer_output = np.dot(one_hot, self.weights[0]) + self.biases[0]
        layer_output = self._apply_activation(self.activation_functions[0], layer_output)
        
        # Process through remaining layers
        for l in range(1, len(self.layer_sizes) - 1):
            layer_output = np.dot(layer_output, self.weights[l]) + self.biases[l]
            layer_output = self._apply_activation(self.activation_functions[l], layer_output)
        
        return layer_output


    # def forward_propagation(self, inputs, masks=None):
    #     """Forward pass through the network with simplified embedding handling"""
    #     self.activations = []
    #     self.weighted_inputs = []
    #     self.dropout_masks = []
        
    #     # Handle embeddings for text data
    #     if self.use_embedding and isinstance(inputs, np.ndarray) and np.issubdtype(inputs.dtype, np.integer):
    #         # Ensure inputs are integers and within vocab range
    #         inputs = np.clip(inputs, 0, self.vocab_size-1)
            
    #         # Apply embeddings
    #         embedded = self.embedding[inputs]  # Shape: (batch_size, seq_len, embed_dim)
            
    #         # Simple pooling to get fixed-size representation (use mean pooling)
    #         if masks is not None:
    #             # Create expanded mask for proper broadcasting
    #             expanded_masks = np.expand_dims(masks, -1)  # Shape: (batch_size, seq_len, 1)
                
    #             # Apply mask to zero out padding tokens
    #             masked_embedded = embedded * expanded_masks
                
    #             # Sum and normalize by sequence length
    #             seq_lengths = np.sum(masks, axis=1, keepdims=True)
    #             seq_lengths = np.clip(seq_lengths, 1, None)  # Avoid division by zero
    #             current_input = np.sum(masked_embedded, axis=1) / seq_lengths
    #         else:
    #             # Simple mean pooling if no masks
    #             current_input = np.mean(embedded, axis=1)
    #     else:
    #         # For non-text data, use inputs directly
    #         current_input = inputs
            
    #     self.activations.append(current_input)
        
    #     # Forward pass through dense layers
    #     for i in range(self.num_layers):
    #         weighted_input = np.dot(current_input, self.weights[i]) + self.biases[i]
    #         self.weighted_inputs.append(weighted_input)
            
    #         # Apply activation function
    #         if self.activation_functions[i] == 'softmax':
    #             activation_output = softmax(weighted_input)
    #         else:
    #             activation_output = self.activation(weighted_input, self.activation_functions[i])
                
    #         # Apply dropout except for the output layer
    #         if self.training and self.dropout_rate > 0 and i < self.num_layers - 1:
    #             mask = np.random.binomial(1, 1 - self.dropout_rate, size=activation_output.shape)
    #             activation_output *= mask / (1 - self.dropout_rate)
    #             self.dropout_masks.append(mask)
    #         else:
    #             self.dropout_masks.append(None)
                
    #         self.activations.append(activation_output)
    #         current_input = activation_output
            
    #     return current_input


    def backward_propagation(self, inputs, targets, output, attention_mask=None):
        """
        Backward propagation compatible with both text generation and sentiment analysis models
        
        Args:
            inputs: Input data
            targets: Target data
            output: Model output from forward pass
            attention_mask: Optional mask for padded sequences
        """
        batch_size = inputs.shape[0]
        
        # Detect model type and input type
        is_sequence_data = len(targets.shape) == 3
        is_language_model = self.use_embedding and self.vocab_size is not None and self.layer_sizes[0] == self.vocab_size
        is_sentiment_model = self.use_embedding and not is_sequence_data and len(inputs.shape) > 1
        
        # For sentiment analysis (tokenized input, classification output)
        if is_sentiment_model:
            # Calculate output error
            if self.activation_functions[-1] == 'softmax':
                output_error = output - targets
            else:
                output_error = (output - targets) * self._activation_derivative(
                    self.activation_functions[-1], output
                )
            
            # Initialize gradients
            dweights = [None] * len(self.weights)
            dbiases = [None] * len(self.biases)
            
            # Backpropagate through layers
            layer_error = output_error
            for l in range(len(self.layer_sizes) - 2, 0, -1):  # Skip the first layer (embedding)
                # Get inputs to this layer (stored during forward pass)
                layer_input = self.layer_outputs[l-1]
                
                # Update gradients
                dweights[l] = np.dot(layer_input.T, layer_error)
                dbiases[l] = np.sum(layer_error, axis=0)
                
                # Propagate error to previous layer
                if l > 0:
                    layer_error = np.dot(layer_error, self.weights[l].T)
                    layer_error *= self._activation_derivative(
                        self.activation_functions[l-1], layer_input
                    )
            
            # We don't update the first layer for sentiment analysis
            # since we're using simulated embeddings
            
            # Update weights and biases (skip the first layer)
            for l in range(1, len(self.weights)):
                if dweights[l] is not None:
                    self.weights[l] -= self.learning_rate * dweights[l]
                    self.biases[l] -= self.learning_rate * dbiases[l]
        
        # For language model with sequence data
        elif is_language_model and is_sequence_data:
            seq_length = inputs.shape[1]
            
            # Initialize gradients
            dweights = [np.zeros_like(w) for w in self.weights]
            dbiases = [np.zeros_like(b) for b in self.biases]
            
            # Process each position in the sequence
            for pos in range(seq_length):
                # Skip padded positions
                if attention_mask is not None and np.sum(attention_mask[:, pos]) == 0:
                    continue
                    
                # Get token IDs for this position
                pos_inputs = inputs[:, pos]
                
                # Get outputs and targets for this position
                pos_output = output[:, pos, :]
                pos_targets = targets[:, pos, :]
                
                # Calculate error at output
                if self.activation_functions[-1] == 'softmax':
                    pos_error = pos_output - pos_targets
                else:
                    pos_error = (pos_output - pos_targets) * self._activation_derivative(
                        self.activation_functions[-1], pos_output
                    )
                
                # Create one-hot encoding for this position's inputs
                one_hot = np.zeros((batch_size, self.vocab_size))
                for i in range(batch_size):
                    if attention_mask is None or attention_mask[i, pos] > 0:
                        if pos_inputs[i] < self.vocab_size:
                            one_hot[i, int(pos_inputs[i])] = 1
                
                # Reconstruct the activations from scratch
                # First layer: vocab_size -> embed_dim
                first_layer_out = np.dot(one_hot, self.weights[0]) + self.biases[0]
                first_layer_out = self._apply_activation(self.activation_functions[0], first_layer_out)
                
                # Second layer: embed_dim -> hidden_dim
                second_layer_out = np.dot(first_layer_out, self.weights[1]) + self.biases[1]
                second_layer_out = self._apply_activation(self.activation_functions[1], second_layer_out)
                
                # Gradient for the last layer (output -> hidden)
                dweights[2] += np.dot(second_layer_out.T, pos_error)
                dbiases[2] += np.sum(pos_error, axis=0)
                
                # Backpropagate error to second layer (hidden -> embedding)
                hidden_error = np.dot(pos_error, self.weights[2].T)
                hidden_error *= self._activation_derivative(
                    self.activation_functions[1], second_layer_out
                )
                
                # Gradient for the second layer (hidden -> embedding)
                dweights[1] += np.dot(first_layer_out.T, hidden_error)
                dbiases[1] += np.sum(hidden_error, axis=0)
                
                # Backpropagate error to first layer (embedding -> one-hot)
                embed_error = np.dot(hidden_error, self.weights[1].T)
                embed_error *= self._activation_derivative(
                    self.activation_functions[0], first_layer_out
                )
                
                # Gradient for the first layer (embedding)
                dweights[0] += np.dot(one_hot.T, embed_error)
                dbiases[0] += np.sum(embed_error, axis=0)
            
            # Update weights and biases with average gradients
            for l in range(len(self.weights)):
                self.weights[l] -= self.learning_rate * dweights[l] / (seq_length * batch_size)
                self.biases[l] -= self.learning_rate * dbiases[l] / (seq_length * batch_size)
        
        # For standard classification (non-sequence, non-embedding)
        else:
            # Calculate error at output
            if self.activation_functions[-1] == 'softmax':
                output_error = output - targets
            else:
                output_error = (output - targets) * self._activation_derivative(
                    self.activation_functions[-1], output
                )
            
            # Initialize gradients
            dweights = [None] * len(self.weights)
            dbiases = [None] * len(self.biases)
            
            # Backpropagate through layers
            layer_error = output_error
            for l in range(len(self.layer_sizes) - 2, -1, -1):
                # Get inputs to this layer
                if l == 0:
                    layer_input = inputs
                else:
                    layer_input = self.layer_outputs[l-1]
                
                # Update gradients
                dweights[l] = np.dot(layer_input.T, layer_error)
                dbiases[l] = np.sum(layer_error, axis=0)
                
                # Propagate error to previous layer
                if l > 0:
                    layer_error = np.dot(layer_error, self.weights[l].T)
                    layer_error *= self._activation_derivative(
                        self.activation_functions[l-1], layer_input
                    )
            
            # Update weights and biases
            for l in range(len(self.weights)):
                if dweights[l] is not None:
                    self.weights[l] -= self.learning_rate * dweights[l]
                    self.biases[l] -= self.learning_rate * dbiases[l]
                
                
    # def backward_propagation(self, inputs, targets, output, masks=None):
    #     """Backward pass to update weights and biases with fixed broadcasting issue"""
    #     batch_size = len(inputs) if isinstance(inputs, list) else inputs.shape[0]
    #     d_weights = [np.zeros_like(w) for w in self.weights]
    #     d_biases = [np.zeros_like(b) for b in self.biases]
    #     deltas = []

    #     # Calculate error based on output activation
    #     if self.activation_functions[-1] == 'softmax':
    #         # Cross-entropy derivative is (output - target) for softmax
    #         output_error = output - targets
    #     else:
    #         # For other activation functions, use derivative
    #         output_error = (output - targets) * self.activation_derivative(
    #             self.weighted_inputs[-1], self.activation_functions[-1])
            
    #     deltas.append(output_error)
        
    #     # Backpropagate through dense layers
    #     for i in reversed(range(self.num_layers)):
    #         delta = deltas[-1]
            
    #         # Apply dropout mask if applicable
    #         if i < len(self.dropout_masks) and self.dropout_masks[i] is not None:
    #             delta *= self.dropout_masks[i] / (1 - self.dropout_rate)
                
    #         # Calculate weight and bias gradients
    #         d_weights[i] = np.dot(self.activations[i].T, delta) / batch_size
    #         d_biases[i] = np.mean(delta, axis=0)
            
    #         # Calculate error for previous layer if not at input layer
    #         if i > 0:
    #             delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(
    #                 self.weighted_inputs[i-1], self.activation_functions[i-1])
    #             deltas.append(delta)

    #     # Update embedding layer if used - FIXED: Create appropriate embedding error
    #     if self.use_embedding and isinstance(inputs, np.ndarray) and np.issubdtype(inputs.dtype, np.integer):
    #         # Create a placeholder for embedding gradients
    #         delta_embedding = np.zeros_like(self.embedding)  # Shape: (vocab_size, embed_dim)
            
    #         # We need to adapt to the mismatch here. The last delta is for the first hidden layer,
    #         # but we need to transform it to match the embedding dimension
    #         # Since we're using the model with layer_sizes=[768, 128, 3], the delta is (batch_size, 128)
            
    #         # We need to transform it back to the embedding dimension (768)
    #         if len(deltas) > 0 and deltas[-1].shape[1] != self.embed_dim:
    #             # Get the error at the first hidden layer (which has shape batch_size x hidden_size)
    #             hidden_delta = deltas[-1]  # Shape: (batch_size, hidden_size)
                
    #             # We need to project this error back to the embedding space
    #             # This is a simple approach - using the first layer weights to project
    #             # If the hidden size is smaller than embed_dim, we need to map it back
    #             # For demonstration, we'll use a simple linear projection
    #             # logging.info(f"Projecting delta from shape {hidden_delta.shape} back to embedding dimension {self.embed_dim}")
                                
    #             # Only update embeddings used in this batch
    #             if masks is not None:
    #                 # Use masks to determine which tokens were actually used (not padding)
    #                 for b in range(batch_size):
    #                     # Create a projected delta for this example
    #                     # We'll use a simple method: repeat the hidden delta values to fill the embedding dimension
    #                     hidden_size = hidden_delta.shape[1]
    #                     repeat_factor = self.embed_dim // hidden_size + 1
    #                     example_delta = np.tile(hidden_delta[b], repeat_factor)[:self.embed_dim]
                        
    #                     # Count valid tokens (non-padding) in this example
    #                     valid_tokens = np.sum(masks[b])
    #                     if valid_tokens == 0:  # Skip if all tokens were padding
    #                         continue
                            
    #                     # Distribute error evenly across all valid token positions
    #                     token_delta = example_delta / valid_tokens  # Shape: (embed_dim,)
                        
    #                     # Update embeddings for each token in this example
    #                     for t in range(inputs.shape[1]):
    #                         if masks[b, t] == 0:  # Skip padding tokens
    #                             continue
                                
    #                         token_id = inputs[b, t]
    #                         if token_id < self.vocab_size:
    #                             # Add the distributed error to this token's embedding
    #                             delta_embedding[token_id] += token_delta
    #             else:
    #                 # No masks, assume all tokens are valid
    #                 for b in range(batch_size):
    #                     # Create a projected delta for this example
    #                     hidden_size = hidden_delta.shape[1]
    #                     repeat_factor = self.embed_dim // hidden_size + 1
    #                     example_delta = np.tile(hidden_delta[b], repeat_factor)[:self.embed_dim]
                        
    #                     # Distribute error evenly across all token positions
    #                     token_delta = example_delta / inputs.shape[1]  # Shape: (embed_dim,)
                        
    #                     # Update embeddings for each token in this example
    #                     for t in range(inputs.shape[1]):
    #                         token_id = inputs[b, t]
    #                         if token_id < self.vocab_size:
    #                             # Add the distributed error to this token's embedding
    #                             delta_embedding[token_id] += token_delta
            
    #         # Update embeddings with gradient clipping
    #         np.clip(delta_embedding, -1.0, 1.0, out=delta_embedding)
    #         self.embedding -= self.learning_rate * delta_embedding / batch_size

    #     # Update dense layer weights and biases with gradient clipping
    #     for i in range(self.num_layers):
    #         np.clip(d_weights[i], -1.0, 1.0, out=d_weights[i])
    #         np.clip(d_biases[i], -1.0, 1.0, out=d_biases[i])
            
    #         self.weights[i] -= self.learning_rate * d_weights[i]
    #         self.biases[i] -= self.learning_rate * d_biases[i]

    # Also modify train_qa to include early stopping and learning rate scheduling
    def train_qa(self, questions, answers, validation_questions=None, validation_answers=None, verbose=True):
        """Train the model for QA tasks with early stopping and learning rate scheduling"""
        # Reset training metrics
        self.training_losses = []
        self.validation_metrics = []
        
        # Number of examples and batches
        num_examples = len(questions)
        batch_size = min(self.batch_size, num_examples)
        num_batches = (num_examples + batch_size - 1) // batch_size
        
        if verbose:
            logging.info(f"Training QA model on {num_examples} examples for {self.epochs} epochs")
        
        # Setup early stopping
        best_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        # Initial learning rate
        initial_lr = self.learning_rate
        
        # Training loop
        for epoch in range(self.epochs):
            # Learning rate scheduling - reduce by 10% every 100 epochs
            if epoch > 0 and epoch % 100 == 0:
                self.learning_rate = initial_lr * (0.9 ** (epoch // 100))
                if verbose:
                    print(f"Reducing learning rate to {self.learning_rate:.6f}")
            
            # Shuffle indices
            indices = np.arange(num_examples)
            np.random.shuffle(indices)
            
            epoch_loss = 0.0
            
            for batch in range(num_batches):
                # Get batch indices
                start_idx = batch * batch_size
                end_idx = min(start_idx + batch_size, num_examples)
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_questions = [questions[i] for i in batch_indices]
                batch_answers = answers[batch_indices]
                
                # Tokenize questions
                tokenized = self.tokenize(batch_questions)
                input_ids = tokenized['input_ids']
                attention_mask = tokenized['attention_mask']
                
                # Forward pass
                batch_output = self._qa_forward(input_ids, attention_mask)
                
                # Calculate loss
                epsilon = 1e-15
                batch_output = np.clip(batch_output, epsilon, 1 - epsilon)
                batch_loss = -np.sum(batch_answers * np.log(batch_output)) / len(batch_indices)
                
                # Backward pass 
                self._qa_backward(input_ids, batch_answers, batch_output, attention_mask)
                
                # Accumulate loss
                epoch_loss += batch_loss
            
            # Average loss for the epoch
            avg_loss = epoch_loss / num_batches
            self.training_losses.append(avg_loss)
            
            # Log progress
            if verbose and (epoch % 10 == 0 or epoch == self.epochs - 1):
                print(f"Epoch {epoch+1}/{self.epochs}: Loss = {avg_loss:.6f}")
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Switch to evaluation mode
        self.training = False
        
        return self.training_losses


    # Modify the _qa_forward method to use a better weight initialization
    def _qa_forward(self, input_ids, attention_mask=None):
        """Forward pass for QA model with improved weight initialization"""
        # If QA weights don't exist yet, initialize them with better scaling
        if not hasattr(self, 'qa_weights') or self.qa_weights is None:
            self.qa_weights = []
            self.qa_biases = []
            
            for i in range(len(self.layer_sizes) - 1):
                # He initialization for ReLU networks
                scale = np.sqrt(2.0 / self.layer_sizes[i])
                self.qa_weights.append(np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * scale)
                self.qa_biases.append(np.zeros(self.layer_sizes[i+1]))
        
        # Rest of the forward method (unchanged)
        batch_size = input_ids.shape[0]
        
        # Create a pooled representation from token embeddings
        pooled_representation = np.zeros((batch_size, self.layer_sizes[0]))
        token_counts = np.zeros(batch_size)
        
        # Process each token position
        for pos in range(input_ids.shape[1]):
            for i in range(batch_size):
                # Skip padded tokens
                if attention_mask is None or attention_mask[i, pos] > 0:
                    token_id = input_ids[i, pos]
                    if token_id < self.vocab_size:
                        # Use a deterministic embedding based on token ID
                        np.random.seed(int(token_id) % 10000)
                        token_embedding = np.random.randn(self.layer_sizes[0]) * 0.1
                        pooled_representation[i] += token_embedding
                        token_counts[i] += 1
        
        # Average the token embeddings
        for i in range(batch_size):
            if token_counts[i] > 0:
                pooled_representation[i] /= token_counts[i]
        
        # Forward through the network
        layer_activations = [pooled_representation]
        
        # Process through each layer
        for l in range(len(self.qa_weights)):
            # Get previous activation
            prev_activation = layer_activations[-1]
            
            # Apply dropout during training
            if self.training and self.dropout_rate > 0:
                dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, 
                                                size=prev_activation.shape) / (1 - self.dropout_rate)
                prev_activation = prev_activation * dropout_mask
            
            # Compute next layer
            next_layer = np.dot(prev_activation, self.qa_weights[l]) + self.qa_biases[l]
            next_activation = self._apply_activation(self.activation_functions[l], next_layer)
            
            # Store activation
            layer_activations.append(next_activation)
        
        return layer_activations[-1]

    def _qa_backward(self, input_ids, targets, output, attention_mask=None):
        """Backward pass for QA model"""
        batch_size = input_ids.shape[0]
        
        # Create a pooled representation again for gradient calculation
        pooled_representation = np.zeros((batch_size, self.layer_sizes[0]))
        token_counts = np.zeros(batch_size)
        
        for pos in range(input_ids.shape[1]):
            for i in range(batch_size):
                if attention_mask is None or attention_mask[i, pos] > 0:
                    token_id = input_ids[i, pos]
                    if token_id < self.vocab_size:
                        np.random.seed(int(token_id) % 10000)
                        token_embedding = np.random.randn(self.layer_sizes[0]) * 0.1
                        pooled_representation[i] += token_embedding
                        token_counts[i] += 1
        
        for i in range(batch_size):
            if token_counts[i] > 0:
                pooled_representation[i] /= token_counts[i]
        
        # Forward pass to get all activations
        layer_activations = [pooled_representation]
        
        # Process through each layer (without dropout for consistency)
        for l in range(len(self.qa_weights)):
            prev_activation = layer_activations[-1]
            next_layer = np.dot(prev_activation, self.qa_weights[l]) + self.qa_biases[l]
            next_activation = self._apply_activation(self.activation_functions[l], next_layer)
            layer_activations.append(next_activation)
        
        # Calculate output error
        output_error = output - targets
        
        # Initialize gradients
        dweights = [np.zeros_like(w) for w in self.qa_weights]
        dbiases = [np.zeros_like(b) for b in self.qa_biases]
        
        # Backpropagate error
        layer_error = output_error
        
        for l in range(len(self.qa_weights)-1, -1, -1):
            # Get inputs to this layer
            layer_input = layer_activations[l]
            
            # Compute gradients
            dweights[l] = np.dot(layer_input.T, layer_error)
            dbiases[l] = np.sum(layer_error, axis=0)
            
            # Propagate error to previous layer
            if l > 0:
                layer_error = np.dot(layer_error, self.qa_weights[l].T)
                layer_error *= self._activation_derivative(
                    self.activation_functions[l-1], layer_activations[l]
                )
        
        # Update weights with learning rate
        for l in range(len(self.qa_weights)):
            self.qa_weights[l] -= self.learning_rate * dweights[l]
            self.qa_biases[l] -= self.learning_rate * dbiases[l]


    def predict_answer(self, question, return_probs=False):
        """
        Predict answer for a question
        
        Args:
            question: Input question (string or list of strings)
            return_probs: Whether to return probabilities or class index
            
        Returns:
            Predicted answer index or probabilities
        """
        # Handle single question input
        is_single = isinstance(question, str)
        questions = [question] if is_single else question
        
        # Tokenize questions
        tokenized = self.tokenize(questions)
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        
        # Forward pass using QA-specific weights
        if hasattr(self, 'qa_weights'):
            predictions = self._qa_forward(input_ids, attention_mask)
        else:
            # Fall back to standard prediction if QA weights don't exist
            predictions = self.forward_propagation(input_ids, attention_mask)
            if len(predictions.shape) == 3:
                predictions = predictions[:, 0, :]
        
        # Return single prediction for single input
        if is_single:
            predictions = predictions[0]
        
        if return_probs:
            return predictions
        else:
            # Return class indices
            if is_single:
                return np.argmax(predictions)
            else:
                return np.argmax(predictions, axis=1)




    def activation(self, x, func_name):
        """Apply activation function"""
        if func_name == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif func_name == 'tanh':
            return np.tanh(x)
        elif func_name == 'relu':
            return np.maximum(0, x)
        elif func_name == 'softmax':
            return softmax(x)
        elif func_name == 'leaky_relu':
            return np.maximum(0.01 * x, x)
        else:
            raise ValueError(f"Unsupported activation function: {func_name}")

    def activation_derivative(self, x, func_name):
        """Calculate derivative of activation function"""
        if func_name == 'sigmoid':
            sig = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            return sig * (1 - sig)
        elif func_name == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif func_name == 'relu':
            return (x > 0).astype(float)
        elif func_name == 'leaky_relu':
            return np.where(x > 0, 1.0, 0.01)
        else:
            raise ValueError(f"Unsupported activation function derivative: {func_name}")

    def enable_text_generation(self):
        """Enable the model for text generation"""
        if not self.use_embedding:
            raise ValueError("Text generation requires embedding layer")
            
        if not self.tokenizer:
            raise ValueError("Text generation requires tokenizer")
            
        # Set flag to indicate the model is ready for generation
        self.is_generative = True
        logging.info("Model enabled for text generation")
        
    def generate(self, prompt, max_length=30, temperature=1.0):
        """Simple text generation from a prompt
        
        Args:
            prompt (str): Starting text
            max_length (int): Maximum number of new tokens to generate
            temperature (float): Controls randomness (lower = more deterministic)
            
        Returns:
            str: Generated text
        """
        if not self.use_embedding:
            raise ValueError("Text generation requires embedding layer")
            
        if not self.tokenizer:
            raise ValueError("Text generation requires tokenizer")
        
        # Set to evaluation mode
        self.training = False
        
        # Tokenize the prompt
        tokenized = self.tokenize(prompt)
        input_ids = tokenized['input_ids'][0]  # Take first sequence
        attention_mask = tokenized['attention_mask'][0]
        
        # Generate tokens auto-regressively
        all_tokens = input_ids.tolist()
        current_length = 0
        
        while current_length < max_length:
            # Prepare current sequence
            current_input_ids = np.array([all_tokens])
            current_mask = np.ones((1, len(all_tokens)))
            
            # Forward pass to get next token probabilities
            # This now returns shape (batch_size, seq_length, vocab_size)
            logits = self.forward_propagation(current_input_ids, current_mask)
            
            # Get logits for the last position in the sequence
            next_token_logits = logits[0, -1, :]  # Shape: (vocab_size,)
            
            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / max(temperature, 1e-8)
            
            # Convert to probabilities
            next_token_probs = self._apply_activation('softmax', next_token_logits)
            
            # Make sure probabilities sum to 1
            next_token_probs = next_token_probs / np.sum(next_token_probs)
            
            # Sample next token 
            try:
                next_token_id = np.random.choice(len(next_token_probs), p=next_token_probs)
            except ValueError as e:
                # Debug information
                print(f"Error with probability distribution: {e}")
                print(f"Shape: {next_token_probs.shape}, Sum: {np.sum(next_token_probs)}")
                print(f"Contains NaN: {np.isnan(next_token_probs).any()}")
                print(f"Min: {np.min(next_token_probs)}, Max: {np.max(next_token_probs)}")
                
                # Fix any NaN or invalid values
                next_token_probs = np.nan_to_num(next_token_probs)
                next_token_probs = np.clip(next_token_probs, 1e-10, 1.0)
                next_token_probs = next_token_probs / np.sum(next_token_probs)
                
                # Try again
                next_token_id = np.random.choice(len(next_token_probs), p=next_token_probs)
            
            # Append to generated tokens
            all_tokens.append(next_token_id)
            current_length += 1
            
            # Stop if we generate an end-of-sequence token
            if hasattr(self.tokenizer, 'eos_token_id') and next_token_id == self.tokenizer.eos_token_id:
                break
        
        # Convert token IDs back to text
        generated_text = self.tokenizer.decode(all_tokens, skip_special_tokens=True)
        return generated_text
        
    def train(self, training_data, labels, validation_data=None, validation_labels=None, 
              learning_rate_schedule=None, early_stopping=False, patience=5, verbose=True):
        """Train the model on data and labels with additional options"""
        # Reset training metrics
        self.training_losses = []
        self.validation_metrics = []
        
        # Prepare the labels
        processed_labels = self._prepare_labels(labels)
        
        # Process validation data if provided
        if validation_data is not None and validation_labels is not None:
            validation_labels = self._prepare_labels(validation_labels)
        
        # Store initial learning rate
        initial_lr = self.learning_rate
        
        # Early stopping variables
        best_val_metric = float('inf')
        patience_counter = 0
        
        # Training loop
        self.training = True
        
        if verbose:
            logging.info(f"Training on {len(training_data)} examples for {self.epochs} epochs")
        
        for epoch in range(self.epochs):
            # Apply learning rate schedule if provided
            if learning_rate_schedule:
                self.learning_rate = learning_rate_schedule(epoch, initial_lr)
            
            # Shuffle training data
            indices = np.arange(len(training_data))
            np.random.shuffle(indices)
            
            # Handle different data types
            if isinstance(training_data, list):
                shuffled_data = [training_data[i] for i in indices]
            else:
                shuffled_data = training_data[indices]
                
            shuffled_labels = processed_labels[indices]
            
            total_batches = (len(training_data) + self.batch_size - 1) // self.batch_size
            batch_losses = []
            
            for i in range(0, len(training_data), self.batch_size):
                batch_data = shuffled_data[i:i+self.batch_size]
                batch_labels = shuffled_labels[i:i+self.batch_size]
                
                # Process inputs based on data type
                if self.use_embedding and isinstance(batch_data, list) and all(isinstance(item, str) for item in batch_data):
                    tokenized = self.tokenize(batch_data)
                    inputs = tokenized['input_ids']
                    masks = tokenized['attention_mask']
                else:
                    inputs = np.array(batch_data)
                    masks = None
                    
                # Forward pass
                output = self.forward_propagation(inputs, masks)
                
                # Calculate loss
                if self.activation_functions[-1] == 'softmax':
                    # Cross-entropy loss
                    epsilon = 1e-15  # To avoid log(0)
                    output = np.clip(output, epsilon, 1 - epsilon)
                    batch_loss = -np.sum(batch_labels * np.log(output)) / len(batch_data)
                else:
                    # Mean squared error
                    batch_loss = np.mean((output - batch_labels) ** 2)
                
                batch_losses.append(batch_loss)
                
                # Backward pass
                self.backward_propagation(inputs, batch_labels, output, masks)
            
            # Calculate average loss for the epoch
            avg_loss = np.mean(batch_losses)
            self.training_losses.append(avg_loss)
            
            # Log progress
            if verbose and (epoch % 10 == 0 or epoch == self.epochs - 1):
                log_msg = f"Epoch {epoch+1}/{self.epochs}: Loss = {avg_loss:.6f}"
                logging.info(log_msg)

    def _prepare_labels(self, labels):
        """Prepare labels for training"""
        # If labels are strings, encode them
        if isinstance(labels, np.ndarray) and labels.dtype.kind in ('U', 'S'):
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                labels = self.label_encoder.fit_transform(labels)
            else:
                labels = self.label_encoder.transform(labels)
                
            # Convert to one-hot encoding
            n_classes = len(self.label_encoder.classes_)
            one_hot_labels = np.zeros((len(labels), n_classes))
            for i, label in enumerate(labels):
                one_hot_labels[i, label] = 1
            return one_hot_labels
        elif isinstance(labels, list) and all(isinstance(l, str) for l in labels):
            # List of strings
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                labels = self.label_encoder.fit_transform(labels)
            else:
                labels = self.label_encoder.transform(labels)
                
            # Convert to one-hot encoding
            n_classes = len(self.label_encoder.classes_)
            one_hot_labels = np.zeros((len(labels), n_classes))
            for i, label in enumerate(labels):
                one_hot_labels[i, label] = 1
            return one_hot_labels
        else:
            # Convert to numpy array if necessary
            return np.array(labels) if not isinstance(labels, np.ndarray) else labels
        
    def train_language_model(self, text_sequences, validation_sequences=None, 
                            learning_rate_schedule=None, early_stopping=False, patience=5, verbose=True):
        """Train the model on text sequences for language modeling"""
        if not self.use_embedding or not self.tokenizer:
            raise ValueError("Language model training requires embedding layer and tokenizer")
        
        # Reset training metrics
        self.training_losses = []
        self.validation_metrics = []
        
        # Store initial learning rate
        initial_lr = self.learning_rate
        
        # Early stopping variables
        best_val_metric = float('inf')
        patience_counter = 0
        
        # Training loop
        self.training = True
        
        if verbose:
            logging.info(f"Training on {len(text_sequences)} sequences for {self.epochs} epochs")
        
        for epoch in range(self.epochs):
            # Apply learning rate schedule if provided
            if learning_rate_schedule:
                self.learning_rate = learning_rate_schedule(epoch, initial_lr)
            
            # Shuffle training data
            indices = np.arange(len(text_sequences))
            np.random.shuffle(indices)
            shuffled_sequences = [text_sequences[i] for i in indices]
            
            total_batches = (len(text_sequences) + self.batch_size - 1) // self.batch_size
            batch_losses = []
            
            for i in range(0, len(text_sequences), self.batch_size):
                batch_sequences = shuffled_sequences[i:i+self.batch_size]
                
                # Tokenize the sequences
                tokenized = self.tokenize(batch_sequences)
                input_ids = tokenized['input_ids']
                attention_mask = tokenized['attention_mask']
                
                # For language modeling, targets are the input sequences shifted right
                # We'll use the input_ids as targets (predict next token)
                target_ids = np.zeros_like(input_ids)
                
                # Shift right: for each sequence, the target for position i is the token at position i+1
                for seq_idx in range(input_ids.shape[0]):
                    seq_length = np.sum(attention_mask[seq_idx])
                    if seq_length > 1:  # Only process if sequence has at least 2 tokens
                        # Copy tokens from position 1 to end as targets for positions 0 to end-1
                        target_ids[seq_idx, 0:seq_length-1] = input_ids[seq_idx, 1:seq_length]
                        # Last position target can be the EOS token or padded
                        if hasattr(self.tokenizer, 'eos_token_id'):
                            target_ids[seq_idx, seq_length-1] = self.tokenizer.eos_token_id
                
                # Convert target_ids to one-hot encoded format
                batch_labels = np.zeros((input_ids.shape[0], input_ids.shape[1], self.vocab_size))
                for seq_idx in range(input_ids.shape[0]):
                    for pos_idx in range(input_ids.shape[1]):
                        if attention_mask[seq_idx, pos_idx] > 0:  # Only for non-padded positions
                            if target_ids[seq_idx, pos_idx] < self.vocab_size:  # Ensure token ID is valid
                                batch_labels[seq_idx, pos_idx, target_ids[seq_idx, pos_idx]] = 1
                
                # Forward pass - shape should be (batch_size, seq_length, vocab_size)
                output = self.forward_propagation(input_ids, attention_mask)
                
                # Calculate loss (cross-entropy for each token position)
                epsilon = 1e-15  # To avoid log(0)
                output = np.clip(output, epsilon, 1 - epsilon)
                
                # Compute loss only for non-padded positions
                loss = 0.0
                total_tokens = 0
                for seq_idx in range(input_ids.shape[0]):
                    for pos_idx in range(input_ids.shape[1]):
                        if attention_mask[seq_idx, pos_idx] > 0:  # Only for non-padded positions
                            loss -= np.sum(batch_labels[seq_idx, pos_idx] * np.log(output[seq_idx, pos_idx]))
                            total_tokens += 1
                
                batch_loss = loss / max(total_tokens, 1)  # Avoid division by zero
                batch_losses.append(batch_loss)
                
                # Backward pass
                self.backward_propagation(input_ids, batch_labels, output, attention_mask)
            
            # Calculate average loss for the epoch
            avg_loss = np.mean(batch_losses)
            self.training_losses.append(avg_loss)
            
            # Log progress
            if verbose and (epoch % 1 == 0 or epoch == self.epochs - 1):
                log_msg = f"Epoch {epoch+1}/{self.epochs}: Loss = {avg_loss:.6f}"
                logging.info(log_msg)
                
        return self.training_losses        
        

    def train_sentiment(self, texts, labels, validation_data=None, validation_labels=None, verbose=True):
        """
        Special training method just for sentiment analysis models.
        Creates new weights specifically for this task to avoid dimension mismatches.
        
        Args:
            texts: List of text strings
            labels: Target labels (one-hot encoded)
            validation_data: Optional validation texts
            validation_labels: Optional validation labels
            verbose: Whether to log training progress
        """
        # Reset training metrics
        self.training_losses = []
        self.validation_metrics = []
        
        if verbose:
            logging.info(f"Training sentiment model on {len(texts)} examples for {self.epochs} epochs")
        
        # Get the correct layer sizes for sentiment analysis
        # Format: [embed_dim, hidden1, hidden2, output] 
        # In our case [512, 256, 256, 3]
        sentiment_layer_sizes = self.layer_sizes.copy()
        
        # Create separate weights specifically for sentiment analysis with the right dimensions
        sentiment_weights = []
        sentiment_biases = []
        
        for i in range(len(sentiment_layer_sizes) - 1):
            # Print dimensions for debugging
            print(f"Creating weight matrix of shape: ({sentiment_layer_sizes[i]}, {sentiment_layer_sizes[i+1]})")
            sentiment_weights.append(np.random.randn(sentiment_layer_sizes[i], sentiment_layer_sizes[i+1]) * 0.01)
            sentiment_biases.append(np.zeros(sentiment_layer_sizes[i+1]))
        
        # Number of batches
        num_examples = len(texts)
        batch_size = min(self.batch_size, num_examples)
        num_batches = (num_examples + batch_size - 1) // batch_size
        
        # Training loop
        for epoch in range(self.epochs):
            # Shuffle indices
            indices = np.arange(num_examples)
            np.random.shuffle(indices)
            
            epoch_loss = 0.0
            
            for batch in range(num_batches):
                # Get batch indices
                start_idx = batch * batch_size
                end_idx = min(start_idx + batch_size, num_examples)
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_texts = [texts[i] for i in batch_indices]
                batch_labels = labels[batch_indices]
                
                # Tokenize texts
                tokenized = self.tokenize(batch_texts)
                input_ids = tokenized['input_ids']
                attention_mask = tokenized['attention_mask']
                
                # Custom forward pass with our sentiment-specific weights
                batch_size_actual = input_ids.shape[0]
                
                # Create a pooled representation from token embeddings
                pooled_representation = np.zeros((batch_size_actual, sentiment_layer_sizes[0]))
                token_counts = np.zeros(batch_size_actual)
                
                # Process each token position
                for pos in range(input_ids.shape[1]):
                    for i in range(batch_size_actual):
                        # Skip padded tokens
                        if attention_mask is None or attention_mask[i, pos] > 0:
                            token_id = input_ids[i, pos]
                            if token_id < self.vocab_size:
                                # Simple deterministic embedding
                                np.random.seed(int(token_id) % 10000)
                                token_embedding = np.random.randn(sentiment_layer_sizes[0]) * 0.1
                                pooled_representation[i] += token_embedding
                                token_counts[i] += 1
                
                # Average the token embeddings
                for i in range(batch_size_actual):
                    if token_counts[i] > 0:
                        pooled_representation[i] /= token_counts[i]
                
                # Forward through the network one layer at a time
                # The first "layer activation" is just the pooled embeddings
                layer_activations = [pooled_representation]
                
                # Process through each layer
                for l in range(len(sentiment_weights)):
                    # Get the previous layer's activation
                    prev_activation = layer_activations[-1]
                    
                    # Compute the current layer
                    # Shape check before dot product
                    print(f"Layer {l}: prev_activation shape: {prev_activation.shape}, weights shape: {sentiment_weights[l].shape}")
                    
                    # Compute next layer
                    next_layer = np.dot(prev_activation, sentiment_weights[l]) + sentiment_biases[l]
                    
                    # Apply activation function
                    next_activation = self._apply_activation(self.activation_functions[l], next_layer)
                    
                    # Store for backpropagation
                    layer_activations.append(next_activation)
                
                # The final activation is our output
                batch_output = layer_activations[-1]
                
                # Calculate loss
                if self.activation_functions[-1] == 'softmax':
                    # Cross-entropy loss
                    epsilon = 1e-15  # To avoid log(0)
                    batch_output_safe = np.clip(batch_output, epsilon, 1 - epsilon)
                    batch_loss = -np.sum(batch_labels * np.log(batch_output_safe)) / len(batch_indices)
                else:
                    # Mean squared error
                    batch_loss = np.mean((batch_output - batch_labels) ** 2)
                
                # Accumulate loss
                epoch_loss += batch_loss
                
                # Backward pass
                # Initialize gradients
                dweights = [np.zeros_like(w) for w in sentiment_weights]
                dbiases = [np.zeros_like(b) for b in sentiment_biases]
                
                # Calculate output error
                if self.activation_functions[-1] == 'softmax':
                    # For softmax with cross-entropy loss
                    output_error = batch_output - batch_labels
                else:
                    # For MSE loss
                    output_error = (batch_output - batch_labels) * self._activation_derivative(
                        self.activation_functions[-1], batch_output
                    )
                
                # Backpropagate through layers
                layer_error = output_error
                
                for l in range(len(sentiment_weights)-1, -1, -1):
                    # Get layer inputs (activations from previous layer)
                    layer_input = layer_activations[l]
                    
                    # Compute gradients
                    dweights[l] = np.dot(layer_input.T, layer_error)
                    dbiases[l] = np.sum(layer_error, axis=0)
                    
                    # Propagate error to previous layer (if not at input layer)
                    if l > 0:
                        layer_error = np.dot(layer_error, sentiment_weights[l].T)
                        layer_error *= self._activation_derivative(
                            self.activation_functions[l-1], layer_activations[l]
                        )
                
                # Update weights
                for l in range(len(sentiment_weights)):
                    sentiment_weights[l] -= self.learning_rate * dweights[l]
                    sentiment_biases[l] -= self.learning_rate * dbiases[l]
            
            # Average loss for the epoch
            avg_loss = epoch_loss / num_batches
            self.training_losses.append(avg_loss)
            
            # Log progress
            if verbose and (epoch % 100 == 0 or epoch == self.epochs - 1):
                logging.info(f"Epoch {epoch+1}/{self.epochs}: Loss = {avg_loss:.6f}")
        
        # Save the trained weights to the model
        self.sentiment_weights = sentiment_weights
        self.sentiment_biases = sentiment_biases
        
        return self.training_losses


    def predict(self, inputs, return_probs=False):
        """
        Make predictions on input data
        
        Args:
            inputs: Input data (can be text, numerical, or sequences)
            return_probs: Whether to return probabilities or class index
            
        Returns:
            Model predictions
        """
        # Set to evaluation mode
        self.training = False
        
        # Check if inputs are numerical (XOR case)
        if isinstance(inputs, np.ndarray) and inputs.dtype.kind in ('i', 'f') or isinstance(inputs, list) and all(isinstance(x, (int, float)) for x in inputs):
            # Convert to numpy array if it's a list
            if isinstance(inputs, list):
                inputs = np.array(inputs)
            
            # Reshape if single example
            if len(inputs.shape) == 1:
                inputs = inputs.reshape(1, -1)
            
            # Use standard forward propagation for numerical inputs
            predictions = self.forward_propagation(inputs)
            
            # If single example, return the first prediction
            if predictions.shape[0] == 1:
                predictions = predictions[0]
            
            return predictions
            
        # Handle single text input
        is_single = isinstance(input, str)
        texts = [inputs] if is_single else inputs
        
        # Tokenize text
        tokenized = self.tokenize(texts)
        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']
        
        # If this looks like a sentiment model and we have sentiment weights
        if hasattr(self, 'sentiment_weights') and self.sentiment_weights is not None:
            # Use sentiment-specific prediction
            batch_size = input_ids.shape[0]
            
            # Create pooled representation
            pooled_representation = np.zeros((batch_size, self.layer_sizes[0]))
            token_counts = np.zeros(batch_size)
            
            # Process each token position
            for pos in range(input_ids.shape[1]):
                for i in range(batch_size):
                    # Skip padded tokens
                    if attention_mask is None or attention_mask[i, pos] > 0:
                        token_id = input_ids[i, pos]
                        if token_id < self.vocab_size:
                            # Simple deterministic embedding
                            np.random.seed(int(token_id) % 10000)
                            token_embedding = np.random.randn(self.layer_sizes[0]) * 0.1
                            pooled_representation[i] += token_embedding
                            token_counts[i] += 1
            
            # Average the token embeddings
            for i in range(batch_size):
                if token_counts[i] > 0:
                    pooled_representation[i] /= token_counts[i]
            
            # Forward through the network one layer at a time
            current_activation = pooled_representation
            
            # Process through each layer
            for l in range(len(self.sentiment_weights)):
                # Compute next layer
                next_layer = np.dot(current_activation, self.sentiment_weights[l]) + self.sentiment_biases[l]
                
                # Apply activation function
                current_activation = self._apply_activation(self.activation_functions[l], next_layer)
            
            predictions = current_activation
        else:
            # Use standard forward pass
            predictions = self.forward_propagation(input_ids, attention_mask)
        
        # Handle sequence model output (take first position)
        if len(predictions.shape) == 3:
            predictions = predictions[:, 0, :]
        
        # Return single prediction for single input
        if is_single:
            predictions = predictions[0]
        
        if return_probs:
            return predictions
        else:
            # Return class indices
            if is_single:
                return np.argmax(predictions)
            else:
                return np.argmax(predictions, axis=1)
        
    def tokenize(self, texts):
        """Tokenize text data for embedding layer"""
        if not self.tokenizer:
            raise ValueError("Tokenizer not initialized")
            
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]
            
        try:
            tokenized = self.tokenizer(
                list(texts),  # Ensure it's a proper list
                padding="max_length", 
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="np",
                return_attention_mask=True
            )
            return {
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask']
            }
        except Exception as e:
            logging.error(f"Tokenization error: {e}")
            raise
        
    # def predict(self, inputs, return_probs=False):
    #     """Make predictions on new data"""
    #     self.training = False
        
    #     # Debug logging to trace the flow
    #     logging.info(f"Input type: {type(inputs)}")
        
    #     # Handle different input types
    #     if isinstance(inputs, str):
    #         inputs = [inputs]
    #         logging.info("Converted string input to list")
    #     elif isinstance(inputs, np.ndarray) and inputs.dtype.kind in ('U', 'S'):
    #         inputs = inputs.tolist()
    #         logging.info("Converted numpy string array to list")
            
    #     # For text data, we need to tokenize and get embeddings
    #     if self.use_embedding:
    #         logging.info(f"Use embedding is True, checking if input is text list")
            
    #         if isinstance(inputs, list) and all(isinstance(item, str) for item in inputs):
    #             logging.info("Input confirmed as text list, proceeding with tokenization")
                
    #             # Check if we have a tokenizer
    #             if not self.tokenizer:
    #                 logging.error("No tokenizer available for text processing")
    #                 raise ValueError("Tokenizer is required for text prediction but not available")
                    
    #             # Tokenize the inputs
    #             try:
    #                 logging.info(f"Tokenizing input: {inputs[:50]}...")  # Log first part of input
    #                 tokenized = self.tokenize(inputs)
                    
    #                 if tokenized is None:
    #                     logging.error("Tokenization returned None")
    #                     raise ValueError("Tokenization failed")
                        
    #                 inputs = tokenized['input_ids']
    #                 masks = tokenized['attention_mask']
                    
    #                 logging.info(f"Tokenized input shape: {inputs.shape}")
    #                 logging.info(f"Attention mask shape: {masks.shape}")
                    
    #                 predictions = self.forward_propagation(inputs, masks)
    #                 logging.info(f"Predictions shape after forward pass: {predictions.shape}")
                    
    #             except Exception as e:
    #                 logging.error(f"Error in text prediction process: {e}")
    #                 import traceback
    #                 traceback.print_exc()
    #                 raise
    #         else:
    #             logging.warning("Input is not a text list but use_embedding is True. Treating as numeric input.")
    #             if not isinstance(inputs, np.ndarray):
    #                 inputs = np.array(inputs)
    #             logging.info(f"Numeric input shape: {inputs.shape}")
    #             predictions = self.forward_propagation(inputs)
    #     else:
    #         # For non-text data
    #         logging.info("Processing as non-text data (use_embedding is False)")
    #         if not isinstance(inputs, np.ndarray):
    #             inputs = np.array(inputs)
                
    #         logging.info(f"Non-text input shape: {inputs.shape}")
            
    #         # Ensure input has the right shape for the first layer
    #         expected_dim = self.layer_sizes[0]
    #         if len(inputs.shape) == 1 and inputs.shape[0] != expected_dim:
    #             logging.error(f"Dimension mismatch: got {inputs.shape}, expected first dimension to be {expected_dim}")
    #             raise ValueError(f"Input dimension mismatch. Expected {expected_dim}, got {inputs.shape[0]}")
                
    #         predictions = self.forward_propagation(inputs)
        
    #     # Return probabilities if requested
    #     if return_probs:
    #         return predictions
            
    #     # Convert class probabilities to class labels if using a label encoder
    #     if self.label_encoder is not None and len(predictions.shape) > 1 and predictions.shape[1] > 1:
    #         # For classification problems
    #         class_indices = np.argmax(predictions, axis=1)
    #         return self.label_encoder.inverse_transform(class_indices)
        
    #     return predictions
    
    def evaluate(self, test_data, test_labels):
        """Evaluate model performance on test data"""
        predictions = self.predict(test_data, return_probs=True)
        
        # Process test labels if needed
        processed_labels = self._prepare_labels(test_labels)
        
        # Calculate metrics based on the task
        if len(processed_labels.shape) > 1 and processed_labels.shape[1] > 1:
            # Classification task
            pred_classes = np.argmax(predictions, axis=1)
            true_classes = np.argmax(processed_labels, axis=1)
            
            # Calculate accuracy
            accuracy = np.mean(pred_classes == true_classes)
            
            # Calculate precision, recall, f1 for each class
            n_classes = processed_labels.shape[1]
            precision = np.zeros(n_classes)
            recall = np.zeros(n_classes)
            f1 = np.zeros(n_classes)
            
            for c in range(n_classes):
                true_positives = np.sum((pred_classes == c) & (true_classes == c))
                predicted_positives = np.sum(pred_classes == c)
                actual_positives = np.sum(true_classes == c)
                
                precision[c] = true_positives / predicted_positives if predicted_positives > 0 else 0
                recall[c] = true_positives / actual_positives if actual_positives > 0 else 0
                f1[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c]) if (precision[c] + recall[c]) > 0 else 0
            
            metrics = {
                'accuracy': accuracy,
                'precision': np.mean(precision),
                'recall': np.mean(recall),
                'f1_score': np.mean(f1)
            }
        else:
            # Regression task
            mse = np.mean((predictions - processed_labels) ** 2)
            mae = np.mean(np.abs(predictions - processed_labels))
            r2 = 1 - np.sum((processed_labels - predictions) ** 2) / np.sum((processed_labels - np.mean(processed_labels)) ** 2)
            
            metrics = {
                'mse': mse,
                'mae': mae,
                'r2_score': r2
            }
        
        return metrics
    
    def save(self, filepath):
        """Save model to file"""
        try:
            # Create a dictionary of model attributes to save
            model_data = {
                'layer_sizes': self.layer_sizes,
                'weights': self.weights,
                'biases': self.biases,
                'activation_functions': self.activation_functions,
                'learning_rate': self.learning_rate,
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'dropout_rate': self.dropout_rate,
                'use_embedding': self.use_embedding,
            }
            
            # Add embedding related data if used
            if self.use_embedding:
                model_data['vocab_size'] = self.vocab_size
                model_data['embed_dim'] = self.embed_dim
                if hasattr(self, 'max_seq_length'):
                    model_data['max_seq_length'] = self.max_seq_length
            
            # Add sentiment weights if they exist
            if hasattr(self, 'sentiment_weights') and self.sentiment_weights is not None:
                model_data['sentiment_weights'] = self.sentiment_weights
                model_data['sentiment_biases'] = self.sentiment_biases
            
            # Add tokenizer info if available
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                # We don't save the tokenizer object itself, just its name/path
                if hasattr(self.tokenizer, 'name_or_path'):
                    model_data['tokenizer_name'] = self.tokenizer.name_or_path
                elif hasattr(self.tokenizer, '_name_or_path'):
                    model_data['tokenizer_name'] = self.tokenizer._name_or_path
            
            # Add generative model flag if it exists
            if hasattr(self, 'is_generative'):
                model_data['is_generative'] = self.is_generative
            
            # Save the label encoder if it exists
            if hasattr(self, 'label_encoder') and self.label_encoder is not None:
                model_data['label_encoder'] = self.label_encoder
            
            
            if hasattr(self, 'qa_weights') and self.qa_weights is not None:
                model_data['qa_weights'] = self.qa_weights
                model_data['qa_biases'] = self.qa_biases
            
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            return True
        except Exception as e:
            logging.error(f"Error saving model: {e}")
            return False
    
    @classmethod
    def load(cls, filepath):
        """Load model from file"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Create a new instance with minimal initialization
            model = cls(
                layer_sizes=model_data['layer_sizes'],
                learning_rate=model_data['learning_rate'],
                epochs=model_data['epochs'],
                batch_size=model_data['batch_size'],
                dropout_rate=model_data['dropout_rate'],
                activation_functions=model_data['activation_functions'],
                use_embedding=model_data['use_embedding']
            )
            
            # Load embedding-related attributes if they exist
            if model.use_embedding:
                if 'vocab_size' in model_data:
                    model.vocab_size = model_data['vocab_size']
                if 'embed_dim' in model_data:
                    model.embed_dim = model_data['embed_dim']
                if 'max_seq_length' in model_data:
                    model.max_seq_length = model_data['max_seq_length']
                
                # Initialize tokenizer if name is available
                if 'tokenizer_name' in model_data:
                    try:
                        from transformers import AutoTokenizer
                        model.tokenizer = AutoTokenizer.from_pretrained(model_data['tokenizer_name'])
                        
                        # Set pad token to eos token if not already set
                        if model.tokenizer.pad_token is None:
                            model.tokenizer.pad_token = model.tokenizer.eos_token
                    except Exception as e:
                        logging.warning(f"Failed to load tokenizer: {e}")
                        model.tokenizer = None
            
            # Load weights and biases
            model.weights = model_data['weights']
            model.biases = model_data['biases']
            
            # Load sentiment weights if they exist
            if 'sentiment_weights' in model_data:
                model.sentiment_weights = model_data['sentiment_weights']
                model.sentiment_biases = model_data['sentiment_biases']
            
            # Load generative flag if it exists
            if 'is_generative' in model_data:
                model.is_generative = model_data['is_generative']
            
            # Load label encoder if it exists
            if 'label_encoder' in model_data:
                model.label_encoder = model_data['label_encoder']
            
            if 'qa_weights' in model_data:
                model.qa_weights = model_data['qa_weights']
                model.qa_biases = model_data['qa_biases']
            
            # Initialize layer outputs storage
            model.layer_outputs = [None] * (len(model.layer_sizes) - 1)
            
            return model
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return None
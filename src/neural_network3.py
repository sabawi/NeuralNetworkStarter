import numpy as np
import pickle
# import os
import pprint
import logging
# from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def softmax(x, axis=-1):
    """Numerically stable softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class AdamOptimizer:
    """
    Implementation of Adam optimizer (Adaptive Moment Estimation)
    
    Adam combines ideas from RMSProp and momentum optimization to provide
    parameter-specific adaptive learning rates.
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize Adam optimizer
        
        Args:
            learning_rate: Base learning rate (alpha in the paper)
            beta1: Exponential decay rate for first moment estimates (default: 0.9)
            beta2: Exponential decay rate for second moment estimates (default: 0.999)
            epsilon: Small constant for numerical stability (default: 1e-8)
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment variables (momentum)
        self.v = None  # Second moment variables (velocity)
        self.t = 0     # Timestep counter for bias correction
        
    def initialize(self, parameters):
        """
        Initialize moment variables for parameters
        
        Args:
            parameters: List of parameter tensors (weights and biases)
        """
        self.m = [np.zeros_like(param) for param in parameters]
        self.v = [np.zeros_like(param) for param in parameters]
        self.t = 0
    
    def update(self, parameters, gradients):
        """
        Update parameters using Adam optimization
        
        Args:
            parameters: List of parameter tensors to update
            gradients: List of gradient tensors for each parameter
            
        Returns:
            Updated parameters
        """
        # Initialize moment variables if not done yet
        if self.m is None or len(self.m) != len(parameters):
            self.initialize(parameters)
        
        # Increment timestep
        self.t += 1
        
        # Updated parameters
        updated_parameters = []
        
        # Process each parameter tensor
        for i in range(len(parameters)):
            # Update biased first moment estimate (momentum)
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * gradients[i]
            
            # Update biased second moment estimate (velocity)
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * np.square(gradients[i])
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            updated_param = parameters[i] - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            updated_parameters.append(updated_param)
        
        return updated_parameters

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.001, epochs=100, batch_size=32, dropout_rate=0.0,
                activation_functions=None, use_embedding=False, vocab_size=None, embed_dim=None,
                max_seq_length=None, tokenizer_name=None, eos_token='<|endoftext|>', pad_token='<|pad|>', 
                bos_token='<|startoftext|>', optimizer='adam', use_attention=False, num_attention_heads=8, 
                attention_dropout=0.1, use_positional_embedding=True, positional_embedding_type='sinusoidal',
                use_layer_norm=True):  
        """Initialize the neural network with fixes
        
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
            optimizer: Optimization algorithm to use ('sgd', 'adam', etc.)
            use_attention: Whether to use self-attention mechanism
            num_attention_heads: Number of attention heads to use
            attention_dropout: Dropout rate for attention weights
            use_positional_embedding: Whether to use positional embeddings
            positional_embedding_type: Type of positional embedding ('sinusoidal' or 'learned')
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
        
        # Add optimizer parameter and options
        self.optimizer_name = optimizer.lower()
        
        # Embedding settings
        self.use_embedding = use_embedding
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_length = max_seq_length
        
        
        # Self-attention parameters
        self.use_attention = use_attention
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        
        # Positional embedding parameters
        self.use_positional_embedding = use_positional_embedding
        self.positional_embedding_type = positional_embedding_type
        
        
        # Initialize optimizer based on the selected type
        if self.optimizer_name == 'adam':
            self.optimizer = AdamOptimizer(learning_rate=learning_rate)
        elif self.optimizer_name == 'sgd':
            self.optimizer = None  # Use basic SGD implementation
        else:
            logging.warning(f"Unknown optimizer '{optimizer}', falling back to SGD")
            self.optimizer = None
            self.optimizer_name = 'sgd'
        
        # Initialize tokenizer if name provided
        if tokenizer_name:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                self.fix_tokenizer()
                logging.info(f"Loaded tokenizer and fixed: {tokenizer_name}")
                        
            except Exception as e:
                logging.error(f"Failed to load tokenizer: {e}")
                self.tokenizer = None
        else:
            self.tokenizer = None
        
        # Set default activation functions if not provided
        if activation_functions is None:
            # Use tanh for hidden layers and softmax for output layer when training language models
            if use_embedding and vocab_size is not None and layer_sizes[0] == vocab_size:
                activation_functions = ['tanh'] * (len(layer_sizes) - 2) + ['softmax']
            else:
                # Use relu for hidden layers and softmax for output layer
                activation_functions = ['relu'] * (len(layer_sizes) - 2) + ['softmax']
        
        # Ensure there's one activation function per layer transition
        assert len(activation_functions) == len(layer_sizes) - 1, \
            f"Expected {len(layer_sizes) - 1} activation functions, got {len(activation_functions)}"
        
        self.activation_functions = activation_functions
                
        # Initialize weights and biases with improved initialization
        self.weights = []
        self.biases = []
        
        # Determine if we have a language model architecture
        self.is_lm_architecture = use_embedding and vocab_size is not None and layer_sizes[0] == vocab_size
        
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i+1]
            
            # Improved initialization based on activation function
            if self.activation_functions[i] == 'relu':
                # He initialization for ReLU
                scale = np.sqrt(2.0 / input_size)
            elif self.activation_functions[i] in ['tanh', 'sigmoid']:
                # Xavier/Glorot initialization for tanh/sigmoid
                scale = np.sqrt(6.0 / (input_size + output_size))
            else:
                # Conservative initialization for other activations
                scale = 0.01
            
            # Initialize weights
            self.weights.append(np.random.randn(input_size, output_size) * scale)
            
            # Initialize biases to zeros
            self.biases.append(np.zeros(output_size))
        
        # Initialize attention-specific weights if attention is enabled
        if self.use_attention:
            self._initialize_attention_weights()
            
        # Initialize positional embeddings if enabled
        if self.use_positional_embedding:
            self._initialize_positional_embeddings()

        # Add this parameter to the class
        self.use_layer_norm = use_layer_norm
        
        # Initialize layer normalization parameters if enabled
        if self.use_layer_norm:
            self._initialize_layer_norm_parameters()
        
        
        # Storage for layer outputs during forward pass
        self.layer_outputs = []
        self.layer_inputs = []
        self.dropout_masks = []
        self.attention_weights = [] # Store attention weights for visualization or gradient computation

    
    def fix_tokenizer(self):
        """Fix tokenizer by ensuring all special tokens have unique IDs, avoiding recursion"""
        if self.tokenizer is None:
            logging.warning("No tokenizer to fix")
            return None
            
        try:
            # Create a dictionary for all special tokens we need to add
            special_tokens_dict = {}
            
            # Handle BOS token
            if self.tokenizer.bos_token is None:
                special_tokens_dict['bos_token'] = '<|startoftext|>'
            
            # Handle EOS token
            if self.tokenizer.eos_token is None:
                special_tokens_dict['eos_token'] = '<|endoftext|>'
            
            # Handle PAD token
            if self.tokenizer.pad_token is None:
                special_tokens_dict['pad_token'] = '<|padding|>'
            
            # Add the prompt separator only if it doesn't already exist
            prompt_sep = '<|promptend|>'
            prompt_sep_exists = False
            
            # Safely check if the token exists in additional_special_tokens
            if hasattr(self.tokenizer, 'additional_special_tokens'):
                prompt_sep_exists = prompt_sep in self.tokenizer.additional_special_tokens
                
            if not prompt_sep_exists:
                if 'additional_special_tokens' not in special_tokens_dict:
                    special_tokens_dict['additional_special_tokens'] = [prompt_sep]
                else:
                    if prompt_sep not in special_tokens_dict['additional_special_tokens']:
                        special_tokens_dict['additional_special_tokens'].append(prompt_sep)
            
            # Add all special tokens at once if needed
            num_added = 0
            if special_tokens_dict:
                try:
                    num_added = self.tokenizer.add_special_tokens(special_tokens_dict)
                    logging.info(f"Added {num_added} special tokens")
                except Exception as e:
                    logging.error(f"Error adding special tokens: {e}")
            
            return self.tokenizer
        except Exception as e:
            logging.error(f"Error in fix_tokenizer: {e}")
            return self.tokenizer  # Return the original tokenizer if there's an error    
    
    def _apply_activation(self, activation_name, x):
        """Apply activation function to input with improved numerical stability"""
        if activation_name == 'sigmoid':
            # Clip to prevent overflow
            x = np.clip(x, -30, 30)
            return 1 / (1 + np.exp(-x))
        elif activation_name == 'relu':
            return np.maximum(0, x)
        elif activation_name == 'leaky_relu':
            return np.where(x > 0, x, 0.01 * x)  # Alpha = 0.01
        elif activation_name == 'tanh':
            # Clip to prevent overflow
            x = np.clip(x, -30, 30)
            return np.tanh(x)
        elif activation_name == 'softmax':
            # Numerically stable softmax
            return softmax(x)
        else:
            # Linear activation (no transformation)
            return x
    
    def _activation_derivative(self, activation_name, x, activated_x=None):
        """Calculate derivative of activation function with improved numerical stability"""
        if activation_name == 'sigmoid':
            if activated_x is not None:
                # Use the sigmoid output directly (more stable)
                return np.clip(activated_x * (1 - activated_x), -1.0, 1.0)
            x = np.clip(x, -30, 30)
            sigmoid_x = 1 / (1 + np.exp(-x))
            return np.clip(sigmoid_x * (1 - sigmoid_x), -1.0, 1.0)
        elif activation_name == 'relu':
            return np.where(x > 0, 1, 0)
        elif activation_name == 'leaky_relu':
            return np.where(x > 0, 1, 0.01)  # Alpha = 0.01
        elif activation_name == 'tanh':
            if activated_x is not None:
                # Clip the result to prevent overflow
                tanh_deriv = 1 - np.square(np.clip(activated_x, -0.99, 0.99))
                return np.clip(tanh_deriv, -1.0, 1.0)
            x = np.clip(x, -20, 20)  # Prevent extreme values
            tanh_x = np.tanh(x)
            return np.clip(1 - np.square(tanh_x), -1.0, 1.0)
        elif activation_name == 'softmax':
            # For softmax with cross-entropy loss, this is handled separately
            return np.ones_like(x)
        else:
            # Linear activation derivative is 1
            return np.ones_like(x)
    

    def _initialize_layer_norm_parameters(self):
        """Initialize layer normalization parameters"""
        if not self.use_embedding or self.embed_dim is None:
            return
            
        # Initialize scale and bias parameters for layer normalization
        # We need two sets: one for attention output and one for feed-forward output
        self.attention_norm_scale = np.ones(self.embed_dim)
        self.attention_norm_bias = np.zeros(self.embed_dim)
        
        self.ffn_norm_scale = np.ones(self.embed_dim)
        self.ffn_norm_bias = np.zeros(self.embed_dim)
        
        # For gradient storage
        self.norm_gradients = {
            'attention_scale': None,
            'attention_bias': None,
            'ffn_scale': None,
            'ffn_bias': None
        }

    def layer_norm(self, x, scale, bias, epsilon=1e-5):
        """Apply layer normalization
        
        Args:
            x: Input tensor of shape [..., features]
            scale: Scale parameter
            bias: Bias parameter
            epsilon: Small constant for numerical stability
            
        Returns:
            Normalized tensor of the same shape as input
        """
        # Compute mean and variance along the last dimension
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        
        # Normalize
        x_norm = (x - mean) / np.sqrt(variance + epsilon)
        
        # Scale and shift
        return x_norm * scale + bias



    def _initialize_attention_weights(self):
        """Initialize weights for self-attention mechanism with proper Xavier/Glorot init"""
        # Only initialize if we have embedding dimension defined
        if not self.use_embedding or self.embed_dim is None:
            logging.warning("Cannot initialize attention weights without embeddings")
            return
            
        # For multi-head attention, calculate dimensions
        self.head_dim = self.embed_dim // self.num_attention_heads
        if self.head_dim * self.num_attention_heads != self.embed_dim:
            logging.warning(f"Embedding dimension ({self.embed_dim}) is not divisible by number of heads ({self.num_attention_heads})")
            self.head_dim = self.embed_dim // self.num_attention_heads
            
        # Scaling factor for dot-product attention
        self.attention_scale = np.sqrt(self.head_dim)
        
        # Xavier/Glorot initialization for attention projections
        # This is critical for stable training in transformers
        xavier_scale = np.sqrt(6.0 / (self.embed_dim + self.embed_dim))
        
        # Initialize projection matrices with Xavier/Glorot
        self.query_projection = np.random.uniform(-xavier_scale, xavier_scale, (self.embed_dim, self.embed_dim))
        self.key_projection = np.random.uniform(-xavier_scale, xavier_scale, (self.embed_dim, self.embed_dim))
        self.value_projection = np.random.uniform(-xavier_scale, xavier_scale, (self.embed_dim, self.embed_dim))
        self.output_projection = np.random.uniform(-xavier_scale, xavier_scale, (self.embed_dim, self.embed_dim))
        
        # Bias terms
        # Initialize to very small values, not zeros
        bias_scale = 0.01
        self.query_bias = np.random.uniform(-bias_scale, bias_scale, self.embed_dim)
        self.key_bias = np.random.uniform(-bias_scale, bias_scale, self.embed_dim)
        self.value_bias = np.random.uniform(-bias_scale, bias_scale, self.embed_dim)
        self.output_bias = np.random.uniform(-bias_scale, bias_scale, self.embed_dim)
        
        # Storage for gradients
        self.attention_gradients = {
            'query_projection': None,
            'key_projection': None,
            'value_projection': None,
            'output_projection': None,
            'query_bias': None,
            'key_bias': None,
            'value_bias': None,
            'output_bias': None
        }


    def _initialize_positional_embeddings(self):
        """Initialize positional embeddings based on the selected type"""
        if not self.use_embedding or self.embed_dim is None or self.max_seq_length is None:
            logging.warning("Cannot initialize positional embeddings without embeddings")
            return
            
        if self.positional_embedding_type == 'learned':
            # Initialize learnable positional embeddings
            scale = np.sqrt(2.0 / self.embed_dim)
            self.position_embeddings = np.random.randn(self.max_seq_length, self.embed_dim) * scale
        elif self.positional_embedding_type == 'sinusoidal':
            # Initialize fixed sinusoidal embeddings (as in the original Transformer paper)
            self.position_embeddings = self._get_sinusoidal_embeddings(self.max_seq_length, self.embed_dim)
        else:
            logging.warning(f"Unknown positional embedding type: {self.positional_embedding_type}, falling back to sinusoidal")
            self.position_embeddings = self._get_sinusoidal_embeddings(self.max_seq_length, self.embed_dim)

    def _get_sinusoidal_embeddings(self, max_seq_length, embed_dim):
        """Create sinusoidal positional embeddings as in the Transformer paper"""
        # Initialize the positional encodings
        position_enc = np.zeros((max_seq_length, embed_dim))
        
        # Compute the positional encodings
        position = np.arange(max_seq_length)[:, np.newaxis]
        div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
        
        # Apply sine to even indices and cosine to odd indices
        position_enc[:, 0::2] = np.sin(position * div_term)
        position_enc[:, 1::2] = np.cos(position * div_term)
        
        return position_enc

    def self_attention(self, inputs, attention_mask=None):
        """
        Compute self-attention over a sequence of embeddings
        
        Args:
            inputs: Input tensor of shape [batch_size, seq_length, embed_dim]
            attention_mask: Optional mask for padding tokens [batch_size, seq_length]
                
        Returns:
            Attention output of shape [batch_size, seq_length, embed_dim]
        """
        batch_size, seq_length, embed_dim = inputs.shape
        
        # Project inputs to queries, keys, and values
        # Linear projections: [batch_size, seq_length, embed_dim]
        queries = np.dot(inputs, self.query_projection) + self.query_bias
        keys = np.dot(inputs, self.key_projection) + self.key_bias
        values = np.dot(inputs, self.value_projection) + self.value_bias
        
        # Reshape for multi-head attention
        # Shape: [batch_size, seq_length, num_heads, head_dim]
        queries = queries.reshape(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        keys = keys.reshape(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        values = values.reshape(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        
        # Transpose to [batch_size, num_heads, seq_length, head_dim]
        queries = np.transpose(queries, (0, 2, 1, 3))
        keys = np.transpose(keys, (0, 2, 1, 3))
        values = np.transpose(values, (0, 2, 1, 3))
        
        # Compute scaled dot-product attention
        # keys_transpose: [batch_size, num_heads, head_dim, seq_length]
        keys_transpose = np.transpose(keys, (0, 1, 3, 2))
        
        # Attention scores: [batch_size, num_heads, seq_length, seq_length]
        attention_scores = np.matmul(queries, keys_transpose) / self.attention_scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand attention_mask to [batch_size, 1, 1, seq_length]
            # to broadcast across heads and query positions
            expanded_mask = attention_mask[:, np.newaxis, np.newaxis, :]
            
            # Create a mask for invalid positions (0 in attention_mask) with large negative values
            attention_mask_expanded = (1.0 - expanded_mask) * -10000.0
            
            # Apply the mask to attention scores
            attention_scores = attention_scores + attention_mask_expanded
        
        # Apply softmax to get attention weights
        attention_weights = softmax(attention_scores, axis=-1)
        
        # Store attention weights for backward pass or visualization
        self.attention_weights.append(attention_weights)
        
        # Apply attention dropout during training
        if self.training and self.attention_dropout > 0:
            dropout_mask = np.random.binomial(1, 1 - self.attention_dropout, 
                                             size=attention_weights.shape) / (1 - self.attention_dropout)
            attention_weights = attention_weights * dropout_mask
        
        # Apply attention weights to values
        # context: [batch_size, num_heads, seq_length, head_dim]
        context = np.matmul(attention_weights, values)
        
        # Transpose and reshape back to [batch_size, seq_length, embed_dim]
        context = np.transpose(context, (0, 2, 1, 3)).reshape(batch_size, seq_length, embed_dim)
        
        # Apply output projection
        output = np.dot(context, self.output_projection) + self.output_bias
        
        return output
        
    
    
    def forward_propagation(self, inputs, attention_mask=None):
        """
        Fixed forward propagation with consistent matrix shapes
        
        Args:
            inputs: Input data
            attention_mask: Optional mask for padded sequences
                
        Returns:
            Network output
        """
        # Reset layer storage
        self.layer_outputs = []
        self.layer_inputs = []
        self.dropout_masks = []
        
        batch_size = inputs.shape[0]
        
        # Language model with sequence data
        if self.is_lm_architecture and len(inputs.shape) > 1:
            seq_length = inputs.shape[1]
            
            # Create embeddings for each token
            embeddings = np.zeros((batch_size, seq_length, self.embed_dim))
            for b in range(batch_size):
                for p in range(seq_length):
                    # Skip padded tokens
                    if attention_mask is None or attention_mask[b, p] > 0:
                        token_id = inputs[b, p]
                        if token_id < self.vocab_size:
                            # Embedding lookup
                            embeddings[b, p] = self.weights[0][token_id]
            
            # Add positional encodings if enabled
            if self.use_positional_embedding:
                # Use only the positions we need from position_embeddings
                max_pos = min(seq_length, self.max_seq_length)
                pos_enc = self.position_embeddings[:max_pos]
                
                # Add positional encodings to token embeddings
                # Broadcasting: adds the same position encoding to each item in batch
                embeddings[:, :max_pos] += pos_enc
            
            # Apply embedding activation
            embeddings_activated = self._apply_activation(self.activation_functions[0], embeddings)
            
            # Apply dropout to embeddings
            if self.training and self.dropout_rate > 0:
                dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, 
                                              size=embeddings_activated.shape) / (1 - self.dropout_rate)
                self.dropout_masks.append(dropout_mask)
                embeddings_activated = embeddings_activated * dropout_mask
            else:
                self.dropout_masks.append(None)
            
            # Store initial embeddings
            self.layer_outputs.append(embeddings_activated)
            
            current_layer = embeddings_activated
            
            # Apply self-attention if enabled
            if self.use_attention:
                # Apply layer norm before attention (pre-norm architecture)
                if self.use_layer_norm:
                    normalized_input = self.layer_norm(
                        current_layer,
                        self.attention_norm_scale,
                        self.attention_norm_bias
                    )
                else:
                    normalized_input = current_layer
                
                # Use normalized_input for attention calculation (this was missing!)
                attention_output = self.self_attention(normalized_input, attention_mask)
                
                # Apply dropout to attention output
                if self.training and self.dropout_rate > 0:
                    dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, 
                                                size=attention_output.shape) / (1 - self.dropout_rate)
                    attention_output = attention_output * dropout_mask
                
                # Residual connection: Add attention output to input (skip connection)
                current_layer = current_layer + attention_output
                
                # Store attention output
                self.layer_outputs.append(current_layer)
            
            
            # Process through remaining feed-forward layers
            for i in range(1, len(self.weights)):
                # Store original input for residual connection
                ffn_input = current_layer
                
                # Apply layer norm before feed-forward (pre-norm for FFN)
                if i == 1 and self.use_layer_norm:  # Only for the first FFN layer after attention
                    normalized_input = self.layer_norm(
                        current_layer,
                        self.ffn_norm_scale,
                        self.ffn_norm_bias
                    )
                else:
                    normalized_input = current_layer
                
                layer_input = np.zeros((batch_size, seq_length, self.layer_sizes[i+1]))
                
                # Process each position in the sequence
                for b in range(batch_size):
                    for p in range(seq_length):
                        # Skip padding tokens
                        if attention_mask is None or attention_mask[b, p] > 0:
                            # Apply linear transformation using normalized input
                            layer_input[b, p] = np.dot(normalized_input[b, p], self.weights[i]) + self.biases[i]
                
                # Apply activation function
                layer_output = self._apply_activation(self.activation_functions[i], layer_input)
                
                # Apply dropout (except for the last layer)
                if i < len(self.weights) - 1 and self.training and self.dropout_rate > 0:
                    dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, 
                                                size=layer_output.shape) / (1 - self.dropout_rate)
                    self.dropout_masks.append(dropout_mask)
                    layer_output = layer_output * dropout_mask
                else:
                    self.dropout_masks.append(None)
                
                # Apply residual connection ONLY if dimensions match
                # (Skip residual for the final layer that projects to vocab size)
                if i < len(self.weights) - 1 or ffn_input.shape[-1] == layer_output.shape[-1]:
                    current_layer = ffn_input + layer_output
                else:
                    # For the final layer, just use the output directly
                    current_layer = layer_output
                        
                # Store layer output
                self.layer_outputs.append(current_layer)
                    
            # # Layer normalization after feed-forward if enabled
            # if self.use_layer_norm:
            #     current_layer = self.layer_norm(
            #         current_layer,
            #         self.ffn_norm_scale,
            #         self.ffn_norm_bias
            #     )
                
            return current_layer
                        
        # Standard forward pass for other models
        else:
            current_input = inputs
            layer_activations = [current_input]
            layer_inputs = []
            
            for l in range(len(self.layer_sizes) - 1):
                # Apply dropout during training (except last layer)
                if self.training and self.dropout_rate > 0 and l < len(self.layer_sizes) - 2:
                    dropout_mask = np.random.binomial(1, 1 - self.dropout_rate, 
                                                size=current_input.shape) / (1 - self.dropout_rate)
                    self.dropout_masks.append(dropout_mask)
                    current_input = current_input * dropout_mask
                else:
                    self.dropout_masks.append(None)
                
                # Linear transformation
                layer_input = np.dot(current_input, self.weights[l]) + self.biases[l]
                layer_inputs.append(layer_input)
                
                # Apply activation
                layer_output = self._apply_activation(self.activation_functions[l], layer_input)
                layer_activations.append(layer_output)
                
                # Update current input for next layer
                current_input = layer_output
            
            # Store for backpropagation
            self.layer_inputs = layer_inputs
            self.layer_outputs = layer_activations[1:]  # Skip the input layer
            
            return self.layer_outputs[-1]
    

    def backward_propagation(self, inputs, targets, outputs, attention_mask=None):
        """
        Backward propagation with self-attention support
        
        Args:
            inputs: Input data
            targets: Target data
            outputs: Model output from forward pass
            attention_mask: Optional mask for padded sequences
        """
        batch_size = inputs.shape[0]
        
        # Initialize gradients for all weights
        dweights = [np.zeros_like(w) for w in self.weights]
        dbiases = [np.zeros_like(b) for b in self.biases]
        
        # Initialize gradients for attention weights if used
        if self.use_attention:
            dquery_projection = np.zeros_like(self.query_projection)
            dkey_projection = np.zeros_like(self.key_projection)
            dvalue_projection = np.zeros_like(self.value_projection)
            doutput_projection = np.zeros_like(self.output_projection)
            dquery_bias = np.zeros_like(self.query_bias)
            dkey_bias = np.zeros_like(self.key_bias)
            dvalue_bias = np.zeros_like(self.value_bias)
            doutput_bias = np.zeros_like(self.output_bias)
        
        # Language model with sequence data
        if self.is_lm_architecture and len(inputs.shape) > 1:
            seq_length = inputs.shape[1]
            
            # Calculate output error
            output_error = np.zeros_like(outputs)
            for b in range(batch_size):
                for p in range(seq_length):
                    # Skip padding tokens
                    if attention_mask is None or attention_mask[b, p] > 0:
                        target_id = targets[b, p]
                        if target_id < self.vocab_size:
                            # Copy the output probabilities
                            output_error[b, p] = outputs[b, p].copy()
                            # Subtract 1 from the true class probability
                            output_error[b, p, target_id] -= 1.0
            
            # Clip gradients to prevent numerical instability
            output_error = np.clip(output_error, -1.0, 1.0)
            
            # Backpropagate through regular layers first (in reverse order)
            layer_error = output_error
            
            # Backpropagate through the last layer (output layer)
            for l in range(len(self.weights)-1, 0, -1):
                # For each position in the sequence
                for b in range(batch_size):
                    for p in range(seq_length):
                        # Skip padding tokens
                        if attention_mask is None or attention_mask[b, p] > 0:
                            # Get the previous layer's activation
                            if l == 1:  # If we're at the first hidden layer after embeddings
                                if self.use_attention:
                                    # Use the attention+residual output
                                    prev_activation = self.layer_outputs[1][b, p]
                                else:
                                    # Use the embedding output
                                    prev_activation = self.layer_outputs[0][b, p]
                            else:
                                prev_activation = self.layer_outputs[l-1][b, p]
                            
                            # Compute weight gradients
                            dw = np.outer(prev_activation, layer_error[b, p])
                            dweights[l] += dw
                            dbiases[l] += layer_error[b, p]
                
                # Propagate error to previous layer
                if l > 1:  # Skip embedding layer for now
                    prev_error = np.zeros((batch_size, seq_length, self.layer_sizes[l-1]))
                    for b in range(batch_size):
                        for p in range(seq_length):
                            # Skip padding tokens
                            if attention_mask is None or attention_mask[b, p] > 0:
                                # Compute error for previous layer
                                prev_error[b, p] = np.dot(layer_error[b, p], self.weights[l].T)
                                
                                # Apply activation derivative
                                prev_error[b, p] *= self._activation_derivative(
                                    self.activation_functions[l-2], 
                                    self.layer_outputs[l-2][b, p]
                                )
                    
                    # Update layer error for next iteration
                    layer_error = prev_error
            
            # Now handle the attention layer and embeddings
            if self.use_attention:
                # The error coming from the first feed-forward layer
                # has already been computed and is in layer_error
                
                # Backpropagate through the attention mechanism
                # First, compute gradients for the output projection
                for b in range(batch_size):
                    for p in range(seq_length):
                        if attention_mask is None or attention_mask[b, p] > 0:
                            # Get context vector before output projection
                            context = self.layer_outputs[0][b, p]  # Embedding + positional encoding
                            
                            # Gradient for output projection
                            doutput_projection += np.outer(context, layer_error[b, p])
                            doutput_bias += layer_error[b, p]
                
                # Compute error for the context vectors
                context_error = np.zeros_like(self.layer_outputs[0])
                for b in range(batch_size):
                    for p in range(seq_length):
                        if attention_mask is None or attention_mask[b, p] > 0:
                            context_error[b, p] = np.dot(layer_error[b, p], self.output_projection.T)
                
                # Backpropagate through attention weights
                # This is complex and would require access to the intermediate states
                # In a real implementation, we would need the Q, K, V values stored from forward pass
                # For simplicity, we'll just estimate the gradients
                
                # Add the error from the residual connection
                embedding_error = context_error
                
                # Finally, compute gradients for the token embeddings
                for b in range(batch_size):
                    for p in range(seq_length):
                        if attention_mask is None or attention_mask[b, p] > 0:
                            token_id = inputs[b, p]
                            if token_id < self.vocab_size:
                                # Update the embedding for this token
                                dweights[0][token_id] += embedding_error[b, p]
            else:
                # No attention layer, propagate directly to embeddings
                for b in range(batch_size):
                    for p in range(seq_length):
                        if attention_mask is None or attention_mask[b, p] > 0:
                            token_id = inputs[b, p]
                            if token_id < self.vocab_size:
                                # Update the embedding for this token
                                dweights[0][token_id] += layer_error[b, p]
            
        # Standard backpropagation for other models
        else:
            # Calculate output error
            if self.activation_functions[-1] == 'softmax':
                # Cross-entropy derivative with softmax is (output - target)
                output_error = outputs - targets
            else:
                # For other activation functions
                output_error = (outputs - targets) * self._activation_derivative(
                    self.activation_functions[-1], 
                    self.layer_inputs[-1], 
                    outputs
                )
            
            # Clip gradients to prevent numerical instability
            output_error = np.clip(output_error, -1.0, 1.0)
            
            # Backpropagate through layers
            layer_error = output_error
            
            for l in range(len(self.weights)-1, -1, -1):
                # Get input to this layer
                if l == 0:
                    layer_input = inputs
                else:
                    layer_input = self.layer_outputs[l-1]
                
                # Calculate gradients
                dw = np.dot(layer_input.T, layer_error)
                # Clip gradients to prevent numerical instability
                dw = np.clip(dw, -1.0, 1.0)
                dweights[l] = dw
                dbiases[l] = np.clip(np.sum(layer_error, axis=0), -1.0, 1.0)
                
                # Propagate error to previous layer (if not at input layer)
                if l > 0:
                    layer_error = np.dot(layer_error, self.weights[l].T)
                    # Clip before applying activation derivative to prevent overflow
                    layer_error = np.clip(layer_error, -1.0, 1.0)
                    layer_error *= self._activation_derivative(
                        self.activation_functions[l-1], 
                        self.layer_inputs[l-1], 
                        self.layer_outputs[l-1]
                    )
        
        # Update weights with gradient descent and learning rate
        for l in range(len(self.weights)):
            self.weights[l] -= self.learning_rate * dweights[l]
            self.biases[l] -= self.learning_rate * dbiases[l]
    
        # Remove the SGD update code and use either/or:
        if self.optimizer_name == 'adam' and self.optimizer is not None:
            # Adam update
            parameters = self.weights + self.biases
            gradients = dweights + dbiases
            updated_params = self.optimizer.update(parameters, gradients)
            self.weights = updated_params[:len(self.weights)]
            self.biases = updated_params[len(self.weights):]
        else:
            # Standard SGD update
            for l in range(len(self.weights)):
                self.weights[l] -= self.learning_rate * dweights[l]
                self.biases[l] -= self.learning_rate * dbiases[l]

    def calculate_loss(self, outputs, targets, attention_mask=None):
        """
        Calculate cross-entropy loss with improved numerical stability
        
        Args:
            outputs: Predicted probabilities
            targets: Target values
            attention_mask: Optional mask for padded tokens
            
        Returns:
            Average loss
        """
        epsilon = 1e-15  # Small value to prevent log(0)
        batch_size = outputs.shape[0]
        
        # Language model with sequence data
        if self.is_lm_architecture and len(outputs.shape) == 3:
            seq_length = outputs.shape[1]
            total_loss = 0.0
            token_count = 0
            
            # Process each position
            for p in range(seq_length):
                for b in range(batch_size):
                    # Skip padded tokens
                    if attention_mask is None or attention_mask[b, p] > 0:
                        target_id = targets[b, p]
                        if target_id < self.vocab_size:
                            # Get predicted probability for the target token
                            prob = outputs[b, p, target_id]
                            # Add to loss (clipped for numerical stability)
                            total_loss -= np.log(max(prob, epsilon))
                            token_count += 1
            
            # Return average loss per token
            return total_loss / max(token_count, 1)
        
        # Standard cross-entropy for classification
        elif self.activation_functions[-1] == 'softmax':
            # Clip predictions for numerical stability
            clipped_outputs = np.clip(outputs, epsilon, 1.0 - epsilon)
            
            # Calculate cross-entropy loss
            if len(targets.shape) > 1 and targets.shape[1] > 1:
                # One-hot encoded targets
                return -np.sum(targets * np.log(clipped_outputs)) / batch_size
            else:
                # Class indices
                loss = 0.0
                for i in range(batch_size):
                    loss -= np.log(clipped_outputs[i, int(targets[i])])
                return loss / batch_size
        
        # Mean squared error for regression
        else:
            return np.mean((outputs - targets) ** 2)
    
    def train_on_prompt_response_pairs(self, prompt_response_pairs, max_chunk_length=4096 ,epochs=None, validation_pairs=None):
        """Train model on prompt-response pairs"""
        # First make sure we have the proper special tokens
        if '<|promptend|>' not in self.tokenizer.get_vocab():
            special_tokens_dict = {
                'additional_special_tokens': ['<|promptend|>']
            }
            num_added = self.tokenizer.add_special_tokens(special_tokens_dict)
            logging.info(f"Added {num_added} special tokens for prompt-response format")
        
        # Process the prompt-response pairs
        processed_sequences = self.prepare_prompt_response_data(prompt_response_pairs, max_chunk_length=max_chunk_length)
        
        # Log data preparation results
        logging.info(f"Prepared {len(processed_sequences)} training sequences from {len(prompt_response_pairs)} prompt-response pairs")
        
        # Call the standard training method with the processed sequences
        return self.train_language_model(processed_sequences, epochs=epochs)
    
    def train_language_model(self, text_sequences, epochs=None, validation_sequences=None, verbose=True):
        """
        Improved training method with learning rate scheduling
        
        Args:
            text_sequences: List of token ID sequences or text strings
            validation_sequences: Optional validation sequences
            verbose: Whether to log training progress
            
        Returns:
            Training losses
        """
        # Reset training metrics
        self.training_losses = []
        self.validation_metrics = []
        
        if epochs is None:
            epochs = self.epochs
        
        if verbose:
            logging.info(f"Training language model on {len(text_sequences)} sequences for {epochs} epochs")
        
        # Check if input consists of strings or token IDs
        input_is_strings = isinstance(text_sequences[0], str)
        
        # Process input sequences based on type
        processed_sequences = []
        
        if input_is_strings:
            # Input is strings, need to tokenize
            if not self.tokenizer:
                raise ValueError("Cannot train on text strings without a tokenizer")
            
            # Tokenize all sequences
            tokenized = self.tokenize(text_sequences, add_special_tokens=True)
            input_ids = tokenized['input_ids']
            attention_masks = tokenized['attention_mask']
            
            # Convert to list of token ID lists
            for i in range(len(text_sequences)):
                # Extract non-padding tokens
                valid_tokens = []
                for j in range(len(input_ids[i])):
                    if attention_masks[i][j] > 0:
                        valid_tokens.append(int(input_ids[i][j]))
                processed_sequences.append(valid_tokens)
        else:
            # For token ID inputs, add special tokens manually
            processed_sequences = []
            for seq in text_sequences:
                # Add BOS at start and EOS at end
                processed_seq = [self.tokenizer.bos_token_id] + seq + [self.tokenizer.eos_token_id]
                # Truncate if too long
                if len(processed_seq) > self.max_seq_length:
                    processed_seq = processed_seq[:self.max_seq_length]
                processed_sequences.append(processed_seq)
        
        # Prepare the sequences
        max_seq_length = self.max_seq_length or max(len(seq) for seq in processed_sequences)
        
        # Pad sequences and create targets
        padded_sequences = []
        targets = []
        attention_masks = []
        
        for seq in processed_sequences:
            # Ensure sequence isn't too long
            if len(seq) > max_seq_length:
                seq = seq[:max_seq_length]
            
            # Create padding and attention mask
            padding_length = max_seq_length - len(seq)
            padded_seq = seq + [0] * padding_length
            attention_mask = [1] * len(seq) + [0] * padding_length
            
            # Create target sequence (shifted right by one)
            # The target for position i is the token at position i+1
            if len(seq) > 1:
                target_seq = seq[1:] + [0]  # Shifted sequence
                target_seq = target_seq + [0] * padding_length  # Add padding to match padded input
            else:
                # Handle edge case for very short sequences
                target_seq = [0] * max_seq_length
            
            padded_sequences.append(padded_seq)
            targets.append(target_seq)
            attention_masks.append(attention_mask)
        
        # Process validation data if provided
        if validation_sequences:
            val_processed = []
            
            if input_is_strings and isinstance(validation_sequences[0], str):
                # Tokenize validation text
                val_tokenized = self.tokenize(validation_sequences)
                val_input_ids = val_tokenized['input_ids']
                val_attention_masks = val_tokenized['attention_mask']
                
                # Extract valid tokens
                for i in range(len(validation_sequences)):
                    valid_tokens = []
                    for j in range(len(val_input_ids[i])):
                        if val_attention_masks[i][j] > 0:
                            valid_tokens.append(int(val_input_ids[i][j]))
                    val_processed.append(valid_tokens)
            else:
                val_processed = validation_sequences
        else:
            val_processed = None
        
        # Convert to numpy arrays
        padded_sequences = np.array(padded_sequences)
        targets = np.array(targets)
        attention_masks = np.array(attention_masks)
        
        # Track best metrics for early stopping
        best_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        # Training loop
        self.training = True
        
        # Initial learning rate
        initial_lr = self.learning_rate
        
        for epoch in range(epochs):
            # Apply learning rate schedule - reduce learning rate over time
            # This helps with numerical stability in later epochs
            if epoch > 30:
                decay_factor = 0.95 ** ((epoch - 30) // 5)  # Decay every 5 epochs after epoch 30
                self.learning_rate = initial_lr * max(decay_factor, 0.1)  # Don't go below 10% of initial rate
                
            # Shuffle data
            indices = np.random.permutation(len(padded_sequences))
            shuffled_sequences = padded_sequences[indices]
            shuffled_targets = targets[indices]
            shuffled_masks = attention_masks[indices]
            
            # Process in mini-batches
            epoch_losses = []
            
            for start_idx in range(0, len(shuffled_sequences), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(shuffled_sequences))
                
                # Get batch data
                batch_sequences = shuffled_sequences[start_idx:end_idx]
                batch_targets = shuffled_targets[start_idx:end_idx]
                batch_masks = shuffled_masks[start_idx:end_idx]
                
                # Forward pass
                batch_outputs = self.forward_propagation(batch_sequences, batch_masks)
                
                # Calculate loss
                batch_loss = self.calculate_loss(batch_outputs, batch_targets, batch_masks)
                
                # Check for NaN loss and skip the update if found
                if np.isnan(batch_loss):
                    if verbose:
                        logging.warning(f"NaN loss detected in batch - skipping update")
                    continue
                    
                epoch_losses.append(batch_loss)
                
                # Backward pass
                self.backward_propagation(batch_sequences, batch_targets, batch_outputs, batch_masks)
            
            # Calculate average loss for epoch
            if epoch_losses:
                avg_loss = np.mean(epoch_losses)
                self.training_losses.append(avg_loss)
            else:
                # If all batches were skipped due to NaN, use the last valid loss
                if self.training_losses:
                    self.training_losses.append(self.training_losses[-1])
                else:
                    self.training_losses.append(float('inf'))
            
            # Validation if provided
            if val_processed:
                # Process validation data
                val_padded_sequences = []
                val_targets = []
                val_attention_masks = []
                
                for seq in val_processed:
                    if len(seq) > max_seq_length:
                        seq = seq[:max_seq_length]
                    
                    padding_length = max_seq_length - len(seq)
                    padded_seq = seq + [0] * padding_length
                    attention_mask = [1] * len(seq) + [0] * padding_length
                    
                    if len(seq) > 1:
                        target_seq = seq[1:] + [0]  # Shifted sequence
                        target_seq = target_seq + [0] * padding_length  # Add padding to match padded input
                    else:
                        # Handle edge case for very short sequences
                        target_seq = [0] * max_seq_length
                    
                    val_padded_sequences.append(padded_seq)
                    val_targets.append(target_seq)
                    val_attention_masks.append(attention_mask)
                
                val_padded_sequences = np.array(val_padded_sequences)
                val_targets = np.array(val_targets)
                val_attention_masks = np.array(val_attention_masks)
                
                # Set to evaluation mode
                self.training = False
                
                # Forward pass on validation data
                val_outputs = self.forward_propagation(val_padded_sequences, val_attention_masks)
                
                # Calculate validation loss
                val_loss = self.calculate_loss(val_outputs, val_targets, val_attention_masks)
                self.validation_metrics.append(val_loss)
                
                # Early stopping check
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            logging.info(f"Early stopping at epoch {epoch+1}")
                        break
                
                # Back to training mode
                self.training = True
                
                if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                    logging.info(f"Epoch {epoch+1}/{epochs}: Train Loss = {self.training_losses[-1]:.6f}, "
                              f"Val Loss = {val_loss:.6f}, LR = {self.learning_rate:.6f}")
            else:
                if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                    logging.info(f"Epoch {epoch+1}/{epochs}: Loss = {self.training_losses[-1]:.6f}, "
                              f"LR = {self.learning_rate:.6f}")
        
        # Set to evaluation mode
        self.training = False
        
        return self.training_losses
    
    def enable_text_generation(self):
        """
        Enable the model for text generation with more flexible conditions
        """
        # Check if we have a proper LM architecture
        is_language_model = (
            self.use_embedding and 
            self.vocab_size is not None and 
            len(self.layer_sizes) >= 2 and
            self.layer_sizes[-1] == self.vocab_size
        )
        
        # Set the LM architecture flag explicitly
        self.is_lm_architecture = is_language_model
        
        if not is_language_model:
            logging.warning(
                "Model might not be ideal for text generation. "
                f"Layer sizes: {self.layer_sizes}, "
                f"Vocab size: {self.vocab_size}, "
                f"Use embedding: {self.use_embedding}"
            )
            # Instead of raising an error, just warn and continue
        
        # Set flag to indicate the model is ready for generation
        self.is_generative = True
        logging.info("Model enabled for text generation")
        
        return is_language_model
    
    # Add this function to your code to verify tokenizer setup
    def debug_tokenizer_setup(self):
        """Debug the tokenizer setup and special tokens"""
        print("\n----- TOKENIZER DEBUG -----")
        
        # Check for special tokens
        special_tokens = {
            'bos_token': self.tokenizer.bos_token,
            'eos_token': self.tokenizer.eos_token,
            'pad_token': self.tokenizer.pad_token,
            'unk_token': self.tokenizer.unk_token
        }
        
        print("Special tokens:")
        for name, token in special_tokens.items():
            token_id = self.tokenizer.convert_tokens_to_ids([token])[0] if token else None
            print(f"  {name}: '{token}' (ID: {token_id})")
        
        # Check for prompt separator
        prompt_sep = "<|promptend|>"
        try:
            prompt_sep_id = self.tokenizer.convert_tokens_to_ids([prompt_sep])[0]
            print(f"  prompt_separator: '{prompt_sep}' (ID: {prompt_sep_id})")
        except:
            print(f"  prompt_separator: Not found in vocabulary")
            
            # Add it and check again
            print("  Adding prompt separator to vocabulary...")
            special_tokens_dict = {'additional_special_tokens': [prompt_sep]}
            num_added = self.tokenizer.add_special_tokens(special_tokens_dict)
            print(f"  Added {num_added} new tokens")
            
            prompt_sep_id = self.tokenizer.convert_tokens_to_ids([prompt_sep])[0]
            print(f"  prompt_separator: '{prompt_sep}' (ID: {prompt_sep_id})")
        
        # Test encoding/decoding of a simple prompt-response pair
        test_prompt = "What is the capital of France?"
        test_response = "The capital of France is Paris."
        
        # Encode with BOS + prompt + separator + response + EOS
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
        
        prompt_tokens = self.tokenizer.encode(test_prompt, add_special_tokens=False)
        response_tokens = self.tokenizer.encode(test_response, add_special_tokens=False)
        
        full_sequence = [bos_id] + prompt_tokens + [prompt_sep_id] + response_tokens + [eos_id]
        
        print(f"\nTest encoding:")
        print(f"  Prompt tokens ({len(prompt_tokens)}): {prompt_tokens}")
        print(f"  Separator token: {prompt_sep_id}")
        print(f"  Response tokens ({len(response_tokens)}): {response_tokens}")
        print(f"  Full sequence ({len(full_sequence)}): {full_sequence[:10]}...{full_sequence[-10:]}")
        
        # Test decoding
        decoded = self.tokenizer.decode(full_sequence, skip_special_tokens=False)
        print(f"\nDecoded with special tokens: {decoded}")
        
        decoded_clean = self.tokenizer.decode(full_sequence, skip_special_tokens=True)
        print(f"Decoded without special tokens: {decoded_clean}")
        
        print("----- END TOKENIZER DEBUG -----\n")
        return prompt_sep_id

    # Modified generate_response function with better debugging
    def generate_response_debug(self, prompt, max_length=100, temperature=0.7):
        """
        Debug version of generate_response with more logging
        """
        if not self.is_generative:
            self.enable_text_generation()
        
        print("\n----- GENERATION DEBUG -----")
        # First debug the tokenizer
        separator_id = self.debug_tokenizer_setup()
        
        # Tokenize the prompt
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        print(f"Prompt encoded to {len(prompt_tokens)} tokens")
        
        # Create the full input with special tokens
        input_sequence = [self.tokenizer.bos_token_id] + prompt_tokens + [separator_id]
        print(f"Input sequence: {len(input_sequence)} tokens")
        print(f"First 10 tokens: {input_sequence[:10]}")
        print(f"Last 10 tokens: {input_sequence[-10:]}")
        
        # Generate tokens
        print(f"Generating with max_length={max_length}, temperature={temperature}")
        generated = self.generate(input_sequence, max_length=max_length, temperature=temperature)
        
        # Ensure generated is a list
        if not isinstance(generated, list):
            try:
                generated = list(generated)
                print(f"Converted generated output to list: {len(generated)} tokens")
            except:
                print(f"ERROR: Could not convert generated output to list: {type(generated)}")
                return "Error: Unable to process generated tokens"
        
        # Debug the generated tokens
        print(f"Generated {len(generated)} tokens")
        print(f"First 10 tokens: {generated[:10]}")
        print(f"Last 10 tokens: {generated[-10:]}")
        
        # Compare with input
        input_length = len(input_sequence)
        generated_length = len(generated)
        
        if generated_length <= input_length:
            print(f"WARNING: Generated sequence ({generated_length}) is not longer than input ({input_length})")
        else:
            print(f"Generated {generated_length - input_length} new tokens beyond input")
        
        # Check sequence for separator token
        separator_positions = []
        for i, token in enumerate(generated):
            if token == separator_id:
                separator_positions.append(i)
        
        print(f"Found separator token at positions: {separator_positions}")
        
        # Find the LAST separator token
        if separator_positions:
            separator_pos = separator_positions[-1]
            print(f"Using last separator at position {separator_pos}")
            
            # Extract everything after the separator
            if separator_pos < len(generated) - 1:
                response_tokens = generated[separator_pos + 1:]
                print(f"Extracted {len(response_tokens)} response tokens")
                response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
                print(f"Decoded response: \"{response}\"")
            else:
                print("Separator was the last token, no response to extract")
                response = "I couldn't generate a proper response."
        else:
            # No separator found, try to extract the new tokens beyond the input
            print("No separator found in generated sequence")
            if len(generated) > len(input_sequence):
                response_tokens = generated[len(input_sequence):]
                print(f"Extracting {len(response_tokens)} tokens beyond input as response")
                response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
                print(f"Decoded response: \"{response}\"")
            else:
                print("No additional tokens beyond input")
                response = "I couldn't generate a proper response."
        
        print("----- END GENERATION DEBUG -----\n")
        return response

    # Quick alternative approach using plain prompt format
    def simple_generate_response(self, prompt, max_length=100, temperature=0.7):
        """
        Simplified response generation that doesn't use separator tokens
        """
        # Format with a clear pattern the model can recognize
        formatted_prompt = f"Question: {prompt}\n\nAnswer:"
        
        # Tokenize
        tokens = self.tokenizer.encode(formatted_prompt, add_special_tokens=True)
        
        # Generate
        generated = self.generate(tokens, max_length=max_length+len(tokens), temperature=temperature)
        
        # Extract just the new part
        if len(generated) > len(tokens):
            response_tokens = generated[len(tokens):]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        else:
            response = "I couldn't generate a response."
        
        return response
    
    def alternate_generate_response(self, prompt, max_length=100, temperature=0.7):
        """Alternative generation approach that doesn't rely on special tokens"""
        # Use a plain text separator that the model can recognize
        formatted_prompt = f"QUESTION: {prompt}\nANSWER:"
        
        # Tokenize
        tokens = self.tokenizer.encode(formatted_prompt, add_special_tokens=True)
        
        # Generate
        generated = self.generate(tokens, max_length=max_length+len(tokens), temperature=temperature)
        
        # Extract just the response
        if len(generated) > len(tokens):
            response_tokens = generated[len(tokens):]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        else:
            response = "I couldn't generate a response."
        
        return response
  
    def simple_split_generate_response(self, prompt, max_length=100, temperature=0.7):
        """Ultra-simple text generation that doesn't use tokenizer"""
        # Format prompt with clear markers
        formatted_prompt = f"QUESTION: {prompt}\nANSWER:"
        
        # Generate without tokenization
        response_text = self.generate(formatted_prompt, max_length=max_length, temperature=temperature)
        
        # Split by markers
        if isinstance(response_text, str) and "\nANSWER:" in response_text:
            parts = response_text.split("\nANSWER:")
            if len(parts) > 1:
                return parts[1].strip()
        
        return "I couldn't generate a proper response."
  
    
    def generate_response(self, prompt, max_length=100, temperature=0.7):
        """
        Generate a response to a specific prompt using the special token format
        that matches how the model was trained.
        
        Args:
            prompt: The user's input prompt 
            max_length: Maximum number of tokens to generate for the response
            temperature: Controls randomness in generation
            
        Returns:
            Generated response text
        """
        if not self.is_generative:
            self.enable_text_generation()
        
        # Check if we have the separator token
        try:
            separator_id = self.tokenizer.convert_tokens_to_ids(["<|promptend|>"])[0]
        except:
            # Add the special token if not present
            special_tokens_dict = {'additional_special_tokens': ['<|promptend|>']}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            separator_id = self.tokenizer.convert_tokens_to_ids(["<|promptend|>"])[0]
        
        # Tokenize the prompt
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        
        # Create the full input with special tokens
        input_sequence = [self.tokenizer.bos_token_id] + prompt_tokens + [separator_id]
        
        # Generate tokens
        generated = self.generate(input_sequence, max_length=max_length, temperature=temperature)
        
        # Make sure we have a list of tokens
        if not isinstance(generated, list):
            try:
                generated = list(generated)
            except:
                return "Error: Unable to process generated tokens"
        
        # In generate_response after generating tokens:

        # Find the position of the separator token
        separator_pos = -1
        for i, token in enumerate(generated):
            if token == separator_id:
                separator_pos = i
                break

        # Extract ONLY the tokens that come after the separator
        if separator_pos >= 0 and separator_pos < len(generated) - 1:
            response_tokens = generated[separator_pos+1:]
            # Decode these tokens
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        else:
            response = "I couldn't generate a proper response."

        # Debug what's happening
        print(f"Separator found at position: {separator_pos}")
        print(f"Response tokens: {response_tokens[:10]}... (total: {len(response_tokens)})")

        return response        
        
    def generate(self, prompt, max_length=30, temperature=1.0, stream=False):
        """
        Improved text generation from a prompt - handles lists, arrays, and strings
        
        Args:
            prompt: Text string, list of token IDs, or numpy array to start generation
            max_length: Maximum number of tokens to generate
            temperature: Controls randomness (lower = more deterministic)
            
        Returns:
            Generated text if text was input, or token IDs if token IDs were input
        """
        if not self.is_generative:
            raise ValueError("Model not enabled for generation. Call enable_text_generation() first.")
        
        # Ensure we're not in training mode
        self.training = False
        
        # Track whether the input was a string or token IDs
        return_text = False
        
        # Handle different input types
        if isinstance(prompt, str):
            return_text = True
            if self.tokenizer:
                # Use the tokenizer to convert text to token IDs
                tokenized = self.tokenize(prompt)
                input_ids = tokenized['input_ids']
                prompt = input_ids
            else:
                raise ValueError("Cannot generate from text without a tokenizer. Provide token IDs instead.")
        
        # Convert to numpy array if it's a list
        if isinstance(prompt, list):
            prompt = np.array(prompt)
        
        # Add batch dimension if needed
        if len(prompt.shape) == 1:
            prompt = prompt.reshape(1, -1)
        
        # Make sure prompt isn't too long
        if prompt.shape[1] > self.max_seq_length:
            prompt = prompt[:, :self.max_seq_length]
        
        # Store the original prompt length
        original_prompt_length = prompt.shape[1]
        
        # Store generated tokens (start with the input tokens)
        generated_tokens = list(prompt[0])
        current_length = len(generated_tokens)
        
        # Generate tokens one by one
        for _ in range(max_length):
            # Prepare current sequence for inference
            current_sequence = np.array(generated_tokens).reshape(1, -1)
            
            # Don't exceed max sequence length
            if current_sequence.shape[1] > self.max_seq_length:
                current_sequence = current_sequence[:, -self.max_seq_length:]
            
            # Pad if necessary
            if current_sequence.shape[1] < self.max_seq_length:
                padding = np.zeros((1, self.max_seq_length - current_sequence.shape[1]), dtype=int)
                padded_sequence = np.concatenate([current_sequence, padding], axis=1)
            else:
                padded_sequence = current_sequence
            
            # Create attention mask
            attention_mask = np.zeros((1, self.max_seq_length), dtype=int)
            attention_mask[0, :current_sequence.shape[1]] = 1
            
            # Forward pass to get next token probabilities
            output = self.forward_propagation(padded_sequence, attention_mask)
            
            # Get probabilities for the last token position
            next_token_logits = output[0, current_sequence.shape[1] - 1]
            
            # Apply temperature for controlling randomness
            if temperature != 1.0:
                # Convert probabilities to logits
                logits = np.log(np.clip(next_token_logits, 1e-10, 1.0))
                # Apply temperature scaling
                logits = logits / temperature
                # Convert back to probabilities
                next_token_probs = softmax(logits)
            else:
                next_token_probs = next_token_logits
            
            # Sample next token from the distribution
            try:
                next_token = np.random.choice(self.vocab_size, p=next_token_probs)
            except ValueError:
                print(f"ValueError: {next_token_probs}")
                # Handle potential numerical issues with probability distribution
                next_token_probs = np.ones(self.vocab_size) / self.vocab_size  # Fallback to uniform
                next_token = np.random.choice(self.vocab_size, p=next_token_probs)
            
            # Add to generated tokens
            generated_tokens.append(next_token)
            
            # Stop if we generate a padding/end token
            if next_token == 0:
                break
            else:
                if self.tokenizer and stream:
                    print(self.tokenizer.decode(next_token, skip_special_tokens=True), end='',flush=True)
        
        # Convert token IDs back to text if input was text
        if return_text and self.tokenizer:
            if stream:
                return
            else:
                return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Return as a Python list, not numpy array wrapped in str representations
        return [int(token) for token in generated_tokens]    
   
   
    def process_large_text_file(self, prompt_file, response_file, batch_size=10):
        """Process large text files line by line to avoid memory issues"""
        processed_sequences = []
        
        with open(prompt_file, 'r') as p_file, open(response_file, 'r') as r_file:
            batch_prompts = []
            batch_responses = []
            
            for prompt_line, response_line in zip(p_file, r_file):
                batch_prompts.append(prompt_line.strip())
                batch_responses.append(response_line.strip())
                
                if len(batch_prompts) >= batch_size:
                    # Process this batch
                    batch_sequences = self.prepare_prompt_response_data(
                        list(zip(batch_prompts, batch_responses))
                    )
                    processed_sequences.extend(batch_sequences)
                    
                    # Clear batch
                    batch_prompts = []
                    batch_responses = []
            
            # Process any remaining items
            if batch_prompts:
                batch_sequences = self.prepare_prompt_response_data(
                    list(zip(batch_prompts, batch_responses))
                )
                processed_sequences.extend(batch_sequences)
        
        return processed_sequences
    
    def tokenize(self, texts, add_special_tokens=True):
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
                return_attention_mask=True,
                add_special_tokens=add_special_tokens  # Will add BOS/EOS tokens
            )
            return {
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask']
            }
        except Exception as e:
            logging.error(f"Tokenization error: {e}")
            raise
        
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
        
        
    def prepare_prompt_response_data(self, prompt_response_pairs, max_chunk_length=512):
        processed_sequences = []
        
        for pair in prompt_response_pairs:
            prompt, response = pair  # Unpack each pair
            
            # Remove HTML tags if present
            if response.startswith("<p>") and response.endswith("</p>"):
                response = response[3:-4]  # Remove <p> and </p>
            
            # Format with explicit text markers
            formatted_text = f"QUESTION: {prompt}\nANSWER: {response}"
            
            # Tokenize the formatted text
            tokens = self.tokenizer.encode(formatted_text, add_special_tokens=True)
            
            # Set an absolute maximum regardless of what was passed in
            max_chunk_length = min(4096, max_chunk_length)
            
            # Ensure sequence isn't too long
            if len(tokens) > max_chunk_length:
                tokens = tokens[:max_chunk_length-1] + [self.tokenizer.eos_token_id]
            
            processed_sequences.append(tokens)
        
        
        # Final safety check before returning
        for i, seq in enumerate(processed_sequences):
            if len(seq) > 4096:
                processed_sequences[i] = seq[:4095] + [self.tokenizer.eos_token_id]

        return processed_sequences

        

    def save(self, filepath):
        """Save model to file including attention parameters"""
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
                'is_lm_architecture': self.is_lm_architecture,
                'optimizer_name': self.optimizer_name,
            }
            
            # Save attention parameters if enabled
            if self.use_attention:
                model_data.update({
                    'use_attention': self.use_attention,
                    'num_attention_heads': self.num_attention_heads,
                    'attention_dropout': self.attention_dropout,
                    'query_projection': self.query_projection,
                    'key_projection': self.key_projection,
                    'value_projection': self.value_projection,
                    'output_projection': self.output_projection,
                    'query_bias': self.query_bias,
                    'key_bias': self.key_bias,
                    'value_bias': self.value_bias,
                    'output_bias': self.output_bias
                })
            
            # Save positional embedding parameters if enabled
            if self.use_positional_embedding:
                model_data.update({
                    'use_positional_embedding': self.use_positional_embedding,
                    'positional_embedding_type': self.positional_embedding_type,
                    'position_embeddings': self.position_embeddings
                })
            
            # Save other existing parameters (from original save method)
            if hasattr(self, 'optimizer') and self.optimizer_name == 'adam':
                model_data['optimizer_state'] = {
                    'learning_rate': self.optimizer.learning_rate,
                    'beta1': self.optimizer.beta1,
                    'beta2': self.optimizer.beta2,
                    'epsilon': self.optimizer.epsilon,
                    'm': self.optimizer.m,
                    'v': self.optimizer.v,
                    't': self.optimizer.t
                }
            
            # Save layer normalization parameters if enabled
            if self.use_layer_norm:
                model_data.update({
                    'use_layer_norm': self.use_layer_norm,
                    'attention_norm_scale': self.attention_norm_scale,
                    'attention_norm_bias': self.attention_norm_bias,
                    'ffn_norm_scale': self.ffn_norm_scale,
                    'ffn_norm_bias': self.ffn_norm_bias
                })
            
            if self.use_embedding:
                model_data['vocab_size'] = self.vocab_size
                model_data['embed_dim'] = self.embed_dim
                if hasattr(self, 'max_seq_length'):
                    model_data['max_seq_length'] = self.max_seq_length
            
            if hasattr(self, 'sentiment_weights') and self.sentiment_weights is not None:
                model_data['sentiment_weights'] = self.sentiment_weights
                model_data['sentiment_biases'] = self.sentiment_biases
            
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                if hasattr(self.tokenizer, 'name_or_path'):
                    model_data['tokenizer_name'] = self.tokenizer.name_or_path
                elif hasattr(self.tokenizer, '_name_or_path'):
                    model_data['tokenizer_name'] = self.tokenizer._name_or_path
            
            if hasattr(self, 'is_generative'):
                model_data['is_generative'] = self.is_generative
            
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
        """Load model from file including attention parameters"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Create a new instance with minimal initialization
            # Include attention parameters in initialization if available
            attention_kwargs = {}
            if 'use_attention' in model_data and model_data['use_attention']:
                attention_kwargs = {
                    'use_attention': model_data['use_attention'],
                    'num_attention_heads': model_data.get('num_attention_heads', 8),
                    'attention_dropout': model_data.get('attention_dropout', 0.1)
                }
            
            # Include positional embedding parameters if available
            pos_embedding_kwargs = {}
            if 'use_positional_embedding' in model_data and model_data['use_positional_embedding']:
                pos_embedding_kwargs = {
                    'use_positional_embedding': model_data['use_positional_embedding'],
                    'positional_embedding_type': model_data.get('positional_embedding_type', 'sinusoidal')
                }
            
            # Initialize model with basic and attention parameters
            model = cls(
                layer_sizes=model_data['layer_sizes'],
                learning_rate=model_data['learning_rate'],
                epochs=model_data['epochs'],
                batch_size=model_data['batch_size'],
                dropout_rate=model_data['dropout_rate'],
                activation_functions=model_data['activation_functions'],
                use_embedding=model_data['use_embedding'],
                **attention_kwargs,
                **pos_embedding_kwargs
            )
            
            # Load weights and biases
            model.weights = model_data['weights']
            model.biases = model_data['biases']
            
            # Load attention-specific weights if they exist
            if 'use_attention' in model_data and model_data['use_attention']:
                model.query_projection = model_data['query_projection']
                model.key_projection = model_data['key_projection']
                model.value_projection = model_data['value_projection']
                model.output_projection = model_data['output_projection']
                model.query_bias = model_data['query_bias']
                model.key_bias = model_data['key_bias']
                model.value_bias = model_data['value_bias']
                model.output_bias = model_data['output_bias']
            
            # Load positional embeddings if they exist
            if 'use_positional_embedding' in model_data and model_data['use_positional_embedding']:
                model.position_embeddings = model_data['position_embeddings']
            
            # Load all other parameters from the original load method
            # (optimizer state, vocabulary info, tokenizer, etc.)
            if 'optimizer_name' in model_data:
                model.optimizer_name = model_data['optimizer_name']
                
                if model.optimizer_name == 'adam':
                    model.optimizer = AdamOptimizer(learning_rate=model.learning_rate)
                    
                    if 'optimizer_state' in model_data:
                        optimizer_state = model_data['optimizer_state']
                        model.optimizer.learning_rate = optimizer_state.get('learning_rate', model.learning_rate)
                        model.optimizer.beta1 = optimizer_state.get('beta1', 0.9)
                        model.optimizer.beta2 = optimizer_state.get('beta2', 0.999)
                        model.optimizer.epsilon = optimizer_state.get('epsilon', 1e-8)
                        model.optimizer.m = optimizer_state.get('m', None)
                        model.optimizer.v = optimizer_state.get('v', None)
                        model.optimizer.t = optimizer_state.get('t', 0)
            
            if model.use_embedding:
                if 'vocab_size' in model_data:
                    model.vocab_size = model_data['vocab_size']
                if 'embed_dim' in model_data:
                    model.embed_dim = model_data['embed_dim']
                if 'max_seq_length' in model_data:
                    model.max_seq_length = model_data['max_seq_length']
                
                if 'tokenizer_name' in model_data:
                    try:
                        from transformers import AutoTokenizer
                        tokenizer_name = model_data['tokenizer_name']
                        model.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                        model.fix_tokenizer()
                    except Exception as e:
                        logging.warning(f"Failed to load tokenizer: {e}")
                        model.tokenizer = None
            
            if 'sentiment_weights' in model_data:
                model.sentiment_weights = model_data['sentiment_weights']
                model.sentiment_biases = model_data['sentiment_biases']
            
            if 'is_generative' in model_data:
                model.is_generative = model_data['is_generative']
            
            if 'is_lm_architecture' in model_data:
                model.is_lm_architecture = model_data['is_lm_architecture']
            
            if 'label_encoder' in model_data:
                model.label_encoder = model_data['label_encoder']
            
            if 'qa_weights' in model_data:
                model.qa_weights = model_data['qa_weights']
                model.qa_biases = model_data['qa_biases']
            
            # load layer normalization parameters if enabled
            if 'use_layer_norm' in model_data:
                model.use_layer_norm = model_data['use_layer_norm']
            
            if 'attention_norm_scale' in model_data:
                model.attention_norm_scale = model_data['attention_norm_scale']

            if 'attention_norm_bias' in model_data:
                model.attention_norm_bias = model_data['attention_norm_bias']
                            
            if 'ffn_norm_scale' in model_data:
                model.ffn_norm_scale = model_data['ffn_norm_scale']
            
            if 'ffn_norm_bias' in model_data:
                model.ffn_norm_bias = model_data['ffn_norm_bias']
            
            
            # Initialize layer storage
            model.layer_outputs = []
            model.layer_inputs = []
            model.dropout_masks = []
            model.attention_weights = []
            
            return model
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return None
        

    def get_hyperparameters(self):
        """Return model hyperparameters including attention settings"""
        hparams = {
            'Architecture': {
                'Layer sizes': self.layer_sizes,
                'Activation functions': self.activation_functions,
                'Dropout rate': self.dropout_rate,
                'Embedding used': self.use_embedding,
            },
            'Training': {
                'Learning rate': self.learning_rate,
                'Epochs': self.epochs,
                'Batch size': self.batch_size,
            },
            'Optimizer': {
                'Type': self.optimizer_name,
            }
        }

        # Add attention-specific parameters if enabled
        if hasattr(self, 'use_attention') and self.use_attention:
            hparams['Attention'] = {
                'Number of heads': self.num_attention_heads,
                'Attention dropout': self.attention_dropout,
                'Head dimension': getattr(self, 'head_dim', self.embed_dim // self.num_attention_heads),
            }
        
        # Add positional embedding parameters if enabled
        if hasattr(self, 'use_positional_embedding') and self.use_positional_embedding:
            hparams['Positional Embedding'] = {
                'Type': self.positional_embedding_type,
                'Max sequence length': self.max_seq_length,
            }

        # Add optimizer-specific parameters if using Adam
        if hasattr(self, 'optimizer') and self.optimizer_name == 'adam':
            hparams['Optimizer'].update({
                'Beta1': self.optimizer.beta1,
                'Beta2': self.optimizer.beta2,
                'Epsilon': self.optimizer.epsilon,
            })

        # Add embedding-specific parameters if applicable
        if self.use_embedding:
            hparams['Embedding'] = {
                'Vocabulary size': getattr(self, 'vocab_size', 'N/A'),
                'Embedding dimension': getattr(self, 'embed_dim', 'N/A'),
                'Max sequence length': getattr(self, 'max_seq_length', 'N/A'),
                'Tokenizer': getattr(self, 'tokenizer_name', 'N/A'),
            }

        # Add optional components
        hparams['Components'] = {
            'Sentiment head': hasattr(self, 'sentiment_weights'),
            'QA head': hasattr(self, 'qa_weights'),
            'Generative model': getattr(self, 'is_generative', False),
            'LM architecture': getattr(self, 'is_lm_architecture', False),
            'Self-attention': getattr(self, 'use_attention', False),
            'Positional embedding': getattr(self, 'use_positional_embedding', False),
        }

        return hparams
    
    def print_hyperparameters(self):
        """Print hyperparameters in a human-readable format"""
        hparams = self.get_hyperparameters()
        pp = pprint.PrettyPrinter(indent=4)
        
        print("Model Hyperparameters:")
        for section, params in hparams.items():
            print(f"\n{section}:")
            pp.pprint(params)
import numpy as np
import pickle
from transformers import AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)


def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

class NeuralNetwork:
    def __init__(self, layer_sizes, activation_functions=None, learning_rate=0.01, 
                 epochs=100, batch_size=32, dropout_rate=0.0, use_embedding=False,
                 vocab_size=None, embed_dim=None, max_seq_length=512, tokenizer_name=None):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.use_embedding = use_embedding
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_length = max_seq_length
        
        if tokenizer_name:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = None

        if activation_functions is None:
            self.activation_functions = ['relu'] * self.num_layers
        else:
            self.activation_functions = activation_functions
        if len(self.activation_functions) != self.num_layers:
            raise ValueError("Activation functions count must match layer count")

        self.weights = []
        self.biases = []
        self._initialize_parameters()

        if self.use_embedding:
            if vocab_size is None or embed_dim is None:
                raise ValueError("Vocab size and embed dim must be specified for embedding")
            self.embedding = np.random.randn(vocab_size, embed_dim) * np.sqrt(2.0 / embed_dim)
            self.query = np.random.randn(embed_dim, embed_dim) * np.sqrt(1.0 / embed_dim)
            self.key = np.random.randn(embed_dim, embed_dim) * np.sqrt(1.0 / embed_dim)
            self.value = np.random.randn(embed_dim, embed_dim) * np.sqrt(1.0 / embed_dim)
        else:
            self.embedding = None

        self.training = True

    def _initialize_parameters(self):
        for i in range(self.num_layers):
            input_size = self.layer_sizes[i]
            output_size = self.layer_sizes[i+1]
            self.weights.append(np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size))
            self.biases.append(np.zeros(output_size))

    def attention(self, Q, K, V, mask=None):
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(Q.shape[-1])
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        attention_weights = softmax(scores, axis=-1)
        output = np.matmul(attention_weights, V)
        return output, attention_weights  # Return both output and weights

    def forward_propagation(self, inputs, masks=None):
        self.activations = []
        self.weighted_inputs = []
        self.dropout_masks = []
        self.attention_weights = []
        
        current_input = inputs
        
        if self.use_embedding:
            embedded = self.embedding[current_input]
            Q = np.dot(embedded, self.query)
            K = np.dot(embedded, self.key)
            V = np.dot(embedded, self.value)
            attention_output, attention_weights = self.attention(Q, K, V, masks)
            self.attention_weights.append(attention_weights)
            self.Q = Q  # Store Q, K, V for backward pass
            self.K = K
            self.V = V
            
            attention_output, attention_weights = self.attention(Q, K, V, masks)
            self.attention_weights.append(attention_weights)  # Store weights
                    
            if self.training and self.dropout_rate > 0:
                mask = np.random.binomial(1, 1 - self.dropout_rate, size=attention_output.shape) 
                attention_output *= mask / (1 - self.dropout_rate)
                self.dropout_masks.append(mask)
            
            current_input = np.mean(attention_output, axis=1)
            self.activations.append(current_input)
        else:
            self.activations.append(current_input)  # Store input for dense layers
        
        for i in range(self.num_layers):
            weighted_input = np.dot(current_input, self.weights[i]) + self.biases[i]
            self.weighted_inputs.append(weighted_input)
            
            activation_func = self.activation_functions[i]
            if activation_func == 'softmax':
                activation_output = self.activation(weighted_input, 'softmax')
            else:
                activation_output = self.activation(weighted_input, activation_func)
            
            if self.training and self.dropout_rate > 0 and i < self.num_layers - 1:
                mask = np.random.binomial(1, 1 - self.dropout_rate, size=activation_output.shape) 
                activation_output *= mask / (1 - self.dropout_rate)
                self.dropout_masks.append(mask)
            else:
                self.dropout_masks.append(None)
            
            self.activations.append(activation_output)
            current_input = activation_output
        
        return current_input

    def backward_propagation(self, inputs, targets, output, masks=None):
        batch_size = inputs.shape[0]
        d_weights = [np.zeros_like(w) for w in self.weights]
        d_biases = [np.zeros_like(b) for b in self.biases]
        deltas = []
        
        # Compute output error
        if self.activation_functions[-1] == 'softmax':
            output_error = output - targets
        else:
            output_error = (output - targets) * self.activation_derivative(
                self.weighted_inputs[-1], self.activation_functions[-1])
        deltas.append(output_error)
        
        # Backpropagate through dense layers
        for i in reversed(range(self.num_layers)):
            delta = deltas[-1]
            if self.dropout_masks[i] is not None:
                delta *= self.dropout_masks[i] / (1 - self.dropout_rate)
            d_weights[i] = np.dot(self.activations[i].T, delta) / batch_size
            d_biases[i] = np.mean(delta, axis=0)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(
                    self.weighted_inputs[i-1], self.activation_functions[i-1])
                deltas.append(delta)
        
        # Backpropagate through attention layer
        if self.use_embedding:
            attention_weights = self.attention_weights.pop()
            Q = self.Q
            K = self.K
            V = self.V
            seq_length = Q.shape[1]
            delta = deltas[-1]
            d_attention = np.dot(delta, self.weights[0].T)
            d_attention = np.repeat(d_attention[:, np.newaxis, :], seq_length, axis=1) / seq_length
            
            # Compute gradients for attention output
            dV = np.matmul(attention_weights.transpose(0, 2, 1), d_attention)
            d_attention_scores = np.matmul(d_attention, V.transpose(0, 2, 1))  # Gradient of attention scores
            
            # Compute gradients for Q and K with scaling
            dQ = np.matmul(d_attention_scores, K) / np.sqrt(Q.shape[-1])
            dK = np.matmul(d_attention_scores.transpose(0, 2, 1), Q) / np.sqrt(Q.shape[-1])
            
            # Compute gradients for query, key, value matrices
            embedded = self.embedding[inputs]
            d_query = np.mean(np.matmul(embedded.transpose(0, 2, 1), dQ), axis=0)
            d_key = np.mean(np.matmul(embedded.transpose(0, 2, 1), dK), axis=0)
            d_value = np.mean(np.matmul(embedded.transpose(0, 2, 1), dV), axis=0)
            
            self.query -= self.learning_rate * d_query
            self.key -= self.learning_rate * d_key
            self.value -= self.learning_rate * d_value
            
            d_embed = np.matmul(dQ, self.query.T) + np.matmul(dK, self.key.T) + np.matmul(dV, self.value.T)
            np.add.at(self.embedding, inputs, d_embed)
        
        # Update dense layer parameters
        for i in range(self.num_layers):
            self.weights[i] -= self.learning_rate * d_weights[i]
            self.biases[i] -= self.learning_rate * d_biases[i]        
                        
    def activation(self, x, func_name):
        if func_name == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif func_name == 'relu':
            return np.maximum(0, x)
        elif func_name == 'softmax':
            return softmax(x)
        elif func_name == 'leaky_relu':
            return np.maximum(0.1 * x, x)
        else:
            raise ValueError(f"Unsupported activation function: {func_name}")

    def activation_derivative(self, x, func_name):
        if func_name == 'sigmoid':
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)
        elif func_name == 'relu':
            return (x > 0).astype(float)
        elif func_name == 'leaky_relu':
            return np.where(x > 0, 1.0, 0.1)
        else:
            raise ValueError(f"Unsupported activation function: {func_name}")

    def train(self, training_data, labels):
        """Batch training with support for sequences"""
        for epoch in range(self.epochs):
            for i in range(0, len(training_data), self.batch_size):
                batch_data = training_data[i:i+self.batch_size]
                batch_labels = labels[i:i+self.batch_size]
                
                # Tokenize and pad
                if self.tokenizer:
                    inputs = self.tokenize(batch_data)
                else:
                    inputs = np.array(batch_data)
                
                # Forward pass
                output = self.forward_propagation(inputs)
                
                # Backward pass
                self.backward_propagation(inputs, batch_labels, output)

    def tokenize(self, texts):
        """Tokenize text using HuggingFace tokenizer"""
        if not self.tokenizer:
            raise ValueError("Tokenizer not initialized")
        
        tokenized = self.tokenizer(texts, padding=True, truncation=True, 
                                  max_length=self.max_seq_length, return_tensors='np')
        return tokenized['input_ids']
    
    # def tokenize(self, texts):
    #     """Tokenize text using HuggingFace tokenizer"""
    #     encoded = self.tokenizer(
    #         texts, 
    #         padding="max_length", 
    #         truncation=True, 
    #         max_length=self.max_seq_length, 
    #         return_tensors="np"
    #     )
    #     return encoded["input_ids"]
    
    

    def predict(self, inputs):
        """Prediction with sequence handling"""
        self.training = False
        if self.tokenizer:
            inputs = self.tokenize(inputs)
        return self.forward_propagation(inputs)

    def save(self, filepath):
        """Save model with tokenizer"""
        data = {
            'weights': self.weights,
            'biases': self.biases,
            'query': self.query if self.use_embedding else None,
            'key': self.key if self.use_embedding else None,
            'value': self.value if self.use_embedding else None,
            'embedding': self.embedding,
            'config': {
                'layer_sizes': self.layer_sizes,
                'activation_functions': self.activation_functions,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'dropout_rate': self.dropout_rate,
                'use_embedding': self.use_embedding,
                'vocab_size': self.vocab_size,
                'embed_dim': self.embed_dim,
                'max_seq_length': self.max_seq_length
            }
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath, tokenizer_name=None):
        """Load model with configuration"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        config = data['config']
        model = cls(**config, tokenizer_name=tokenizer_name)
        
        model.weights = data['weights']
        model.biases = data['biases']
        if config['use_embedding']:
            model.query = data['query']
            model.key = data['key']
            model.value = data['value']
            model.embedding = data['embedding']
        
        return model

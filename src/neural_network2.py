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
    def __init__(self, layer_sizes, activation_functions=None, learning_rate=0.001, 
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
        self.tokenizer_name = tokenizer_name
        
        if tokenizer_name:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                logging.info(f"Loaded tokenizer: {tokenizer_name}")
            except Exception as e:
                logging.error(f"Failed to load tokenizer: {e}")
                self.tokenizer = None
        else:
            self.tokenizer = None
            
        if activation_functions is None:
            self.activation_functions = ['relu'] * (self.num_layers - 1) + ['softmax']
        else:
            self.activation_functions = activation_functions
            
        if len(self.activation_functions) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} activation functions, got {len(self.activation_functions)}")
            
        self.weights = []
        self.biases = []
        self._initialize_parameters()
        
        if self.use_embedding:
            if vocab_size is None or embed_dim is None:
                raise ValueError("Vocab size and embed dim must be specified for embedding")
                
            # Simple embedding matrix
            self.embedding = np.random.randn(vocab_size, embed_dim) * np.sqrt(2.0 / (vocab_size + embed_dim))
                
        self.training = True
        self.label_encoder = None
        
        # For generation task
        self.is_generative = False
        
        # Keep track of losses and metrics for monitoring
        self.training_losses = []
        self.validation_metrics = []

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

    def forward_propagation(self, inputs, masks=None):
        """Forward pass through the network with simplified embedding handling"""
        self.activations = []
        self.weighted_inputs = []
        self.dropout_masks = []
        
        # Handle embeddings for text data
        if self.use_embedding and isinstance(inputs, np.ndarray) and np.issubdtype(inputs.dtype, np.integer):
            # Ensure inputs are integers and within vocab range
            inputs = np.clip(inputs, 0, self.vocab_size-1)
            
            # Apply embeddings
            embedded = self.embedding[inputs]  # Shape: (batch_size, seq_len, embed_dim)
            
            # Simple pooling to get fixed-size representation (use mean pooling)
            if masks is not None:
                # Create expanded mask for proper broadcasting
                expanded_masks = np.expand_dims(masks, -1)  # Shape: (batch_size, seq_len, 1)
                
                # Apply mask to zero out padding tokens
                masked_embedded = embedded * expanded_masks
                
                # Sum and normalize by sequence length
                seq_lengths = np.sum(masks, axis=1, keepdims=True)
                seq_lengths = np.clip(seq_lengths, 1, None)  # Avoid division by zero
                current_input = np.sum(masked_embedded, axis=1) / seq_lengths
            else:
                # Simple mean pooling if no masks
                current_input = np.mean(embedded, axis=1)
        else:
            # For non-text data, use inputs directly
            current_input = inputs
            
        self.activations.append(current_input)
        
        # Forward pass through dense layers
        for i in range(self.num_layers):
            weighted_input = np.dot(current_input, self.weights[i]) + self.biases[i]
            self.weighted_inputs.append(weighted_input)
            
            # Apply activation function
            if self.activation_functions[i] == 'softmax':
                activation_output = softmax(weighted_input)
            else:
                activation_output = self.activation(weighted_input, self.activation_functions[i])
                
            # Apply dropout except for the output layer
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
        """Backward pass to update weights and biases with fixed broadcasting issue"""
        batch_size = len(inputs) if isinstance(inputs, list) else inputs.shape[0]
        d_weights = [np.zeros_like(w) for w in self.weights]
        d_biases = [np.zeros_like(b) for b in self.biases]
        deltas = []

        # Calculate error based on output activation
        if self.activation_functions[-1] == 'softmax':
            # Cross-entropy derivative is (output - target) for softmax
            output_error = output - targets
        else:
            # For other activation functions, use derivative
            output_error = (output - targets) * self.activation_derivative(
                self.weighted_inputs[-1], self.activation_functions[-1])
            
        deltas.append(output_error)
        
        # Backpropagate through dense layers
        for i in reversed(range(self.num_layers)):
            delta = deltas[-1]
            
            # Apply dropout mask if applicable
            if i < len(self.dropout_masks) and self.dropout_masks[i] is not None:
                delta *= self.dropout_masks[i] / (1 - self.dropout_rate)
                
            # Calculate weight and bias gradients
            d_weights[i] = np.dot(self.activations[i].T, delta) / batch_size
            d_biases[i] = np.mean(delta, axis=0)
            
            # Calculate error for previous layer if not at input layer
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(
                    self.weighted_inputs[i-1], self.activation_functions[i-1])
                deltas.append(delta)

        # Update embedding layer if used - FIXED: Create appropriate embedding error
        if self.use_embedding and isinstance(inputs, np.ndarray) and np.issubdtype(inputs.dtype, np.integer):
            # Create a placeholder for embedding gradients
            delta_embedding = np.zeros_like(self.embedding)  # Shape: (vocab_size, embed_dim)
            
            # We need to adapt to the mismatch here. The last delta is for the first hidden layer,
            # but we need to transform it to match the embedding dimension
            # Since we're using the model with layer_sizes=[768, 128, 3], the delta is (batch_size, 128)
            
            # We need to transform it back to the embedding dimension (768)
            if len(deltas) > 0 and deltas[-1].shape[1] != self.embed_dim:
                # Get the error at the first hidden layer (which has shape batch_size x hidden_size)
                hidden_delta = deltas[-1]  # Shape: (batch_size, hidden_size)
                
                # We need to project this error back to the embedding space
                # This is a simple approach - using the first layer weights to project
                # If the hidden size is smaller than embed_dim, we need to map it back
                # For demonstration, we'll use a simple linear projection
                # logging.info(f"Projecting delta from shape {hidden_delta.shape} back to embedding dimension {self.embed_dim}")
                                
                # Only update embeddings used in this batch
                if masks is not None:
                    # Use masks to determine which tokens were actually used (not padding)
                    for b in range(batch_size):
                        # Create a projected delta for this example
                        # We'll use a simple method: repeat the hidden delta values to fill the embedding dimension
                        hidden_size = hidden_delta.shape[1]
                        repeat_factor = self.embed_dim // hidden_size + 1
                        example_delta = np.tile(hidden_delta[b], repeat_factor)[:self.embed_dim]
                        
                        # Count valid tokens (non-padding) in this example
                        valid_tokens = np.sum(masks[b])
                        if valid_tokens == 0:  # Skip if all tokens were padding
                            continue
                            
                        # Distribute error evenly across all valid token positions
                        token_delta = example_delta / valid_tokens  # Shape: (embed_dim,)
                        
                        # Update embeddings for each token in this example
                        for t in range(inputs.shape[1]):
                            if masks[b, t] == 0:  # Skip padding tokens
                                continue
                                
                            token_id = inputs[b, t]
                            if token_id < self.vocab_size:
                                # Add the distributed error to this token's embedding
                                delta_embedding[token_id] += token_delta
                else:
                    # No masks, assume all tokens are valid
                    for b in range(batch_size):
                        # Create a projected delta for this example
                        hidden_size = hidden_delta.shape[1]
                        repeat_factor = self.embed_dim // hidden_size + 1
                        example_delta = np.tile(hidden_delta[b], repeat_factor)[:self.embed_dim]
                        
                        # Distribute error evenly across all token positions
                        token_delta = example_delta / inputs.shape[1]  # Shape: (embed_dim,)
                        
                        # Update embeddings for each token in this example
                        for t in range(inputs.shape[1]):
                            token_id = inputs[b, t]
                            if token_id < self.vocab_size:
                                # Add the distributed error to this token's embedding
                                delta_embedding[token_id] += token_delta
            
            # Update embeddings with gradient clipping
            np.clip(delta_embedding, -1.0, 1.0, out=delta_embedding)
            self.embedding -= self.learning_rate * delta_embedding / batch_size

        # Update dense layer weights and biases with gradient clipping
        for i in range(self.num_layers):
            np.clip(d_weights[i], -1.0, 1.0, out=d_weights[i])
            np.clip(d_biases[i], -1.0, 1.0, out=d_biases[i])
            
            self.weights[i] -= self.learning_rate * d_weights[i]
            self.biases[i] -= self.learning_rate * d_biases[i]

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
            logits = self.forward_propagation(current_input_ids, current_mask)
            
            # Apply temperature
            next_token_logits = logits[0]
            if temperature != 1.0:
                next_token_logits = next_token_logits / max(temperature, 1e-8)
            
            # Convert to probabilities
            next_token_probs = softmax(next_token_logits)
            
            # Sample next token
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
        
    def predict(self, inputs, return_probs=False):
        """Make predictions on new data"""
        self.training = False
        
        # Debug logging to trace the flow
        logging.info(f"Input type: {type(inputs)}")
        
        # Handle different input types
        if isinstance(inputs, str):
            inputs = [inputs]
            logging.info("Converted string input to list")
        elif isinstance(inputs, np.ndarray) and inputs.dtype.kind in ('U', 'S'):
            inputs = inputs.tolist()
            logging.info("Converted numpy string array to list")
            
        # For text data, we need to tokenize and get embeddings
        if self.use_embedding:
            logging.info(f"Use embedding is True, checking if input is text list")
            
            if isinstance(inputs, list) and all(isinstance(item, str) for item in inputs):
                logging.info("Input confirmed as text list, proceeding with tokenization")
                
                # Check if we have a tokenizer
                if not self.tokenizer:
                    logging.error("No tokenizer available for text processing")
                    raise ValueError("Tokenizer is required for text prediction but not available")
                    
                # Tokenize the inputs
                try:
                    logging.info(f"Tokenizing input: {inputs[:50]}...")  # Log first part of input
                    tokenized = self.tokenize(inputs)
                    
                    if tokenized is None:
                        logging.error("Tokenization returned None")
                        raise ValueError("Tokenization failed")
                        
                    inputs = tokenized['input_ids']
                    masks = tokenized['attention_mask']
                    
                    logging.info(f"Tokenized input shape: {inputs.shape}")
                    logging.info(f"Attention mask shape: {masks.shape}")
                    
                    predictions = self.forward_propagation(inputs, masks)
                    logging.info(f"Predictions shape after forward pass: {predictions.shape}")
                    
                except Exception as e:
                    logging.error(f"Error in text prediction process: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
            else:
                logging.warning("Input is not a text list but use_embedding is True. Treating as numeric input.")
                if not isinstance(inputs, np.ndarray):
                    inputs = np.array(inputs)
                logging.info(f"Numeric input shape: {inputs.shape}")
                predictions = self.forward_propagation(inputs)
        else:
            # For non-text data
            logging.info("Processing as non-text data (use_embedding is False)")
            if not isinstance(inputs, np.ndarray):
                inputs = np.array(inputs)
                
            logging.info(f"Non-text input shape: {inputs.shape}")
            
            # Ensure input has the right shape for the first layer
            expected_dim = self.layer_sizes[0]
            if len(inputs.shape) == 1 and inputs.shape[0] != expected_dim:
                logging.error(f"Dimension mismatch: got {inputs.shape}, expected first dimension to be {expected_dim}")
                raise ValueError(f"Input dimension mismatch. Expected {expected_dim}, got {inputs.shape[0]}")
                
            predictions = self.forward_propagation(inputs)
        
        # Return probabilities if requested
        if return_probs:
            return predictions
            
        # Convert class probabilities to class labels if using a label encoder
        if self.label_encoder is not None and len(predictions.shape) > 1 and predictions.shape[1] > 1:
            # For classification problems
            class_indices = np.argmax(predictions, axis=1)
            return self.label_encoder.inverse_transform(class_indices)
        
        return predictions
    
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
        """Save the model to a file using pickle with improved attribute handling"""
        # Create a dictionary with all the model components and parameters
        model_dict = {
            # Architecture parameters - critical for recreating the model
            'layer_sizes': self.layer_sizes,
            'activation_functions': self.activation_functions,
            
            # Training parameters
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'dropout_rate': self.dropout_rate,
            
            # Embedding parameters - critical for text processing
            'use_embedding': self.use_embedding,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'max_seq_length': self.max_seq_length,
            'tokenizer_name': self.tokenizer_name,
            
            # Model weights and parameters
            'weights': self.weights,
            'biases': self.biases,
            'embedding': self.embedding if self.use_embedding else None,
            'label_encoder': self.label_encoder,
            'is_generative': self.is_generative,
            
            # Training history
            'training_losses': self.training_losses,
            'validation_metrics': self.validation_metrics
        }
        
        # Log what we're saving
        logging.info(f"Saving model with parameters: layer_sizes={self.layer_sizes}, use_embedding={self.use_embedding}")
        
        # Save tokenizer separately if it exists
        if self.tokenizer:
            tokenizer_path = filepath + "_tokenizer"
            try:
                self.tokenizer.save_pretrained(tokenizer_path)
                logging.info(f"Tokenizer saved to {tokenizer_path}")
                
                # Add flag to indicate tokenizer was saved
                model_dict['tokenizer_saved'] = True
                model_dict['tokenizer_path'] = tokenizer_path
            except Exception as e:
                logging.error(f"Error saving tokenizer: {e}")
                model_dict['tokenizer_saved'] = False
        else:
            logging.warning("No tokenizer to save")
            model_dict['tokenizer_saved'] = False
        
        # Save model components to file
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_dict, f)
            logging.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logging.error(f"Error saving model: {e}")
            return False

    @classmethod
    def load(cls, filepath):
        """Load a model from a file with improved error handling and parameter restoration"""
        try:
            with open(filepath, 'rb') as f:
                model_dict = pickle.load(f)
            
            # Log what we found
            logging.info(f"Model keys found: {list(model_dict.keys())}")
            
            # Create a new instance with parameters from the saved model
            # Use get() with defaults to handle missing keys
            model = cls(
                layer_sizes=model_dict.get('layer_sizes', [768, 128, 3]),
                activation_functions=model_dict.get('activation_functions', ['relu', 'softmax']),
                learning_rate=model_dict.get('learning_rate', 0.001),
                epochs=model_dict.get('epochs', 100),
                batch_size=model_dict.get('batch_size', 32),
                dropout_rate=model_dict.get('dropout_rate', 0.0),
                use_embedding=model_dict.get('use_embedding', False),
                vocab_size=model_dict.get('vocab_size', 30522),
                embed_dim=model_dict.get('embed_dim', 768),
                max_seq_length=model_dict.get('max_seq_length', 512),
                tokenizer_name=model_dict.get('tokenizer_name', None)
            )
            
            # Log critical parameters that were loaded
            logging.info(f"Loaded model: layer_sizes={model.layer_sizes}, use_embedding={model.use_embedding}")
            
            # Load weights and biases if available
            if 'weights' in model_dict and 'biases' in model_dict:
                model.weights = model_dict['weights']
                model.biases = model_dict['biases']
            else:
                logging.warning("No weights or biases found in model file - using initialized values")
            
            # Load embedding-related weights if applicable
            if model.use_embedding and 'embedding' in model_dict and model_dict['embedding'] is not None:
                model.embedding = model_dict['embedding']
                logging.info("Loaded embedding matrix")
            
            # Load label encoder
            if 'label_encoder' in model_dict:
                model.label_encoder = model_dict['label_encoder']
                if model.label_encoder is not None:
                    logging.info(f"Loaded label encoder with classes: {model.label_encoder.classes_}")
            
            # Load generative model components
            if 'is_generative' in model_dict:
                model.is_generative = model_dict['is_generative']
            
            # Load training history
            if 'training_losses' in model_dict:
                model.training_losses = model_dict['training_losses']
            if 'validation_metrics' in model_dict:
                model.validation_metrics = model_dict['validation_metrics']
            
            # Try to load tokenizer if it was saved or if tokenizer_name is available
            tokenizer_loaded = False
            
            # First try loading from saved tokenizer path
            if model_dict.get('tokenizer_saved', False) and 'tokenizer_path' in model_dict:
                tokenizer_path = model_dict['tokenizer_path']
                if os.path.exists(tokenizer_path):
                    try:
                        from transformers import AutoTokenizer
                        model.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                        logging.info(f"Tokenizer loaded from saved path: {tokenizer_path}")
                        tokenizer_loaded = True
                    except Exception as e:
                        logging.error(f"Error loading tokenizer from saved path: {e}")
            
            # If that didn't work, try the default location
            if not tokenizer_loaded:
                tokenizer_path = filepath + "_tokenizer"
                if os.path.exists(tokenizer_path):
                    try:
                        from transformers import AutoTokenizer
                        model.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                        logging.info(f"Tokenizer loaded from default path: {tokenizer_path}")
                        tokenizer_loaded = True
                    except Exception as e:
                        logging.error(f"Error loading tokenizer from default path: {e}")
            
            # If still not loaded, try using the tokenizer_name
            if not tokenizer_loaded and model.tokenizer_name:
                try:
                    from transformers import AutoTokenizer
                    model.tokenizer = AutoTokenizer.from_pretrained(model.tokenizer_name)
                    logging.info(f"Tokenizer loaded from name: {model.tokenizer_name}")
                    tokenizer_loaded = True
                except Exception as e:
                    logging.error(f"Failed to load tokenizer from name: {e}")
            
            # Warn if tokenizer should be available but isn't
            if model.use_embedding and not tokenizer_loaded:
                logging.warning("Model uses embedding but no tokenizer was loaded. Text input won't work.")
            
            logging.info(f"Model loaded from {filepath}")
            return model
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            raise
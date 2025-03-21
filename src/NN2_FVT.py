import numpy as np
import unittest
from transformers import AutoTokenizer
from neural_network2 import NeuralNetwork  # Assuming the previous code is saved as neural_network.py

class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        # Simple configuration for testing
        self.layer_sizes = [10, 20, 2]  # Example for classification
        self.activation_functions = ['relu', 'softmax']
        self.learning_rate = 0.01
        self.epochs = 2
        self.batch_size = 4
        self.dropout_rate = 0.2
        self.vocab_size = 1000
        self.embed_dim = 8
        self.max_seq_length = 10
        self.tokenizer_name = 'bert-base-uncased'

    def test_initialization(self):
        # Test basic initialization
        model = NeuralNetwork(
            layer_sizes=self.layer_sizes,
            activation_functions=self.activation_functions,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            dropout_rate=self.dropout_rate
        )
        self.assertEqual(len(model.weights), 2)
        self.assertEqual(model.weights[0].shape, (10, 20))
        self.assertEqual(model.weights[1].shape, (20, 2))
        self.assertEqual(len(model.biases), 2)

    def test_embedding_initialization(self):
        # Test model with embedding layer
        model = NeuralNetwork(
            layer_sizes=[self.embed_dim, 16, 2],
            use_embedding=True,
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            max_seq_length=self.max_seq_length,
            tokenizer_name=self.tokenizer_name
        )
        self.assertIsNotNone(model.embedding)
        self.assertEqual(model.embedding.shape, (self.vocab_size, self.embed_dim))
        self.assertIsNotNone(model.tokenizer)

    def test_tokenization(self):
        # Test text tokenization
        model = NeuralNetwork(
            layer_sizes=[self.embed_dim, 16, 2],
            use_embedding=True,
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            max_seq_length=self.max_seq_length,
            tokenizer_name=self.tokenizer_name
        )
        
        texts = ["Hello world", "Test sentence", "Another example"]
        tokenized = model.tokenize(texts)
        self.assertEqual(tokenized.shape[0], 3)
        self.assertLessEqual(tokenized.shape[1], self.max_seq_length)

    def test_forward_pass(self):
        # Test forward propagation with embedding
        model = NeuralNetwork(
            layer_sizes=[self.embed_dim, 16, 2],
            use_embedding=True,
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            max_seq_length=self.max_seq_length,
            tokenizer_name=self.tokenizer_name
        )
        
        # Create dummy input (batch_size=2, seq_length=5)
        input_ids = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])
        output = model.forward_propagation(input_ids)
        self.assertEqual(output.shape, (2, 2))

    def test_dropout(self):
        # Test dropout functionality
        model = NeuralNetwork(
            layer_sizes=[10, 20, 2],
            dropout_rate=0.5,
            batch_size=2
        )
        
        # Training mode
        model.training = True
        inputs = np.random.rand(2, 10)
        output_train = model.forward_propagation(inputs)
        
        # Evaluation mode
        model.training = False
        output_eval = model.forward_propagation(inputs)
        
        # Check that outputs are different in training vs evaluation
        self.assertFalse(np.array_equal(output_train, output_eval))

    def test_attention_mechanism(self):
        # Test self-attention layer
        model = NeuralNetwork(
            layer_sizes=[self.embed_dim, 16, 2],
            use_embedding=True,
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            max_seq_length=self.max_seq_length,
            tokenizer_name=self.tokenizer_name
        )
        
        # Create input with sequence length 5
        input_ids = np.array([[1, 2, 3, 4, 5]])
        masks = np.array([[1, 1, 1, 0, 0]])  # Mask last two tokens
        
        # Forward pass with attention
        output = model.forward_propagation(input_ids, masks)
        self.assertEqual(output.shape, (1, 2))

    def test_training_loop(self):
        model = NeuralNetwork(
            layer_sizes=[self.embed_dim, 16, 2],
            use_embedding=True,
            vocab_size=30522,  # BERT's actual vocab size
            embed_dim=self.embed_dim,
            max_seq_length=self.max_seq_length,
            tokenizer_name=self.tokenizer_name,
            epochs=2
        )        
        # Create dummy dataset
        texts = ["Sample text 1", "Example input", "Test data", "Another example"]
        labels = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
        
        # Train the model
        model.train(texts, labels)
        
        # Test prediction
        predictions = model.predict(["New input text"])
        self.assertEqual(predictions.shape, (1, 2))

    def test_rl_setup(self):
        model = NeuralNetwork(
            layer_sizes=[4, 8, 2],
            activation_functions=['relu', 'softmax'],
            use_embedding=False
        )
        
        states = np.random.rand(5, 4)
        actions = model.predict(states)
        self.assertEqual(actions.shape, (5, 2))
        
        rewards = np.array([0.5, 0.3, 0.8, 0.1, 0.9])
        advantages = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-8)
        
        action_indices = np.argmax(actions, axis=1)
        action_masks = np.eye(2)[action_indices]
        
        # Reshape advantages to match action_masks
        advantages = advantages[:, np.newaxis] * action_masks
        model.train(states, advantages)
    
    def test_save_load(self):
        # Test model serialization
        model = NeuralNetwork(
            layer_sizes=self.layer_sizes,
            activation_functions=self.activation_functions,
            use_embedding=True,
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            max_seq_length=self.max_seq_length,
            tokenizer_name=self.tokenizer_name
        )
        
        model.save("test_model.pkl")
        loaded_model = NeuralNetwork.load("test_model.pkl", self.tokenizer_name)
        
        # Check configuration
        self.assertEqual(loaded_model.layer_sizes, self.layer_sizes)
        self.assertEqual(loaded_model.activation_functions, self.activation_functions)
        self.assertEqual(loaded_model.use_embedding, True)
        self.assertIsNotNone(loaded_model.embedding)

if __name__ == '__main__':
    unittest.main()
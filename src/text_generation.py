import numpy as np
import matplotlib.pyplot as plt
from neural_network2 import NeuralNetwork

def run_language_model_test():
    """
    Run a test of the improved language model implementation
    to demonstrate learning progress
    """
    print("Testing improved neural network language model...")
    
    # Create a small vocabulary for testing
    vocab_size = 100
    embed_dim = 64
    hidden_dim = 32
    
    # Layer sizes for language model: [vocab_size, embed_dim, hidden_dim, vocab_size]
    layer_sizes = [vocab_size, embed_dim, hidden_dim, vocab_size]
    
    # Create test sequences with patterns
    sequences = create_test_sequences()
    print(f"Created {len(sequences)} test sequences with patterns")
    print("Example sequences:")
    for seq in sequences[:5]:
        print(seq)
    
    # Create the model with improved parameters
    model = NeuralNetwork(
        layer_sizes=layer_sizes,
        learning_rate=0.005,  # Reduced learning rate
        epochs=50,            # Enough epochs to see progress
        batch_size=8,         # Smaller batch size
        dropout_rate=0.2,     # Increased dropout for regularization
        activation_functions=['tanh', 'tanh', 'softmax'],  # Better activations for language modeling
        use_embedding=True,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        max_seq_length=15,
        tokenizer_name='gpt2',  # Use GPT-2 tokenizer
        optimizer='adam'
    )
    
    # Train the language model
    print("Training language model...")
    losses = model.train_language_model(sequences)
    
    # Report training results
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Loss reduction: {losses[0] - losses[-1]:.4f} ({(losses[0] - losses[-1]) / losses[0] * 100:.2f}%)")
    
    # Plot the training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Language Model Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('language_model_loss.png')
    plt.close()
    print("Loss plot saved to 'language_model_loss.png'")
    
    # Enable text generation
    model.enable_text_generation()
    
    # Test generation with pattern prompts
    print("\nTesting generation with pattern prompts:")
    
    # Pattern 1: Token 5 should be followed by token 10
    prompt_1 = [5]
    generated_1 = model.generate(prompt_1, max_length=10, temperature=0.5)
    print(f"Pattern 1 - Prompt [5] generated: {generated_1}")
    
    # Pattern 2: Tokens [20, 21] should be followed by token 22
    prompt_2 = [20, 21]
    generated_2 = model.generate(prompt_2, max_length=10, temperature=0.5)
    print(f"Pattern 2 - Prompt [20, 21] generated: {generated_2}")
    
    # Pattern 3: Token 30 should start sequence [30, 31, 32, 33]
    prompt_3 = [30]
    generated_3 = model.generate(prompt_3, max_length=10, temperature=0.5)
    print(f"Pattern 3 - Prompt [30] generated: {generated_3}")
    
    # Random prompt
    random_prompt = [np.random.randint(1, vocab_size)]
    generated_random = model.generate(random_prompt, max_length=10, temperature=0.5)
    print(f"Random prompt {random_prompt} generated: {generated_random}")
    
    return model, losses

def create_test_sequences():
    """Create test sequences with clear patterns for language model training"""
    # Define a small vocabulary for testing
    vocab_size = 100
    sequences = []
    
    # Pattern 1: Token 5 is always followed by token 10
    for _ in range(20):
        seq = [5, 10]
        # Add some random tokens
        seq.extend(np.random.randint(11, vocab_size, size=np.random.randint(3, 6)).tolist())
        sequences.append(seq)
    
    # Pattern 2: Tokens [20, 21] are always followed by token 22
    for _ in range(20):
        seq = [20, 21, 22]
        # Add some random tokens
        seq.extend(np.random.randint(23, vocab_size, size=np.random.randint(3, 6)).tolist())
        sequences.append(seq)
    
    # Pattern 3: Token 30 at the start always means a sequence of [30, 31, 32, 33]
    for _ in range(20):
        seq = [30, 31, 32, 33]
        # Add some random tokens
        seq.extend(np.random.randint(34, vocab_size, size=np.random.randint(2, 5)).tolist())
        sequences.append(seq)
    
    # Random sequences for variety
    for _ in range(40):
        seq_len = np.random.randint(4, 10)
        # Avoid using pattern tokens in random sequences
        allowed_tokens = list(range(1, 5)) + list(range(6, 10)) + list(range(11, 20)) + \
                        list(range(23, 30)) + list(range(34, vocab_size))
        seq = np.random.choice(allowed_tokens, size=seq_len).tolist()
        sequences.append(seq)
    
    return sequences

if __name__ == "__main__":
    model, losses = run_language_model_test()
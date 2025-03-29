import numpy as np
import matplotlib.pyplot as plt
from neural_network2 import NeuralNetwork
from transformers import AutoTokenizer

def compare_optimizers():
    """
    Compare SGD vs Adam optimization on a simple language modeling task
    """
    print("Comparing SGD vs Adam optimization for language modeling...")
    
    # Setup tokenizer and model parameters
    tokenizer_name = 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    vocab_size = len(tokenizer)
    embed_dim = 128
    hidden_dim = 64
    
    # Layer sizes for language model
    layer_sizes = [vocab_size, embed_dim, hidden_dim, vocab_size]
    
    # Create training data
    # Use simple patterns that are easy to learn
    text_sequences = [
        "The cat sat on the mat.",
        "Dogs play in the park.",
        "The sun rises in the east.",
        "Birds sing in the morning.",
        "The stars shine at night.",
        "Rain falls from the clouds.",
        "People walk in the streets.",
        "Children play in the yard.",
        "The ocean is deep and blue.",
        "Mountains are covered in snow.",
        "Forests are full of trees.",
        "Deserts are hot and dry.",
        "Rivers flow through the land.",
        "Lakes are calm and peaceful.",
        "Cities are busy and noisy.",
        "Villages are quiet and small.",
        "The world is a big place.",
        "The universe is vast and dark.",
        "Life is a journey of discovery.",
        "Dreams are a window to the soul.",
        "Hope is a light in the darkness.",
        "Love is a gift from the heart.",
        "Time is a river without end.",
        "Change is the only constant.", 
        "Wisdom comes from experience.",
        "Knowledge is power in action.",
        "Truth is the path to freedom.",
        "Peace is the way to happiness.",
        "War is a road to destruction.",
        "Science is a search for truth.",
        "Art is an expression of beauty.",
        "Music is a language of emotions.",
        "Dance is a movement of joy.",
        "Theater is a stage for stories.",
        "Books are windows to the mind.",
        "Poetry is a song of the soul.",
        "History is a mirror of the past.",
        "The future is an open book.",
        "Imagination is a key to dreams.",
        "Creativity is a spark of life.",
        "Innovation is a bridge to progress.",
        "Technology is a tool for change.",
        "Nature is a source of wonder.",
        "Ecology is a balance of life.",
        "Environment is a home for all.",
        "Humanity is a family of beings.",
        "Animals are friends of nature.",
        "Plants are guardians of the earth.",
        "Water is a source of life.",
        "Air is a breath of fresh air.",
        "Fire is a force of nature.",
        "Earth is a planet of life.",
        "Stars are lights in the sky.",
        "Galaxies are islands of stars.",
        "Cosmos is a universe of wonders.",
        "Infinity is a mystery of time.",
        "Eternity is a journey without end.",
        "Reality is a dream of the mind.",
        "Illusion is a trick of the eye.",
        "Magic is a touch of wonder.",
        "Mystery is a puzzle of secrets.",
        "Adventure is a quest for treasure.",
        "Quest is a search for meaning.",
        "Journey is a path to discovery.",
        "Discovery is a key to knowledge.",
        "Knowledge is a door to wisdom.",
        "Wisdom is a guide to truth.",
        "Truth is a light in the dark.",
        "Light is a source of energy.",
        "Energy is a force of nature.",
        "Force is a power of change.",
        "Power is a tool for action.",
        "Action is a step to progress.",
        "Progress is a road to success.",
        "Success is a goal of life.",
        "Life is a journey of the soul.",
        "Soul is a spirit of the heart.",
        "Heart is a center of love.",
        "Love is a gift of the spirit.",
        "Spirit is a spark of life.",
        "Life is a circle of being.",
        "Being is a state of mind.",
        "Mind is a mirror of reality.", 
        "Reality is a mirror of mind.",
        "Mind is a key to the universe.",
        "Universe is a mystery of mind.",
        "Mind is a mystery of the universe.",
        "Universe is a key to the mind.",
        "Mind is a mirror of the soul.",
        "Soul is a mirror of the mind.",
        "Soul is a key to the heart.",
        "Heart is a key to the soul.",
        "Soul is a mystery of the heart.",
        "Heart is a mystery of the soul.",
        "Heart is a mirror of the spirit.",
        "Spirit is a mirror of the heart.",
        "Spirit is a key to the soul.",
        "Soul is a key to the spirit.",
        "Spirit is a mystery of the soul.",
        "Soul is a mystery of the spirit.",
        "Spirit is a mirror of the mind.",
        "Mind is a mirror of the spirit.",
        "Mind is a key to the heart.",
        "Heart is a key to the mind.",
        "Heart is a mystery of the mind.",
        "Mind is a mystery of the heart.",
        "Mind is a mirror of the soul.",
        "Soul is a mirror of the mind.",
        "Soul is a key to the universe.",
        "Universe is a key to the soul.",
        "Soul is a mystery of the universe.",
        "Universe is a mystery of the soul.",
        "Universe is a mirror of the spirit.",
        "Spirit is a mirror of the universe.",
        "Spirit is a key to the mind.",
        "Mind is a key to the spirit.",
        "Spirit is a mystery of the mind.",
        "Mind is a mystery of the spirit.",
        "Mind is a mirror of the heart.",
        "Heart is a mirror of the mind.",
        "Heart is a key to the soul.",
        "Soul is a key to the heart.",
        "Heart is a mystery of the soul.",
        "Soul is a mystery of the heart.",
        "Soul is a mirror of the spirit.",
        "Spirit is a mirror of the soul.",
        "Spirit is a key to the heart.",
        "Heart is a key to the spirit.",
        "Spirit is a mystery of the heart.",
        "Heart is a mystery of the spirit.",
        "Heart is a mirror of the mind.",
        "Mind is a mirror of the heart.",
        "Mind is a key to the universe.",
        "Universe is a key to the mind.",
        "Mind is a mystery of the universe.",
        "Universe is a mystery of the soul."
    ]
    
    # Tokenize sequences
    tokenized_sequences = [tokenizer.encode(text) for text in text_sequences]
    
    # Create SGD model
    model_sgd = NeuralNetwork(
        layer_sizes=layer_sizes,
        learning_rate=0.005,
        epochs=50,
        batch_size=4,
        dropout_rate=0.1,
        activation_functions=['tanh', 'tanh', 'softmax'],
        use_embedding=True,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        max_seq_length=20,
        tokenizer_name=tokenizer_name,
        optimizer='sgd'  # Use standard SGD
    )
    
    # Create Adam model with identical parameters
    model_adam = NeuralNetwork(
        layer_sizes=layer_sizes,
        learning_rate=0.005,  # Same learning rate for fair comparison
        epochs=50,
        batch_size=4,
        dropout_rate=0.1,
        activation_functions=['tanh', 'tanh', 'softmax'],
        use_embedding=True,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        max_seq_length=20,
        tokenizer_name=tokenizer_name,
        optimizer='adam'  # Use Adam optimization
    )
    
    # Explicitly set LM architecture flag (if needed)
    model_sgd.is_lm_architecture = True
    model_adam.is_lm_architecture = True
    
    # Train both models and track losses
    print("Training with SGD...")
    sgd_losses = model_sgd.train_language_model(tokenized_sequences)
    
    print("Training with Adam...")
    adam_losses = model_adam.train_language_model(tokenized_sequences)
    
    # Enable text generation for both models
    model_sgd.enable_text_generation()
    model_adam.enable_text_generation()
    
    # Test generation on a simple prompt
    prompt_text = "The cat"
    prompt_tokens = tokenizer.encode(prompt_text)
    
    print("\nGeneration with SGD model:")
    sgd_generated = model_sgd.generate(prompt_tokens, max_length=10, temperature=0.8)
    sgd_text = tokenizer.decode(sgd_generated)
    print(f"Generated: {sgd_text}")
    
    print("\nGeneration with Adam model:")
    adam_generated = model_adam.generate(prompt_tokens, max_length=10, temperature=0.8)
    adam_text = tokenizer.decode(adam_generated)
    print(f"Generated: {adam_text}")
    
    # Plot training curves for comparison
    plt.figure(figsize=(10, 6))
    plt.plot(sgd_losses, label='SGD')
    plt.plot(adam_losses, label='Adam')
    plt.title('Training Loss Comparison: SGD vs Adam')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('optimizer_comparison.png')
    plt.close()
    
    print("\nTraining curves saved to 'optimizer_comparison.png'")
    
    # Calculate improvement metrics
    sgd_final_loss = sgd_losses[-1]
    adam_final_loss = adam_losses[-1]
    
    improvement_pct = ((sgd_final_loss - adam_final_loss) / sgd_final_loss) * 100
    
    print(f"\nFinal loss comparison:")
    print(f"SGD final loss: {sgd_final_loss:.4f}")
    print(f"Adam final loss: {adam_final_loss:.4f}")
    print(f"Improvement with Adam: {improvement_pct:.2f}%")
    
    return model_sgd, model_adam, sgd_losses, adam_losses

if __name__ == "__main__":
    sgd_model, adam_model, sgd_losses, adam_losses = compare_optimizers()
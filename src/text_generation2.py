import numpy as np
from transformers import AutoTokenizer
from neural_network2 import NeuralNetwork  # Your fixed implementation

def test_text_generation():
    """
    Test the fixed text generation model with proper text input
    """
    print("Testing fixed text generation model...")
    
    # Initialize tokenizer - using gpt2 as in your original code
    tokenizer_name = 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print(f"BOS token: {tokenizer.bos_token}, ID: {tokenizer.bos_token_id}")
    print(f"EOS token: {tokenizer.eos_token}, ID: {tokenizer.eos_token_id}")
    vocab_size = tokenizer.vocab_size
    embed_dim = 128
    hidden_dim = 256
    
    # Layer sizes for language model: [vocab_size, embed_dim, hidden_dim, vocab_size]
    layer_sizes = [vocab_size, embed_dim, hidden_dim, vocab_size]
    
    # Create the model
    model = NeuralNetwork(
        layer_sizes=layer_sizes,
        learning_rate=0.001,  # Lower learning rate
        epochs=200,            # Fewer epochs for testing
        batch_size=8,         # Smaller batch size
        dropout_rate=0.2,     # Increased dropout for regularization
        activation_functions=['tanh', 'tanh', 'softmax'],  # Better activations for LM
        use_embedding=True,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        max_seq_length=30,
        optimizer='adam',
        tokenizer_name=tokenizer_name  # Will load the tokenizer automatically
    )
    
    # Create a small text dataset for training
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
    
    # Train the model on text sequences
    print(f"Training on {len(text_sequences)} text sequences...")
    model.train_language_model(text_sequences)
    
    model.save("text_generation_model_fixed.pkl")  # Save the model
    
    model = NeuralNetwork.load("text_generation_model_fixed.pkl")  # Load the model
    
    # Enable text generation
    model.enable_text_generation()
    
    # Test generating from text
    prompt = "The cat sat on the"
    print(f"\nGenerating from text prompt: '{prompt}'")
    generated_text = model.generate(prompt, max_length=10, temperature=0.8)
    print(f"Generated: {generated_text}")
    
    # Generate from token IDs (original approach)
    # Get token IDs for a prompt
    # tokenized = tokenizer("The dog", return_tensors="np")
    # token_ids = tokenized["input_ids"][0].tolist()
    
    # print(f"\nGenerating from token IDs: {token_ids}")
    prompt = "Villages are quiet and "    
    generated_text = model.generate(prompt, max_length=10, temperature=0.8)
    # generated_from_tokens = tokenizer.decode(generated_tokens.append(0))
    print(f"Generated: {generated_text}")
    
    return model

if __name__ == "__main__":
    model = test_text_generation()
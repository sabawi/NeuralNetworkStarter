import numpy as np
from neural_network2 import NeuralNetwork
from chat_interface import chat, interactive_chat

# Example of loading a model and using the chat interface
def chat_demo():
    # Load a trained model (replace with your actual model path)
    model_path = "sabawi_chatbot_model2.pkl"
    try:
        model = NeuralNetwork.load(model_path)
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Create a demo model for testing if loading fails
        print("Creating a minimal test model instead")
        model = create_test_model()
    
    # Enable text generation mode if not already enabled
    if not hasattr(model, 'is_generative') or not model.is_generative:
        model.enable_text_generation()
    
    # Define system prompt
    system_prompt = """You are a helpful AI assistant built with a neural network. 
    Answer questions concisely and accurately. If you don't know something, 
    just say so rather than making up information."""
    
    # Start interactive chat session
    print("\nStarting interactive chat session...\n")
    interactive_chat(
        model=model,
        system_prompt=system_prompt,
        max_length=100,          # Maximum tokens per response
        temperature=0.8,         # Controls randomness (0.8 = somewhat creative)
        memory_tokens=1024,      # Maximum context size in tokens 
        use_special_tokens=True, # Use special formatting if model was trained with it
        max_turns=5,             # Keep only last 5 conversation turns in context
        verbose=False            # Set to True for debugging information
    )

# Create a minimal test model for demo purposes
def create_test_model():
    from transformers import AutoTokenizer
    
    # Create a small language model
    model = NeuralNetwork(
        layer_sizes=[50000, 512, 512, 50000],  # Example sizes
        learning_rate=0.001,
        epochs=10,
        batch_size=32,
        dropout_rate=0.1,
        activation_functions=['tanh', 'tanh', 'softmax'],
        use_embedding=True,
        vocab_size=50000,
        embed_dim=512,
        max_seq_length=128,
        tokenizer_name='gpt2'  # Use GPT-2 tokenizer
    )
    
    # Mock the generation functions for testing
    def mock_generate(prompt, max_length=30, temperature=1.0):
        """Mock generation function that returns a canned response"""
        responses = [
            "I'm an AI assistant. I can help answer questions and provide information.",
            "That's an interesting question. Let me think about it.",
            "I don't have enough information to answer that completely.",
            "I'd be happy to help with that. What specifically would you like to know?",
            "Based on my training, I believe the answer is...",
        ]
        import random
        return random.choice(responses)
    
    def mock_generate_response(prompt, max_length=100, temperature=0.7):
        """Mock response generation that handles prompt formatting"""
        if "how are you" in prompt.lower():
            return "I'm functioning well, thank you for asking! How can I help you today?"
        elif "your name" in prompt.lower():
            return "I'm a neural network assistant. You can call me Neural Assistant."
        elif "joke" in prompt.lower():
            return "Why don't scientists trust atoms? Because they make up everything!"
        else:
            return mock_generate(prompt, max_length, temperature)
    
    # Attach mock methods to the model
    # model.generate = mock_generate
    # model.generate_response = mock_generate_response
    model.is_generative = True
    
    return model

# Example of non-interactive chat usage
def non_interactive_example():
    # Create a test model
    model = create_test_model()
    
    # Initialize conversation
    conversation = []
    system_prompt = "You are a helpful AI assistant."
    
    # First exchange
    response, conversation = chat(
        model=model, 
        prompt="Hello! How are you today?",
        conversation_history=conversation,
        system_prompt=system_prompt
    )
    print("User: Hello! How are you today?")
    print(f"Assistant: {response}\n")
    
    # Second exchange
    response, conversation = chat(
        model=model, 
        prompt="Can you tell me your name?",
        conversation_history=conversation,
        system_prompt=system_prompt
    )
    print("User: Can you tell me your name?")
    print(f"Assistant: {response}\n")
    
    # Third exchange
    response, conversation = chat(
        model=model, 
        prompt="Tell me a joke!",
        conversation_history=conversation,
        system_prompt=system_prompt
    )
    print("User: Tell me a joke!")
    print(f"Assistant: {response}\n")
    
    print("Conversation history:")
    for role, text in conversation:
        print(f"{role.capitalize()}: {text}")

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Interactive chat demo")
    print("2. Non-interactive chat example")
    
    choice = input("Enter choice (1 or 2): ")
    
    if choice == "1":
        chat_demo()
    elif choice == "2":
        non_interactive_example()
    else:
        print("Invalid choice!")

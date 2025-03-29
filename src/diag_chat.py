import numpy as np
import logging
from neural_network2 import NeuralNetwork  # Your implementation

def diagnose_model(model_path):
    """
    Diagnose issues with a text generation model
    """
    print(f"Loading model from {model_path}...")
    model = NeuralNetwork.load(model_path)
    
    # Check model attributes
    print("\n=== Model Configuration ===")
    print(f"Is generative: {getattr(model, 'is_generative', False)}")
    print(f"Is language model architecture: {getattr(model, 'is_lm_architecture', False)}")
    print(f"Layer sizes: {model.layer_sizes}")
    print(f"Vocab size: {getattr(model, 'vocab_size', 'Not set')}")
    print(f"Has tokenizer: {hasattr(model, 'tokenizer') and model.tokenizer is not None}")
    
    if hasattr(model, 'tokenizer') and model.tokenizer is not None:
        print(f"Tokenizer vocabulary size: {len(model.tokenizer.get_vocab())}")
        print(f"Special tokens: {model.tokenizer.all_special_tokens}")
    
    # Try simple generation
    print("\n=== Testing Generation ===")
    
    # Enable text generation if needed
    if not getattr(model, 'is_generative', False):
        print("Enabling text generation mode...")
        model.enable_text_generation()
    
    # Test different temperature settings
    prompts = ["Hello, how are you?", "Tell me about yourself"]
    temperatures = [0.1, 0.5, 1.0]
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        for temp in temperatures:
            try:
                print(f"\nTemperature: {temp}")
                
                # Test generate method
                print("Using generate():")
                output = model.generate(prompt, max_length=30, temperature=temp)
                if isinstance(output, str):
                    print(f"  Result: '{output}'")
                else:
                    print(f"  Result (token IDs): {output[:10]}...")
                
                # Test generate_response method
                print("Using generate_response():")
                response = model.generate_response(prompt, max_length=30, temperature=temp)
                print(f"  Result: '{response}'")
                
            except Exception as e:
                print(f"  Error: {e}")
    
    return model

def fix_chat_interface(model_path, system_prompt=None):
    """
    A simpler chat interface that works with potentially problematic models
    """
    model = NeuralNetwork.load(model_path)
    
    # Enable generation mode
    if not getattr(model, 'is_generative', False):
        model.enable_text_generation()
    
    # Print model configuration
    print(f"Model info:")
    print(f"- Architecture: {'language model' if getattr(model, 'is_lm_architecture', False) else 'general'}")
    print(f"- Layer sizes: {model.layer_sizes}")
    print(f"- Has tokenizer: {hasattr(model, 'tokenizer') and model.tokenizer is not None}")
    
    # Set up conversation context
    context = []
    if system_prompt:
        print(f"System: {system_prompt}")
        context.append(f"System: {system_prompt}\n")
    
    print("\nStarting chat (type 'exit' to quit)...")
    
    # Main chat loop
    while True:
        # Get user input
        user_input = input("User: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            break
        
        # Add to context
        context.append(f"User: {user_input}\n")
        
        # Keep context under control (last 5 exchanges)
        if len(context) > 10:
            context = context[-10:]
        
        # Create full prompt with context
        full_prompt = "".join(context)
        
        # Try to generate a response
        try:
            # Try generate_response first with lower temperature for stability
            try:
                response = model.generate_response(full_prompt, max_length=50, temperature=0.2)
            except:
                # Fall back to generate
                response = model.generate(full_prompt, max_length=50, temperature=0.2)
                
                # If it's token IDs, convert to text if possible
                if not isinstance(response, str) and hasattr(model, 'tokenizer') and model.tokenizer is not None:
                    try:
                        response = model.tokenizer.decode(response, skip_special_tokens=True)
                    except:
                        response = str(response)
            
            # Clean up response
            if not response or response.isspace():
                response = "I don't have a specific response for that."
            
            # Display and save response
            print(f"Assistant: {response}")
            context.append(f"Assistant: {response}\n")
            
        except Exception as e:
            print(f"Error generating response: {e}")
            response = "I encountered an error while generating a response."
            context.append(f"Assistant: {response}\n")

# Add improved model initialization for mock testing
def create_mock_language_model():
    """Create a mock language model with working generation"""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    except:
        tokenizer = None
        print("Could not load tokenizer")
    
    # Create a small model with function overrides
    model = NeuralNetwork(
        layer_sizes=[50000, 256, 256, 50000],
        learning_rate=0.001,
        epochs=1,
        batch_size=16,
        activation_functions=['tanh', 'tanh', 'softmax'],
        use_embedding=True,
        vocab_size=50000,
        embed_dim=256,
        max_seq_length=128
    )
    
    # Set tokenizer
    model.tokenizer = tokenizer
    
    # Override generation with simple mock functions
    def mock_generate(prompt, max_length=30, temperature=1.0):
        """Simple mock generation function"""
        responses = [
            "I'm an AI assistant that's still learning.",
            "I can help answer questions based on my training.",
            "That's an interesting question.",
            "Let me think about how to respond to that.",
            "I'm not sure I understand the question. Could you rephrase it?"
        ]
        import random
        return responses[random.randint(0, len(responses) - 1)]
    
    def mock_generate_response(prompt, max_length=30, temperature=1.0):
        """Response-specific generation"""
        if "hello" in prompt.lower() or "hi" in prompt.lower():
            return "Hello! How can I help you today?"
        elif "how are you" in prompt.lower():
            return "I'm functioning well, thank you for asking!"
        elif "your name" in prompt.lower():
            return "You can call me Neural Assistant."
        else:
            return mock_generate(prompt, max_length, temperature)
    
    # Attach the mock functions
    model.generate = mock_generate
    model.generate_response = mock_generate_response
    model.is_generative = True
    model.is_lm_architecture = True
    
    return model

if __name__ == "__main__":
    import sys
    
    print("Choose an option:")
    print("1. Diagnose your model")
    print("2. Start a simple chat interface")
    print("3. Use a mock model for testing")
    
    choice = input("Enter choice (1-3): ")
    
    if choice == "1":
        model_path = input("Enter model path (default: text_generation_model_fixed.pkl): ") or "text_generation_model_fixed.pkl"
        diagnose_model(model_path)
    
    elif choice == "2":
        model_path = input("Enter model path (default: text_generation_model_fixed.pkl): ") or "text_generation_model_fixed.pkl"
        system_prompt = input("Enter system prompt (optional): ")
        fix_chat_interface(model_path, system_prompt)
    
    elif choice == "3":
        print("Creating and using a mock model for testing...")
        model = create_mock_language_model()
        print("Model created. Starting chat...")
        
        system_prompt = input("Enter system prompt (optional): ")
        context = []
        if system_prompt:
            print(f"System: {system_prompt}")
            context.append(f"System: {system_prompt}\n")
        
        while True:
            user_input = input("User: ")
            if user_input.lower() in ['exit', 'quit', 'bye']:
                break
            
            context.append(f"User: {user_input}\n")
            full_prompt = "".join(context)
            
            response = model.generate_response(user_input, max_length=50, temperature=0.5)
            print(f"Assistant: {response}")
            context.append(f"Assistant: {response}\n")
    
    else:
        print("Invalid choice.")
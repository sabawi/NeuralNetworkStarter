def simple_chat(model, prompt, conversation_history=None, max_length=100, temperature=0.7):
    """
    Simple chat function that works with early-stage models without relying on special tokens.
    
    Args:
        model: The trained NeuralNetwork model
        prompt: Current user input
        conversation_history: Optional list to store conversation history
        max_length: Maximum number of tokens to generate
        temperature: Controls randomness (lower = more deterministic)
        
    Returns:
        response: Whatever the model generates, even if nonsensical
        updated_history: Updated conversation history
    """
    # Initialize history if not provided
    if conversation_history is None:
        conversation_history = []
    
    # Enable generation mode if not already enabled
    if not hasattr(model, 'is_generative') or not model.is_generative:
        model.enable_text_generation()
    
    # Format prompt with history
    formatted_prompt = ""
    for i, (role, text) in enumerate(conversation_history):
        if role == 'user':
            formatted_prompt += f"User: {text}\n"
        else:
            formatted_prompt += f"Assistant: {text}\n"
    
    # Add current prompt
    full_prompt = formatted_prompt + f"User: {prompt}\nAssistant: "
    print(f"Sending prompt to model: {full_prompt}")
    
    # Try generate_response first
    try:
        print("Trying generate_response()...")
        response = model.generate_response(full_prompt, max_length=max_length, temperature=temperature)
        print(f"Raw generate_response output: '{response}'")
    except Exception as e:
        print(f"Error with generate_response: {e}")
        response = None
    
    # If empty or error, try regular generate
    if not response or response.isspace():
        try:
            generated = None
            print("Trying generate()...")
            generated = model.generate(full_prompt, max_length=max_length, temperature=temperature)
            
            # Convert from tokens to text if needed
            if not isinstance(generated, str) and hasattr(model, 'tokenizer'):
                response = model.tokenizer.decode(generated, skip_special_tokens=True)
            else:
                response = str(generated)
                
            print(f"Raw generate output: '{response}'")
            
            # Extract the model's contribution (if possible)
            if "Assistant: " in response and response.index("Assistant: ") < len(response) - 12:
                parts = response.split("Assistant: ", 1)
                if len(parts) > 1:
                    response = parts[1].strip()
                    
        except Exception as e:
            print(f"Error with generate: {e}")
            response = "Error generating response."
    
    # Fallback for completely empty responses
    if not response or response.isspace():
        response = "(The model is still learning to generate coherent text.)"
    
    # Update history
    conversation_history.append(('user', prompt))
    conversation_history.append(('assistant', response))
    
    return response, conversation_history


def interactive_simple_chat(model, max_length=100, temperature=0.5):
    """
    Interactive chat session that works with early-stage models.
    
    Args:
        model: The trained NeuralNetwork model
        max_length: Maximum length of generated responses
        temperature: Controls randomness (lower = more deterministic)
    """
    conversation = []
    
    print("Starting chat with early-stage model.")
    print("Type 'exit' to end the conversation.")
    print("Note: Model responses may be nonsensical during early training.")
    print("-----")
    
    while True:
        user_input = input("User: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            break
        
        response, conversation = simple_chat(
            model, 
            user_input, 
            conversation, 
            max_length=max_length, 
            temperature=temperature
        )
        
        print(f"Assistant: {response}")
        print("-----")


# Example usage
if __name__ == "__main__":
    import sys
    from neural_network2 import NeuralNetwork
    
    model_path = input("Enter model path (default: text_generation_model_fixed.pkl): ") or "text_generation_model_fixed.pkl"
    
    try:
        model = NeuralNetwork.load(model_path)
        print(f"Loaded model from {model_path}")
        
        # Set temperature - lower is more stable for early models
        temperature = 0.3  # Lower temperature for more deterministic output
        
        # Start chat
        interactive_simple_chat(model, max_length=50, temperature=temperature)
        
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
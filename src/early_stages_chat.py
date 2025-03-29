import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def early_stage_chat(model, prompt, max_length=50, temperature=0.5, prefix_mode=False):
    """
    Chat function specifically designed for early-stage language models.
    
    Args:
        model: The neural network model
        prompt: User input
        max_length: Maximum length of response
        temperature: Randomness factor (lower = more deterministic)
        prefix_mode: Whether to use a prefix to help guide the model
        
    Returns:
        The model's raw output, regardless of quality
    """
    # Ensure model is ready for generation
    if not hasattr(model, 'is_generative') or not model.is_generative:
        model.enable_text_generation()
    
    # Format the prompt differently based on mode
    if prefix_mode:
        # Add a guiding prefix to encourage the model
        formatted_prompt = f"Q: {prompt}\nA:"
    else:
        # Use the raw prompt
        formatted_prompt = prompt
    
    # Try direct generation with lower temperature for better results
    logging.info(f"Generating with prompt: '{formatted_prompt}'")
    
    try:
        # For early models, direct generate() often works better than generate_response()
        output = model.generate(formatted_prompt, max_length=max_length, temperature=temperature)
        
        # If output is not a string, try to convert it
        if not isinstance(output, str) and hasattr(model, 'tokenizer'):
            try:
                output = model.tokenizer.decode(output, skip_special_tokens=True)
            except:
                output = str(output)
        
        logging.info(f"Raw output: '{output}'")
        
        # Try to extract just the response part if prefix_mode is used
        if prefix_mode and "A:" in output:
            response_part = output.split("A:", 1)[1].strip()
            logging.info(f"Extracted response: '{response_part}'")
            return response_part
        
        # Return the raw output if we can't extract cleanly
        return output
    
    except Exception as e:
        logging.error(f"Error generating: {e}")
        return f"Error: {str(e)}"

def run_interactive_chat():
    """Start an interactive chat session with an early-stage model."""
    import sys
    
    try:
        # Import your neural network class
        from neural_network2 import NeuralNetwork
        
        # Get model path
        model_path = input("Enter model path (default: text_generation_model_fixed.pkl): ") or "text_generation_model_fixed.pkl"
        
        try:
            # Load model
            model = NeuralNetwork.load(model_path)
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
        
        # Configuration options
        print("\nConfiguration:")
        temp = input("Temperature (0.1-1.0, default 0.3): ") or "0.3"
        temperature = float(temp)
        
        length = input("Max response length (default 50): ") or "50"
        max_length = int(length)
        
        prefix_mode = input("Use Q/A prefix mode? (y/n, default y): ").lower() != "n"
        
        # Display generation parameters
        print(f"\nGeneration settings:")
        print(f"- Temperature: {temperature} (lower = more consistent)")
        print(f"- Max length: {max_length} tokens")
        print(f"- Prefix mode: {'ON' if prefix_mode else 'OFF'}")
        
        # Begin chat session
        print("\nStarting chat session. Type 'exit' to quit.")
        print("Remember: This is an early-stage model, so outputs may be nonsensical.")
        print("----------------------------------------")
        
        # Chat loop
        while True:
            # Get user input
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit', 'bye']:
                break
            
            # Generate response
            response = early_stage_chat(
                model, 
                user_input, 
                max_length=max_length, 
                temperature=temperature,
                prefix_mode=prefix_mode
            )
            
            # Display response
            print(f"\nModel: {response}")
            
    except KeyboardInterrupt:
        print("\nChat session ended by user.")
    except Exception as e:
        print(f"\nError: {e}")

# Simple direct function for testing models
def test_model_generation(model_path, prompt="Hello", temperature=0.3, max_length=50):
    """Quick test function for model generation capabilities"""
    try:
        from neural_network2 import NeuralNetwork
        model = NeuralNetwork.load(model_path)
        
        # Enable generation
        if not hasattr(model, 'is_generative') or not model.is_generative:
            model.enable_text_generation()
        
        # Try several prompt formats
        formats = [
            prompt,                      # Raw prompt
            f"User: {prompt}\nAssistant:", # Chat format
            f"Q: {prompt}\nA:",          # Q&A format
            f"Input: {prompt}\nOutput:"  # Input/Output format
        ]
        
        print(f"Testing generation for prompt: '{prompt}'")
        print("------------------------------------------")
        
        for fmt in formats:
            print(f"\nPrompt format: '{fmt}'")
            try:
                output = model.generate(fmt, max_length=max_length, temperature=temperature)
                if isinstance(output, str):
                    print(f"Output: '{output}'")
                else:
                    print(f"Output (token IDs): {output}")
            except Exception as e:
                print(f"Error: {e}")
        
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    # Choose what to do
    print("Choose an option:")
    print("1. Start interactive chat")
    print("2. Test model generation")
    
    choice = input("Enter choice (1 or 2): ")
    
    if choice == "1":
        run_interactive_chat()
    elif choice == "2":
        model_path = input("Enter model path: ")
        prompt = input("Enter test prompt: ") or "Hello"
        test_model_generation(model_path, prompt)
    else:
        print("Invalid choice.")
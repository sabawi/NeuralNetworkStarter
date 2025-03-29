def chat(model, prompt, conversation_history=None, max_length=100, temperature=0.7, 
         system_prompt=None, memory_tokens=1024, use_special_tokens=True,  # Set default to True
         user_prefix="User: ", assistant_prefix="Assistant: ", 
         max_turns=None, verbose=False):
    """
    Chat function for neural network text generation model that maintains conversation context.
    """
    # Initialize history if not provided
    if conversation_history is None:
        conversation_history = []
    
    # Add current prompt to history
    conversation_history.append(('user', prompt))
    
    if verbose:
        print(f"Added user message to history: {prompt}")
    
    # Make sure model is in evaluation mode and ready for text generation
    if not model.is_generative:
        model.enable_text_generation()
        if verbose:
            print("Enabled text generation mode")
    model.training = False
    
    # Properly format the prompt based on how the model was trained
    if use_special_tokens:
        # For models trained with the <|promptend|> token
        full_prompt = ""
        
        # Add system prompt if provided (as part of user instruction)
        if system_prompt:
            full_prompt = system_prompt + "\n\n"
        
        # Format previous conversation turns if available
        if len(conversation_history) > 1:
            for i in range(0, len(conversation_history) - 1, 2):
                if i+1 < len(conversation_history):  # Make sure we have both user and assistant
                    user_turn = conversation_history[i][1]
                    assistant_turn = conversation_history[i+1][1]
                    
                    # Add to prompt (without prefixes - they'll be added in generate_response)
                    full_prompt += f"{user_turn}\n{assistant_turn}\n\n"
        
        # Add the current user query
        full_prompt += prompt
        
        if verbose:
            print(f"Using special tokens format with prompt: {full_prompt}")
        
        # Use the specialized generation method that handles the <|promptend|> token
        # response_text = model.simple_split_generate_response(full_prompt, max_length=max_length, temperature=temperature)
        response_text = model.simple_split_generate_response(full_prompt, max_length=max_length, temperature=temperature)
    else:
        # This branch should rarely be used since you trained with special tokens
        # But left here for completeness
        formatted_prompt = ""
        if system_prompt:
            formatted_prompt = system_prompt + "\n\n"
        
        # Add all conversation turns
        for role, text in conversation_history:
            if role == 'user':
                formatted_prompt += f"{user_prefix}{text}\n"
            else:  # assistant
                formatted_prompt += f"{assistant_prefix}{text}\n"
        
        # Add prefix for the new assistant response
        formatted_prompt += assistant_prefix
        
        if verbose:
            print(f"Using general format with prompt: {formatted_prompt}")
        
        # Use regular generation
        tokens = model.tokenizer.encode(formatted_prompt, add_special_tokens=False)
        generated_tokens = model.generate(tokens, max_length=max_length, temperature=temperature)
        
        # Extract only the new tokens
        if len(generated_tokens) > len(tokens):
            response_tokens = generated_tokens[len(tokens):]
            response_text = model.tokenizer.decode(response_tokens, skip_special_tokens=True)
        else:
            response_text = "I don't have a response for that."
    
    # Clean up the response if needed
    if isinstance(response_text, str):
        # Remove any accidental prefixes
        if response_text.startswith(assistant_prefix):
            response_text = response_text[len(assistant_prefix):].strip()
    else:
        response_text = str(response_text)
    
    if verbose:
        print(f"Generated response: {response_text}")
    
    # Add response to history
    conversation_history.append(('assistant', response_text))
    
    # Remainder of function (history management) unchanged
    # ...
    
    return response_text, conversation_history

def interactive_chat(model, system_prompt=None, max_length=100, temperature=0.7, 
                    memory_tokens=1024, use_special_tokens=False, max_turns=None,
                    user_prefix="User: ", assistant_prefix="Assistant: ", verbose=False):
    """
    Run an interactive chat session with the model in the console.
    
    Args:
        model: The trained NeuralNetwork model
        system_prompt: Optional system instructions to provide context/personality
        max_length: Maximum length of generated responses
        temperature: Controls randomness in generation
        memory_tokens: Maximum number of tokens to keep in context memory
        use_special_tokens: Whether to use prompt-response formatting with separator tokens
        max_turns: Maximum number of conversation turns to keep (None for unlimited)
        user_prefix: Prefix for user messages
        assistant_prefix: Prefix for assistant messages
        verbose: Whether to print debugging information
    """
    conversation_history = []
    
    print("Starting chat session. Type 'exit', 'quit', or 'bye' to end the conversation.")
    if system_prompt:
        print(f"System: {system_prompt}")
    print("-----")
    
    while True:
        # Get user input
        user_input = input(f"{user_prefix[:-2] if user_prefix.endswith(': ') else user_prefix} ")
        
        # Check for exit command
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Ending chat session.")
            break
        
        # Generate response
        response, conversation_history = chat(
            model, 
            user_input, 
            conversation_history, 
            max_length=max_length, 
            temperature=temperature,
            system_prompt=system_prompt,
            memory_tokens=memory_tokens,
            use_special_tokens=use_special_tokens,
            user_prefix=user_prefix,
            assistant_prefix=assistant_prefix,
            max_turns=max_turns,
            verbose=verbose
        )
        
        # Display response
        print(f"{assistant_prefix[:-2] if assistant_prefix.endswith(': ') else assistant_prefix} {response}")
        print("-----")


# Example usage:
if __name__ == "__main__":
    # Load a trained model
    # model = NeuralNetwork.load('path_to_trained_model.pkl')
    
    # Define a system prompt
    system_prompt = "You are a helpful AI assistant. Be concise and informative in your responses."
    
    # Start an interactive chat session
    # interactive_chat(
    #     model=model,
    #     system_prompt=system_prompt,
    #     max_length=150,          # Maximum tokens per response
    #     temperature=0.7,         # Controls randomness (lower = more deterministic)
    #     memory_tokens=1024,      # Maximum context size in tokens
    #     use_special_tokens=True, # Use special formatting if model was trained with it
    #     max_turns=10,            # Keep the last 10 turns in context
    #     verbose=False            # Set to True for debugging information
    # )
    
    # Alternatively, use the chat function directly:
    # conversation = []
    # response, conversation = chat(model, "Hello! How are you?", conversation)
    # print(f"Assistant: {response}")
    # 
    # response, conversation = chat(model, "Tell me about yourself", conversation)
    # print(f"Assistant: {response}")
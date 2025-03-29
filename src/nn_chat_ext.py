class NeuralNetworkChatbot:
    """
    An extension class that wraps a NeuralNetwork model with chatbot functionality.
    This preserves the original model while adding conversation capabilities.
    """
    
    def __init__(self, model, system_prompt=None, memory_tokens=1024, 
                 use_special_tokens=False, max_turns=None):
        """
        Initialize a chatbot with a neural network model.
        
        Args:
            model: The trained NeuralNetwork model
            system_prompt: Optional system message to set bot personality
            memory_tokens: Maximum number of tokens to keep in context memory
            use_special_tokens: Whether to use prompt-response formatting with separator tokens
            max_turns: Maximum number of conversation turns to keep in history
        """
        self.model = model
        self.system_prompt = system_prompt
        self.memory_tokens = memory_tokens
        self.use_special_tokens = use_special_tokens
        self.max_turns = max_turns
        self.conversation_history = []
        
        # Enable text generation if not already enabled
        if not hasattr(model, 'is_generative') or not model.is_generative:
            model.enable_text_generation()
        
        # Default message formatting
        self.user_prefix = "User: "
        self.assistant_prefix = "Assistant: "
        
    def chat(self, message, max_length=100, temperature=0.7, verbose=False):
        """
        Process a user message and generate a response.
        
        Args:
            message: User input message
            max_length: Maximum number of tokens to generate
            temperature: Controls randomness (lower = more deterministic)
            verbose: Whether to print debugging information
            
        Returns:
            response: The model's response
        """
        from chat_interface import chat as chat_function
        
        response, self.conversation_history = chat_function(
            model=self.model,
            prompt=message,
            conversation_history=self.conversation_history,
            max_length=max_length,
            temperature=temperature,
            system_prompt=self.system_prompt,
            memory_tokens=self.memory_tokens,
            use_special_tokens=self.use_special_tokens,
            user_prefix=self.user_prefix,
            assistant_prefix=self.assistant_prefix,
            max_turns=self.max_turns,
            verbose=verbose
        )
        
        return response
    
    def get_history(self):
        """Get the current conversation history."""
        return self.conversation_history
    
    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
        return True
    
    def set_system_prompt(self, system_prompt):
        """Set a new system prompt."""
        self.system_prompt = system_prompt
        return True
    
    def set_formatting(self, user_prefix=None, assistant_prefix=None):
        """Set custom formatting for user and assistant messages."""
        if user_prefix is not None:
            self.user_prefix = user_prefix
        if assistant_prefix is not None:
            self.assistant_prefix = assistant_prefix
        return True
    
    def start_interactive_session(self, max_length=100, temperature=0.7, verbose=False):
        """Start an interactive chat session in the console."""
        from chat_interface import interactive_chat
        
        # Reset conversation history for a fresh start
        self.conversation_history = []
        
        # Start interactive session
        interactive_chat(
            model=self.model,
            system_prompt=self.system_prompt,
            max_length=max_length,
            temperature=temperature,
            memory_tokens=self.memory_tokens,
            use_special_tokens=self.use_special_tokens,
            max_turns=self.max_turns,
            user_prefix=self.user_prefix,
            assistant_prefix=self.assistant_prefix,
            verbose=verbose
        )
    
    def save_conversation(self, filepath):
        """Save the current conversation to a text file."""
        try:
            with open(filepath, 'w') as f:
                if self.system_prompt:
                    f.write(f"System: {self.system_prompt}\n\n")
                
                for role, text in self.conversation_history:
                    if role == 'user':
                        f.write(f"{self.user_prefix}{text}\n")
                    else:
                        f.write(f"{self.assistant_prefix}{text}\n")
            return True
        except Exception as e:
            print(f"Error saving conversation: {e}")
            return False
    
    def load_conversation(self, filepath):
        """Load a conversation from a text file."""
        try:
            self.conversation_history = []
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            # Skip system prompt if present
            start_idx = 0
            if lines and lines[0].startswith("System: "):
                self.system_prompt = lines[0][8:].strip()
                start_idx = 2  # Skip system line and blank line
            
            # Process conversation lines
            for line in lines[start_idx:]:
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith(self.user_prefix):
                    message = line[len(self.user_prefix):].strip()
                    self.conversation_history.append(('user', message))
                elif line.startswith(self.assistant_prefix):
                    message = line[len(self.assistant_prefix):].strip()
                    self.conversation_history.append(('assistant', message))
            
            return True
        except Exception as e:
            print(f"Error loading conversation: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # This would be your actual model
    # from neural_network import NeuralNetwork
    # model = NeuralNetwork.load('my_model.pkl')
    
    # For demonstration, create a mock model
    def create_mock_model():
        class MockModel:
            def __init__(self):
                self.is_generative = True
            
            def enable_text_generation(self):
                self.is_generative = True
            
            def generate(self, prompt, max_length=30, temperature=1.0):
                import random
                responses = [
                    "I'm an AI assistant. I can help with various tasks.",
                    "That's an interesting question. Let me think about that.",
                    "I don't have enough information to provide a complete answer.",
                    "I'd be happy to assist with that request.",
                ]
                return random.choice(responses)
            
            def generate_response(self, prompt, max_length=100, temperature=0.7):
                import random
                responses = [
                    "I'm here to help! What would you like to know?",
                    "I'll do my best to assist you with that request.",
                    "Based on my training, I think the answer is...",
                    "That's a great question. Let me explain...",
                ]
                return random.choice(responses)
            
            def tokenize(self, text):
                # Simple mock tokenizer
                if isinstance(text, str):
                    text = [text]
                tokens = []
                attention_masks = []
                for t in text:
                    # Just count words as tokens
                    words = t.split()
                    token_count = min(len(words), 128)  # Max 128 tokens
                    tokens.append(list(range(token_count)))
                    attention_masks.append([1] * token_count)
                return {'input_ids': tokens, 'attention_mask': attention_masks}
            
            # Mock tokenizer properties
            @property
            def tokenizer(self):
                class MockTokenizer:
                    def encode(self, text, **kwargs):
                        if isinstance(text, str):
                            return [1] * len(text.split())
                        return [1] * 20  # Arbitrary token count
                return MockTokenizer()
        
        return MockModel()
    
    # Create a chatbot with a mock model
    mock_model = create_mock_model()
    chatbot = NeuralNetworkChatbot(
        model=mock_model,
        system_prompt="You are a helpful AI assistant that provides clear and concise answers.",
        memory_tokens=1024,
        max_turns=5
    )
    
    # Example of programmatic chat
    print("Example conversation:")
    response = chatbot.chat("Hello! Who are you?")
    print(f"User: Hello! Who are you?")
    print(f"Assistant: {response}")
    
    response = chatbot.chat("Can you help me with a problem?")
    print(f"User: Can you help me with a problem?")
    print(f"Assistant: {response}")
    
    # Demonstrate saving and loading conversations
    chatbot.save_conversation("example_conversation.txt")
    print("\nSaved conversation to example_conversation.txt")
    
    # Clear and load the saved conversation
    chatbot.clear_history()
    chatbot.load_conversation("example_conversation.txt")
    print("Loaded conversation from file")
    
    print("\nConversation history:")
    for role, text in chatbot.get_history():
        if role == 'user':
            print(f"User: {text}")
        else:
            print(f"Assistant: {text}")
    
    print("\nYou can also start an interactive session with:")
    print("chatbot.start_interactive_session()")
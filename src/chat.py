from neural_network3 import NeuralNetwork  # Your fixed implementation
import sys

# model_name = "NN3sabawi_chatbot_model2.pkl_checkpoints/final_model_run_20250401_195624.pkl"
model_name = "NN3sabawi_chatbot_model2.pkl"
try:
    model = NeuralNetwork.load(model_name)   # Load the model
    print(f"Mode '{model_name}' loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit()
    
model.enable_text_generation()  # Enable text generation mode

def debug_generate(model, prompt):
    print("--- DEBUG INFO ---")
    print(f"Input prompt: '{prompt}'")
    
    # Tokenize the prompt
    if hasattr(model, 'tokenizer') and model.tokenizer is not None:
        tokens = model.tokenizer.encode(prompt, add_special_tokens=True)
        print(f"Tokenized to {len(tokens)} tokens: {tokens[:10]}...")
    
    # Generate with very simple parameters
    try:
        # Try with temperature=0 for deterministic output
        print("Attempting generation with temperature=0...")
        output = model.generate(prompt, max_length=20, temperature=0, stream=False)
        print(f"Generation result: {output}")
        
        if not output or len(output) < 2:
            print("No meaningful output generated")
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("--- END DEBUG ---")
    return output

def check_model_config(model):
    print("--- MODEL CONFIG ---")
    print(f"Is generative model: {model.is_generative}")
    print(f"Vocab size: {model.vocab_size}")
    print(f"Embedding dimension: {model.embed_dim}")
    print(f"Layer sizes: {model.layer_sizes}")
    print(f"Activation functions: {model.activation_functions}")
    print(f"Has tokenizer: {model.tokenizer is not None}")
    if model.tokenizer:
        print(f"Special tokens: BOS={model.tokenizer.bos_token}, EOS={model.tokenizer.eos_token}")
    print("--- END CONFIG ---")

check_model_config(model)

# Prompt user for input
conversation_history = []

prompt = None
while prompt != "/quit":
    prompt = input("Enter a prompt: ")
    if prompt != "/quit":
        # Add to conversation history
        conversation_history.append(f"User: {prompt}")
        
        # Format full context with history
        if len(conversation_history) > 6:  # Limit history length
            conversation_history = conversation_history[-6:]
        
        full_context = "\n".join(conversation_history) + "\nAlaa's Brain:"
        
        # CHANGE HERE: Use stream=False to get the response
        # generated_response = model.generate(full_context, max_length=100, temperature=0.7, 
                                        #   use_top_k=40, top_p=0.95, stream=False)
                                        
        generated_response = debug_generate(model, full_context)
        
        
        if not generated_response or len(generated_response) < 2:
            print("I'm not sure how to respond to that. Could you try asking something else?")
        else:
            print(generated_response)
            # Add the response to conversation history
            conversation_history.append(f"Alaa's Brain: {generated_response}")
            
        print()
# Simple generation test - add this before starting your chat loop

import numpy as np
from neural_network2 import NeuralNetwork
from chat_interface import chat, interactive_chat

# Load a trained model (replace with your actual model path)
model_path = "sabawi_chatbot_model.pkl"
model = NeuralNetwork.load(model_path)
model.enable_text_generation()

def test_basic_generation(model):
    print("\n----- TESTING BASIC GENERATION -----")
    test_prompt = "Hello, my name is"
    print(f"Test prompt: '{test_prompt}'")
    
    # Encode and generate
    tokens = model.tokenizer.encode(test_prompt, add_special_tokens=False)
    print(f"Encoded as {len(tokens)} tokens: {tokens}")
    
    # Try to generate with minimal processing
    result = model.generate(tokens, max_length=20, temperature=1.0)
    print(f"Generation result: {result}")
    
    # Try to decode
    if isinstance(result, list):
        text = model.tokenizer.decode(result, skip_special_tokens=True)
        print(f"Decoded result: '{text}'")
        
        # Show what new tokens were added
        if len(result) > len(tokens):
            new_tokens = result[len(tokens):]
            new_text = model.tokenizer.decode(new_tokens, skip_special_tokens=True)
            print(f"New tokens added: {new_tokens}")
            print(f"New text added: '{new_text}'")
        else:
            print("No new tokens were generated!")
    else:
        print(f"Unexpected result type: {type(result)}")
    
    print("----- END TEST -----\n")

# Run this test when your app starts
test_basic_generation(model)

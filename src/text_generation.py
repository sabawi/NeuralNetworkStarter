import numpy as np
from neural_network2 import NeuralNetwork

vocab_size = 10000
embed_dim = 768
tokenizer_name = 'gpt2'
# Add vocab_size as the first dimension in layer sizes
layer_sizes_generation = [vocab_size, embed_dim, 128, vocab_size]

model_generator = NeuralNetwork(layer_sizes=layer_sizes_generation, 
                              learning_rate=0.0001, 
                              epochs=3, 
                              batch_size=4,
                              use_embedding=True, 
                              vocab_size=vocab_size, 
                              embed_dim=embed_dim, 
                              tokenizer_name=tokenizer_name,
                              activation_functions=['relu', 'relu', 'softmax'])

# Enable text generation
model_generator.enable_text_generation()

# Try generating some text (note: you'd ideally train this on a large corpus for meaningful output)
prompt = "The cat sat on the"
print(f"\nGenerating text from prompt: 'The cat sat on the'")
generated_text = model_generator.generate(prompt, max_length=20, temperature=0.9)
print(f"Generated text: {generated_text}")

prompt_2 = "Artificial intelligence will"
print(f"\nGenerating text from prompt: 'Artificial intelligence will'")
generated_text_2 = model_generator.generate(prompt_2, max_length=15, temperature=1.0)
print(f"Generated text: {generated_text_2}")


# Train a text generation model on a small dataset
# Let's say we have a small dataset of text sequences
text_sequences = [
    "The cat sat on the mat",
    "The dog barked at the mailman",
    "The sun rose in the east",
    "Birds chirped in the trees",
    "The river flowed gently",
    "The wind rustled the leaves"
]
model_generator.train_language_model(text_sequences)

# Generate text from the trained model
prompt = "The cat sat on the"
print(f"\nGenerating text from prompt: 'The cat sat on the'")
generated_text = model_generator.generate(prompt, max_length=20, temperature=0.9)
print(f"Generated text: {generated_text}")

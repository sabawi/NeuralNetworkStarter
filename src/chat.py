from neural_network2 import NeuralNetwork  # Your fixed implementation
import sys

model_name = "NN3sabawi_chatbot_model2.pkl"
try:
    model = NeuralNetwork.load(model_name)   # Load the model
    print(f"Mode '{model_name}' loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit()
    
model.enable_text_generation()  # Enable text generation mode
# Prompt user for input
prompt = None
while prompt != "/quit":
    prompt = input("Enter a prompt: ")
    if prompt != "/quit":
        print(model.generate(prompt,  max_length=30,temperature=0.1,stream=True))  # Predict 1000 characters after the 

from neural_network2 import NeuralNetwork  # Your fixed implementation

model = NeuralNetwork.load("sabawi_chatbot_model2.pkl")  # Load the model
model.enable_text_generation()  # Enable text generation mode
# Prompt user for input
prompt = None
while prompt != "/quit":
    prompt = input("Enter a prompt: ")
    if prompt != "/quit":
        print(model.generate_response(prompt,  max_length=100,temperature=0.5))  # Predict 1000 characters after the 

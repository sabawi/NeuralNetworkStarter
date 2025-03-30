from neural_network2 import NeuralNetwork  # Your fixed implementation

model = NeuralNetwork.load("NN3sabawi_chatbot_model2.pkl")  # Load the model
model.enable_text_generation()  # Enable text generation mode
# Prompt user for input
prompt = None
while prompt != "/quit":
    prompt = input("Enter a prompt: ")
    if prompt != "/quit":
        print(model.generate(prompt,  max_length=100,temperature=1.0,stream=True))  # Predict 1000 characters after the 

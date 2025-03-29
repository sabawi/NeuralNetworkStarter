import json
import sys
from neural_network2 import NeuralNetwork  # Your fixed implementation

# Setting model name:
model_name = "sabawi_chatbot_model2.pkl"
new_model = False

if new_model:
    print(f"Creating a new model: Model Name {model_name}")
else:
    print(f"Loading an existing model: Model Name {model_name}")

def create_new_model():
    """Creates a new neural network model with specified hyperparameters."""
    model = NeuralNetwork(
        layer_sizes=[50000, 512, 512, 50000],  # Example sizes
        learning_rate=0.001,
        epochs=100,
        batch_size=32,
        dropout_rate=0.2,
        activation_functions=['tanh', 'tanh', 'softmax'],
        use_embedding=True,
        vocab_size=50000,
        embed_dim=512,
        max_seq_length=128,
        tokenizer_name='gpt2'
    )
    return model

def load_text_sequence_data(file_path):
    """Loads text sequence data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)  # Ensure JSON structure
            if not isinstance(data, list):
                raise ValueError("Training data must be a list of prompt-response pairs.")
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file {file_path} is not a valid JSON file.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None

if __name__ == "__main__":
    print("\t1-Continue training an existing model\n\t2-Create and train a new model\n\t3-Exit")
    choice = input("Enter your choice: ").strip()

    if choice == "1":
        new_model = False
        model_name = input("You are retraining an existing model. Enter the model name: ").strip()
    elif choice == "2":
        new_model = True
        model_name = input("You are creating a new model. Enter a model name: ").strip()
    elif choice == "3":
        sys.exit()
    else:
        print("Invalid choice. Exiting.")
        sys.exit()

    # Ensure valid training data file
    training_data_file_name = ""
    while not training_data_file_name.endswith(".json"):
        training_data_file_name = input("Enter the name of the training JSON data file: ").strip()

    # Load training data
    training_data = load_text_sequence_data(training_data_file_name)
    if training_data is None:
        sys.exit("Failed to load training data.")

    # Get start sequence
    start_sequence = input("Enter training start sequence (default: 0): ").strip()
    start_sequence = int(start_sequence) if start_sequence.isdigit() else 0

    # Initialize the model
    if new_model:
        model = create_new_model()
        print("\tNew model created.")
    else:
        try:
            model = NeuralNetwork.load(model_name)
            print("\tExisting model loaded.")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit()

    # Ensure tokenizer parameters are correctly set
    if hasattr(model, "tokenizer"):
        model.tokenizer.model_max_length = 4096

    sequences_in_file = len(training_data)
    print(f"-- File contains {sequences_in_file} sequences")

    batch_size = 100
    sequences_trained = 0  # Initialize variable

    # Train in batches
    for i in range(start_sequence, sequences_in_file, batch_size):
        batch_data = training_data[i:i+batch_size]
        try:
            print(f"\tTraining from {i} to {min(i+batch_size, sequences_in_file)} of {sequences_in_file} sequences")
            model.train_on_prompt_response_pairs(batch_data)
            print(f"\tTraining of {batch_size} sequences completed successfully")
            
            # Save the model
            model.save(model_name)
            print(f"\tModel saved as {model_name}")
        except Exception as e:
            print(f"Error during training: {e}")
            print(f"Partial failure during training of {batch_size} sequences, from {i} to {min(i+batch_size, sequences_in_file)}")
            sys.exit()

        sequences_trained += batch_size

    # Test if model saving and loading work
    print("Testing model loading from disk...")
    try:
        model2 = NeuralNetwork.load(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit()
    
    print("Model loaded successfully.")
    model2.enable_text_generation()  # Enable text generation mode

    # Generate a response to a new prompt
    if isinstance(training_data, list) and training_data:
        first_prompt_pair = training_data[0]
        if isinstance(first_prompt_pair, list) and len(first_prompt_pair) == 2:
            prompt, expected_response = first_prompt_pair
            response = model2.simple_generate_response(prompt, max_length=100, temperature=0.5)
            print(f"Prompt: {prompt}")
            print(f"Expected: {expected_response}")
            print(f"Response: {response}\n")
        else:
            print("Warning: Training data format may be incorrect. Expecting [prompt, response] pairs.")
    else:
        print("Warning: No training data available for generating responses.")

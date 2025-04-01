import json
import pandas as pd
from datasets import load_dataset
import sys
import re
from neural_network3 import NeuralNetwork  # Your fixed implementation

# Setting model name:
model_name = "NN3sabawi_chatbot_model2.pkl"
new_model = False

if new_model:
    print(f"Creating a new model: Model Name {model_name}")
else:
    print(f"Loading an existing model: Model Name {model_name}")

def create_new_model():
    """Creates a new neural network model with specified hyperparameters."""
    model = NeuralNetwork(
        layer_sizes=[50000, 512, 512, 50000],  
        learning_rate=0.0005,
        epochs=100,
        batch_size=16,
        dropout_rate=0.1,
        activation_functions=['tanh', 'tanh', 'softmax'],
        use_embedding=True,
        vocab_size=50000,
        embed_dim=512,
        max_seq_length=128,
        tokenizer_name='gpt2',        
        use_attention=True,
        use_layer_norm=True,
        num_attention_heads=8,
        attention_dropout=0.1,
        use_positional_embedding=True,
        positional_embedding_type='sinusoidal'

    )
    return model

def print_model_hyperparameters(model):
    """Prints the hyperparameters of the given model."""
    print("Model Hyperparameters:")
    if hasattr(model, "layer_sizes"):
        print(f"Layer Sizes: {model.layer_sizes}")

    if hasattr(model, "learning_rate"):
        print(f"Learning Rate: {model.learning_rate}")
    
    if hasattr(model, "epochs"):
        print(f"Epochs: {model.epochs}")
    
    if hasattr(model, "batch_size"):
        print(f"Batch Size: {model.batch_size}")
        
    if hasattr(model, "dropout_rate"):
        print(f"Dropout Rate: {model.dropout_rate}")
        
    if hasattr(model, "activation_functions"):
        print(f"Activation Functions: {model.activation_functions}")
        
    if hasattr(model, "use_embedding"):
        print(f"Use Embedding: {model.use_embedding}")
        
    if hasattr(model, "vocab_size"):
        print(f"Vocab Size: {model.vocab_size}")
        
    if hasattr(model, "embed_dim"):
        print(f"Embed Dim: {model.embed_dim}")
    
    if hasattr(model, "max_seq_length"):
        print(f"Max Seq Length: {model.max_seq_length}")
        
    if hasattr(model, "tokenizer_name"):
        print(f"Tokenizer Name: {model.tokenizer_name}")
        
    if hasattr(model, "use_attention"):
        print(f"Use Attention: {model.use_attention}")
        
    if hasattr(model, "num_attention_heads"):
        print(f"Num Attention Heads: {model.num_attention_heads}")
        
    if hasattr(model, "attention_dropout"):
        print(f"Attention Dropout: {model.attention_dropout}")  
        
    
        
def chunk_text_into_sentences(text, max_length=1024):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    
    for sentence in sentences:
        if len(sentence) > max_length:
            sub_chunks = [sentence[i:i+max_length] for i in range(0, len(sentence), max_length)]
            chunks.extend(sub_chunks)
        else:
            chunks.append(sentence)
    
    return [chunk for chunk in chunks]

# Example usage
# json_output = json.dumps(chunk_text_into_sentences(conversations), indent=2)


def load_from_local_file(file_path):
    """Loads text conversations from a local file."""
    try:
        with open(file_path, 'r') as file:
            conversations = file.read().splitlines()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
        
    # Chunk text into sentences 
    # sequences_in_file = chunk_text_into_sentences(conversations)
    # return sequences_in_file
    return conversations
    

def load_sample_from_dailydialog(number_sequences=100):
    """Loads a sample of text conversations from the DailyDialog dataset."""
    try:
        data = load_dataset("li2017dailydialog/daily_dialog")
        # Convert dataset to pandas DataFrame for easier inspection
        split = list(data.keys())[0]  # Use the first available split
        df = data[split].to_pandas()
        
        text_column = None
        for col in df.columns:
            if df[col].dtype == 'object':  # Look for textual data
                text_column = col
                break

        if text_column:
            print(f"\nUsing column '{text_column}' for text sequences.")
            text_data = df[text_column].sample(number_sequences).dropna().astype(str).tolist()
            combined_texts = []
            temp_text = ""
            
            # random.shuffle(text_data)
            print(f"Number of sequences: {len(text_data)}")
            for text in text_data:
                combined_texts.append(text[0].strip())
                # if len(temp_text) + len(text) <= 128:
                #     temp_text += " " + text
                # else:
                #     combined_texts.append(temp_text.strip())
                #     temp_text = text
            if temp_text:
                combined_texts.append(temp_text.strip())
            
            print(f"Total sequences: {len(combined_texts)}")
            
            # print("\nGenerated Text Sequences :")
            # for snippet in combined_texts:
            #     print("\n---\n", snippet)
        else:
            print("\nNo suitable text column found in the dataset.")
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        return None
    
    return combined_texts

    
    
def load_text_conversations_data(hugging_face_dataset_name = "roskoN/dailydialog"):
    """Loads text conversations dataset from HuggingFace"""
    try:
        data = load_dataset(hugging_face_dataset_name)
        df = pd.DataFrame(data['train']['utterances'])

        conversations = []

        for _, row in df.iterrows():
            valid_values = row.dropna().values  # Remove None values
            conversation = ' '.join(map(str, valid_values))
            conversations.append(conversation)
            
        return conversations
    except FileNotFoundError:
        print(f"File not found: {hugging_face_dataset_name}")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file {hugging_face_dataset_name} is not a valid JSON file.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return None

def load_user_entered_text():
    """Prompt the user for a sequence of text conversations. Use it to train the model."""
    try:
        chunks = []
                
        data = input("Enter a text sequence to train the mode on: ")
        if(len(data) > 128):
            chunks = chunk_text_into_sentences(data)
        else:
            chunks = [data]
            
        print(f"\n\n**** Chunks: {chunks}")

        return chunks
    except Exception:
        print(f"Exception in load_user_entered_text. e={e}")

    
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
    
    print("\tChoose Training Data Source: \n\t1-Local File\n\t2-Hugging Face Dataset\n\t3-User entered text sequences\n\t4-Exit")
    choice = input("Enter your choice: ").strip()
    if choice == "2":
        training_data_file_name = input("Enter the name of Hugging Face dataset file: ").strip()
        # Load training data
        # training_data = load_text_conversations_data(training_data_file_name)
        training_data = load_sample_from_dailydialog(number_sequences=100)
        print(f"First Conversation : {training_data[0]}")
        print(f"Last Conversation : {training_data[-1]}")
    elif choice == "1":
        training_data_file_name = input("Enter local file path : ").strip()
        training_data = load_from_local_file(training_data_file_name)
        print(f"First Conversation : {training_data[0]}")
        print(f"Last Conversation : {training_data[-1]}")
    elif choice == "3": 
        training_data = load_user_entered_text()
        print(f"First sequence : {training_data[0]}")
        print(f"Last sequence : {training_data[-1]}")
    elif choice == "4":
        sys.exit()
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

    if hasattr(model, "learning_rate"):
        model.learning_rate = 0.0005
        
    print_model_hyperparameters(model)

    # Interactive modification of hyperparameters
    Done = False
    while not Done: 
        hyperparamters = input("Enter one of the model's hyperparameters to modify or 'done' to continue: ")
        if hyperparamters.lower() == 'done':
            Done = True
        else:
            try:
                if hasattr(model, hyperparamters):
                    new_value = input(f"Enter the new value for {hyperparamters}: ")
                    setattr(model, hyperparamters, new_value)
                    print(f"{hyperparamters} has been updated to {new_value}")
                else:
                    print(f"Hyperparameter '{hyperparamters}' not found in the model.")
                    
            except Exception as e:
                print(f"Error updating hyperparameter: {e}")

    print_model_hyperparameters(model)


    sequences_in_file = len(training_data)
    print(f"-- File contains {sequences_in_file} sequences")

    batch_size = 200
    sequences_trained = 0  # Initialize variable

    # Train in batches
    for i in range(start_sequence, sequences_in_file, batch_size):
        batch_data = training_data[i:i+batch_size]
        print(f"Batch {i}-{(i+batch_size)} of {sequences_in_file} sequences:\n\tData--> {batch_data}\n")
        try:
            print(f"\tTraining from {i} to {min(i+batch_size, sequences_in_file)} of {sequences_in_file} sequences")
            model.train_language_model(
                batch_data,
                epochs=100,
                verbose=True,
                validation_sequences=None,
                stride=None,
                checkpoint_dir= f"{model_name}_checkpoints",
                checkpoint_interval=10,
                max_checkpoints=3,           # Keep only 3 most recent checkpoints during training
                cleanup_on_completion=True,   # Remove all but the final checkpoint when done
                save_best_only=True
                )
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
            response = model2.generate(prompt, max_length=100, temperature=0.5)
            print(f"Prompt: {prompt}")
            print(f"Expected: {expected_response}")
            print(f"Response: {response}\n")
        else:
            print("Warning: Training data format may be incorrect. Expecting [prompt, response] pairs.")
    else:
        print("Warning: No training data available for generating responses.")

# Sentiment Analysis Model text
from neural_network2 import NeuralNetwork
import numpy as np
# from transformers import AutoTokenizer, AutoModel

if __name__ == "__main__":
    # Load the model
    model = NeuralNetwork.load("sentiment_model_simple.pkl")
    model.enable_text_generation()
    # print(f"Model use_embedding: {model.use_embedding}")
    # print(f"Model tokenizer available: {model.tokenizer is not None}")
    # print(f"Model layer sizes: {model.layer_sizes}")

    # Fix the missing attributes
    # model.use_embedding = True
    # model.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # Initialize embedding matrix if it doesn't exist
    # if not hasattr(model, 'embedding') or model.embedding is None:
        # print("Initializing embedding matrix")
        # vocab_size = 30522  # BERT vocabulary size
        # embed_dim = 768     # BERT embedding dimension
        # model.embedding = np.random.randn(vocab_size, embed_dim) * np.sqrt(2.0 / (vocab_size + embed_dim))
   
    
    # Test tokenization if possible
    # if model.tokenizer:
        # try:
            # test_tokens = model.tokenize(["This is a test"])
            # print(f"Tokenization test successful, shape: {test_tokens['input_ids'].shape}")
        # except Exception as e:
            # print(f"Tokenization test failed: {e}")


    # print("tokenizer_name", model.tokenizer)
    # Define the class labels   
    class_labels = ["Positive", "Negative", "Neutral"]

    #Prediction 1
    #============
    print("\nPrediction 1:")
    # Make prediction
    pred = model.predict("I really enjoyed this")
    predicted_class_index = np.argmax(pred)
    # Get the corresponding class label
    print(f"\tPrediction for 'I really enjoyed this': {class_labels[predicted_class_index]}")
    
    print(f"\tGenerated Text: I really enjoyed this, this product is {class_labels[predicted_class_index]}")
    
    #Prediction 2
    #============
    print("\nPrediction 2:")
    # Make prediction
    pred = model.predict("I did not like this product")
    predicted_class_index = np.argmax(pred)
    # Get the corresponding class label
    print(f"\tPrediction for 'I did not like this product': {class_labels[predicted_class_index]}")
    
    # Get the corresponding class label
    predicted_class_label = class_labels[predicted_class_index]

    print(f"\tGenerated Text: I did not like this product, I find it  {predicted_class_label}")

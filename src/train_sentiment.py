import numpy as np
import logging
from neural_network2 import NeuralNetwork

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    # Create a simpler sentiment classification model
    model = NeuralNetwork(
        layer_sizes=[512, 256, 256, 3],  # Simplified architecture: embedding → hidden → output
        activation_functions=['relu','relu', 'softmax'],
        learning_rate=0.05,
        epochs=2000,  # Fewer epochs for faster training
        batch_size=32,  # Small batch size for this small dataset
        dropout_rate=0.02,  # Reduced dropout
        use_embedding=True,
        vocab_size=30522,  # BERT vocabulary size
        embed_dim=512,
        max_seq_length=64,  # Shorter sequences for our simple sentences
        tokenizer_name="bert-base-uncased"
    )
    
    # Example data
    texts = [
        "This is a positive review", 
        "I did not like this product", 
        "The service was excellent", 
        "It was a terrible experience", 
        'I loved the food', 
        'The movie was boring', 
        'The product was great', 
        'I did not like the service', 
        'The food was awful', 
        'The movie was amazing', 
        'The service was terrible', 
        'The product was terrible', 
        'I loved the experience'
    ]
    
    labels = [
        [1, 0, 0],  # Positive
        [0, 1, 0],  # Negative
        [1, 0, 0],  # Positive
        [0, 1, 0],  # Negative
        [1, 0, 0],  # Positive
        [0, 1, 0],  # Negative
        [1, 0, 0],  # Positive
        [0, 1, 0],  # Negative
        [0, 1, 0],  # Negative
        [1, 0, 0],  # Positive
        [0, 1, 0],  # Negative
        [0, 1, 0],  # Negative
        [1, 0, 0]   # Positive
    ]
    
    # Define class labels for output
    class_labels = ["Positive", "Negative", "Neutral"]
    
    # Train model
    try:
        logging.info("Starting simplified model training...")
        model.train(texts, np.array(labels), verbose=True)
        logging.info("Training completed successfully")
    except Exception as e:
        logging.error(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Test on examples
    test_examples = [
        "I really enjoyed this",
        "This is the best product ever",
        "I did not like this product",
        "The quality was poor",
        "It was just okay"
    ]
    
    logging.info("\nTesting model on examples:")
    for text in test_examples:
        # Get probabilities
        probs = model.predict(text, return_probs=True)
        
        # Get predicted class
        pred_class_idx = np.argmax(probs)
        confidence = probs[0][pred_class_idx] if len(probs.shape) > 1 else probs[pred_class_idx]
        
        logging.info(f'Text: "{text}"')
        logging.info(f'Prediction: {class_labels[pred_class_idx]} (Confidence: {confidence:.2%})')
    
    # Save model
    save_path = "sentiment_model_simple.pkl"
    if model.save(save_path):
        logging.info(f"Model saved to {save_path}")
        
        # Test loading the model
        try:
            loaded_model = NeuralNetwork.load(save_path)
            logging.info("Model loaded successfully")
            
            # Test the loaded model
            test_text = "I really like this product"
            loaded_prediction = loaded_model.predict(test_text, return_probs=True)
            loaded_class_idx = np.argmax(loaded_prediction)
            logging.info(f'Loaded model prediction for "{test_text}": {class_labels[loaded_class_idx]}')
        except Exception as e:
            logging.error(f"Error loading model: {e}")
    else:
        logging.error("Failed to save model")
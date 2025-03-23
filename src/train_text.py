from neural_network2 import NeuralNetwork
from sklearn.preprocessing import LabelEncoder
import numpy as np
import logging

train_questions = [
    "What is the capital of France?",
    "What is 2 + 2?",
    "Who wrote 'Romeo and Juliet'?",
    "What is the chemical symbol for water?",
    "How many continents are there?",
    "What planet is known as the Red Planet?",
    "Who painted the Mona Lisa?",
    "What is the largest organ of the human body?",
    "What is the main ingredient in guacamole?",
    "Which element has the atomic number 1?",
    "What is the largest planet in our solar system?",
    "Who is the author of 'To Kill a Mockingbird'?",
    "What is the capital of Spain?",
    "What is the capital of Italy?",
    "What is the capital of Japan?",
    "What is the capital of Australia?",
    "What is the capital of Brazil?",
    "What is the capital of Argentina?",
    "What is the capital of Canada?",
    "What is the capital of Mexico?",
    "What is the capital of Russia?",
    "What is the capital of China?",
    "What is the capital of India?",
    "What is the capital of South Africa?",
]

train_answers = [
    "Paris",
    "4",
    "William Shakespeare",
    "H2O",
    "Seven",
    "Mars",
    "Leonardo da Vinci",
    "Skin",
    "Avocado",
    "Hydrogen",
    "Jupiter",
    "Harper Lee",
    "Madrid",
    "Rome",
    "Tokyo",
    "Canberra",
    "Brasília",
    "Buenos Aires",
    "Ottawa",
    "Mexico City",
    "Moscow",
    "Beijing",
    "New Delhi",
    "Pretoria"
]

test_questions = [
    "What is the capital of Germany?",
    "What is 5 multiplied by 3?",
    "Who discovered penicillin?",
    "What is the tallest mountain in the world?",
    "What gas do plants absorb from the atmosphere?",
    "Who is the author of 'To Kill a Mockingbird'?"
]

test_answers = [
    "Berlin",
    "15",
    "Alexander Fleming",
    "Mount Everest",
    "Carbon dioxide",
    "Harper Lee"
]

# Combine all answers before encoding
all_answers = train_answers + test_answers
le = LabelEncoder().fit(all_answers)
train_answers_encoded = le.transform(train_answers)
test_answers_encoded = le.transform(test_answers)

# Data augmentation - ask the same questions with different phrasing
augmented_questions = train_questions.copy()
augmented_answers_encoded = train_answers_encoded.copy()

# Add variations of questions
augmented_questions.append("What's the capital city of France?")
augmented_answers_encoded = np.append(augmented_answers_encoded, le.transform(["Paris"]))

augmented_questions.append("What equals 2 plus 2?")
augmented_answers_encoded = np.append(augmented_answers_encoded, le.transform(["4"]))

augmented_questions.append("Who is the author of Romeo and Juliet?")
augmented_answers_encoded = np.append(augmented_answers_encoded, le.transform(["William Shakespeare"]))

# Initialize model with simpler architecture
model = NeuralNetwork(
    layer_sizes=[768, 256, len(le.classes_)],  # Simplified architecture
    activation_functions=['relu', 'softmax'],  # Fewer layers
    use_embedding=True,
    vocab_size=30522,
    embed_dim=768,
    max_seq_length=128,
    tokenizer_name='bert-base-uncased',
    dropout_rate=0.1,  # Reduced dropout
    learning_rate=0.01,  # Increased learning rate
    epochs=5000,  # Fewer epochs but we'll monitor progress
    batch_size=8)  # Smaller batch size for better gradient estimates


# Train with one-hot encoded labels
train_answers_onehot = np.eye(len(le.classes_))[augmented_answers_encoded]  # Use augmented data
model.train_qa(augmented_questions, train_answers_onehot, verbose=True)  # Train with improved method

# Save model
model.save("qa_model.pkl")
model2 = NeuralNetwork.load("qa_model.pkl")

# Evaluate
for i in range(len(test_questions)):
    print(f"Q: {test_questions[i]}")
    prediction = model2.predict_answer(test_questions[i], return_probs=True)
    predicted_index = np.argmax(prediction)
    answer = le.inverse_transform([predicted_index])[0]
    print(f"P: {answer}")
    print(f"A: {test_answers[i]}")
    print()
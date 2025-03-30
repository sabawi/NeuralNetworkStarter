import json
import random
import pandas as pd
from datasets import load_dataset
import sys


# config_name = "20231101.en"  # Use 'en' for English Wikipedia
config_name = None  
# Load a dataset from Huggingface datasets and randomly show snippets of it for review
ds_name = input("Enter Huggingface Dataset name: ")

# get and convert number of sequences to integer
number_sequences = int(input("Enter number of sequences to generate:"))

# Load the dataset

data = load_dataset(ds_name, config_name)

# Display dataset structure
print("\nDataset Structure:")
for split in data.keys():
    print(f"Split: {split}, Rows: {len(data[split])}")
    print("Features:", data[split].features, "\n")

# Convert dataset to pandas DataFrame for easier inspection
split = list(data.keys())[0]  # Use the first available split
df = data[split].to_pandas()

# Display data types
print("\nData Types:")
print(df.dtypes)

# Show some random samples
print("\nRandom Samples:")
print(df.sample(5))


# Attempt to combine text columns into sequences of approximately 1024 characters
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
        combined_texts.append(text.strip())
        # if len(temp_text) + len(text) <= 128:
        #     temp_text += " " + text
        # else:
        #     combined_texts.append(temp_text.strip())
        #     temp_text = text
    if temp_text:
        combined_texts.append(temp_text.strip())
    
    print(f"Total sequences: {len(combined_texts)}")
    
    print("\nGenerated Text Sequences :")
    for snippet in combined_texts:
        print("\n---\n", snippet)
else:
    print("\nNo suitable text column found in the dataset.")



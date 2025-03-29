import pandas as pd

# Replace 'train-00000-of-00010.parquet' with the actual path to your file
file_path = 'data/train-00000-of-00010.parquet'

try:
    # Read the Parquet file into a pandas DataFrame
    df = pd.read_parquet(file_path)

    # Print the first few rows of the DataFrame
    print("First few rows of the Parquet file:")
    print(df.head())

    # You can also print the entire DataFrame if it's not too large
    print("\nEntire DataFrame:")
    print(df)

    # Get some basic information about the DataFrame
    print("\nDataFrame Info:")
    print(df.info())

except FileNotFoundError:
    print(f"Error: File not found at path: {file_path}")
except Exception as e:
    print(f"An error occurred: {e}")

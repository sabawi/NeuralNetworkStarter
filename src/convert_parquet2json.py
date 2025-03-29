import pandas as pd
import json

file_name = input("Enter parquet file name from ./data directory: ").strip()
if not file_name:
    print("No file name entered, exiting.")
else:
    try:
        # Read the Parquet file
        df = pd.read_parquet(f"../data/{file_name}")

        # Ensure column names are stripped of spaces
        df.columns = df.columns.str.strip()

        # Print original column names
        print("Original Columns:", df.columns.tolist())

        # Define the renaming map, but only keep keys that exist in df
        rename_map = {"problem": "prompt", "answer": "response"}
        rename_map = {k: v for k, v in rename_map.items() if k in df.columns}

        # Perform renaming if needed
        if rename_map:
            df = df.rename(columns=rename_map)
            print("Renamed Columns:", df.columns.tolist())
        else:
            print("No matching columns found to rename.")

        # Ensure the required columns exist
        required_columns = ["prompt", "response"]
        available_columns = [col for col in required_columns if col in df.columns]

        if len(available_columns) < 2:
            raise KeyError(f"Missing required columns: {set(required_columns) - set(available_columns)}")

        # Extract prompt-response pairs
        data = df[available_columns].dropna().values.tolist()

        # Save as JSON
        json_file = f"../data/{file_name}.json"
        with open(json_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Converted {len(data)} pairs. First entry:")
        if data:
            print(data[0])
        else:
            print("No data available.")

    except Exception as e:
        print(f"Exception: {e}")

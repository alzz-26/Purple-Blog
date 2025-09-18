import pandas as pd

print("--- Starting Data Diagnostic ---")

try:
    # We will try loading with both comma and semicolon delimiters to be sure
    try:
        df = pd.read_csv('twitter_dataset.csv', encoding='utf-8')
        print("Successfully loaded with comma delimiter.")
    except Exception:
        df = pd.read_csv('final_twitter_dataset.csv', encoding='utf-8', delimiter=';')
        print("Successfully loaded with semicolon delimiter.")

    print("\n1. First 5 Rows (df.head()):")
    print("---------------------------------")
    print(df.head())

    print("\n2. Column Names (df.columns):")
    print("---------------------------------")
    print(df.columns.tolist())

    print("\n3. Data Info (df.info()):")
    print("---------------------------------")
    df.info()
    
    # Select only numeric columns for describe(), otherwise it can throw an error
    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty:
        print("\n4. Numerical Summary (df.describe()):")
        print("---------------------------------")
        print(numeric_df.describe())
    else:
        print("\n4. Numerical Summary (df.describe()):")
        print("---------------------------------")
        print("No numeric columns found to describe.")


except FileNotFoundError:
    print("\nError: 'final_twitter_dataset.csv' not found.")
except Exception as e:
    print(f"\nAn error occurred: {e}")

print("\n--- Diagnostic Complete ---")
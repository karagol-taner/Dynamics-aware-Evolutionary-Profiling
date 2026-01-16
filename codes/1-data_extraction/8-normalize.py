
import pandas as pd
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-human/data_tsv/All_Proteins_AM_Combined_Master.tsv'       # Replace with your actual file name
OUTPUT_FILE = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-human/data_tsv/All_Proteins_DCS_normalized_scores.tsv' # The name of the new file to be created

SEPARATOR = '\t'                        # <--- UPDATED: Uses Tab separator now
TARGET_COLUMN = 'Dynamic_Conserved_Score_0_10' # <--- UPDATED: Correct column name
# ==========================================

def process_scores():
    try:
        # Load the file
        df = pd.read_csv(INPUT_FILE, sep=SEPARATOR)
        print(f"Successfully loaded {INPUT_FILE} with {len(df)} rows.")
        
        # Clean column names (removes accidental spaces)
        df.columns = df.columns.str.strip()
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{INPUT_FILE}'.")
        return

    # Check if the specific column exists
    if TARGET_COLUMN not in df.columns:
        print(f"Error: Column '{TARGET_COLUMN}' not found.")
        print("Columns found:", df.columns.tolist())
        return

    # 1. Convert to Numeric (Force errors to NaN)
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors='coerce')

    # 2. Drop rows where the score is missing
    df_clean = df.dropna(subset=[TARGET_COLUMN]).copy()

    # 3. Apply Log-Normalization
    # Log transform to fix the density issue (0-1 vs 1-10)
    df_clean['DCS_Log_Score'] = np.log1p(df_clean[TARGET_COLUMN])

    # Min-Max Scale the Log values
    min_log = df_clean['DCS_Log_Score'].min()
    max_log = df_clean['DCS_Log_Score'].max()

    if max_log - min_log == 0:
        df_clean['Normalized_Score'] = 0.0
    else:
        df_clean['Normalized_Score'] = (df_clean['DCS_Log_Score'] - min_log) / (max_log - min_log)

    # 4. Save
    df_clean.to_csv(OUTPUT_FILE, index=False, sep='\t') # Saves as standard CSV
    print(f"Success! Normalized data saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_scores()

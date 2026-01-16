import pandas as pd
import numpy as np
import os
import glob

def calculate_dynamics_aware_score(file_path, output_folder):
    """
    Processes a protein CSV to calculate the Dynamic Conserved Score and Dynamics Coupling Score.
    """
    # Load data
    try:
        # Assuming the CSV format matches the image provided (headers on first row)
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    # 1. Data Cleaning
    # We strictly need Dynamics data (RMSF) to proceed.
    # Conservation and Enrichment are processed if available, allowing for rows where one might be missing.
    rmsf_cols = ['RMSF_R1', 'RMSF_R2', 'RMSF_R3']
    score_cols = ['conservation', 'enrichment'] # We check for these
    
    # Ensure numeric conversion for all relevant columns
    for col in rmsf_cols + score_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows only if RMSF data is missing (we can't do dynamics analysis without dynamics)
    clean_df = df.dropna(subset=[c for c in rmsf_cols if c in df.columns]).copy()

    if clean_df.empty:
        print(f"Skipping {file_path}: No valid dynamics data found.")
        return

    # --- FORMATTING ADJUSTMENT ---
    # Ensure sequence column 'i' is integer (1, 2, 3) not float (1.0, 2.0),
    # but keep N/A for missing indices.
    if 'i' in clean_df.columns:
        # Force numeric first (coerces existing 'N/A' strings to NaN)
        clean_df['i'] = pd.to_numeric(clean_df['i'], errors='coerce')
        # Convert to Nullable Integer type (allows ints and NaNs)
        clean_df['i'] = clean_df['i'].astype('Int64')

    # 2. Calculate Average Dynamics
    # We average the 3 replicas to get a robust measure of flexibility
    clean_df['Avg_RMSF'] = clean_df[rmsf_cols].mean(axis=1)
    
    # Optional: We also calculate Avg Neq if you want to use it later
    if all(col in clean_df.columns for col in ['Neq_R1', 'Neq_R2', 'Neq_R3']):
        clean_df['Avg_Neq'] = clean_df[['Neq_R1', 'Neq_R2', 'Neq_R3']].mean(axis=1)

    # 3. Normalize Dynamics (0 to 1 scale)
    # 0 = Least dynamic residue in this protein
    # 1 = Most dynamic residue in this protein
    rmsf_min = clean_df['Avg_RMSF'].min()
    rmsf_max = clean_df['Avg_RMSF'].max()
    
    # Avoid division by zero if flat line
    if rmsf_max - rmsf_min == 0:
        clean_df['Norm_Dynamics'] = 0
    else:
        clean_df['Norm_Dynamics'] = (clean_df['Avg_RMSF'] - rmsf_min) / (rmsf_max - rmsf_min)

    # 4. Establish the "Dynamic Conserved Score" (DCS)
    # Logic: Conservation * Normalized_Dynamics
    if 'conservation' in clean_df.columns:
        clean_df['Raw_DCS'] = clean_df['conservation'] * clean_df['Norm_Dynamics']

        # Normalize DCS to 0-10 scale
        dcs_min = clean_df['Raw_DCS'].min()
        dcs_max = clean_df['Raw_DCS'].max()

        if dcs_max - dcs_min == 0:
            clean_df['Dynamic_Conserved_Score_0_10'] = 0
        else:
            clean_df['Dynamic_Conserved_Score_0_10'] = (
                (clean_df['Raw_DCS'] - dcs_min) / (dcs_max - dcs_min)
            ) * 10
            
    # 5. Establish the "Dynamics Coupling Score" (Enrichment-based)
    # Logic: Enrichment * Normalized_Dynamics
    if 'enrichment' in clean_df.columns:
        clean_df['Raw_Coupling'] = clean_df['enrichment'] * clean_df['Norm_Dynamics']

        # Normalize Coupling Score to 0-10 scale
        cpl_min = clean_df['Raw_Coupling'].min()
        cpl_max = clean_df['Raw_Coupling'].max()

        if cpl_max - cpl_min == 0:
            clean_df['Dynamics_Coupling_Score_0_10'] = 0
        else:
            clean_df['Dynamics_Coupling_Score_0_10'] = (
                (clean_df['Raw_Coupling'] - cpl_min) / (cpl_max - cpl_min)
            ) * 10

    # 6. Save Output
    base_name = os.path.basename(file_path)
    output_name = f"Processed_DCS_{base_name}"
    output_path = os.path.join(output_folder, output_name)
    
    # Save the dataframe, using 'N/A' for any missing values (blanks/NaNs)
    clean_df.to_csv(output_path, index=False, na_rep='N/A')
    print(f"Successfully processed: {base_name} -> Saved to {output_path}")

def main():
    # --- CONFIGURATION ---
    # Update these paths to match your folder structure
    input_folder = '/content/drive/MyDrive/All Project Files/dynamics aware evo/combined_aligned_files'  # Where your 93 CSVs are
    output_folder = '/content/drive/MyDrive/All Project Files/dynamics aware evo/dynamics_aware_results_3' # Where you want the new files
    # --

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Find all CSV files
    files = glob.glob(os.path.join(input_folder, '*.csv'))
    
    if not files:
        print("No CSV files found. Please check the input_folder path.")
        return

    print(f"Found {len(files)} protein files. Starting analysis...")

    for file_path in files:
        calculate_dynamics_aware_score(file_path, output_folder)

    print("Analysis complete.")

if __name__ == "__main__":
    main()

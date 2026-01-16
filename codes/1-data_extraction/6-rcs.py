import pandas as pd
import numpy as np
import os

def add_rcs_to_file(file_path):
    """
    Reads a Master TSV, calculates:
      1. Rigid Conserved Score (RCS) = Conservation * (1 - Dynamics)
      2. Rigid Coupling Score (RCopS) = Enrichment * (1 - Dynamics)
    Normalizes both 0-10 per protein, and overwrites the file.
    """
    if not os.path.exists(file_path):
        print(f"Skipping: File not found at {file_path}")
        return

    print(f"\nProcessing: {os.path.basename(file_path)}...")
    
    try:
        # Read Master File
        df = pd.read_csv(file_path, sep='\t', na_values=['N/A', 'NA', ''])
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Check for necessary columns
    required_cols = ['conservation', 'Norm_Dynamics', 'Protein_ID']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Missing one of {required_cols}")
        return

    # Ensure numeric types
    df['conservation'] = pd.to_numeric(df['conservation'], errors='coerce')
    df['Norm_Dynamics'] = pd.to_numeric(df['Norm_Dynamics'], errors='coerce')
    
    # Check for Enrichment (for Rigid Coupling)
    has_enrichment = 'enrichment' in df.columns
    if has_enrichment:
        df['enrichment'] = pd.to_numeric(df['enrichment'], errors='coerce')

    # --- CALCULATE RAW SCORES ---
    print("  Calculating Raw Scores...")
    # RCS: High Cons + Low Dyn
    df['raw_rcs'] = df['conservation'] * (1 - df['Norm_Dynamics'])
    
    # RCopS: High Enrichment + Low Dyn
    if has_enrichment:
        df['raw_rcops'] = df['enrichment'] * (1 - df['Norm_Dynamics'])

    # --- NORMALIZE 0-10 PER PROTEIN ---
    print("  Normalizing 0-10 per Protein...")
    
    grouped = df.groupby('Protein_ID')
    
    # 1. Normalize RCS
    rcs_min = grouped['raw_rcs'].transform('min')
    rcs_max = grouped['raw_rcs'].transform('max')
    denom_rcs = rcs_max - rcs_min
    
    df['Rigid_Conserved_Score_0_10'] = np.where(
        denom_rcs > 0,
        ((df['raw_rcs'] - rcs_min) / denom_rcs) * 10,
        0
    )
    
    # 2. Normalize RCopS
    if has_enrichment:
        rcops_min = grouped['raw_rcops'].transform('min')
        rcops_max = grouped['raw_rcops'].transform('max')
        denom_rcops = rcops_max - rcops_min
        
        df['Rigid_Coupling_Score_0_10'] = np.where(
            denom_rcops > 0,
            ((df['raw_rcops'] - rcops_min) / denom_rcops) * 10,
            0
        )

    # Cleanup temporary columns
    cols_to_drop = ['raw_rcs']
    if has_enrichment:
        cols_to_drop.append('raw_rcops')
    
    df.drop(columns=cols_to_drop, inplace=True)

    # --- SAVE ---
    df.to_csv(file_path, sep='\t', index=False, na_rep='N/A')
    print(f"  Success! Updated {os.path.basename(file_path)} with RCS and RCopS.")

def main():
    # --- CONFIGURATION ---
    # Paths to your EXISTING Master Files
    file_1 = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data/data_tsv/All_Proteins_DCS_Master.tsv'
    file_2 = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data/data_tsv/All_Proteins_AM_Combined_Master.tsv'
    
    # Process both
    add_rcs_to_file(file_1)
    add_rcs_to_file(file_2)

if __name__ == "__main__":
    main()

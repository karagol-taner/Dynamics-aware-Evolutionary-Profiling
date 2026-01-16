import pandas as pd
import numpy as np
import os
import glob
import re

def extract_protein_number(filename):
    """
    Robustly extracts the protein number as an integer.
    Regex 'protein(\d+)' is greedy, so 'protein12' captures '12', not '1'.
    Returns integer ID or None.
    """
    match = re.search(r'protein(\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

def parse_alphamissense(am_file_path):
    """
    Parses an AlphaMissense file to calculate the mean pathogenicity per residue.
    Returns a DataFrame with ['i', 'mean_am_pathogenicity'].
    """
    try:
        # Check delimiter based on extension
        sep = '\t' if am_file_path.endswith('.tsv') else ','
        df = pd.read_csv(am_file_path, sep=sep)
        
        # Clean column names (strip whitespace, lowercase)
        df.columns = [c.strip().lower() for c in df.columns]
        
        # 1. Find Residue Index Column
        idx_col = None
        for candidate in ['position', 'residue', 'resi']:
            if candidate in df.columns:
                idx_col = candidate
                break
        
        # 2. Find Pathogenicity Score Column
        score_col = None
        for candidate in ['pathogenicity', 'pathogenicity score', 'score', 'am_pathogenicity']:
             if candidate in df.columns:
                 score_col = candidate
                 break
        if not score_col:
             for c in df.columns:
                 if 'pathogenicity' in c:
                     score_col = c
                     break
        
        if not idx_col or not score_col:
            print(f"  Warning: Could not parse columns for {os.path.basename(am_file_path)}")
            return None

        # Standardize Data
        df['i'] = pd.to_numeric(df[idx_col], errors='coerce')
        df = df.dropna(subset=['i'])
        df['i'] = df['i'].astype(int)
        
        # Calculate mean pathogenicity per residue
        grouped = df.groupby('i')[score_col].mean().reset_index()
        grouped.columns = ['i', 'mean_am_pathogenicity']
        
        return grouped

    except Exception as e:
        print(f"  Error reading {os.path.basename(am_file_path)}: {e}")
        return None

def merge_datasets(dynamics_folder, alphamissense_folder, output_folder):
    """
    Merges DCS files with matching AlphaMissense files using strict Integer ID matching.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # --- STEP 1: INDEX ALPHAMISSENSE FILES ---
    am_map = {} # ID -> Path
    am_files = glob.glob(os.path.join(alphamissense_folder, '*'))
    
    for fpath in am_files:
        pid = extract_protein_number(os.path.basename(fpath))
        if pid is not None:
            am_map[pid] = fpath

    print(f"Indexed {len(am_map)} AlphaMissense files with valid IDs.")

    # --- STEP 2: INDEX DYNAMICS FILES ---
    dyn_map = {} # ID -> Path
    dyn_files = glob.glob(os.path.join(dynamics_folder, '*.csv'))
    
    for fpath in dyn_files:
        pid = extract_protein_number(os.path.basename(fpath))
        if pid is not None:
            dyn_map[pid] = fpath
            
    print(f"Indexed {len(dyn_map)} Dynamics files with valid IDs.")

    # --- STEP 3: FIND INTERSECTION & DIAGNOSE ---
    am_ids = set(am_map.keys())
    dyn_ids = set(dyn_map.keys())
    
    common_ids = sorted(list(am_ids.intersection(dyn_ids)))
    missing_in_dyn = am_ids - dyn_ids
    missing_in_am = dyn_ids - am_ids

    print("\n--- MATCHING REPORT ---")
    print(f"Matches found: {len(common_ids)}")
    if missing_in_dyn:
        print(f"WARNING: {len(missing_in_dyn)} Proteins have AlphaMissense data but NO Dynamics file:")
        print(f"  IDs: {sorted(list(missing_in_dyn))}")
    if missing_in_am:
        print(f"Info: {len(missing_in_am)} Proteins have Dynamics but NO AlphaMissense data (Skipping).")

    # --- STEP 4: PROCESS MATCHES ---
    print("\n--- STARTING MERGE ---")
    merged_count = 0

    for pid in common_ids:
        dyn_path = dyn_map[pid]
        am_path = am_map[pid]
        dyn_fname = os.path.basename(dyn_path)
        
        print(f"Processing Protein {pid}...")
        
        try:
            # Read Dynamics (Base file)
            df_dyn = pd.read_csv(dyn_path)
            
            # Read AM Scores
            df_am = parse_alphamissense(am_path)
            
            if df_am is None or df_am.empty:
                print(f"  Error: AlphaMissense data empty for Protein {pid}")
                continue

            # Ensure join keys are integers
            if 'i' not in df_dyn.columns:
                 print(f"  Error: 'i' column missing in {dyn_fname}")
                 continue
                 
            df_dyn['i'] = pd.to_numeric(df_dyn['i'], errors='coerce').astype('Int64')
            df_am['i'] = pd.to_numeric(df_am['i'], errors='coerce').astype('Int64')
            
            # MERGE (Left join aligns everything to the Dynamics sequence)
            merged_df = pd.merge(df_dyn, df_am, on='i', how='left')
            
            # Save result
            output_name = f"Combined_AM_{dyn_fname}"
            output_path = os.path.join(output_folder, output_name)
            merged_df.to_csv(output_path, index=False, na_rep='N/A')
            merged_count += 1
            
        except Exception as e:
            print(f"  Critical error merging Protein {pid}: {e}")

    print(f"\nProcessing Complete.")
    print(f"Successfully created {merged_count} combined files.")
    print(f"Results saved to: {output_folder}")

def main():
    # --- CONFIGURATION ---
    dynamics_folder = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data/dynamics_aware_results_final'
    alphamissense_folder = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data/AlphaMissense'
    output_folder = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data/dynamics_aware_alphamissense_2'
    # ---------------------

    merge_datasets(dynamics_folder, alphamissense_folder, output_folder)

if __name__ == "__main__":
    main()

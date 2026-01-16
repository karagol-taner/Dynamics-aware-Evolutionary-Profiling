import pandas as pd
import os
import glob
import re

def aggregate_folder(input_folder, output_file, dataset_name, mapping_file=None):
    """
    Aggregates all CSV files in a specific folder into one master TSV,
    adding PDB and UniProt IDs from a mapping file.
    """
    print(f"\n--- Processing: {dataset_name} ---")
    print(f"Looking for files in: {input_folder}")
    
    # Load Mapping File if provided
    mapping_dict = {}
    if mapping_file and os.path.exists(mapping_file):
        try:
            print(f"Loading mapping from: {mapping_file}")
            map_df = pd.read_csv(mapping_file)
            # clean column names
            map_df.columns = [c.strip() for c in map_df.columns]
            
            # Create a lookup dictionary: ID (int) -> {'PDB': ..., 'UniProt': ...}
            # Assumes columns are 'ID', 'PDB', 'UniProt'
            for _, row in map_df.iterrows():
                try:
                    # Convert ID to int just in case
                    pid = int(row['ID'])
                    mapping_dict[pid] = {
                        'PDB': row.get('PDB', 'N/A'),
                        'UniProt': row.get('UniProt', 'N/A')
                    }
                except ValueError:
                    continue # Skip non-integer IDs
            print(f"Loaded mapping for {len(mapping_dict)} proteins.")
        except Exception as e:
            print(f"Error reading mapping file: {e}")
            return
    elif mapping_file:
        print(f"Warning: Mapping file not found at {mapping_file}")

    # Find all CSV files in the folder
    files = glob.glob(os.path.join(input_folder, '*.csv'))
    
    if not files:
        print(f"No CSV files found in {input_folder}.")
        return

    print(f"Found {len(files)} files. Beginning aggregation...")

    all_data = []

    for file_path in files:
        try:
            # Read the individual protein file
            df = pd.read_csv(file_path)
            
            # Extract Protein ID from filename
            filename = os.path.basename(file_path)
            
            # Extract 'proteinX' string and the number X
            match = re.search(r'protein(\d+)', filename.lower())
            
            protein_str = "Unknown"
            pdb_val = "N/A"
            uniprot_val = "N/A"

            if match:
                # 'protein9'
                protein_str = f"protein{match.group(1)}"
                # 9 (int)
                pid_num = int(match.group(1))
                
                # Look up in mapping dict
                if pid_num in mapping_dict:
                    pdb_val = mapping_dict[pid_num]['PDB']
                    uniprot_val = mapping_dict[pid_num]['UniProt']
            else:
                # Fallback: use filename if pattern doesn't match
                protein_str = os.path.splitext(filename)[0]

            # Insert metadata columns (at the start)
            df.insert(0, 'UniProt_ID', uniprot_val)
            df.insert(0, 'PDB_ID', pdb_val)
            df.insert(0, 'Protein_ID', protein_str)
            
            all_data.append(df)
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if all_data:
        # Concatenate all dataframes
        master_df = pd.concat(all_data, ignore_index=True)
        
        # Save as TSV
        master_df.to_csv(output_file, sep='\t', index=False, na_rep='N/A')
        
        print(f"Successfully aggregated {len(files)} proteins.")
        print(f"Total rows: {len(master_df)}")
        print(f"Master file saved to: {output_file}")
    else:
        print("No data was aggregated.")

def main():
    # --- GLOBAL CONFIGURATION ---
    # Path to your ID mapping CSV (ID, PDB, UniProt)
    mapping_csv_path = '/content/drive/MyDrive/All Project Files/dynamics aware evo/lists/pdb-uniprot-list.csv'
    # ----------------------------

    # --- CONFIGURATION 1: Dynamics Only (No AlphaMissense) ---
    dcs_input_folder = '/content/drive/MyDrive/All Project Files/dynamics aware evo/dynamics_aware_results_final'
    dcs_output_file = '/content/drive/MyDrive/All Project Files/dynamics aware evo/All_Proteins_DCS_Master_2.tsv'
    
    aggregate_folder(dcs_input_folder, dcs_output_file, "Dynamics Only Dataset", mapping_csv_path)

    # --- CONFIGURATION 2: With AlphaMissense ---
    am_input_folder = '/content/drive/MyDrive/All Project Files/dynamics aware evo/dynamics_aware_alphamissense'
    am_output_file = '/content/drive/MyDrive/All Project Files/dynamics aware evo/All_Proteins_AM_Combined_Master_2.tsv'
    
    aggregate_folder(am_input_folder, am_output_file, "AlphaMissense Combined Dataset", mapping_csv_path)

if __name__ == "__main__":
    main()

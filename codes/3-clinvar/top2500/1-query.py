import pandas as pd
import os
import re
import sys

# --- GLOBAL CONFIGURATION ---
# 1. Your Master Data File
ALL_PROTEINS_MASTER_PATH = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-human/data_tsv/All_Proteins_DCS_normalized_scores.tsv'

# 2. Your Gene Mapping File
UNIPROT_TO_GENE_PATH = '/content/drive/MyDrive/All Project Files/dynamics aware evo/lists/human/idmapping.csv'

# --- CONSTANTS ---
TOP_N_RESIDUES = 2500
OUTPUT_QUERY_FILE = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-human/Clinvar/ClinVar_Query_Top2500_Gene.txt'
OUTPUT_PREP_FILE = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-human/Clinvar/Top2500_Validation_Prep_Gene.csv'

# STRICT COLUMN NAMES (Must match your file exactly)
DCS_RANKING_COL = 'Dynamic_Conserved_Score_0_10'
NORMALIZED_COL = 'Normalized_Score'
AM_SCORE_COL = 'mean_am_pathogenicity' # New column in master file
POS_COL = 'i'
PROT_ID_COL = 'Protein_ID'

def read_gene_map_fixed(file_path):
    """Reads the gene map safely."""
    try:
        gene_df = pd.read_csv(file_path, sep=',', encoding='utf-8', on_bad_lines='skip')
        gene_df.columns = gene_df.columns.str.strip().str.replace('\ufeff', '', regex=False)
        
        id_col = 'From'
        name_col = 'Gene Names'

        if id_col not in gene_df.columns or name_col not in gene_df.columns:
            return {}

        gene_df['Gene_Names_Clean'] = gene_df[name_col].astype(str).fillna('').str.strip()
        return dict(zip(gene_df[id_col].astype(str), gene_df['Gene_Names_Clean']))
    except Exception as e:
        print(f"Warning reading gene map: {e}")
        return {}

def generate_clinvar_query_files():
    print("Step 1: Loading Master File...")

    try:
        # Load file
        df_raw = pd.read_csv(ALL_PROTEINS_MASTER_PATH, sep='\t', encoding='latin1')
        # Clean whitespace from column headers
        df_raw.columns = df_raw.columns.str.strip()
    except Exception as e:
        print(f"CRITICAL ERROR: Could not read file at {ALL_PROTEINS_MASTER_PATH}")
        print(f"Python Error: {e}")
        return

    # --- ERROR PROTECTION & DIAGNOSTICS ---
    print("Checking columns...")
    
    # 1. Define expectations
    required_columns = {
        'Protein ID': PROT_ID_COL,
        'Residue Position': POS_COL,
        'Ranking Score': DCS_RANKING_COL,
        'Normalized Score': NORMALIZED_COL,
        'AM Pathogenicity': AM_SCORE_COL
    }

    missing_cols = []
    for label, col_name in required_columns.items():
        if col_name not in df_raw.columns:
            missing_cols.append(f"{label} ('{col_name}')")

    # 2. If any are missing, print a detailed report and STOP
    if missing_cols:
        print("\n" + "="*60)
        print("❌ CRITICAL ERROR: MISSING COLUMNS")
        print("="*60)
        print(f"The following required columns were NOT found in your file:")
        for missing in missing_cols:
            print(f" - {missing}")
        
        print("-" * 60)
        print("Here are the columns actually found in your file:")
        print(list(df_raw.columns))
        print("-" * 60)
        print("Please check your input file or update the variable names in the script.")
        print("="*60 + "\n")
        return # Stop execution here

    # --- PROCESSING ---
    print("Columns verified. Processing data...")

    # Create working DataFrame
    df = pd.DataFrame()
    df['Protein_ID'] = df_raw[PROT_ID_COL].astype(str).str.strip()
    
    # Handle UniProt ID (Optional but recommended)
    if 'UniProt_ID' in df_raw.columns:
        df['UniProt_ID'] = df_raw['UniProt_ID'].astype(str).str.strip()
    else:
        print("Note: 'UniProt_ID' column not found, defaulting to 'Protein_ID'.")
        df['UniProt_ID'] = df['Protein_ID']

    df['Residue_Pos'] = pd.to_numeric(df_raw[POS_COL], errors='coerce')
    df['DCS_Score'] = pd.to_numeric(df_raw[DCS_RANKING_COL], errors='coerce')
    df['Normalized_DCS'] = pd.to_numeric(df_raw[NORMALIZED_COL], errors='coerce')
    
    # Load AM Score directly from master file
    df['Mean_AlphaMissense_Score'] = pd.to_numeric(df_raw[AM_SCORE_COL], errors='coerce')

    # Drop invalid rows (Ranking requires DCS and Pos)
    initial_count = len(df)
    df.dropna(subset=['DCS_Score', 'Residue_Pos'], inplace=True)
    df['Residue_Pos'] = df['Residue_Pos'].astype(int)
    final_count = len(df)
    
    if initial_count != final_count:
        print(f"Dropped {initial_count - final_count} rows due to NaN values in Score or Position.")

    # --- RANKING: Top 2500 High to Low ---
    df_top = df.sort_values(by='DCS_Score', ascending=False).head(TOP_N_RESIDUES).copy()

    # --- GENE MAPPING ---
    print("Step 2: Mapping Genes...")
    gene_lookup = read_gene_map_fixed(UNIPROT_TO_GENE_PATH)
    df_top['Gene_Name'] = df_top['UniProt_ID'].apply(lambda x: gene_lookup.get(x, x))

    # --- DATA COLLECTION ---
    print("Step 3: Preparing Output Data...")
    
    # No loop needed for AM scores anymore, we can just export the dataframe directly
    # Re-arranging columns for output
    cols_to_write = ['Protein_ID', 'UniProt_ID', 'Gene_Name', 'Residue_Pos', 'DCS_Score', 'Normalized_DCS', 'Mean_AlphaMissense_Score']
    out_df = df_top[cols_to_write].copy()
    
    # Format AM Score to 4 decimal places for cleanliness (optional)
    out_df['Mean_AlphaMissense_Score'] = out_df['Mean_AlphaMissense_Score'].apply(lambda x: f"{x:.4f}" if pd.notnull(x) else '?')

    # --- SAVE OUTPUT ---
    out_df.to_csv(OUTPUT_PREP_FILE, index=False)
    print(f"\nSUCCESS: Validation Prep File saved to: {OUTPUT_PREP_FILE}")
    
    # --- SAVE QUERY FILE ---
    query_parts = []
    gene_split_pattern = re.compile(r'[;,\s]+')
    
    for idx, row in out_df.iterrows():
        g_str = str(row['Gene_Name'])
        p = str(row['Residue_Pos'])
        
        g_list = [x.strip() for x in gene_split_pattern.split(g_str) if x.strip() and x != row['UniProt_ID']]
        
        for g in g_list:
            query_parts.append(f"({g}[gene] AND {p}[pos])")
            
    if query_parts:
        with open(OUTPUT_QUERY_FILE, 'w') as f:
            f.write(" OR ".join(query_parts))
        print(f"SUCCESS: Query file saved to: {OUTPUT_QUERY_FILE}")

if __name__ == "__main__":
    generate_clinvar_query_files()

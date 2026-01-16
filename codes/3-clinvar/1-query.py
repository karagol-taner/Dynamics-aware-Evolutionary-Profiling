import pandas as pd
import os
import re

# --- GLOBAL CONFIGURATION ---
# 1. Your Master Data File
ALL_PROTEINS_MASTER_PATH = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-human/data_tsv/All_Proteins_DCS_normalized_scores.tsv'

# 2. Your Gene Mapping File
UNIPROT_TO_GENE_PATH = '/content/drive/MyDrive/All Project Files/dynamics aware evo/lists/human/idmapping.csv'

# 3. Output Query File
OUTPUT_QUERY_FILE = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-human/Clinvar/ClinVar_Query_All_Genes.txt'

def read_gene_map_fixed(file_path):
    """Reads the gene map safely to create a dictionary."""
    print("Loading Gene Map...")
    try:
        gene_df = pd.read_csv(file_path, sep=',', encoding='utf-8', on_bad_lines='skip')
        gene_df.columns = gene_df.columns.str.strip().str.replace('\ufeff', '', regex=False)
        
        id_col = 'From'
        name_col = 'Gene Names'

        if id_col not in gene_df.columns or name_col not in gene_df.columns:
            print("Error: ID Mapping file missing required columns.")
            return {}

        gene_df['Gene_Names_Clean'] = gene_df[name_col].astype(str).fillna('').str.strip()
        # Create dictionary: UniProtID -> GeneName String
        return dict(zip(gene_df[id_col].astype(str), gene_df['Gene_Names_Clean']))
    except Exception as e:
        print(f"Warning reading gene map: {e}")
        return {}

def generate_full_gene_query():
    print("Step 1: Loading Master File to identify UniProt IDs...")

    try:
        df_raw = pd.read_csv(ALL_PROTEINS_MASTER_PATH, sep='\t', encoding='latin1')
        df_raw.columns = df_raw.columns.str.strip()
    except Exception as e:
        print(f"CRITICAL ERROR: Could not read file at {ALL_PROTEINS_MASTER_PATH}")
        print(f"Python Error: {e}")
        return

    # --- IDENTIFY THE CORRECT COLUMN ---
    # Prioritize UniProt_ID
    if 'UniProt_ID' in df_raw.columns:
        target_col = 'UniProt_ID'
    elif 'Protein_ID' in df_raw.columns:
        print("Warning: 'UniProt_ID' column not found. Trying 'Protein_ID'...")
        target_col = 'Protein_ID'
    else:
        print("CRITICAL ERROR: Neither 'UniProt_ID' nor 'Protein_ID' columns found.")
        print(f"Available columns: {list(df_raw.columns)}")
        return

    # Extract Unique IDs
    unique_ids = df_raw[target_col].astype(str).str.strip().unique()
    print(f"Found {len(unique_ids)} unique IDs in column '{target_col}'.")

    # --- MAPPING GENES ---
    print("Step 2: Mapping UniProt IDs to Gene Names...")
    gene_lookup = read_gene_map_fixed(UNIPROT_TO_GENE_PATH)
    
    found_genes = set()
    gene_split_pattern = re.compile(r'[;,\s]+')

    for uid in unique_ids:
        # Lookup using the ID from the file
        g_str = gene_lookup.get(uid)
        
        if g_str:
            # Split if multiple names exist (e.g. "TP53; P53")
            names = [x.strip() for x in gene_split_pattern.split(g_str) if x.strip()]
            for name in names:
                # Exclude names that look like IDs
                if name != uid:
                    found_genes.add(name)
        else:
            # Debugging info for the first few failures
            if len(found_genes) < 3:
                 print(f"Debug: Could not find gene name for ID: {uid}")

    # Sort for consistency
    sorted_genes = sorted(list(found_genes))
    print(f"Identified {len(sorted_genes)} unique Gene Names.")

    if len(sorted_genes) == 0:
        print("ERROR: No genes mapped. Please check if your ID Mapping file matches the IDs in your Master File.")
        return

    # --- GENERATING QUERY ---
    print("Step 3: Writing Query File...")
    
    # Format: (GeneA[gene]) OR (GeneB[gene]) ...
    query_parts = [f"({gene}[gene])" for gene in sorted_genes]
    full_query = " OR ".join(query_parts)

    try:
        with open(OUTPUT_QUERY_FILE, 'w') as f:
            f.write(full_query)
        print(f"\nSUCCESS: Query file saved to: {OUTPUT_QUERY_FILE}")
        print(f"File Size: {len(full_query)} characters.")
        
        if len(full_query) > 8000:
            print("NOTE: This query is very large. ClinVar web search might truncate it.")
            
    except Exception as e:
        print(f"Error writing output file: {e}")

if __name__ == "__main__":
    generate_full_gene_query()

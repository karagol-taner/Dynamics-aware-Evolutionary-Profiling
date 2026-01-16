import pandas as pd
import os
import glob
import re
import numpy as np

# --- 1. CONFIGURATION ---
CLINVAR_RESULTS_PATH = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-human/Clinvar/clinvar_result-3.txt'
ALL_PROTEINS_MASTER_PATH = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-human/data_tsv/All_Proteins_DCS_normalized_scores.tsv'
UNIPROT_TO_GENE_PATH = '/content/drive/MyDrive/All Project Files/dynamics aware evo/lists/human/idmapping.csv'
AM_FILES_FOLDER = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-human/AlphaMissense'

OUTPUT_RESULT_CSV = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-human/Clinvar/ClinVar_AlphaMissense_DCS_Analysis.csv'

# Column Names in Master File
COL_PROT_ID = 'Protein_ID'
COL_UNIPROT = 'UniProt_ID'
COL_POS = 'i'
COL_DCS = 'Dynamic_Conserved_Score_0_10'
COL_NORM = 'Normalized_Score'
COL_MEAN_AM = 'mean_am_pathogenicity'

# Amino Acid Map: 3-Letter to 1-Letter (for parsing p.Arg586Gln)
AA_3_TO_1 = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C', 'Glu': 'E',
    'Gln': 'Q', 'Gly': 'G', 'His': 'H', 'Ile': 'I', 'Leu': 'L', 'Lys': 'K',
    'Met': 'M', 'Phe': 'F', 'Pro': 'P', 'Ser': 'S', 'Thr': 'T', 'Trp': 'W',
    'Tyr': 'Y', 'Val': 'V', 'Ter': '*'
}

# 1-Letter to 3-Letter (for Output)
AA_1_TO_3 = {v: k for k, v in AA_3_TO_1.items()}

def load_master_data(path):
    print(f"Loading Master Data from: {path}")
    try:
        df = pd.read_csv(path, sep='\t', encoding='latin1')
        df.columns = df.columns.str.strip()
        
        # Rename columns to standard internal names
        rename_map = {
            COL_PROT_ID: 'Protein_ID',
            COL_UNIPROT: 'UniProt_ID',
            COL_POS: 'Residue_Pos',
            COL_DCS: 'DCS_Score',
            COL_NORM: 'Normalized_DCS',
            COL_MEAN_AM: 'Mean_AM_Score'
        }
        
        # Handle case where UniProt_ID missing (default to Protein_ID)
        if COL_UNIPROT not in df.columns and COL_PROT_ID in df.columns:
            rename_map[COL_PROT_ID] = 'UniProt_ID'
            
        df = df.rename(columns=rename_map)
        
        # Ensure Protein_ID exists
        if 'Protein_ID' not in df.columns and 'UniProt_ID' in df.columns:
            df['Protein_ID'] = df['UniProt_ID']

        # Clean Data
        df['UniProt_ID'] = df['UniProt_ID'].astype(str).str.strip()
        df['Residue_Pos'] = pd.to_numeric(df['Residue_Pos'], errors='coerce')
        
        df.dropna(subset=['UniProt_ID', 'Residue_Pos'], inplace=True)
        df['Residue_Pos'] = df['Residue_Pos'].astype(int)
        
        # Create lookup key
        df['lookup_key'] = df['UniProt_ID'] + '_' + df['Residue_Pos'].astype(str)
        
        # --- CRITICAL FIX: REMOVE DUPLICATES ---
        # This prevents the "ValueError: DataFrame index must be unique"
        df = df.drop_duplicates(subset=['lookup_key'])
        
        return df
    except Exception as e:
        print(f"Error loading master data: {e}")
        return pd.DataFrame()

def load_gene_map(path):
    print("Loading Gene Map...")
    try:
        df = pd.read_csv(path, sep=',', encoding='utf-8', on_bad_lines='skip')
        df.columns = df.columns.str.strip().str.replace('\ufeff', '', regex=False)
        id_col = 'From'
        name_col = 'Gene Names'
        
        if id_col not in df.columns or name_col not in df.columns:
            return {}
            
        gene_to_id = {}
        for _, row in df.iterrows():
            uid = str(row[id_col]).strip()
            names = re.split(r'[;,\s]+', str(row[name_col]))
            for name in names:
                if name.strip():
                    gene_to_id[name.strip()] = uid
        return gene_to_id
    except Exception as e:
        print(f"Error loading gene map: {e}")
        return {}

def get_am_specific_score(uniprot_id, position, ref_aa_3, alt_aa_3):
    """Fetches the SPECIFIC variant score (e.g. A123T) from raw files."""
    search_pattern = os.path.join(AM_FILES_FOLDER, f"*{uniprot_id}*.tsv")
    files = glob.glob(search_pattern)
    
    if not files:
        return None, "File not found"
        
    try:
        am_df = pd.read_csv(files[0], sep='\t')
        am_df.columns = am_df.columns.str.replace('^#', '', regex=True).str.strip()
        
        score_col = next((c for c in am_df.columns if 'pathogenicity score' in c or 'pathogenicity' in c), None)
        variant_col = 'protein variant'
        
        if not score_col:
            return None, "Score column missing"
            
        target_variant_str = f"p.{ref_aa_3}{position}{alt_aa_3}"
        specific_row = am_df[am_df[variant_col] == target_variant_str]
        
        if specific_row.empty:
            return None, "Variant not found"
            
        val = pd.to_numeric(specific_row.iloc[0][score_col], errors='coerce')
        return val, "Found"
        
    except Exception:
        return None, "Error reading file"

# --- MAIN ---
def main():
    # 1. Load Master Data
    master_df = load_master_data(ALL_PROTEINS_MASTER_PATH)
    if master_df.empty:
        return

    # Convert to Dict for O(1) Access
    master_lookup = master_df.set_index('lookup_key').to_dict('index')
    print(f"Master data loaded: {len(master_lookup)} unique residues indexed.")

    # 2. Load Gene Map
    gene_map = load_gene_map(UNIPROT_TO_GENE_PATH)
    
    # 3. Load ClinVar Results
    print("Processing ClinVar Results...")
    try:
        clinvar_df = pd.read_csv(CLINVAR_RESULTS_PATH, sep='\t')
    except Exception as e:
        print(f"Error reading ClinVar file: {e}")
        return

    results = []
    
    for index, row in clinvar_df.iterrows():
        # --- PARSE GENE ---
        gene_raw = str(row.get('Gene(s)', ''))
        # Take the first gene. If it's a pipe-separated list (CNV), we likely won't find a valid protein change anyway.
        gene = gene_raw.split('|')[0].strip()
        
        # --- PARSE PROTEIN CHANGE (Handle Multiple Formats) ---
        protein_change = str(row.get('Protein change', '')).strip()
        
        ref_aa_1, pos, alt_aa_1 = None, None, None
        
        # Regex 1: Simple (R586Q)
        match_simple = re.match(r'^([A-Z])(\d+)([A-Z])$', protein_change)
        
        # Regex 2: Formal (p.Arg586Gln)
        match_formal = re.match(r'^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$', protein_change)

        if match_simple:
            ref_aa_1 = match_simple.group(1)
            pos = int(match_simple.group(2))
            alt_aa_1 = match_simple.group(3)
        elif match_formal:
            ref_aa_3_in = match_formal.group(1)
            pos = int(match_formal.group(2))
            alt_aa_3_in = match_formal.group(3)
            # Convert 3-letter to 1-letter for internal consistency
            ref_aa_1 = AA_3_TO_1.get(ref_aa_3_in)
            alt_aa_1 = AA_3_TO_1.get(alt_aa_3_in)

        # If we couldn't parse a valid missense change, SKIP this row (Handles CNVs/Deletions)
        if not ref_aa_1 or not pos or not alt_aa_1:
            continue
        
        # Get 3-Letter codes for filenames
        ref_aa_3 = AA_1_TO_3.get(ref_aa_1, '???')
        alt_aa_3 = AA_1_TO_3.get(alt_aa_1, '???')
        
        # --- MAP TO UNIPROT ---
        uniprot_id = gene_map.get(gene, None)
        
        # Defaults
        prot_id = "Not Found"
        dcs = None
        norm_dcs = None
        mean_am = None
        specific_am = None
        am_status = "Skipped"

        if uniprot_id:
            # LOOKUP in Master Data
            lookup_key = f"{uniprot_id}_{pos}"
            data = master_lookup.get(lookup_key)
            
            if data:
                prot_id = data.get('Protein_ID', uniprot_id)
                dcs = data.get('DCS_Score')
                norm_dcs = data.get('Normalized_DCS')
                mean_am = data.get('Mean_AM_Score')
                
                # Fetch Specific Score
                specific_am, am_status = get_am_specific_score(uniprot_id, pos, ref_aa_3, alt_aa_3)
            else:
                am_status = "Residue not in Master File"
        else:
            am_status = "No UniProt ID for Gene"

        # --- STORE RESULT ---
        # Note: We store it if we found a match in your master file (DCS score exists)
        if dcs is not None:
             results.append({
                'Protein_ID': prot_id,
                'UniProt_ID': uniprot_id,
                'Gene_Name': gene,
                'Protein_Change': f"p.{ref_aa_3}{pos}{alt_aa_3}", # Standardize output format
                'Residue_Pos': pos,
                'DCS_Score': dcs,
                'Normalized_DCS': norm_dcs,
                'AM_Specific_Score': specific_am,
                'AM_Mean_Residue_Score': mean_am,
                'AM_Status': am_status,
                'ClinVar_Significance': row.get('Germline classification', row.get('Clinical significance', '')),
                'ClinVar_Accession': row.get('Accession', '')
            })

    if not results:
        print("No matches found. (Check: Do your Genes map to UniProt? Do the residues exist in your Master File?)")
        return

    final_df = pd.DataFrame(results)
    final_df = final_df.sort_values(by='DCS_Score', ascending=False)
    
    # Output columns
    cols = ['Protein_ID', 'UniProt_ID', 'Gene_Name', 'Protein_Change', 'Residue_Pos',
            'DCS_Score', 'Normalized_DCS', 'AM_Specific_Score', 'AM_Mean_Residue_Score',
            'ClinVar_Significance', 'ClinVar_Accession', 'AM_Status']
    
    final_df = final_df[[c for c in cols if c in final_df.columns]]
    
    final_df.to_csv(OUTPUT_RESULT_CSV, index=False)
    print(f"\nSUCCESS! Results saved to: {OUTPUT_RESULT_CSV}")
    print(f"Total Matches: {len(final_df)}")
    print(final_df.head())

if __name__ == "__main__":
    main()

import pandas as pd
import os
import glob
import re
import numpy as np

# --- 1. CONFIGURATION ---
CLINVAR_RESULTS_PATH = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-human/Clinvar/clinvar_result-all.txt'
ALL_PROTEINS_MASTER_PATH = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-human/data_tsv/All_Proteins_DCS_normalized_scores.tsv'
UNIPROT_TO_GENE_PATH = '/content/drive/MyDrive/All Project Files/dynamics aware evo/lists/human/idmapping.csv'
AM_FILES_FOLDER = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-human/AlphaMissense'

OUTPUT_RESULT_CSV = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-human/Clinvar/ClinVar_All_AlphaMissense_DCS_Analysis.csv'

# Column Names in Master File
COL_PROT_ID = 'Protein_ID'
COL_UNIPROT = 'UniProt_ID'
COL_POS = 'i'
COL_DCS = 'Dynamic_Conserved_Score_0_10'
COL_RCS = 'Rigid_Conserved_Score_0_10'
COL_NORM = 'Normalized_Score'
COL_MEAN_AM = 'mean_am_pathogenicity'

# Amino Acid Maps
AA_3_TO_1 = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C', 'Glu': 'E',
    'Gln': 'Q', 'Gly': 'G', 'His': 'H', 'Ile': 'I', 'Leu': 'L', 'Lys': 'K',
    'Met': 'M', 'Phe': 'F', 'Pro': 'P', 'Ser': 'S', 'Thr': 'T', 'Trp': 'W',
    'Tyr': 'Y', 'Val': 'V', 'Ter': '*'
}
AA_1_TO_3 = {v: k for k, v in AA_3_TO_1.items()}

def load_master_data(path):
    print(f"Loading Master Data from: {path}")
    try:
        df = pd.read_csv(path, sep='\t', encoding='latin1')
        df.columns = df.columns.str.strip()
        
        # Check if RCS column exists
        if COL_RCS not in df.columns:
            if 'RCS_Score' in df.columns:
                df = df.rename(columns={'RCS_Score': COL_RCS})
            else:
                df[COL_RCS] = np.nan

        rename_map = {
            COL_PROT_ID: 'Protein_ID',
            COL_UNIPROT: 'UniProt_ID',
            COL_POS: 'Residue_Pos',
            COL_DCS: 'DCS_Score',
            COL_RCS: 'RCS_Score',
            COL_NORM: 'Normalized_DCS',
            COL_MEAN_AM: 'Mean_AM_Score'
        }
        
        if COL_UNIPROT not in df.columns and COL_PROT_ID in df.columns:
            rename_map[COL_PROT_ID] = 'UniProt_ID'
            
        df = df.rename(columns=rename_map)
        
        if 'Protein_ID' not in df.columns and 'UniProt_ID' in df.columns:
            df['Protein_ID'] = df['UniProt_ID']

        df['UniProt_ID'] = df['UniProt_ID'].astype(str).str.strip()
        df['Residue_Pos'] = pd.to_numeric(df['Residue_Pos'], errors='coerce')
        
        df.dropna(subset=['UniProt_ID', 'Residue_Pos'], inplace=True)
        df['Residue_Pos'] = df['Residue_Pos'].astype(int)
        
        df['lookup_key'] = df['UniProt_ID'] + '_' + df['Residue_Pos'].astype(str)
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

def get_am_specific_score(uniprot_id, gene_name, position, ref_aa_3, alt_aa_3):
    """
    Fetches the SPECIFIC variant score robustly.
    """
    # 1. FIND FILE (Prioritize the specific pattern you mentioned)
    # Pattern 1: User specified "proteinh7_AlphaMissense-Search-P17213.tsv"
    files = glob.glob(os.path.join(AM_FILES_FOLDER, f"*AlphaMissense-Search-*{uniprot_id}*.tsv"))
    
    # Pattern 2: Generic UniProt search
    if not files:
        files = glob.glob(os.path.join(AM_FILES_FOLDER, f"*{uniprot_id}*.tsv"))
        
    # Pattern 3: Gene Name search
    if not files and gene_name:
        files = glob.glob(os.path.join(AM_FILES_FOLDER, f"*{gene_name}*.tsv"))
    
    if not files:
        return None, "File not found"
        
    try:
        # Load file. Treat '#' lines as normal headers to strip the '#' later
        am_df = pd.read_csv(files[0], sep='\t')
        
        # Clean headers: remove leading '#' and whitespace
        am_df.columns = am_df.columns.str.replace('^#', '', regex=True).str.strip()
        
        # EXACT COLUMN NAMES from your sample
        score_col = 'pathogenicity score'
        variant_col = 'protein variant'
        
        if score_col not in am_df.columns or variant_col not in am_df.columns:
            # Fallback scan
            score_col = next((c for c in am_df.columns if 'pathogenicity' in c and 'score' in c), None)
            variant_col = next((c for c in am_df.columns if 'variant' in c and 'protein' in c), None)
            
            if not score_col or not variant_col:
                return None, f"Columns missing (Found: {list(am_df.columns)})"
            
        # 2. CONSTRUCT VARIANT STRINGS (Try all formats)
        ref_aa_1 = AA_3_TO_1.get(ref_aa_3, '?')
        alt_aa_1 = AA_3_TO_1.get(alt_aa_3, '?')
        
        candidates = [
            f"p.{ref_aa_3}{position}{alt_aa_3}",  # p.Arg2Ala (Standard)
            f"{ref_aa_3}{position}{alt_aa_3}",     # Arg2Ala
            f"p.{ref_aa_1}{position}{alt_aa_1}",  # p.R2A
            f"{ref_aa_1}{position}{alt_aa_1}"      # R2A
        ]
        
        # Search
        am_df[variant_col] = am_df[variant_col].astype(str).str.strip()
        mask = am_df[variant_col].isin(candidates)
        specific_row = am_df[mask]
        
        if specific_row.empty:
            # --- DIAGNOSTIC: Check what variants ARE in the file ---
            # This helps debug why a match failed (e.g. file covers range 1-100 but variant is 150)
            sample_vars = am_df[variant_col].head(3).tolist()
            return None, f"Variant not found (File starts with: {sample_vars})"
            
        val = pd.to_numeric(specific_row.iloc[0][score_col], errors='coerce')
        return val, "Found"
        
    except Exception as e:
        return None, f"Error reading file: {e}"

# --- MAIN ---
def main():
    # 1. Load Master Data
    master_df = load_master_data(ALL_PROTEINS_MASTER_PATH)
    if master_df.empty:
        return

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
        gene_raw = str(row.get('Gene(s)', ''))
        gene = gene_raw.split('|')[0].strip()
        
        protein_change = str(row.get('Protein change', '')).strip()
        ref_aa_1, pos, alt_aa_1 = None, None, None
        
        match_simple = re.match(r'^([A-Z])(\d+)([A-Z])$', protein_change)
        match_formal = re.match(r'^p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$', protein_change)

        if match_simple:
            ref_aa_1 = match_simple.group(1)
            pos = int(match_simple.group(2))
            alt_aa_1 = match_simple.group(3)
        elif match_formal:
            ref_aa_3_in = match_formal.group(1)
            pos = int(match_formal.group(2))
            alt_aa_3_in = match_formal.group(3)
            ref_aa_1 = AA_3_TO_1.get(ref_aa_3_in)
            alt_aa_1 = AA_3_TO_1.get(alt_aa_3_in)

        if not ref_aa_1 or not pos or not alt_aa_1:
            continue
        
        ref_aa_3 = AA_1_TO_3.get(ref_aa_1, '???')
        alt_aa_3 = AA_1_TO_3.get(alt_aa_1, '???')
        
        uniprot_id = gene_map.get(gene, None)
        
        prot_id = "Not Found"
        dcs = None
        rcs = None
        norm_dcs = None
        mean_am = None
        specific_am = None
        am_status = "Skipped"

        if uniprot_id:
            lookup_key = f"{uniprot_id}_{pos}"
            data = master_lookup.get(lookup_key)
            
            if data:
                prot_id = data.get('Protein_ID', uniprot_id)
                dcs = data.get('DCS_Score')
                rcs = data.get('RCS_Score')
                norm_dcs = data.get('Normalized_DCS')
                mean_am = data.get('Mean_AM_Score')
                
                # Fetch AM Score
                specific_am, am_status = get_am_specific_score(uniprot_id, gene, pos, ref_aa_3, alt_aa_3)
            else:
                am_status = "Residue not in Master File"
        else:
            am_status = "No UniProt ID for Gene"

        if dcs is not None:
             results.append({
                'Protein_ID': prot_id,
                'UniProt_ID': uniprot_id,
                'Gene_Name': gene,
                'Protein_Change': f"p.{ref_aa_3}{pos}{alt_aa_3}",
                'Residue_Pos': pos,
                'DCS_Score': dcs,
                'RCS_Score': rcs,
                'Normalized_DCS': norm_dcs,
                'AM_Specific_Score': specific_am,
                'AM_Mean_Residue_Score': mean_am,
                'AM_Status': am_status,
                'ClinVar_Significance': row.get('Germline classification', row.get('Clinical significance', '')),
                'ClinVar_Accession': row.get('Accession', '')
            })

    if not results:
        print("No matches found.")
        return

    final_df = pd.DataFrame(results)
    final_df = final_df.sort_values(by='DCS_Score', ascending=False)
    
    cols = ['Protein_ID', 'UniProt_ID', 'Gene_Name', 'Protein_Change', 'Residue_Pos',
            'DCS_Score', 'RCS_Score', 'Normalized_DCS', 'AM_Specific_Score',
            'AM_Mean_Residue_Score', 'ClinVar_Significance', 'ClinVar_Accession', 'AM_Status']
    
    final_df = final_df[[c for c in cols if c in final_df.columns]]
    
    final_df.to_csv(OUTPUT_RESULT_CSV, index=False)
    print(f"\nSUCCESS! Results saved to: {OUTPUT_RESULT_CSV}")
    print(f"Total Matches: {len(final_df)}")
    print(final_df.head())

if __name__ == "__main__":
    main()

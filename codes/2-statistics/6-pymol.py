import pandas as pd
import numpy as np
import os

def generate_pymol_script():
    # --- CONFIGURATION ---
    input_file = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data/data_tsv/All_Proteins_AM_Combined_Master.tsv'
    output_pml = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data/visualize_results.pml'
    
    # How many top proteins to visualize?
    TOP_N_PROTEINS = 3

    print("Loading data for visualization...")
    try:
        df = pd.read_csv(input_file, sep='\t', na_values=['N/A', 'NA', ''])
    except:
        df = pd.read_csv(input_file, na_values=['N/A', 'NA', ''])

    # Ensure numeric
    cols = ['Dynamic_Conserved_Score_0_10', 'Dynamics_Coupling_Score_0_10']
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows with no PDB ID
    df = df.dropna(subset=['PDB_ID'])

    # 1. Rank Proteins to find the "Best Examples"
    # We want proteins that have a good mix of High DCS residues
    protein_ranking = df.groupby('PDB_ID')['Dynamic_Conserved_Score_0_10'].mean().sort_values(ascending=False)
    top_pdbs = protein_ranking.head(TOP_N_PROTEINS).index.tolist()

    print(f"Generating PyMOL script for top {TOP_N_PROTEINS} proteins: {top_pdbs}")

    # 2. Start Writing the PML File
    with open(output_pml, 'w') as f:
        f.write("# PyMOL Visualization Script for Dynamics-Aware Evolutionary Profiling\n")
        f.write("# Generated automatically\n")
        f.write("reinitialize\n")
        f.write("bg_color white\n")
        f.write("set cartoon_fancy_helices, 1\n")
        f.write("set specular, 0.5\n\n")

        for pdb_chain in top_pdbs:
            # Parse PDB and Chain (e.g., '2xolB' -> '2xol', 'B')
            pdb_code = pdb_chain[:4].lower()
            chain_id = pdb_chain[4:].upper() if len(pdb_chain) > 4 else 'A'
            
            obj_name = f"struct_{pdb_code}_{chain_id}"
            
            f.write(f"\n# --- Processing {pdb_chain} ---\n")
            f.write(f"fetch {pdb_code}, {obj_name}, async=0\n")
            f.write(f"remove solvent\n")
            
            # Extract only the relevant chain if needed
            if chain_id:
                f.write(f"remove not chain {chain_id}\n")
            
            # Get residues for this protein
            prot_data = df[df['PDB_ID'] == pdb_chain]
            
            f.write("python\n")
            f.write("cmd.alter('all', 'b=0.0')\n") # Reset B-factors to 0
            
            # We map DCS score to B-factor
            # We map DCopS score to occupancy (q) so we can use it for coloring if we want
            for _, row in prot_data.iterrows():
                try:
                    resi = int(row['i'])
                    dcs = row['Dynamic_Conserved_Score_0_10']
                    # Safety check for NaN
                    if pd.isna(dcs): dcs = 0.0
                    
                    # PyMOL command to set B-factor
                    # cmd.alter(selection, expression)
                    f.write(f"cmd.alter('{obj_name} and resi {resi} and chain {chain_id}', 'b={dcs:.2f}')\n")
                except:
                    continue
            
            f.write("python end\n")
            
            # --- VISUALIZATION COMMANDS ---
            f.write(f"show cartoon, {obj_name}\n")
            
            # 1. Putty Cartoon: Thickness = DCS Score (Dynamics)
            f.write(f"cartoon putty, {obj_name}\n")
            f.write(f"set cartoon_putty_scale_min, 0.5, {obj_name}\n")
            f.write(f"set cartoon_putty_scale_max, 4.0, {obj_name}\n")
            
            # 2. Coloring: Spectrum based on B-factor (DCS)
            # Low (0) = White/Blue, High (10) = Red
            f.write(f"spectrum b, blue_white_red, {obj_name}, minimum=0, maximum=10\n")
            
            # 3. Highlight "Hinges" (DCS > 7.0) as Spheres
            f.write(f"select hinges_{pdb_code}, {obj_name} and b > 7.0\n")
            f.write(f"show spheres, hinges_{pdb_code}\n")
            f.write(f"set sphere_scale, 0.5, hinges_{pdb_code}\n")
            f.write(f"color red, hinges_{pdb_code}\n")
            
            f.write(f"zoom {obj_name}\n")
            
        f.write("\n# Arrange views\n")
        f.write("grid_mode 1\n")
        f.write("deselect\n")
        f.write(f"print 'Visualization Loaded. Red/Thick = Dynamic-Conserved (Pathogenic). Blue/Thin = Static.'\n")

    print(f"Done. PyMOL script saved to: {output_pml}")
    print("Download this .pml file and open it in PyMOL!")

if __name__ == "__main__":
    generate_pymol_script()

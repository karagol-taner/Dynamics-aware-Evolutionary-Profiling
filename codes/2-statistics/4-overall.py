import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Global Bold settings
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['font.size'] = 12

def generate_big_data_analysis():
    # --- CONFIGURATION ---
    input_file = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-human/data_tsv/All_Proteins_AM_Combined_Master.tsv'
    output_folder = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-human/big_data_stats_rcs'
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        print(f"Created output folder: {output_folder}")

    print("Loading Big Data...")
    try:
        df = pd.read_csv(input_file, sep='\t', na_values=['N/A', 'NA', ''])
    except:
        df = pd.read_csv(input_file, na_values=['N/A', 'NA', ''])

    # Ensure numeric
    numeric_cols = [
        'Dynamic_Conserved_Score_0_10', 'Dynamics_Coupling_Score_0_10',
        'Rigid_Conserved_Score_0_10', 'Rigid_Coupling_Score_0_10'
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # --- LABEL HANDLING (Use UniProt if available) ---
    if 'UniProt_ID' in df.columns:
        df['Label_ID'] = df['UniProt_ID'].fillna(df['Protein_ID'])
    else:
        df['Label_ID'] = df['Protein_ID']

    # Drop missing based on label and numeric columns
    # We check if at least one score type exists
    df_clean = df.dropna(subset=['Label_ID'])
    print(f"Analyzing {len(df_clean)} residues across {df_clean['Label_ID'].nunique()} proteins...")

    # Initialize Results Text File
    results_path = os.path.join(output_folder, 'big_data_results.txt')
    f_out = open(results_path, 'w')
    f_out.write("BIG DATA ANALYSIS RESULTS (PAN-PROTEOME)\n")
    f_out.write("========================================\n\n")

    # =================================================================================
    # ANALYSIS 1: PROTEIN LANDSCAPE (Ranking by DCS - Dynamic Density)
    # =================================================================================
    print("Generating Protein Landscape (DCS)...")
    
    stats_per_protein = df_clean.groupby('Label_ID').agg(
        Mean_DCS=('Dynamic_Conserved_Score_0_10', 'mean'),
        Mean_RCS=('Rigid_Conserved_Score_0_10', 'mean'),
        Residue_Count=('i', 'count'),
        High_DCS_Count=('Dynamic_Conserved_Score_0_10', lambda x: (x > 5.0).sum()),
        High_RCS_Count=('Rigid_Conserved_Score_0_10', lambda x: (x > 5.0).sum())
    ).reset_index()
    
    stats_per_protein['Dynamic_Density_Pct'] = (stats_per_protein['High_DCS_Count'] / stats_per_protein['Residue_Count']) * 100
    stats_per_protein['Rigid_Density_Pct'] = (stats_per_protein['High_RCS_Count'] / stats_per_protein['Residue_Count']) * 100

    # Save Protein Stats to text
    f_out.write("--- Top 5 Proteins by Dynamic Conservation (Soft Machines) ---\n")
    top_dcs_prots = stats_per_protein.sort_values('Mean_DCS', ascending=False).head(5)
    f_out.write(top_dcs_prots[['Label_ID', 'Mean_DCS', 'Dynamic_Density_Pct']].to_string(index=False))
    f_out.write("\n\n")

    # Plot DCS Ranking
    stats_per_protein = stats_per_protein.sort_values('Mean_DCS', ascending=False)
    top_n = 15
    plot_data_dcs = pd.concat([stats_per_protein.head(top_n), stats_per_protein.tail(top_n)])
    
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Label_ID', y='Mean_DCS', data=plot_data_dcs, hue='Dynamic_Density_Pct', palette='viridis', dodge=False)
    plt.title('Protein Landscape: Which Proteins are "Soft Machines"?\n(Ranked by Mean DCS)', fontsize=16)
    plt.xlabel('Protein (UniProt ID)', fontsize=14)
    plt.ylabel('Average Dynamic Conserved Score', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.legend(title='High-DCS Density (%)', loc='upper right')
    plt.grid(axis='y', alpha=0.3)
    
    save_path = os.path.join(output_folder, 'Protein_DCS_Ranking.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.svg'), format='svg', bbox_inches='tight')
    print(f"Saved Protein Ranking (DCS) to {save_path}")
    plt.close()

    # =================================================================================
    # ANALYSIS 2: PROTEIN LANDSCAPE (Ranking by RCS - Rigid Density)
    # =================================================================================
    print("Generating Protein Landscape (RCS)...")

    # Save Protein Stats to text
    f_out.write("--- Top 5 Proteins by Rigid Conservation (Structural Rocks) ---\n")
    top_rcs_prots = stats_per_protein.sort_values('Mean_RCS', ascending=False).head(5)
    f_out.write(top_rcs_prots[['Label_ID', 'Mean_RCS', 'Rigid_Density_Pct']].to_string(index=False))
    f_out.write("\n\n")

    # Plot RCS Ranking
    stats_per_protein_rcs = stats_per_protein.sort_values('Mean_RCS', ascending=False)
    plot_data_rcs = pd.concat([stats_per_protein_rcs.head(top_n), stats_per_protein_rcs.tail(top_n)])

    plt.figure(figsize=(14, 8))
    sns.barplot(x='Label_ID', y='Mean_RCS', data=plot_data_rcs, hue='Rigid_Density_Pct', palette='magma', dodge=False)
    plt.title('Protein Landscape: Which Proteins are "Structural Rocks"?\n(Ranked by Mean RCS)', fontsize=16)
    plt.xlabel('Protein (UniProt ID)', fontsize=14)
    plt.ylabel('Average Rigid Conserved Score', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.legend(title='High-RCS Density (%)', loc='upper right')
    plt.grid(axis='y', alpha=0.3)

    save_path_rcs = os.path.join(output_folder, 'Protein_RCS_Ranking.png')
    plt.savefig(save_path_rcs, dpi=300, bbox_inches='tight')
    plt.savefig(save_path_rcs.replace('.png', '.svg'), format='svg', bbox_inches='tight')
    print(f"Saved Protein Ranking (RCS) to {save_path_rcs}")
    plt.close()

    # =================================================================================
    # ANALYSIS 3: THE DYNAMIC LANDSCAPE (DCS vs DCopS)
    # =================================================================================
    print("Generating Evolutionary Landscape (DCS vs DCopS)...")
    
    plt.figure(figsize=(10, 8))
    hb = plt.hexbin(
        x=df_clean['Dynamics_Coupling_Score_0_10'],
        y=df_clean['Dynamic_Conserved_Score_0_10'],
        gridsize=40,
        cmap='inferno',
        mincnt=1,
        bins='log'
    )
    cb = plt.colorbar(hb)
    cb.set_label('Residue Count (Log Scale)', fontsize=12, fontweight='bold')
    
    plt.xlabel('Dynamics Coupling Score (DCopS)\n(Evolutionary Buffer / Tolerance)', fontsize=14)
    plt.ylabel('Dynamic Conserved Score (DCS)\n(Functional Hinge / Fragility)', fontsize=14)
    plt.title('The Dynamic Landscape:\nFragility (DCS) vs Tolerance (DCopS)', fontsize=16)
    
    plt.text(1, 9, 'Allosteric Hinges\n(High DCS)', color='cyan', fontsize=12, fontweight='bold', ha='left')
    plt.text(9, 1, 'Flexible Buffers\n(High DCopS)', color='cyan', fontsize=12, fontweight='bold', ha='right')

    save_path_land = os.path.join(output_folder, 'Evolutionary_Landscape_DCS_vs_DCopS.png')
    plt.savefig(save_path_land, dpi=300)
    plt.savefig(save_path_land.replace('.png', '.svg'), format='svg')
    print(f"Saved Dynamic Landscape to {save_path_land}")
    plt.close()

    # =================================================================================
    # ANALYSIS 4: THE RIGID LANDSCAPE (RCS vs RCopS)
    # =================================================================================
    print("Generating Evolutionary Landscape (RCS vs RCopS)...")

    plt.figure(figsize=(10, 8))
    # Check if RCopS exists (it might be missing if enrichment wasn't calculated for all)
    if 'Rigid_Coupling_Score_0_10' in df_clean.columns:
        hb2 = plt.hexbin(
            x=df_clean['Rigid_Coupling_Score_0_10'],
            y=df_clean['Rigid_Conserved_Score_0_10'],
            gridsize=40,
            cmap='viridis',
            mincnt=1,
            bins='log'
        )
        cb2 = plt.colorbar(hb2)
        cb2.set_label('Residue Count (Log Scale)', fontsize=12, fontweight='bold')
        
        plt.xlabel('Rigid Coupling Score (RCopS)\n(Structural Co-evolution)', fontsize=14)
        plt.ylabel('Rigid Conserved Score (RCS)\n(Structural Core / Stability)', fontsize=14)
        plt.title('The Rigid Landscape:\nStability (RCS) vs Co-evolution (RCopS)', fontsize=16)
        
        plt.text(1, 9, 'Structural Core\n(High RCS)', color='yellow', fontsize=12, fontweight='bold', ha='left')
        plt.text(9, 1, 'Rigid Network\n(High RCopS)', color='yellow', fontsize=12, fontweight='bold', ha='right')

        save_path_land_rcs = os.path.join(output_folder, 'Evolutionary_Landscape_RCS_vs_RCopS.png')
        plt.savefig(save_path_land_rcs, dpi=300)
        plt.savefig(save_path_land_rcs.replace('.png', '.svg'), format='svg')
        print(f"Saved Rigid Landscape to {save_path_land_rcs}")
        plt.close()
    else:
        print("Skipping RCS vs RCopS plot (Rigid Coupling data missing).")
        f_out.write("\nNote: Rigid Coupling Score data was missing for plot generation.\n")

    f_out.close()
    print("Done.")

if __name__ == "__main__":
    generate_big_data_analysis()

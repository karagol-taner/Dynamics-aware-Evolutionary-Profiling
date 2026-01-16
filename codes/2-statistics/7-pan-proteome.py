import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set scientific styling
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['font.size'] = 12

def analyze_pan_proteome():
    # --- CONFIGURATION ---
    # Path to the Master TSV file containing all proteins and scores
    input_file = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-human/data_tsv/All_Proteins_DCS_Master.tsv'
    output_folder = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-human/pan_proteome_stats_rcs'
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    print(f"Loading Pan-Proteome Data from {input_file}...")
    
    try:
        # Load the master TSV
        # Use low_memory=False to avoid warnings on mixed types if any
        df_clean = pd.read_csv(input_file, sep='\t', na_values=['N/A', 'NA', ''])
    except Exception as e:
        print(f"Error reading file with tab separator: {e}")
        try:
            # Fallback to comma if tab fails
            df_clean = pd.read_csv(input_file, na_values=['N/A', 'NA', ''])
        except Exception as e2:
            print(f"Error reading file with comma separator: {e2}")
            return

    # Ensure numeric columns
    cols = ['conservation', 'Norm_Dynamics',
            'Dynamic_Conserved_Score_0_10', 'Dynamics_Coupling_Score_0_10',
            'Rigid_Conserved_Score_0_10', 'Rigid_Coupling_Score_0_10']
    
    for c in cols:
        if c in df_clean.columns:
            df_clean[c] = pd.to_numeric(df_clean[c], errors='coerce')
            
    # Drop rows where amino acid type is missing (essential for chemical logic)
    df_clean = df_clean.dropna(subset=['A_i'])
    
    # Filter for standard Amino Acids only
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    df_clean = df_clean[df_clean['A_i'].isin(valid_aa)]

    print(f"Successfully loaded {len(df_clean)} residues across {df_clean['Protein_ID'].nunique()} proteins.")

    # =================================================================================
    # ANALYSIS 0: TEXT REPORT GENERATION
    # =================================================================================
    print("Generating Statistical Report...")
    results_path = os.path.join(output_folder, 'pan_proteome_results.txt')
    
    with open(results_path, 'w') as f:
        f.write("DYNAMICS-AWARE EVOLUTIONARY PROFILING: PAN-PROTEOME STATISTICS\n")
        f.write("============================================================\n\n")
        f.write(f"Total Proteins Analyzed: {df_clean['Protein_ID'].nunique()}\n")
        f.write(f"Total Residues Analyzed: {len(df_clean)}\n\n")
        
        # DCS Stats
        if 'Dynamic_Conserved_Score_0_10' in df_clean.columns:
            f.write("--- Dynamic Conserved Score (DCS) ---\n")
            f.write(f"Mean: {df_clean['Dynamic_Conserved_Score_0_10'].mean():.4f}\n")
            f.write(f"Median: {df_clean['Dynamic_Conserved_Score_0_10'].median():.4f}\n")
            f.write(f"Standard Deviation: {df_clean['Dynamic_Conserved_Score_0_10'].std():.4f}\n")
            f.write(f"95th Percentile (High Confidence): {df_clean['Dynamic_Conserved_Score_0_10'].quantile(0.95):.4f}\n")
            f.write(f"99th Percentile (Top Hits): {df_clean['Dynamic_Conserved_Score_0_10'].quantile(0.99):.4f}\n\n")
        
        # DCopS Stats
        if 'Dynamics_Coupling_Score_0_10' in df_clean.columns:
            f.write("--- Dynamics Coupling Score (DCopS) ---\n")
            f.write(f"Mean: {df_clean['Dynamics_Coupling_Score_0_10'].mean():.4f}\n")
            f.write(f"Median: {df_clean['Dynamics_Coupling_Score_0_10'].median():.4f}\n")
            f.write(f"95th Percentile: {df_clean['Dynamics_Coupling_Score_0_10'].quantile(0.95):.4f}\n\n")

        # RCS Stats
        if 'Rigid_Conserved_Score_0_10' in df_clean.columns:
            f.write("--- Rigid Conserved Score (RCS) ---\n")
            f.write(f"Mean: {df_clean['Rigid_Conserved_Score_0_10'].mean():.4f}\n")
            f.write(f"Median: {df_clean['Rigid_Conserved_Score_0_10'].median():.4f}\n")
            f.write(f"95th Percentile: {df_clean['Rigid_Conserved_Score_0_10'].quantile(0.95):.4f}\n\n")

        # RCopS Stats
        if 'Rigid_Coupling_Score_0_10' in df_clean.columns:
            f.write("--- Rigid Coupling Score (RCopS) ---\n")
            f.write(f"Mean: {df_clean['Rigid_Coupling_Score_0_10'].mean():.4f}\n")
            f.write(f"Median: {df_clean['Rigid_Coupling_Score_0_10'].median():.4f}\n")
            f.write(f"95th Percentile: {df_clean['Rigid_Coupling_Score_0_10'].quantile(0.95):.4f}\n\n")
        
        if 'Dynamic_Conserved_Score_0_10' in df_clean.columns:
            f.write("--- Top 5 Amino Acids by Mean DCS ---\n")
            aa_means = df_clean.groupby('A_i')['Dynamic_Conserved_Score_0_10'].mean().sort_values(ascending=False).head(5)
            for aa, score in aa_means.items():
                f.write(f"{aa}: {score:.4f}\n")
            f.write("\n")

        if 'Rigid_Conserved_Score_0_10' in df_clean.columns:
            f.write("--- Top 5 Amino Acids by Mean RCS ---\n")
            aa_means_rcs = df_clean.groupby('A_i')['Rigid_Conserved_Score_0_10'].mean().sort_values(ascending=False).head(5)
            for aa, score in aa_means_rcs.items():
                f.write(f"{aa}: {score:.4f}\n")

    print(f"Results saved to {results_path}")

    # =================================================================================
    # FIGURE 1: THE UNIVERSAL DISTRIBUTION (Long Tail Proof)
    # =================================================================================
    print("Generating Universal Distribution Plots...")
    
    # Plot 1: Dynamic Scores
    plt.figure(figsize=(10, 6))
    if 'Dynamic_Conserved_Score_0_10' in df_clean.columns:
        sns.histplot(df_clean['Dynamic_Conserved_Score_0_10'], bins=50, kde=True, color='orange', alpha=0.6, label='DCS (Hinges)')
    if 'Dynamics_Coupling_Score_0_10' in df_clean.columns:
        sns.histplot(df_clean['Dynamics_Coupling_Score_0_10'], bins=50, kde=True, color='cyan', alpha=0.4, label='DCopS (Buffers)')
    
    plt.title('Universal Distribution: Dynamic Scores\n(N = ~200,000 Residues)', fontsize=16)
    plt.xlabel('Score (0-10)', fontsize=14)
    plt.ylabel('Frequency (Log Scale)', fontsize=14)
    plt.yscale('log') # Log scale proves the "Rare Event" hypothesis
    plt.legend()
    plt.grid(alpha=0.3)
    
    save_dist = os.path.join(output_folder, 'Universal_Score_Distribution_Dynamic.png')
    plt.savefig(save_dist, dpi=300)
    plt.savefig(save_dist.replace('.png', '.svg'), format='svg') # Save SVG
    print(f"Saved Dynamic Distribution to {save_dist}")
    plt.close()

    # Plot 2: Rigid Scores (New)
    plt.figure(figsize=(10, 6))
    if 'Rigid_Conserved_Score_0_10' in df_clean.columns:
        sns.histplot(df_clean['Rigid_Conserved_Score_0_10'], bins=50, kde=True, color='purple', alpha=0.6, label='RCS (Core)')
    if 'Rigid_Coupling_Score_0_10' in df_clean.columns:
        sns.histplot(df_clean['Rigid_Coupling_Score_0_10'], bins=50, kde=True, color='green', alpha=0.4, label='RCopS (Rigid Network)')
    
    plt.title('Universal Distribution: Rigid Scores\n(N = ~200,000 Residues)', fontsize=16)
    plt.xlabel('Score (0-10)', fontsize=14)
    plt.ylabel('Frequency (Log Scale)', fontsize=14)
    plt.yscale('log')
    plt.legend()
    plt.grid(alpha=0.3)
    
    save_dist_rigid = os.path.join(output_folder, 'Universal_Score_Distribution_Rigid.png')
    plt.savefig(save_dist_rigid, dpi=300)
    plt.savefig(save_dist_rigid.replace('.png', '.svg'), format='svg')
    print(f"Saved Rigid Distribution to {save_dist_rigid}")
    plt.close()

    # =================================================================================
    # FIGURE 2: THE UNIVERSAL PHASE SPACE (Cons vs Dyn)
    # =================================================================================
    print("Generating Universal Phase Space...")
    if 'conservation' in df_clean.columns and 'Norm_Dynamics' in df_clean.columns:
        plt.figure(figsize=(10, 8))
        
        # Hexbin of ALL data
        hb = plt.hexbin(
            x=df_clean['conservation'],
            y=df_clean['Norm_Dynamics'],
            gridsize=40,
            cmap='inferno',
            mincnt=1,
            bins='log'
        )
        
        cb = plt.colorbar(hb)
        cb.set_label('Residue Count (Log Scale)', fontsize=12, fontweight='bold')
        
        plt.xlabel('Evolutionary Conservation', fontsize=14)
        plt.ylabel('Normalized Dynamics', fontsize=14)
        plt.title('The Universal Landscape of Protein Physics\n(Cross-Species Analysis)', fontsize=16)
        
        # Annotation
        plt.text(0.95, 0.95, 'Universal Dynamic\nPharmacophores', color='cyan', ha='right', va='top', fontsize=12, fontweight='bold')

        save_phase = os.path.join(output_folder, 'Universal_Phase_Space.png')
        plt.savefig(save_phase, dpi=300)
        plt.savefig(save_phase.replace('.png', '.svg'), format='svg') # Save SVG
        print(f"Saved Phase Space to {save_phase}")
        plt.close()

    # =================================================================================
    # FIGURE 3: CHEMICAL CONSISTENCY (Amino Acid Boxplot)
    # =================================================================================
    print("Generating Chemical Consistency Plots...")
    
    # Plot 1: DCS (Dynamic)
    if 'Dynamic_Conserved_Score_0_10' in df_clean.columns:
        # Calculate order by median DCS
        order = df_clean.groupby("A_i")["Dynamic_Conserved_Score_0_10"].median().sort_values(ascending=False).index
        
        plt.figure(figsize=(14, 6))
        sns.boxplot(x='A_i', y='Dynamic_Conserved_Score_0_10', data=df_clean, order=order, palette='magma', showfliers=False)
        
        plt.title('Chemical Consistency: DCS Scores by Amino Acid', fontsize=16)
        plt.xlabel('Amino Acid', fontsize=14)
        plt.ylabel('DCS Score', fontsize=14)
        plt.grid(axis='y', alpha=0.3)
        
        save_chem = os.path.join(output_folder, 'Universal_Chemical_Logic_DCS.png')
        plt.savefig(save_chem, dpi=300)
        plt.savefig(save_chem.replace('.png', '.svg'), format='svg') # Save SVG
        print(f"Saved Chemical Logic (DCS) to {save_chem}")
        plt.close()

    # Plot 2: RCS (Rigid)
    if 'Rigid_Conserved_Score_0_10' in df_clean.columns:
        # Calculate order by median RCS
        order_rcs = df_clean.groupby("A_i")["Rigid_Conserved_Score_0_10"].median().sort_values(ascending=False).index
        
        plt.figure(figsize=(14, 6))
        sns.boxplot(x='A_i', y='Rigid_Conserved_Score_0_10', data=df_clean, order=order_rcs, palette='viridis', showfliers=False)
        
        plt.title('Chemical Consistency: RCS Scores by Amino Acid', fontsize=16)
        plt.xlabel('Amino Acid', fontsize=14)
        plt.ylabel('RCS Score', fontsize=14)
        plt.grid(axis='y', alpha=0.3)
        
        save_chem_rcs = os.path.join(output_folder, 'Universal_Chemical_Logic_RCS.png')
        plt.savefig(save_chem_rcs, dpi=300)
        plt.savefig(save_chem_rcs.replace('.png', '.svg'), format='svg') # Save SVG
        print(f"Saved Chemical Logic (RCS) to {save_chem_rcs}")
        plt.close()

    print("Analysis Complete.")

if __name__ == "__main__":
    analyze_pan_proteome()

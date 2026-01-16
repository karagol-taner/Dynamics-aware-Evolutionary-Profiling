import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats  # Added for statistical tests
import os

# Set global font weight to bold for all plots
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['font.size'] = 12

def generate_biophysical_analysis():
    # --- CONFIGURATION ---
    input_file = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-human/data_tsv/All_Proteins_AM_Combined_Master.tsv'
    
    # Absolute path for output
    output_folder = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-human/biophysical_stats_rcs'
    
    # Create folder
    if not os.path.exists(output_folder):
        try:
            os.makedirs(output_folder)
            print(f"Created output folder: {output_folder}")
        except OSError:
            output_folder = './biophysical_plots'
            os.makedirs(output_folder, exist_ok=True)
            print("Fallback: Created local output folder.")

    print("Loading Master Dataset...")
    try:
        df = pd.read_csv(input_file, sep='\t', na_values=['N/A', 'NA', ''])
    except:
        df = pd.read_csv(input_file, na_values=['N/A', 'NA', ''])

    # Ensure numeric columns
    numeric_cols = [
        'conservation', 'Norm_Dynamics',
        'Dynamic_Conserved_Score_0_10', 'Dynamics_Coupling_Score_0_10',
        'Rigid_Conserved_Score_0_10', 'Rigid_Coupling_Score_0_10',
        'mean_am_pathogenicity'
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            
    # Clean data (must have AA type and scores)
    if 'A_i' not in df.columns:
        print("Error: Column 'A_i' (Amino Acid Type) not found in master file.")
        return

    # Filter for standard Amino Acids
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    df_clean = df.dropna(subset=['A_i', 'mean_am_pathogenicity'])
    df_clean = df_clean[df_clean['A_i'].isin(valid_aa)]

    print(f"Analyzing {len(df_clean)} residues...")
    
    # Initialize Results Text File
    results_path = os.path.join(output_folder, 'biophysical_results.txt')
    with open(results_path, 'w') as f:
        f.write("BIOPHYSICAL ANALYSIS RESULTS\n")
        f.write("============================\n\n")

    # =================================================================================
    # ANALYSIS 1: AMINO ACID PROPENSITY (Loop through all scores)
    # =================================================================================
    print("Generating Amino Acid Propensity Plots...")
    
    scores_to_plot = {
        'DCS': ('Dynamic_Conserved_Score_0_10', 'magma'),
        'DCopS': ('Dynamics_Coupling_Score_0_10', 'viridis'),
        'RCS': ('Rigid_Conserved_Score_0_10', 'rocket'),
        'RCopS': ('Rigid_Coupling_Score_0_10', 'mako')
    }

    for name, (col, palette) in scores_to_plot.items():
        if col not in df_clean.columns:
            print(f"Skipping {name}: Column {col} not found.")
            continue
            
        print(f"  Processing {name}...")
        
        # 1. Determine Sort Order by Mean
        aa_order = df_clean.groupby('A_i')[col].mean().sort_values(ascending=False).index

        # 2. Statistical Test: Kruskal-Wallis
        groups = [df_clean[df_clean['A_i'] == aa][col].values for aa in aa_order]
        kw_stat, kw_p = stats.kruskal(*groups)
        
        # Write to file
        with open(results_path, 'a') as f:
            f.write(f"--- {name} ({col}) ---\n")
            f.write(f"Kruskal-Wallis Test (AA Bias): P = {kw_p:.4e}\n")
            f.write("Top 3 Enriched Amino Acids:\n")
            for aa in aa_order[:3]:
                val = df_clean[df_clean['A_i'] == aa][col].mean()
                f.write(f"  {aa}: {val:.4f}\n")
            f.write("\n")

        # 3. Plot
        plt.figure(figsize=(12, 6))
        sns.barplot(
            x='A_i',
            y=col,
            data=df_clean,
            order=aa_order,
            palette=palette,
            errorbar=('ci', 95),
            capsize=0.1
        )
        
        plt.title(f'{name} Amino Acid Propensity\n(Kruskal-Wallis P < {kw_p:.1e})', fontsize=15, fontweight='bold')
        plt.xlabel('Amino Acid Type', fontsize=14, fontweight='bold')
        plt.ylabel(f'Average {name} (95% CI)', fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(fontweight='bold')
        plt.yticks(fontweight='bold')

        save_name = f'Amino_Acid_Propensity_{name}'
        plt.savefig(os.path.join(output_folder, f'{save_name}.png'), dpi=300)
        plt.savefig(os.path.join(output_folder, f'{save_name}.svg'), format='svg')
        plt.close()

    # =================================================================================
    # ANALYSIS 2: PHASE SPACE MAP (2D Enrichment)
    # =================================================================================
    print("Generating Phase Space Map...")
    
    # Ensure inputs are valid
    phase_df = df_clean.dropna(subset=['conservation', 'Norm_Dynamics', 'mean_am_pathogenicity'])
    
    if len(phase_df) > 0:
        # 1. Statistical Comparison: Danger Zone vs Buffer Zone
        # Danger: High Cons (>0.6) & High Dyn (>0.4)
        # Buffer: Low Cons (<0.4) & High Dyn (>0.4)
        danger_zone = phase_df[(phase_df['conservation'] > 0.6) & (phase_df['Norm_Dynamics'] > 0.4)]
        buffer_zone = phase_df[(phase_df['conservation'] < 0.4) & (phase_df['Norm_Dynamics'] > 0.4)]
        
        mw_p = 1.0
        if len(danger_zone) > 0 and len(buffer_zone) > 0:
            mw_stat, mw_p = stats.mannwhitneyu(
                danger_zone['mean_am_pathogenicity'],
                buffer_zone['mean_am_pathogenicity'],
                alternative='greater'
            )
        
        with open(results_path, 'a') as f:
            f.write("--- Phase Space Analysis ---\n")
            f.write(f"Danger Zone (High Cons/High Dyn) N={len(danger_zone)}\n")
            f.write(f"Buffer Zone (Low Cons/High Dyn) N={len(buffer_zone)}\n")
            f.write(f"Mann-Whitney U Test (Pathogenicity: Danger > Buffer): P = {mw_p:.4e}\n")

        plt.figure(figsize=(10, 8))
        
        hb = plt.hexbin(
            x=phase_df['conservation'],
            y=phase_df['Norm_Dynamics'],
            C=phase_df['mean_am_pathogenicity'],
            gridsize=30,
            cmap='RdYlBu_r', # Red = High Pathogenicity
            reduce_C_function=np.mean,
            mincnt=10
        )
        
        cb = plt.colorbar(hb)
        cb.set_label('Mean AlphaMissense Pathogenicity', fontsize=12, fontweight='bold')
        cb.ax.tick_params(labelsize=10, width=2)
        for l in cb.ax.yaxis.get_ticklabels(): l.set_weight('bold')

        plt.xlabel('Evolutionary Conservation (0=Variable, 1=Conserved)', fontsize=14, fontweight='bold')
        plt.ylabel('Normalized Dynamics (0=Rigid, 1=Flexible)', fontsize=14, fontweight='bold')
        plt.title('Pathogenicity Landscape: Dynamics vs. Conservation', fontsize=16, fontweight='bold')
        plt.xticks(fontweight='bold')
        plt.yticks(fontweight='bold')
        
        # Statistical Annotation
        stats_text = f"Statistical Comparison:\nHigh-Risk (Top Right) vs Buffer (Top Left)\nMann-Whitney P < {mw_p:.1e}"
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, ha='left', va='top',
                 fontsize=11, fontweight='bold', bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.5'))

        plt.savefig(os.path.join(output_folder, 'Phase_Space_Pathogenicity_Map.png'), dpi=300)
        plt.savefig(os.path.join(output_folder, 'Phase_Space_Pathogenicity_Map.svg'), format='svg')
        plt.close()
    
    print(f"Done. All results saved to {output_folder}")

if __name__ == "__main__":
    generate_biophysical_analysis()

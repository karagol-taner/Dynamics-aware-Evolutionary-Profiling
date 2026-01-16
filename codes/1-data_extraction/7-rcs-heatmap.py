import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def generate_heatmap_from_df(df, protein_id, output_folder):
    """
    Generates a heatmap for RCS and RCopS (Rigid Scores Only) for a specific protein DataFrame.
    """
    # 1. Prepare Data for Plotting
    cols_to_plot = []
    
    # Add Rigid Scores ONLY (as requested)
    if 'Rigid_Conserved_Score_0_10' in df.columns:
        cols_to_plot.append('Rigid_Conserved_Score_0_10')
    if 'Rigid_Coupling_Score_0_10' in df.columns:
        cols_to_plot.append('Rigid_Coupling_Score_0_10')

    if not cols_to_plot:
        print(f"Skipping {protein_id}: No rigid score columns found.")
        return

    # Create a subset for plotting
    plot_df = df.copy()

    # Ensure residue index 'i' is the index and sorted
    if 'i' in plot_df.columns:
        plot_df['i'] = pd.to_numeric(plot_df['i'], errors='coerce')
        plot_df = plot_df.dropna(subset=['i'])
        plot_df['i'] = plot_df['i'].astype(int)
        plot_df = plot_df.sort_values('i') # Ensure sequence order
        plot_df = plot_df.set_index('i')
    else:
        print(f"Skipping {protein_id}: No residue index column 'i' found.")
        return

    # Select only the score columns
    heatmap_data = plot_df[cols_to_plot].transpose()

    # 2. Setup Plot
    # Width: Dynamic based on protein length
    n_residues = len(plot_df)
    figsize_width = max(10, min(n_residues * 0.15, 50))

    # Reduced height since we are plotting fewer rows
    plt.figure(figsize=(figsize_width, 3))

    # 3. Create Heatmap
    sns.heatmap(heatmap_data, cmap='viridis', vmin=0, vmax=10,
                cbar_kws={'label': 'Score (0-10)'}, linewidths=0.0)

    plt.title(f'Rigid Evolutionary Profile: {protein_id}')
    plt.xlabel('Residue Index')
    plt.ylabel('Score Metric')
    plt.tight_layout()

    # 4. Save
    output_name_svg = f"Heatmap_Rigid_{protein_id}.svg"
    output_path_svg = os.path.join(output_folder, output_name_svg)
    plt.savefig(output_path_svg, format='svg')

    output_name_png = f"Heatmap_Rigid_{protein_id}.png"
    output_path_png = os.path.join(output_folder, output_name_png)
    plt.savefig(output_path_png, format='png', dpi=300)

    plt.close()

    print(f"Heatmap generated for: {protein_id}")

def main():
    # --- CONFIGURATION ---
    input_file = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-human/data_tsv/All_Proteins_DCS_Master.tsv'
    output_folder = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-human/RCS_heatmaps_final'
    # ---------------------

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"Loading Master File: {input_file}...")
    try:
        # Load the master TSV
        df_master = pd.read_csv(input_file, sep='\t', na_values=['N/A', 'NA', ''])
    except Exception as e:
        print(f"Error reading master file: {e}")
        return

    # Ensure numeric columns
    numeric_cols = [
        'Rigid_Conserved_Score_0_10', 'Rigid_Coupling_Score_0_10', 'i'
    ]
    for c in numeric_cols:
        if c in df_master.columns:
            df_master[c] = pd.to_numeric(df_master[c], errors='coerce')

    # Group by Protein_ID and generate heatmap for each
    if 'Protein_ID' not in df_master.columns:
         print("Error: 'Protein_ID' column missing in master file.")
         return

    unique_proteins = df_master['Protein_ID'].dropna().unique()
    print(f"Found {len(unique_proteins)} proteins. Generating rigid heatmaps...")

    for protein_id in unique_proteins:
        protein_df = df_master[df_master['Protein_ID'] == protein_id].copy()
        generate_heatmap_from_df(protein_df, str(protein_id), output_folder)

    print("Heatmap generation complete.")

if __name__ == "__main__":
    main()

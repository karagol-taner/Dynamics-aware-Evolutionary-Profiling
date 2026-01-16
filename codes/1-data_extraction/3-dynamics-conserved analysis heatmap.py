import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

def generate_heatmap(file_path, output_folder):
    """
    Generates a heatmap for the computed scores in a CSV and saves it as an SVG and PNG.
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    base_name = os.path.basename(file_path)

    # 1. Prepare Data for Plotting
    # specific columns we want to visualize
    cols_to_plot = []
    if 'Dynamic_Conserved_Score_0_10' in df.columns:
        cols_to_plot.append('Dynamic_Conserved_Score_0_10')
    if 'Dynamics_Coupling_Score_0_10' in df.columns:
        cols_to_plot.append('Dynamics_Coupling_Score_0_10')
    
    if not cols_to_plot:
        print(f"Skipping {base_name}: Score columns not found.")
        return

    # Create a subset for plotting
    plot_df = df.copy()
    
    # We need the residue index 'i' to be the X-axis.
    # Drop rows where 'i' is N/A for the visualization
    if 'i' in plot_df.columns:
        # Ensure it's numeric/integer
        plot_df['i'] = pd.to_numeric(plot_df['i'], errors='coerce')
        plot_df = plot_df.dropna(subset=['i'])
        plot_df['i'] = plot_df['i'].astype(int)
        plot_df = plot_df.set_index('i')
    else:
        # Fallback if 'i' is missing: use default index
        plot_df.index.name = 'Residue Index'

    # Select only the score columns
    heatmap_data = plot_df[cols_to_plot].transpose() # Transpose: X=Residues, Y=Scores
    
    # 2. Setup Plot
    # Width: Dynamic based on protein length, but max out to avoid huge files
    # Approx 0.15 inch per residue, min 10 inches, max 50 inches
    n_residues = len(plot_df)
    figsize_width = max(10, min(n_residues * 0.15, 50))
    
    plt.figure(figsize=(figsize_width, 4))
    
    # 3. Create Heatmap
    # cmap='viridis' is good for 0-10 intensity
    sns.heatmap(heatmap_data, cmap='viridis', vmin=0, vmax=10,
                cbar_kws={'label': 'Score (0-10)'}, linewidths=0.0)
    
    plt.title(f'Dynamics-Aware Evolutionary Profile: {base_name}')
    plt.xlabel('Residue Index')
    plt.ylabel('Score Type')
    plt.tight_layout()
    
    # 4. Save
    # Save SVG
    output_name_svg = f"Heatmap_{base_name.replace('.csv', '')}.svg"
    output_path_svg = os.path.join(output_folder, output_name_svg)
    plt.savefig(output_path_svg, format='svg')

    # Save PNG (High Quality)
    output_name_png = f"Heatmap_{base_name.replace('.csv', '')}.png"
    output_path_png = os.path.join(output_folder, output_name_png)
    plt.savefig(output_path_png, format='png', dpi=300)

    plt.close() # Close figure to free memory
    
    print(f"Heatmap generated for: {base_name}")

def main():
    # --- CONFIGURATION ---
    # Point this to the folder containing your ALREADY PROCESSED CSV files (with the scores)
    input_folder = '/content/drive/MyDrive/All Project Files/dynamics aware evo/dynamics_aware_results_3'
    output_folder = '/content/drive/MyDrive/All Project Files/dynamics aware evo/dynamics_aware_heatmaps_2' # Folder for the SVG images
    # ---------------------

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Find all CSV files
    files = glob.glob(os.path.join(input_folder, '*.csv'))
    
    if not files:
        print(f"No CSV files found in {input_folder}. Please check the path.")
        return

    print(f"Found {len(files)} files. Generating heatmaps...")

    for file_path in files:
        generate_heatmap(file_path, output_folder)

    print("Heatmap generation complete.")

if __name__ == "__main__":
    main()

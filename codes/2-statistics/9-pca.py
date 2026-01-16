import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# Global Styles
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['font.size'] = 12

def run_pca_only():
    # --- CONFIGURATION ---
    # We use the Global Data (All Proteins/Species) for unsupervised PCA
    input_file_global = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-human/data_tsv/All_Proteins_DCS_Master.tsv'
    output_folder = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-human/pca_stats_rcs'
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    print("Loading Global Data (All Proteins)...")
    try:
        if os.path.exists(input_file_global):
            try:
                df_global = pd.read_csv(input_file_global, sep='\t', na_values=['N/A', 'NA', ''])
            except:
                df_global = pd.read_csv(input_file_global, na_values=['N/A', 'NA', ''])
            
            # Use all available physical metrics for the landscape
            cols_struct = [
                'conservation', 'Norm_Dynamics',
                'Dynamic_Conserved_Score_0_10', 'Dynamics_Coupling_Score_0_10',
                'Rigid_Conserved_Score_0_10', 'Rigid_Coupling_Score_0_10'
            ]
            
            # Check availability
            valid_cols = [c for c in cols_struct if c in df_global.columns]
            for c in valid_cols:
                df_global[c] = pd.to_numeric(df_global[c], errors='coerce')
            
            df_global_clean = df_global.dropna(subset=valid_cols)
            print(f"  Successfully loaded Global Dataset: {len(df_global_clean)} residues.")
        else:
            print("  Error: Global dataset file not found.")
            return
    except Exception as e:
        print(f"  Error loading global dataset: {e}")
        return

    # Define results file path
    results_path = os.path.join(output_folder, 'pca_results.txt')

    with open(results_path, 'w') as res_file:
        res_file.write("DYNAMICS-AWARE EVOLUTIONARY PROFILING: UNSUPERVISED PCA\n")
        res_file.write("=======================================================\n\n")
        res_file.write(f"Total Residues Analyzed: {len(df_global_clean)}\n")
        res_file.write(f"Features Used: {', '.join(valid_cols)}\n\n")
        
        # =================================================================================
        # ANALYSIS: UNSUPERVISED PCA LANDSCAPE (Global Data)
        # =================================================================================
        print("Running Unsupervised PCA on Global Dataset...")
        
        # Subsample if too large for visualization speed (e.g. 50k)
        if len(df_global_clean) > 50000:
            df_pca = df_global_clean.sample(50000, random_state=42)
        else:
            df_pca = df_global_clean
        
        X_pca = df_pca[valid_cols]
        # Standardize
        X_pca_std = StandardScaler().fit_transform(X_pca)
        
        pca = PCA(n_components=2)
        components = pca.fit_transform(X_pca_std)
        
        # Plot
        plt.figure(figsize=(10, 8))
        
        # Color by DCS to show gradient distribution
        # High DCS residues will show as bright yellow/orange
        sc = plt.scatter(components[:, 0], components[:, 1],
                         c=df_pca['Dynamic_Conserved_Score_0_10'],
                         cmap='inferno', s=2, alpha=0.6)
        
        cb = plt.colorbar(sc)
        cb.set_label('DCS Score (Dynamic Conservation)', fontsize=12, fontweight='bold')
        
        plt.xlabel(f'PC1 (Variance Explained: {pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
        plt.ylabel(f'PC2 (Variance Explained: {pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
        plt.title('PCA Landscape: The Biophysical Space of the Proteome', fontsize=16)
        
        plt.savefig(os.path.join(output_folder, 'Global_PCA_Landscape.png'), dpi=300)
        plt.savefig(os.path.join(output_folder, 'Global_PCA_Landscape.svg'), format='svg')
        plt.close()
        
        res_file.write("\nPCA VARIANCE EXPLAINED\n")
        res_file.write("----------------------\n")
        res_file.write(f"PC1 Variance: {pca.explained_variance_ratio_[0]:.4f}\n")
        res_file.write(f"PC2 Variance: {pca.explained_variance_ratio_[1]:.4f}\n")
        res_file.write(f"Total Variance Explained (PC1+PC2): {np.sum(pca.explained_variance_ratio_):.4f}\n")
        res_file.write("\nInterpretation: This map visualizes the continuous biophysical space occupied by the proteome.\n")
        res_file.write("PC1 typically separates Conserved vs Variable residues.\n")
        res_file.write("PC2 typically separates Rigid vs Dynamic residues.\n")

    print(f"PCA Analysis Complete. Results saved to: {output_folder}")

if __name__ == "__main__":
    run_pca_only()

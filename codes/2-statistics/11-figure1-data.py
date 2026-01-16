import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import os

# --- PUBLICATION STYLE SETTINGS ---
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['font.size'] = 14  # Increased base font size for readability
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['xtick.major.width'] = 2.0
plt.rcParams['ytick.major.width'] = 2.0

def generate_figure_1_combined_c_wider():
    # --- CONFIGURATION ---
    input_file_1a = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-alphahelical/data_tsv/All_Proteins_DCS_Master.tsv'
    input_file_1b = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-human/data_tsv/All_Proteins_AM_Combined_Master.tsv'
    output_folder = '/content/drive/MyDrive/All Project Files/dynamics aware evo/figures/figure_1'
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    print("Loading Datasets...")
    try:
        # Load Dataset A (Alphahelical/Global)
        try:
            df_a = pd.read_csv(input_file_1a, sep='\t', na_values=['N/A', 'NA', ''])
        except:
            df_a = pd.read_csv(input_file_1a, na_values=['N/A', 'NA', ''])
            
        # Load Dataset B (Human/Pathogenicity)
        try:
            df_b = pd.read_csv(input_file_1b, sep='\t', na_values=['N/A', 'NA', ''])
        except:
            df_b = pd.read_csv(input_file_1b, na_values=['N/A', 'NA', ''])

        # --- PREPROCESS ---
        cols_a = ['conservation', 'Norm_Dynamics', 'Dynamic_Conserved_Score_0_10', 'Rigid_Conserved_Score_0_10']
        for c in cols_a:
            if c in df_a.columns: df_a[c] = pd.to_numeric(df_a[c], errors='coerce')
        df_clean_a = df_a.dropna(subset=cols_a).copy()
        
        df_pca = df_clean_a.copy()
        if len(df_pca) > 50000: df_pca = df_pca.sample(50000, random_state=42)
        X_std = StandardScaler().fit_transform(df_pca[['conservation', 'Norm_Dynamics']])
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X_std)
        var_exp = pca.explained_variance_ratio_ * 100

        cols_b = ['Dynamic_Conserved_Score_0_10', 'Rigid_Conserved_Score_0_10',
                  'conservation', 'Norm_Dynamics', 'mean_am_pathogenicity']
        short_labels = ['DCS', 'RCS', 'Conserv.', 'Dynamics', 'Patho. (AM)']
        
        for c in cols_b:
            if c in df_b.columns: df_b[c] = pd.to_numeric(df_b[c], errors='coerce')
        df_corr = df_b.dropna(subset=cols_b).copy()
        corr_mat = df_corr[cols_b].corr(method='spearman')

    except Exception as e:
        print(f"Error loading/processing data: {e}")
        return

    # =================================================================
    # SETUP LAYOUT
    # =================================================================
    fig = plt.figure(figsize=(16, 17))
    
    # GridSpec Config:
    gs = gridspec.GridSpec(2, 4, height_ratios=[1, 1], width_ratios=[1, 1, 1, 0.6],
                           wspace=0.9, hspace=0.15)

    ax1 = fig.add_subplot(gs[0, 0:2]) # Top Left (Cols 0-1)
    ax2 = fig.add_subplot(gs[0, 2:4]) # Top Right (Cols 2-3)
    ax3 = fig.add_subplot(gs[1, 1:3]) # Bottom Center (Cols 1-2)

    # Force Exact Squares
    ax1.set_box_aspect(1)
    ax2.set_box_aspect(1)
    ax3.set_box_aspect(0.90)

    # =================================================================
    # PANEL A: PCA
    # =================================================================
    print("Plotting Panel A...")
    sc = ax1.scatter(coords[:, 0], coords[:, 1],
                     c=df_pca['Dynamic_Conserved_Score_0_10'],
                     cmap='inferno', s=10, alpha=0.75, edgecolors='none')
    
    cbar = plt.colorbar(sc, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('DCS Score', fontsize=14, weight='bold')
    
    ax1.set_xlabel(f'PC1: Conservation ({var_exp[0]:.1f}%)', fontsize=14)
    ax1.set_ylabel(f'PC2: Dynamics ({var_exp[1]:.1f}%)', fontsize=14)
    ax1.set_title('a) The Orthogonality of Evolutionary Constraints', pad=15, fontsize=16)
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    ax1.arrow(0, 0, 2.5, 0, color='white', width=0.12, head_width=0.35, length_includes_head=True, zorder=9)
    ax1.arrow(0, 0, 2.5, 0, color='navy', width=0.08, head_width=0.3, length_includes_head=True, alpha=0.9, zorder=10)
    ax1.text(2.6, 0.3, 'Stability', color='navy', fontsize=16, fontweight='bold', zorder=10, ha='left',
             path_effects=[pe.withStroke(linewidth=3, foreground="white")])
    
    ax1.arrow(0, 0, 0, 2.5, color='white', width=0.12, head_width=0.35, length_includes_head=True, zorder=9)
    ax1.arrow(0, 0, 0, 2.5, color='darkred', width=0.08, head_width=0.3, length_includes_head=True, alpha=0.9, zorder=10)
    ax1.text(0.3, 2.6, 'Dynamics', color='darkred', fontsize=16, fontweight='bold', zorder=10, ha='left',
             path_effects=[pe.withStroke(linewidth=3, foreground="white")])

    # =================================================================
    # PANEL B: CORRELATION
    # =================================================================
    print("Plotting Panel B...")
    sns.heatmap(corr_mat, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                vmin=-1, vmax=1, square=True, cbar=False,
                xticklabels=short_labels, yticklabels=short_labels,
                annot_kws={"size": 13, "weight": "bold"}, ax=ax2)
    
    ax2.set_title('b) Statistical Independence (Human Proteome)', pad=15, fontsize=16)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=12)
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=12)
    ax2.grid(False)
    
    rect = Rectangle((1, 0), 1, 1, fill=False, edgecolor='black', lw=4, clip_on=False)
    ax2.add_patch(rect)

    # =================================================================
    # PANEL C: DISTRIBUTIONAL ASYMMETRY
    # =================================================================
    print("Plotting Panel C...")
    
    # 1. Calculate 80th Percentiles (The "Top 20%")
    rcs_80 = df_clean_a['Rigid_Conserved_Score_0_10'].quantile(0.80)
    dcs_80 = df_clean_a['Dynamic_Conserved_Score_0_10'].quantile(0.80)

    # 2. Define Labels with Values (simplified)
    label_rcs = f'RCS | Top 20% > {rcs_80:.2f}'
    label_dcs = f'DCS | Top 20% > {dcs_80:.2f}'

    # 3. Plot KDEs
    sns.kdeplot(df_clean_a['Rigid_Conserved_Score_0_10'], fill=True,
                color='#440154', label=label_rcs,
                alpha=0.3, linewidth=3, ax=ax3)
    
    sns.kdeplot(df_clean_a['Dynamic_Conserved_Score_0_10'], fill=True,
                color='#FDE725', label=label_dcs,
                alpha=0.4, linewidth=3, ax=ax3)
    
    # 4. Add Vertical Lines for 80th Percentiles
    ax3.axvline(x=rcs_80, color='#440154', linestyle='--', linewidth=2.5, alpha=0.9)
    ax3.axvline(x=dcs_80, color='#FDE725', linestyle='--', linewidth=2.5, alpha=1.0)
    
    ax3.set_title('c) Distributional Asymmetry of Constraints', pad=15, fontsize=16)
    ax3.set_xlabel('Score Magnitude (0-10)', fontsize=14)
    ax3.set_ylabel('Density', fontsize=14)
    ax3.set_xlim(0, 10)
    ax3.grid(linestyle='--', alpha=0.3)
    
    # Text Annotations with background box for readability
    bbox_props = dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8)
    
    ax3.text(4.5, 0.35, "Stability is Global\n(Broad Distribution)",
             color='#440154', fontsize=15, fontweight='bold', bbox=bbox_props)
    
    ax3.text(1.5, 0.8, "Dynamics is Sparse\n(Heavy-Tailed)",
             color='#D4C300', fontsize=15, fontweight='bold', ha='left', bbox=bbox_props)
    
    # Legend
    ax3.legend(loc='upper right', fontsize=13, frameon=True, framealpha=0.95)

    # =================================================================
    # SAVE
    # =================================================================
    plt.tight_layout(pad=3.0, h_pad=2.0)
    
    print(f"Saving Figure 1 (Wider C) to {output_folder}...")
    plt.savefig(os.path.join(output_folder, 'Figure1_Combined_WiderC_Top20.png'), dpi=300)
    plt.savefig(os.path.join(output_folder, 'Figure1_Combined_WiderC_Top20.svg'), format='svg')
    plt.show()
    plt.close()
    
    print("Done.")

if __name__ == "__main__":
    generate_figure_1_combined_c_wider()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import os

# --- CONFIGURATION ---
INPUT_TSV_MAIN = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-human/data_tsv/All_Proteins_AM_Combined_Master.tsv'
INPUT_CSV_CLINVAR = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-human/Clinvar/ClinVar_All_AlphaMissense_DCS_Analysis-check.csv'
OUTPUT_FOLDER = '/content/drive/MyDrive/All Project Files/dynamics aware evo/figures/figure_3'

# --- PUBLICATION STYLE ---
sns.set_context("paper", font_scale=1.4)
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 2.5
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.6

def load_clinvar_data(file_path):
    print(f"Loading ClinVar data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        
        if 'AM_Specific_Score' in df.columns and 'AM_Mean_Residue_Score' in df.columns:
            df['AM_Score'] = df['AM_Specific_Score'].fillna(df['AM_Mean_Residue_Score'])
        elif 'AM_Specific_Score' in df.columns:
            df['AM_Score'] = df['AM_Specific_Score']
        else:
            return None

        if 'DCS_Score' in df.columns:
            df['Plot_DCS'] = df['DCS_Score']
        else:
            return None

        col_class = next((c for c in df.columns if 'ClinVar_Significance' in c or 'ClinVar_Class' in c), None)
        if not col_class:
            return None
        
        df['AM_Score'] = pd.to_numeric(df['AM_Score'], errors='coerce')
        df['Plot_DCS'] = pd.to_numeric(df['Plot_DCS'], errors='coerce')
        df_clean = df.dropna(subset=['AM_Score', 'Plot_DCS', col_class]).copy()

        def simplify_class(c):
            c = str(c).lower()
            if 'pathogenic' in c and 'benign' not in c and 'conflicting' not in c:
                return 'Pathogenic'
            if 'benign' in c and 'pathogenic' not in c and 'conflicting' not in c:
                return 'Benign'
            return 'Ambiguous/Uncertain'

        df_clean['Plot_Class'] = df_clean.apply(lambda row: simplify_class(row[col_class]), axis=1)
        return df_clean
    except Exception as e:
        print(f"Error reading ClinVar file: {e}")
        return None

def generate_figure_3_complete():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 1. LOAD MAIN DATA (Panels a-e)
    print("Loading Human Dataset (Main)...")
    try:
        try:
            df_main = pd.read_csv(INPUT_TSV_MAIN, sep='\t', na_values=['N/A', 'NA', ''])
        except:
            df_main = pd.read_csv(INPUT_TSV_MAIN, na_values=['N/A', 'NA', ''])
            
        cols = ['conservation', 'Norm_Dynamics',
                'Dynamic_Conserved_Score_0_10', 'Rigid_Conserved_Score_0_10',
                'mean_am_pathogenicity']
        for c in cols:
            if c in df_main.columns: df_main[c] = pd.to_numeric(df_main[c], errors='coerce')
        df_main = df_main.dropna(subset=cols).copy()
        df_main['is_pathogenic'] = (df_main['mean_am_pathogenicity'] > 0.90).astype(int)
        
        # Clustering
        X = df_main[['conservation', 'Norm_Dynamics']].values
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        df_main['Cluster'] = kmeans.fit_predict(X_std)
        
        cluster_stats = df_main.groupby('Cluster')[['conservation', 'Norm_Dynamics']].mean()
        cluster_stats['Prod'] = cluster_stats['conservation'] * cluster_stats['Norm_Dynamics']
        
        core_idx = cluster_stats['conservation'].idxmax()
        remaining = list(set(cluster_stats.index) - {core_idx})
        loop_idx = cluster_stats.loc[remaining, 'Norm_Dynamics'].idxmax()
        remaining = list(set(remaining) - {loop_idx})
        hinge_idx = cluster_stats.loc[remaining, 'Prod'].idxmax()
        spacer_idx = list(set(remaining) - {hinge_idx})[0]
        
        label_map = {core_idx: 'Rigid-Conserved (RCS)', hinge_idx: 'Dynamic-Conserved (DCS)',
                     loop_idx: 'Buffer (Loop)', spacer_idx: 'Spacer'}
        df_main['Label'] = df_main['Cluster'].map(label_map)

    except Exception as e:
        print(f"Error processing main data: {e}")
        return

    colors_map = {'Rigid-Conserved (RCS)': '#440154', 'Dynamic-Conserved (DCS)': '#FF5733',
                  'Buffer (Loop)': '#21918c', 'Spacer': 'lightgray'}

    # 2. SETUP LAYOUT
    fig = plt.figure(figsize=(16, 22))
    gs = gridspec.GridSpec(3, 2)
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_c, ax_d = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    ax_e, ax_f = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])

    # PANELS A-C (Standard)
    sc = ax_a.scatter(df_main['conservation'], df_main['Norm_Dynamics'],
                      c=df_main['mean_am_pathogenicity'], cmap='viridis_r', s=15, alpha=0.5, edgecolor='none')
    ax_a.set_title('a) Feature-Weighted Distribution'); ax_a.set_xlabel('Conservation'); ax_a.set_ylabel('Dynamics')
    cax = ax_a.inset_axes([0.55, 0.88, 0.4, 0.03]); plt.colorbar(sc, cax=cax, orientation='horizontal').set_label('AM Score', fontsize=10); cax.set_facecolor('none')

    target_labels = ['Dynamic-Conserved (DCS)', 'Rigid-Conserved (RCS)']
    sns.boxplot(x='Label', y='mean_am_pathogenicity', data=df_main[df_main['Label'].isin(target_labels)],
                order=target_labels, palette=colors_map, showfliers=False, ax=ax_b)
    ax_b.set_title('b) Pathogenicity Distribution'); ax_b.set_ylabel('Pathogenicity (AM)')

    top_n = int(len(df_main) * 0.10)
    idx_cons = set(df_main.nlargest(top_n, 'conservation').index)
    idx_rcs = set(df_main.nlargest(top_n, 'Rigid_Conserved_Score_0_10').index)
    idx_dcs = set(df_main.nlargest(top_n, 'Dynamic_Conserved_Score_0_10').index)
    bars = ax_c.bar(['Cons. Only', 'RCS Only', 'DCS Only\n(Cryptic)'],
                    [len(idx_cons-(idx_rcs|idx_dcs)), len(idx_rcs-(idx_cons|idx_dcs)), len(idx_dcs-(idx_cons|idx_rcs))],
                    color=['gray', colors_map['Rigid-Conserved (RCS)'], colors_map['Dynamic-Conserved (DCS)']])
    ax_c.set_title('c) Unique Residue ID'); ax_c.set_ylabel('Count')
    for b in bars: ax_c.text(b.get_x()+b.get_width()/2, b.get_height(), f'{b.get_height()}', ha='center', va='bottom', fontweight='bold')

    # PANEL D: ROC (FIXED COLORS)
    print("Plotting Panel D...")
    y_true = (df_main['mean_am_pathogenicity'] > 0.90).astype(int)
    
    # Specific color map for ROC labels
    roc_colors = {
        'RCS (Rigid)': colors_map['Rigid-Conserved (RCS)'],
        'DCS (Dynamic)': colors_map['Dynamic-Conserved (DCS)'],
        'Conservation': 'gray'
    }

    for label, col in [('RCS (Rigid)', 'Rigid_Conserved_Score_0_10'),
                       ('DCS (Dynamic)', 'Dynamic_Conserved_Score_0_10'),
                       ('Conservation', 'conservation')]:
        fpr, tpr, _ = roc_curve(y_true, df_main[col])
        # Use the specific map created above
        ax_d.plot(fpr, tpr, lw=3, label=f'{label} (AUC={auc(fpr, tpr):.2f})', color=roc_colors[label])
        
    ax_d.plot([0,1],[0,1], 'k--')
    ax_d.set_title('d) ROC Analysis')
    ax_d.legend(loc='lower right', fontsize=10)

    # PANEL E: PRECISION
    res = []
    for name, col in [('RCS', 'Rigid_Conserved_Score_0_10'), ('DCS', 'Dynamic_Conserved_Score_0_10'), ('Cons', 'conservation')]:
        sdf = df_main.sort_values(col, ascending=False)
        for t in [0.01, 0.02, 0.05, 0.10]:
            res.append({'Metric': name, 'Top_Pct': int(t*100), 'Precision': sdf.head(int(len(sdf)*t))['is_pathogenic'].mean()})
    sns.lineplot(x='Top_Pct', y='Precision', hue='Metric', data=pd.DataFrame(res), marker='o', lw=3, ms=10, ax=ax_e)
    ax_e.set_title('e) Precision at Top K%'); ax_e.set_ylim(0, 1.05); ax_e.set_xticks([1, 2, 5, 10])

    # =================================================================
    # PANEL F: CLINVAR VALIDATION (UPDATED DUAL ZONES)
    # =================================================================
    print("Plotting Panel F...")
    df_clinvar = load_clinvar_data(INPUT_CSV_CLINVAR)
    
    if df_clinvar is not None and not df_clinvar.empty:
        max_val = df_clinvar['Plot_DCS'].max()
        requested_cutoff = 0.095
        dcs_threshold = requested_cutoff * 10 if max_val > 1.5 else requested_cutoff
        ymax = max(10, df_clinvar['Plot_DCS'].max())

        palette_f = {'Pathogenic': '#D50000', 'Benign': '#9E9E9E', 'Ambiguous/Uncertain': '#2962FF'}

        # Scatter Plots
        for cls in ['Benign', 'Ambiguous/Uncertain', 'Pathogenic']:
            subset = df_clinvar[df_clinvar['Plot_Class'] == cls]
            alpha = 0.8 if cls == 'Pathogenic' else 0.4
            edge = 'k' if cls == 'Pathogenic' else 'none'
            sns.scatterplot(data=subset, x='AM_Score', y='Plot_DCS', color=palette_f[cls],
                            s=60 if cls!='Pathogenic' else 100, alpha=alpha, edgecolor=edge, label=cls, ax=ax_f)

        # Lines
        ax_f.axhline(y=dcs_threshold, color='k', linestyle='--', alpha=0.8, linewidth=2.5)
        ax_f.axvline(x=0.6, color='k', linestyle=':', alpha=0.8, linewidth=2.5)

        # --- ZONE 1: CRYPTIC (High DCS, Low AM) ---
        # Color: Light Orange
        ax_f.fill_betweenx([dcs_threshold, ymax + 0.5], 0, 0.6,
                           color='orange', alpha=0.25)
        ax_f.text(0.3, 6,
                  'Cryptic Zone',
                  fontsize=10, color='black', fontweight='bold', ha='center',
                  bbox=dict(facecolor='yellow', alpha=0.8, edgecolor='orange', boxstyle='round,pad=0.3'))

        # --- ZONE 2: AGREEMENT (High DCS, High AM) ---
        # Color: Light Purple
        ax_f.fill_betweenx([dcs_threshold, ymax + 0.5], 0.6, 1.05,
                           color='yellow', alpha=0.15)
        ax_f.text(0.6, dcs_threshold + (ymax * 0.05),
                  'High DCS (>0.95)',
                  fontsize=10, color='black', fontweight='bold', ha='center',
                  bbox=dict(facecolor='white', alpha=0.8, edgecolor='orange', boxstyle='round,pad=0.3'))

        ax_f.set_title('f) DCS vs ClinVar', pad=10)
        ax_f.set_xlabel('AlphaMissense Score')
        ax_f.set_ylabel('DCS Score')
        ax_f.legend(title='ClinVar Class', loc='upper right', framealpha=0.9, edgecolor='black', fontsize=9)
        ax_f.set_xlim(0, 1.05)
        ax_f.set_ylim(-0.5, ymax + 0.5)

    plt.tight_layout(pad=2.0, h_pad=3.0, w_pad=2.5)
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'Figure3_Complete_SixPanel.png'), dpi=300)
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'Figure3_Complete_SixPanel.svg'), format='svg')
    plt.show()

if __name__ == "__main__":
    generate_figure_3_complete()

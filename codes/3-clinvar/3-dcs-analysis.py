import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import os

# --- CONFIGURATION ---
INPUT_CSV = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-human/Clinvar/ClinVar_All_AlphaMissense_DCS_Analysis-check.csv'
OUTPUT_FIGURE = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-human/Clinvar/DCS_Raw_Validation_ClinVar.png'
OUTPUT_STATS = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-human/Clinvar/ClinVar_Stats_Raw_DCS.txt'

# --- PUBLICATION STYLE (BOLD + BLACK BORDERS + DASHED GRID) ---
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
plt.rcParams['grid.color'] = 'gray'

# --- 1. LOAD AND PREPARE DATA ---
def load_and_clean_data(file_path):
    print(f"Loading data from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        
        # --- PREPARE AM SCORE ---
        if 'AM_Specific_Score' in df.columns and 'AM_Mean_Residue_Score' in df.columns:
            df['AM_Score'] = df['AM_Specific_Score'].fillna(df['AM_Mean_Residue_Score'])
        elif 'AM_Specific_Score' in df.columns:
            df['AM_Score'] = df['AM_Specific_Score']
        else:
            print("Error: AM Score columns not found.")
            return None

        # --- PREPARE DCS SCORE (RAW) ---
        if 'DCS_Score' in df.columns:
            df['Plot_DCS'] = df['DCS_Score']
        else:
            print("Error: Raw DCS_Score column not found.")
            return None

        # --- PREPARE CLINVAR CLASS ---
        col_class = next((c for c in df.columns if 'ClinVar_Significance' in c or 'ClinVar_Class' in c), None)
        if not col_class:
            print("Error: ClinVar classification column not found.")
            return None
        
        # Clean Types
        df['AM_Score'] = pd.to_numeric(df['AM_Score'], errors='coerce')
        df['Plot_DCS'] = pd.to_numeric(df['Plot_DCS'], errors='coerce')
        
        # Drop rows with missing values
        df_clean = df.dropna(subset=['AM_Score', 'Plot_DCS', col_class]).copy()

        # Simplify Classes
        def simplify_class(c):
            c = str(c).lower()
            if 'pathogenic' in c and 'benign' not in c and 'conflicting' not in c:
                return 'Pathogenic'
            if 'benign' in c and 'pathogenic' not in c and 'conflicting' not in c:
                return 'Benign'
            return 'Ambiguous/Uncertain'

        df_clean['Plot_Class'] = df_clean.apply(lambda row: simplify_class(row[col_class]), axis=1)
        
        print(f"Data loaded successfully. {len(df_clean)} valid variants found.")
        return df_clean

    except Exception as e:
        print(f"Fatal error reading file: {e}")
        return None

# --- 2. STATISTICAL ANALYSIS ---
def perform_stats(df, dcs_threshold):
    results = []
    results.append("--- STATISTICAL VALIDATION: RAW DCS vs CLINVAR ---\n")
    results.append(f"Total Variants: {len(df)}\n")
    results.append(f"DCS High Constraint Threshold: {dcs_threshold}\n")
    
    # Cryptic Discovery Rate
    cryptic_hits = df[
        (df['Plot_Class'] == 'Pathogenic') &
        (df['AM_Score'] < 0.6) &
        (df['Plot_DCS'] >= dcs_threshold)
    ]
    
    results.append(f"3. Cryptic Pathogenic Discoveries (DCS > {dcs_threshold}, AM < 0.6):\n")
    results.append(f"   Count: {len(cryptic_hits)} variants\n")
    
    if not cryptic_hits.empty:
        results.append("   Top Examples:\n")
        top_hits = cryptic_hits.sort_values(by='Plot_DCS', ascending=False).head(10)
        for _, row in top_hits.iterrows():
            var_str = row.get('Protein_Change', row.get('Variant', 'Unknown'))
            gene = row.get('Gene_Name', 'Unknown')
            results.append(f"   - {gene} {var_str}: DCS={row['Plot_DCS']:.2f}, AM={row['AM_Score']:.2f}")
            
    return "\n".join(results)

# --- 3. GENERATE FIGURE ---
def plot_figure(df, dcs_threshold):
    # Force black borders explicitly
    with plt.rc_context({'axes.edgecolor':'black', 'grid.linestyle':'--'}):
        plt.figure(figsize=(10, 8))
        
        # --- COLOR PALETTE ---
        palette = {
            'Pathogenic': '#D50000',          # Vivid Red
            'Benign': '#9E9E9E',              # Medium Gray
            'Ambiguous/Uncertain': '#2962FF'  # Vivid Blue
        }
        
        # 1. Benign (Background)
        sns.scatterplot(
            data=df[df['Plot_Class'] == 'Benign'],
            x='AM_Score', y='Plot_DCS', color=palette['Benign'],
            s=60, alpha=0.4, edgecolor='none', label='Benign'
        )

        # 2. Ambiguous/Uncertain (Midground)
        sns.scatterplot(
            data=df[df['Plot_Class'] == 'Ambiguous/Uncertain'],
            x='AM_Score', y='Plot_DCS', color=palette['Ambiguous/Uncertain'],
            s=60, alpha=0.5, edgecolor='none', label='Ambiguous/Uncertain'
        )
        
        # 3. Pathogenic (Foreground)
        sns.scatterplot(
            data=df[df['Plot_Class'] == 'Pathogenic'],
            x='AM_Score', y='Plot_DCS', color=palette['Pathogenic'],
            s=100, alpha=0.8, edgecolor='k', linewidth=0.5, label='Pathogenic'
        )
        
        # Lines & Zones
        plt.axhline(y=dcs_threshold, color='k', linestyle='--', alpha=0.8, linewidth=2.5)
        plt.axvline(x=0.6, color='k', linestyle=':', alpha=0.8, linewidth=2.5)
        
        # Highlight Discovery Zone
        ymax = max(10, df['Plot_DCS'].max())
        plt.fill_betweenx([dcs_threshold, ymax + 0.5], 0, 0.6, color='orange', alpha=0.1)
        
        # Text Box (Preserved Location)
        plt.text(0.32, dcs_threshold + 7.5, f'DCS Discovery Zone\n(Cryptic Pathogenic)\nDCS > {dcs_threshold}',
                 fontsize=12, color='#BF360C', fontweight='bold',
                 bbox=dict(facecolor='white', alpha=0.9, edgecolor='orange', boxstyle='round,pad=0.5'))

        plt.title('Validation: Raw DCS vs ClinVar', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('AlphaMissense Score (Sequence/Structure)', fontsize=14, fontweight='bold')
        plt.ylabel('DCS Score', fontsize=14, fontweight='bold')
        
        # Legend
        plt.legend(title='ClinVar Classification', loc='lower right', frameon=True, framealpha=0.9, edgecolor='black')
        
        # Force Bold Ticks
        plt.xticks(fontweight='bold')
        plt.yticks(fontweight='bold')

        # Ensure spines are visible and black
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2.5)

        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.5, ymax + 0.5)
        
        plt.tight_layout()
        
        # --- SAVE AS PNG ---
        plt.savefig(OUTPUT_FIGURE, dpi=300)
        print(f"PNG saved to: {OUTPUT_FIGURE}")
        
        # --- SAVE AS SVG (NEW) ---
        svg_output = OUTPUT_FIGURE.replace('.png', '.svg')
        plt.savefig(svg_output, format='svg')
        print(f"SVG saved to: {svg_output}")

# --- MAIN ---
if __name__ == "__main__":
    if os.path.exists(INPUT_CSV):
        df = load_and_clean_data(INPUT_CSV)
        
        if df is not None and not df.empty:
            
            # --- INTELLIGENT THRESHOLD SCALING (PRESERVED) ---
            max_val = df['Plot_DCS'].max()
            requested_cutoff = 0.095
            
            if max_val > 1.5:
                dcs_threshold = requested_cutoff * 10
                print(f"Detected 0-10 scale. Interpreting threshold '{requested_cutoff}' as {dcs_threshold}.")
            else:
                dcs_threshold = requested_cutoff
                print(f"Detected 0-1 scale. Using threshold {dcs_threshold}.")

            # Stats
            stats_text = perform_stats(df, dcs_threshold)
            with open(OUTPUT_STATS, 'w') as f:
                f.write(stats_text)
            print(f"Stats saved to: {OUTPUT_STATS}")
            
            # Plot
            plot_figure(df, dcs_threshold)
        else:
            print("Dataframe empty.")
    else:
        print(f"File not found: {INPUT_CSV}")

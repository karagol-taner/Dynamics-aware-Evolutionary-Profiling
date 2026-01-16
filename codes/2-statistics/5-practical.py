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

def generate_practical_utility_analysis():
    # --- CONFIGURATION ---
    input_file = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-alphahelical/data_tsv/All_Proteins_AM_Combined_Master.tsv'
    output_folder = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-alphahelical/practical_stats_rcs'
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        print(f"Created output folder: {output_folder}")

    print("Loading Data...")
    try:
        df = pd.read_csv(input_file, sep='\t', na_values=['N/A', 'NA', ''])
    except:
        df = pd.read_csv(input_file, na_values=['N/A', 'NA', ''])

    # Ensure numeric columns
    numeric_cols = [
        'conservation',
        'Dynamic_Conserved_Score_0_10',
        'Rigid_Conserved_Score_0_10',
        'mean_am_pathogenicity'
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Filter for valid ground truth (using AlphaMissense)
    # Pathogenic = AM > 0.90
    PATHOGENIC_CUTOFF = 0.90
    
    df_clean = df.dropna(subset=numeric_cols).copy()
    df_clean['is_pathogenic'] = (df_clean['mean_am_pathogenicity'] >= PATHOGENIC_CUTOFF).astype(int)
    
    print(f"Total Residues: {len(df_clean)}")
    print(f"Total Pathogenic Targets: {df_clean['is_pathogenic'].sum()}")

    # Initialize Results Text File
    results_path = os.path.join(output_folder, 'practical_utility_results.txt')
    f_out = open(results_path, 'w')
    f_out.write("PRACTICAL UTILITY ANALYSIS RESULTS\n")
    f_out.write("==================================\n\n")

    # =================================================================================
    # ANALYSIS 1: PRECISION AT TOP-K% (The "Drug Hunter" Metric)
    # =================================================================================
    print("\nCalculating Precision @ Top K%...")

    # We want to see: Of the top X% scoring residues, what fraction are actually pathogenic?
    thresholds = [0.01, 0.02, 0.05, 0.10, 0.20] # Top 1%, 2%, 5%, 10%, 20%
    results = []

    # Metrics to compare
    metrics = {
        'Conservation': 'conservation',
        'DCS (Dynamic Conserved)': 'Dynamic_Conserved_Score_0_10',
        'RCS (Rigid Conserved)': 'Rigid_Conserved_Score_0_10'
    }

    f_out.write("--- Precision @ Top K% ---\n")
    
    for name, col in metrics.items():
        sorted_df = df_clean.sort_values(col, ascending=False)
        total_n = len(sorted_df)
        
        f_out.write(f"\nMetric: {name}\n")
        
        for t in thresholds:
            top_k_count = int(total_n * t)
            top_k_df = sorted_df.head(top_k_count)
            
            # Precision = (True Positives in Top K) / (K)
            precision = top_k_df['is_pathogenic'].mean()
            
            f_out.write(f"  Top {int(t*100)}%: {precision:.4f}\n")
            
            results.append({
                'Metric': name,
                'Top_Pct': f'Top {int(t*100)}%',
                'Precision': precision,
                'Threshold_Value': t
            })

    res_df = pd.DataFrame(results)

    # Plot
    plt.figure(figsize=(12, 6))
    # Colors: Cons=Gray, DCS=Orange, RCS=Purple
    sns.barplot(x='Top_Pct', y='Precision', hue='Metric', data=res_df, palette=['gray', '#FF5733', '#800080'])
    
    # Add baseline (Random/Background rate)
    baseline = df_clean['is_pathogenic'].mean()
    plt.axhline(baseline, color='navy', linestyle='--', linewidth=2, label=f'Random Chance ({baseline:.2f})')
    
    plt.title('Practical Utility: Precision of Top-Ranked Residues\n(Comparison of Conservation vs. Dynamics vs. Rigidity)', fontsize=16)
    plt.ylabel('Precision (Fraction Pathogenic)', fontsize=14)
    plt.xlabel('Percentile Rank', fontsize=14)
    plt.legend(title='Scoring Method')
    plt.grid(axis='y', alpha=0.3)
    
    save_path = os.path.join(output_folder, 'Precision_At_Top_K.png')
    plt.savefig(save_path, dpi=300)
    plt.savefig(save_path.replace('.png', '.svg'), format='svg')
    print(f"Saved Precision Plot to {save_path}")
    plt.close()

    # =================================================================================
    # ANALYSIS 2: UNIQUE DISCOVERY (Venn Diagram Logic)
    # =================================================================================
    print("\nCalculating Unique Discovery Rate...")
    f_out.write("\n--- Unique Discovery Rate (Top 10%) ---\n")
    
    # Define "High Priority" as Top 10%
    top_10_pct_idx = int(len(df_clean) * 0.10)
    
    # Get Top 10% Sets
    top_cons = set(df_clean.sort_values('conservation', ascending=False).head(top_10_pct_idx).index)
    top_dcs = set(df_clean.sort_values('Dynamic_Conserved_Score_0_10', ascending=False).head(top_10_pct_idx).index)
    top_rcs = set(df_clean.sort_values('Rigid_Conserved_Score_0_10', ascending=False).head(top_10_pct_idx).index)
    
    # Helper to get mean path
    def get_path_rate(indices):
        if not indices: return 0
        return df_clean.loc[list(indices), 'is_pathogenic'].mean()

    # Define Groups
    groups = [
        {'Name': 'Found by DCS Only', 'Indices': top_dcs - top_cons, 'Color': '#FF5733'},
        {'Name': 'Found by RCS Only', 'Indices': top_rcs - top_cons, 'Color': '#800080'},
        {'Name': 'Found by Cons Only', 'Indices': top_cons - (top_dcs | top_rcs), 'Color': 'gray'},
        {'Name': 'Found by All 3', 'Indices': top_cons & top_dcs & top_rcs, 'Color': 'black'}
    ]
    
    venn_data = []
    for g in groups:
        rate = get_path_rate(g['Indices'])
        count = len(g['Indices'])
        venn_data.append({'Group': g['Name'], 'Pathogenicity_Rate': rate, 'Count': count, 'Color': g['Color']})
        f_out.write(f"{g['Name']}: N={count}, Pathogenicity={rate:.4f}\n")

    venn_df = pd.DataFrame(venn_data)
    
    plt.figure(figsize=(12, 6))
    bars = sns.barplot(x='Group', y='Pathogenicity_Rate', data=venn_df, palette=[g['Color'] for g in groups])
    
    # Annotate with counts
    for i, p in enumerate(bars.patches):
        bars.annotate(f"N={venn_df.iloc[i]['Count']}",
                      (p.get_x() + p.get_width() / 2., p.get_height()/2),
                      ha = 'center', va = 'center', color='white', fontsize=12, fontweight='bold')

    plt.axhline(baseline, color='navy', linestyle='--', linewidth=2, label='Random Chance')
    
    plt.title('Unique Discovery: Do DCS and RCS find novel targets?', fontsize=16)
    plt.ylabel('Pathogenicity Rate (Validation)', fontsize=14)
    plt.xlabel('Target Group (Top 10% of Scores)', fontsize=14)
    plt.ylim(0, 1.0)
    plt.legend()
    
    save_path_venn = os.path.join(output_folder, 'Unique_Discovery_Analysis.png')
    plt.savefig(save_path_venn, dpi=300)
    plt.savefig(save_path_venn.replace('.png', '.svg'), format='svg')
    print(f"Saved Unique Discovery Plot to {save_path_venn}")
    
    f_out.close()
    print("Done.")

if __name__ == "__main__":
    generate_practical_utility_analysis()

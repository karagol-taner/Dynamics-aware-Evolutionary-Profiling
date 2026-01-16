import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.utils import resample # Required for bootstrapping statistics
import os

# --- GLOBAL PLOT SETTINGS (Publication Style) ---
# Set all fonts to bold for high-contrast visibility
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['font.size'] = 12

def get_bootstrap_ci(y_true, y_scores, metric_func, n_bootstraps=1000, rng_seed=42):
    """
    Calculates the 95% Confidence Interval for a specific metric using bootstrapping.
    """
    rng = np.random.RandomState(rng_seed)
    bootstrapped_scores = []
    
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_scores), len(y_scores))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = metric_func(y_true[indices], y_scores[indices])
        bootstrapped_scores.append(score)
        
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    
    lower = sorted_scores[int(0.025 * len(sorted_scores))]
    upper = sorted_scores[int(0.975 * len(sorted_scores))]
    
    return lower, upper

# Helper wrappers
def roc_auc_scorer(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return auc(fpr, tpr)

def pr_auc_scorer(y_true, y_score):
    return average_precision_score(y_true, y_score)

def plot_and_save(name, predictors, colors, y_true, output_folder, results_list):
    """
    Helper to generate ROC and PR curves for a specific group of predictors.
    """
    print(f"\n--- Processing Group: {name} ---")
    results_list.append(f"\n=== {name} ANALYSIS ===\n")
    
    # 1. ROC CURVE
    plt.figure(figsize=(10, 8))
    for label, scores in predictors.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        lower, upper = get_bootstrap_ci(y_true, scores, roc_auc_scorer)
        
        res_str = f"ROC AUC: {label} = {roc_auc:.3f} [{lower:.3f}-{upper:.3f}]"
        print(res_str)
        results_list.append(res_str)
        
        plt.plot(fpr, tpr, lw=3, label=f'{label}\nAUC={roc_auc:.2f}', color=colors.get(label, 'black'))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.title(f'ROC Curve: {name} Predictors', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    
    plt.savefig(os.path.join(output_folder, f'ROC_{name}.png'), dpi=300)
    plt.savefig(os.path.join(output_folder, f'ROC_{name}.svg'), format='svg')
    plt.close()

    # 2. PR CURVE
    plt.figure(figsize=(10, 8))
    for label, scores in predictors.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        avg_p = average_precision_score(y_true, scores)
        lower, upper = get_bootstrap_ci(y_true, scores, pr_auc_scorer)
        
        res_str = f"PR AP:   {label} = {avg_p:.3f} [{lower:.3f}-{upper:.3f}]"
        print(res_str)
        results_list.append(res_str)

        plt.plot(recall, precision, lw=3, label=f'{label}\nAP={avg_p:.2f}', color=colors.get(label, 'black'))

    baseline = y_true.mean()
    plt.plot([0, 1], [baseline, baseline], color='navy', lw=2, linestyle='--', label=f'Random ({baseline:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=14, fontweight='bold')
    plt.ylabel('Precision', fontsize=14, fontweight='bold')
    plt.title(f'Precision-Recall: {name} Predictors', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(alpha=0.3)
    
    plt.savefig(os.path.join(output_folder, f'PR_{name}.png'), dpi=300)
    plt.savefig(os.path.join(output_folder, f'PR_{name}.svg'), format='svg')
    plt.close()


def generate_roc_analysis():
    # --- CONFIGURATION ---
    input_file = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-alphahelical/data_tsv/All_Proteins_AM_Combined_Master.tsv'
    output_folder = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-alphahelical/roc_stats_output_rcs_3'
    
    PATHOGENIC_CUTOFF = 0.90
    BENIGN_CUTOFF = 0.20
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    print("Loading data...")
    try:
        df = pd.read_csv(input_file, sep='\t', na_values=['N/A', 'NA', ''])
    except:
        df = pd.read_csv(input_file, na_values=['N/A', 'NA', ''])

    # Ensure numeric columns
    cols_to_fix = [
        'mean_am_pathogenicity', 'conservation',
        'Dynamic_Conserved_Score_0_10', 'Dynamics_Coupling_Score_0_10',
        'Rigid_Conserved_Score_0_10', 'Rigid_Coupling_Score_0_10'
    ]
    for c in cols_to_fix:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')

    df_clean = df.dropna(subset=cols_to_fix)
    
    df_analysis = df_clean[
        (df_clean['mean_am_pathogenicity'] >= PATHOGENIC_CUTOFF) |
        (df_clean['mean_am_pathogenicity'] <= BENIGN_CUTOFF)
    ].copy()

    df_analysis['target'] = (df_analysis['mean_am_pathogenicity'] >= PATHOGENIC_CUTOFF).astype(int)
    y_true = df_analysis['target']

    print(f"Analysis Set: {len(df_analysis)} residues (Pos: {y_true.sum()}, Neg: {len(y_true)-y_true.sum()})")

    results_log = []
    results_log.append("ROC/PR STATISTICAL REPORT\n=========================\n")
    results_log.append(f"Total Residues: {len(df_analysis)}\n")

    # --- GROUP 1: DYNAMIC PROFILE (DCS, DCopS) ---
    pred_dynamic = {
        'Conservation': df_analysis['conservation'],
        'DCS (Dynamic Conserved)': df_analysis['Dynamic_Conserved_Score_0_10'],
        'DCopS (Coupling)': df_analysis['Dynamics_Coupling_Score_0_10'] # Raw score
    }
    colors_dynamic = {
        'Conservation': 'gray',
        'DCS (Dynamic Conserved)': '#FF5733', # Orange
        'DCopS (Coupling)': '#33C1FF'         # Cyan
    }
    plot_and_save("Dynamic_Profile", pred_dynamic, colors_dynamic, y_true, output_folder, results_log)

    # --- GROUP 2: RIGID PROFILE (RCS, RCopS) ---
    pred_rigid = {
        'Conservation': df_analysis['conservation'],
        'RCS (Rigid Conserved)': df_analysis['Rigid_Conserved_Score_0_10'],
        'RCopS (Coupling)': df_analysis['Rigid_Coupling_Score_0_10'] # Raw score
    }
    colors_rigid = {
        'Conservation': 'gray',
        'RCS (Rigid Conserved)': '#800080',   # Purple
        'RCopS (Coupling)': '#008000'         # Green
    }
    plot_and_save("Rigid_Profile", pred_rigid, colors_rigid, y_true, output_folder, results_log)

    # Save Text Results
    with open(os.path.join(output_folder, 'roc_results.txt'), 'w') as f:
        f.writelines([line + "\n" for line in results_log])
    
    print(f"\nDone. Results saved to {output_folder}")

if __name__ == "__main__":
    generate_roc_analysis()

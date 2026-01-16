import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Global Styles
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['font.size'] = 12

def run_ml_validation():
    # --- CONFIGURATION ---
    input_file = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-human/data_tsv/All_Proteins_AM_Combined_Master.tsv'
    output_folder = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-human/ml_stats_rcs'
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    print("Loading Data for Machine Learning Validation...")
    try:
        df = pd.read_csv(input_file, sep='\t', na_values=['N/A', 'NA', ''])
    except:
        df = pd.read_csv(input_file, na_values=['N/A', 'NA', ''])

    # 1. Prepare Data
    # We need rows that have ALL scores: Cons, RMSF, DCS, DCopS, RCS, RCopS and AlphaMissense
    required_cols = [
        'conservation', 'Norm_Dynamics',
        'Dynamic_Conserved_Score_0_10', 'Dynamics_Coupling_Score_0_10',
        'Rigid_Conserved_Score_0_10', 'Rigid_Coupling_Score_0_10',
        'mean_am_pathogenicity'
    ]
    
    # Ensure numeric
    for c in required_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df_clean = df.dropna(subset=required_cols).copy()
    
    # Define Binary Target: Pathogenic (>0.9) vs Benign (<0.2)
    # This ignores ambiguous data for a cleaner signal
    df_ml = df_clean[
        (df_clean['mean_am_pathogenicity'] > 0.90) |
        (df_clean['mean_am_pathogenicity'] < 0.20)
    ].copy()
    
    df_ml['Target'] = (df_ml['mean_am_pathogenicity'] > 0.90).astype(int)

    print(f"Training Model on {len(df_ml)} high-confidence residues...")

    # Define results file path
    results_path = os.path.join(output_folder, 'ml_validation_results.txt')

    # Open file to write results
    with open(results_path, 'w') as res_file:
        res_file.write("DYNAMICS-AWARE EVOLUTIONARY PROFILING: ML VALIDATION RESULTS\n")
        res_file.write("============================================================\n\n")
        res_file.write(f"Total Residues used for ML: {len(df_ml)}\n")
        res_file.write(f"  - Pathogenic (>0.90): {df_ml['Target'].sum()}\n")
        res_file.write(f"  - Benign (<0.20): {len(df_ml) - df_ml['Target'].sum()}\n\n")

        # =================================================================================
        # ANALYSIS 1: RANDOM FOREST FEATURE IMPORTANCE
        # =================================================================================
        print("Running Random Forest Classifier...")
        
        features = [
            'conservation', 'Norm_Dynamics',
            'Dynamic_Conserved_Score_0_10', 'Dynamics_Coupling_Score_0_10',
            'Rigid_Conserved_Score_0_10', 'Rigid_Coupling_Score_0_10'
        ]
        feature_names = [
            'Raw Conservation', 'Raw Dynamics',
            'DCS (Dynamic-Cons)', 'DCopS (Dynamic-Coup)',
            'RCS (Rigid-Cons)', 'RCopS (Rigid-Coup)'
        ]
        
        X = df_ml[features]
        y = df_ml['Target']
        
        # Simple split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train RF
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        # Get Importance
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Calculate Accuracy
        acc = rf.score(X_test, y_test)

        # Write to text file
        res_file.write("1. RANDOM FOREST FEATURE IMPORTANCE\n")
        res_file.write("-----------------------------------\n")
        res_file.write(f"Model Accuracy (Pathogenicity Prediction): {acc:.2%}\n\n")
        res_file.write("Feature Rankings (Relative Importance):\n")
        for i in indices:
            res_file.write(f"  {feature_names[i]}: {importances[i]:.4f}\n")
        res_file.write("\n")

        # Plot
        plt.figure(figsize=(12, 6))
        
        # Map raw names to pretty names for plotting
        sorted_names = [feature_names[i] for i in indices]
        
        sns.barplot(x=importances[indices], y=sorted_names, palette='viridis')
        
        plt.title('Feature Importance: Which metric predicts Pathogenicity best?', fontsize=16)
        plt.xlabel('Relative Importance (Gini Impurity)', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.grid(axis='x', alpha=0.3)
        
        # Accuracy Score annotation
        plt.text(0.5, 0.5, f'Model Accuracy: {acc:.1%}', transform=plt.gca().transAxes,
                 fontsize=14, color='gray', ha='center', bbox=dict(facecolor='white', alpha=0.8))

        save_rf = os.path.join(output_folder, 'ML_Feature_Importance.png')
        plt.savefig(save_rf, dpi=300)
        plt.savefig(save_rf.replace('.png', '.svg'), format='svg')
        print(f"Saved Feature Importance to {save_rf}")
        plt.close()

        # =================================================================================
        # ANALYSIS 2: CORRELATION MATRIX (Independence Check)
        # =================================================================================
        print("Generating Independence Matrix...")
        
        # We want to see correlations between ALL scores and Pathogenicity
        corr_cols = required_cols # Includes all 4 scores + cons + dyn + pathogenicity
        pretty_labels = [
            'Conservation', 'Dynamics',
            'DCS', 'DCopS',
            'RCS', 'RCopS',
            'Pathogenicity'
        ]
        
        corr_matrix = df_clean[corr_cols].corr(method='spearman')
        
        # Write to text file
        res_file.write("2. INDEPENDENCE CHECK (SPEARMAN CORRELATION MATRIX)\n")
        res_file.write("-------------------------------------------------\n")
        # Create a readable version for the text file
        corr_text = corr_matrix.copy()
        corr_text.index = pretty_labels
        corr_text.columns = pretty_labels
        res_file.write(corr_text.to_string(float_format="{:.4f}".format))
        res_file.write("\n")

        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    xticklabels=pretty_labels, yticklabels=pretty_labels, linewidths=1, linecolor='white')
        
        plt.title('Independence Check: Correlations between all metrics', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        save_corr = os.path.join(output_folder, 'Independence_Correlation_Matrix.png')
        plt.savefig(save_corr, dpi=300)
        plt.savefig(save_corr.replace('.png', '.svg'), format='svg')
        print(f"Saved Correlation Matrix to {save_corr}")
        plt.close()

    print(f"ML Validation Complete. Results saved to: {results_path}")

if __name__ == "__main__":
    run_ml_validation()

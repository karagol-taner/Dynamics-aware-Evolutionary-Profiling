import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# Global Styles
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['font.size'] = 12

def run_clustering_analysis():
    # --- CONFIGURATION ---
    # 1. Global Data (Train the Biophysical Model)
    input_file_global = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-human/data_tsv/All_Proteins_DCS_Master.tsv'
    
    # 2. Labeled Data (Validate with Pathogenicity)
    input_file_am = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-human/data_tsv/All_Proteins_AM_Combined_Master.tsv'
    
    output_folder = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-human/clustering_stats'
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # --- STEP 1: LOAD & TRAIN ON GLOBAL DATA ---
    print("Loading Global Data for Clustering (Training)...")
    try:
        try:
            df_global = pd.read_csv(input_file_global, sep='\t', na_values=['N/A', 'NA', ''])
        except:
            df_global = pd.read_csv(input_file_global, na_values=['N/A', 'NA', ''])
            
        cols_phys = ['conservation', 'Norm_Dynamics']
        for c in cols_phys:
            if c in df_global.columns: df_global[c] = pd.to_numeric(df_global[c], errors='coerce')
        
        df_global_clean = df_global.dropna(subset=cols_phys).copy()
        
        # Subsample if too huge for KMeans speed (e.g. 100k)
        if len(df_global_clean) > 100000:
            df_train = df_global_clean.sample(100000, random_state=42)
        else:
            df_train = df_global_clean
            
        print(f"Training K-Means on {len(df_train)} residues from Global Dataset...")
        
        # Standardize
        scaler = StandardScaler()
        X_train = scaler.fit_transform(df_train[cols_phys])
        
        # K=4 Clusters
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        df_train['Cluster'] = kmeans.fit_predict(X_train)
        
        # --- IDENTIFY CLUSTERS AUTOMATICALLY ---
        # 1. Hinge = High Cons * High Dyn
        # 2. Core = High Cons
        # 3. Loop = High Dyn
        cluster_means = df_train.groupby('Cluster')[cols_phys].mean()
        cluster_means['Product'] = cluster_means['conservation'] * cluster_means['Norm_Dynamics']
        
        hinge_id = cluster_means['Product'].idxmax()
        core_id = cluster_means['conservation'].idxmax()
        loop_id = cluster_means['Norm_Dynamics'].idxmax()
        
        def get_label(c_id):
            if c_id == hinge_id: return "Dynamic-Conserved (Hinge)"
            if c_id == core_id: return "Rigid-Conserved (Core)"
            if c_id == loop_id: return "Dynamic-Variable (Loop)"
            return "Rigid-Variable (Spacer)"
            
        df_train['Label'] = df_train['Cluster'].apply(get_label)
        
        # Plot 1: The Global Landscape
        print("Generating Global Cluster Map...")
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=df_train, x='conservation', y='Norm_Dynamics', hue='Label',
                        palette='viridis', s=10, alpha=0.5, linewidth=0)
        
        plt.title('Universal Biophysical States (Global Dataset)', fontsize=16)
        plt.xlabel('Evolutionary Conservation', fontsize=14)
        plt.ylabel('Structural Dynamics (RMSF)', fontsize=14)
        plt.legend(title='Cluster Class', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_folder, 'Global_Cluster_Map.png'), dpi=300)
        plt.savefig(os.path.join(output_folder, 'Global_Cluster_Map.svg'), format='svg')
        plt.close()

    except Exception as e:
        print(f"Error in Global Clustering: {e}")
        return

    # --- STEP 2: PREDICT & VALIDATE ON ALPHAMISSENSE DATA ---
    print("Loading AlphaMissense Data for Validation...")
    try:
        try:
            df_am = pd.read_csv(input_file_am, sep='\t', na_values=['N/A', 'NA', ''])
        except:
            df_am = pd.read_csv(input_file_am, na_values=['N/A', 'NA', ''])
            
        cols_val = ['conservation', 'Norm_Dynamics', 'mean_am_pathogenicity']
        for c in cols_val:
            if c in df_am.columns: df_am[c] = pd.to_numeric(df_am[c], errors='coerce')
            
        df_am_clean = df_am.dropna(subset=cols_val).copy()
        
        # Apply the Global Model to the Labeled Data
        X_val = scaler.transform(df_am_clean[['conservation', 'Norm_Dynamics']])
        df_am_clean['Cluster'] = kmeans.predict(X_val)
        df_am_clean['Label'] = df_am_clean['Cluster'].apply(get_label)
        
        # Plot 2: Validation Boxplot
        print("Validating Pathogenicity of Global Clusters...")
        plt.figure(figsize=(10, 6))
        
        # Sort by median pathogenicity for clean plot
        order = df_am_clean.groupby("Label")["mean_am_pathogenicity"].median().sort_values(ascending=False).index
        
        sns.boxplot(data=df_am_clean, x='Label', y='mean_am_pathogenicity', order=order, palette='magma')
        plt.title('Validation: Do Universal "Hinges" Cause Disease?', fontsize=16)
        plt.xlabel('Biophysical Class (Defined Globally)', fontsize=14)
        plt.ylabel('AlphaMissense Pathogenicity', fontsize=14)
        plt.xticks(rotation=15, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        plt.savefig(os.path.join(output_folder, 'Cluster_Pathogenicity_Validation.png'), dpi=300)
        plt.savefig(os.path.join(output_folder, 'Cluster_Pathogenicity_Validation.svg'), format='svg')
        plt.close()
        
        # Save Stats
        with open(os.path.join(output_folder, 'clustering_results.txt'), 'w') as f:
            f.write("CLUSTERING ANALYSIS RESULTS (TRAIN=GLOBAL, TEST=AM)\n")
            f.write("===================================================\n\n")
            f.write("Cluster Centroids (Global Definition):\n")
            f.write(cluster_means.to_string())
            f.write("\n\nPathogenicity by Cluster (Validation Set):\n")
            summary = df_am_clean.groupby('Label')['mean_am_pathogenicity'].describe()
            f.write(summary.to_string())

    except Exception as e:
        print(f"Error in Validation Step: {e}")

    print(f"Done. Results saved to {output_folder}")

if __name__ == "__main__":
    run_clustering_analysis()

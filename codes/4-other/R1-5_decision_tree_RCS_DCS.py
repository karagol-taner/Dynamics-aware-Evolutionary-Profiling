
import pandas as pd, numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import cross_val_score, StratifiedKFold

# ================= CONFIG =================
INPUT_CSV = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-human/Clinvar/ClinVar_All_AlphaMissense_DCS_Analysis-check.csv'
MAX_DEPTH = 3
OUTPUT_DIR = '/content/drive/MyDrive/All Project Files/dynamics aware evo/revision_results'  # results go here; NEVER overwrites, NEVER edits input files

# =========================================

def find_col(df, options, contains=None):
    for o in options:
        if o in df.columns: return o
    if contains:
        for c in df.columns:
            if contains.lower() in c.lower(): return c
    return None

def simplify_class(c):
    c = str(c).lower()
    if 'pathogenic' in c and 'benign' not in c and 'conflicting' not in c: return 1
    if 'benign' in c and 'pathogenic' not in c and 'conflicting' not in c: return 0
    return np.nan


import sys, os
def _new_output_path(directory, base):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, base)
    root, ext = os.path.splitext(path)
    k = 1
    while os.path.exists(path):          # never overwrite an existing file
        path = f"{root}_{k}{ext}"
        k += 1
    return path

class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for s in self.streams: s.write(data)
    def flush(self):
        for s in self.streams: s.flush()

def _start_capture(base):
    p = _new_output_path(OUTPUT_DIR, base)
    fh = open(p, 'w')                     # 'w' on a guaranteed-new path = safe
    sys.stdout = _Tee(sys.__stdout__, fh)
    print(f"[results will be saved to a NEW file: {p}]")
    return fh
# -------------------------------------------------------------------------------------

def main():
    _fh = _start_capture("R1-5_decision_tree_result.txt")
    df = pd.read_csv(INPUT_CSV); df.columns=[c.strip() for c in df.columns]
    rcs = find_col(df, ['RCS_Score','Rigid_Conserved_Score_0_10'], contains='rcs')
    dcs = find_col(df, ['DCS_Score','Dynamic_Conserved_Score_0_10'], contains='dcs')
    cls = find_col(df, ['ClinVar_Significance','ClinVar_Class'], contains='clinvar')
    df['RCS']=pd.to_numeric(df[rcs],errors='coerce')
    df['DCS']=pd.to_numeric(df[dcs],errors='coerce')
    df['y']=df[cls].apply(simplify_class)
    d=df.dropna(subset=['RCS','DCS','y']).copy(); d['y']=d['y'].astype(int)
    X=d[['RCS','DCS']].values; y=d['y'].values
    print(f"Variants used: {len(d)}  (Pathogenic={int(y.sum())}, Benign={int((1-y).sum())})")

    clf=DecisionTreeClassifier(max_depth=MAX_DEPTH, class_weight='balanced', random_state=0).fit(X,y)
    print("\n--- Decision tree rules ---")
    print(export_text(clf, feature_names=['RCS','DCS']))

    cv=StratifiedKFold(5, shuffle=True, random_state=0)
    acc=cross_val_score(clf,X,y,cv=cv,scoring='balanced_accuracy')
    auc=cross_val_score(clf,X,y,cv=cv,scoring='roc_auc')
    print(f"5-fold balanced accuracy = {acc.mean():.3f} ± {acc.std():.3f}")
    print(f"5-fold ROC-AUC           = {auc.mean():.3f} ± {auc.std():.3f}")
    print("\nPASTE -> R1-5: balanced acc=%.2f, AUC=%.2f; primary split shown above" % (acc.mean(), auc.mean()))

if __name__=="__main__":
    main()

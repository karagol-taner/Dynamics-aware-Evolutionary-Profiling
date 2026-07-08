
import pandas as pd, numpy as np
from scipy.stats import mannwhitneyu, fisher_exact

# ================= CONFIG =================
INPUT_CSV  = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-human/Clinvar/ClinVar_All_AlphaMissense_DCS_Analysis-check.csv'
OUTPUT_DIR = '/content/drive/MyDrive/All Project Files/dynamics aware evo/revision_results'
DCS_ZONE   = 0.95
AM_BLIND   = 0.60
# =========================================

import sys, os
def _new_output_path(directory, base):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, base)
    root, ext = os.path.splitext(path)
    k = 1
    while os.path.exists(path):
        path = f"{root}_{k}{ext}"; k += 1
    return path
class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, d):
        for s in self.streams: s.write(d)
    def flush(self):
        for s in self.streams: s.flush()
def _start_capture(base):
    p = _new_output_path(OUTPUT_DIR, base)
    fh = open(p, 'w')
    sys.stdout = _Tee(sys.__stdout__, fh)
    print(f"[results saved to a NEW file: {p}]")
    return fh
# -------------------------------------------------------------------------------------

def find_col(df, options, contains=None):
    for o in options:
        if o in df.columns: return o
    if contains:
        for c in df.columns:
            if contains.lower() in c.lower(): return c
    return None

def simplify_class(c):
    c=str(c).lower()
    if 'pathogenic' in c and 'benign' not in c and 'conflicting' not in c: return 'Pathogenic'
    if 'benign' in c and 'pathogenic' not in c and 'conflicting' not in c: return 'Benign'
    return 'Ambiguous'

def rank_biserial(u, n1, n2):
    return (2*u)/(n1*n2) - 1   # effect size; positive = pathogenic stochastically greater

def mwu_report(a, b, label):
    a=np.asarray(a); b=np.asarray(b)
    if len(a)<3 or len(b)<3:
        print(f"  {label}: too few points (path={len(a)}, benign={len(b)})"); return
    U,p = mannwhitneyu(a, b, alternative='greater')
    rb = rank_biserial(U, len(a), len(b))
    print(f"  {label}: median DCS path={np.median(a):.3f} vs benign={np.median(b):.3f} | "
          f"MWU p={p:.3e}, rank-biserial={rb:.2f} (n_path={len(a)}, n_benign={len(b)})")

def main():
    _fh=_start_capture("R2-6b_bettertests_result.txt")
    df=pd.read_csv(INPUT_CSV); df.columns=[c.strip() for c in df.columns]
    am_s=find_col(df,['AM_Specific_Score']); am_m=find_col(df,['AM_Mean_Residue_Score'])
    if am_s and am_m:
        df['AM']=pd.to_numeric(df[am_s],errors='coerce').fillna(pd.to_numeric(df[am_m],errors='coerce'))
    else:
        df['AM']=pd.to_numeric(df[am_s or am_m],errors='coerce')
    dcs=find_col(df,['DCS_Score','Dynamic_Conserved_Score_0_10'],contains='dcs')
    cls=find_col(df,['ClinVar_Significance','ClinVar_Class'],contains='clinvar')
    df['DCS']=pd.to_numeric(df[dcs],errors='coerce')
    df['Cls']=df[cls].apply(simplify_class)
    df=df.dropna(subset=['DCS','AM']).copy()
    blind=df[df['AM']<AM_BLIND]

    print("="*66)
    print("(1) THRESHOLD-FREE Mann-Whitney on DCS  (one-sided: path>benign)")
    print("="*66)
    print("Full set:")
    mwu_report(df.loc[df.Cls=='Pathogenic','DCS'], df.loc[df.Cls=='Benign','DCS'], "Pathogenic vs Benign")
    print("AM-blind set (AM<%.2f):"%AM_BLIND)
    mwu_report(blind.loc[blind.Cls=='Pathogenic','DCS'], blind.loc[blind.Cls=='Benign','DCS'], "Pathogenic vs Benign")

    print("\n"+"="*66)
    print("(2) Fisher enrichment of DCS>%.2f in the AM-blind set"%DCS_ZONE)
    print("="*66)
    for label, negclasses in [("P vs B", ['Benign']), ("P vs non-P (Benign+Ambiguous)", ['Benign','Ambiguous'])]:
        sub=blind[blind.Cls.isin(['Pathogenic']+negclasses)]
        inz=sub['DCS']>DCS_ZONE; isp=sub['Cls']=='Pathogenic'
        a=int((inz&isp).sum()); b=int((inz&~isp).sum())
        c=int((~inz&isp).sum()); d=int((~inz&~isp).sum())
        OR,p=fisher_exact([[a,b],[c,d]],alternative='greater')
        print(f"  {label}: in-zone path={a}, in-zone neg={b}, out path={c}, out neg={d} | OR={OR:.2f}, p={p:.3e}")

    print("\nInterpretation guide:")
    print("  - If (1) full-set MWU is significant, report DCS is higher in pathogenic")
    print("    variants (threshold-free) as the primary evidence.")
    print("  - Use the significant, honest test; do NOT tune thresholds to chase p<0.05.")

if __name__=="__main__":
    main()

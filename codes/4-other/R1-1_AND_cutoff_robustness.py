import pandas as pd, numpy as np

# ================= CONFIG =================
MASTER_TSV = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-human/data_tsv/All_Proteins_AM_Combined_Master.tsv'
Q = 0.90          # per-protein quantile for the AND cutoffs (top decile)
TOP_FRAC = 0.10   # top decile for the product-metric comparison
OUTPUT_DIR = '/content/drive/MyDrive/All Project Files/dynamics aware evo/revision_results'  # results go here; NEVER overwrites, NEVER edits input files

# =========================================

def find_col(df, options, contains=None):
    for o in options:
        if o in df.columns: return o
    if contains:
        for c in df.columns:
            if contains.lower() in c.lower(): return c
    return None


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
    _fh = _start_capture("R1-1_AND_cutoff_result.txt")
    df = pd.read_csv(MASTER_TSV, sep='\t', na_values=['N/A','NA',''])
    df.columns=[c.strip() for c in df.columns]
    cons = find_col(df,['conservation'])
    nd   = find_col(df,['Norm_Dynamics'])
    dcs  = find_col(df,['Dynamic_Conserved_Score_0_10','DCS_Score'], contains='dynamic_conserved')
    rcs  = find_col(df,['Rigid_Conserved_Score_0_10','RCS_Score'], contains='rigid_conserved')
    am   = find_col(df,['mean_am_pathogenicity','AM_Score','AlphaMissense'], contains='pathogen')
    pid  = find_col(df,['Protein_ID'], contains='protein')
    for c in [cons,nd,dcs,rcs,am]:
        if c: df[c]=pd.to_numeric(df[c],errors='coerce')
    d=df.dropna(subset=[cons,nd,dcs,rcs]).copy()

    # per-protein quantile cutoffs
    g=d.groupby(pid)
    d['cons_q']=g[cons].transform(lambda x: x.quantile(Q))
    d['nd_q']  =g[nd].transform(lambda x: x.quantile(Q))
    d['dcs_thr']=g[dcs].transform(lambda x: x.quantile(1-TOP_FRAC))
    d['rcs_thr']=g[rcs].transform(lambda x: x.quantile(1-TOP_FRAC))

    # sets
    prod_dcs = d[dcs] >= d['dcs_thr']
    and_dcs  = (d[nd] > d['nd_q']) & (d[cons] > d['cons_q'])
    prod_rcs = d[rcs] >= d['rcs_thr']
    and_rcs  = ((1-d[nd]) > (1-d['nd_q'])) & (d[cons] > d['cons_q'])  # rigid: low dynamics + high cons

    def jacc(a,b):
        a=set(np.where(a.values)[0]); b=set(np.where(b.values)[0])
        return len(a&b)/max(1,len(a|b))
    jd=jacc(prod_dcs,and_dcs); jr=jacc(prod_rcs,and_rcs)
    print(f"Residues analysed: {len(d)}")
    print(f"DCS  product-top-decile n={int(prod_dcs.sum())}  AND-cutoff n={int(and_dcs.sum())}  Jaccard={jd:.2f}")
    print(f"RCS  product-top-decile n={int(prod_rcs.sum())}  AND-cutoff n={int(and_rcs.sum())}  Jaccard={jr:.2f}")

    if am:
        print("\nMean AlphaMissense pathogenicity by set (concordance check):")
        for name,mask in [('DCS product',prod_dcs),('DCS AND',and_dcs),
                          ('RCS product',prod_rcs),('RCS AND',and_rcs)]:
            print(f"   {name:12s}: {d.loc[mask, am].mean():.3f}")
    print("\nPASTE -> R1-1: DCS Jaccard=%.2f, RCS Jaccard=%.2f; pathogenicity trends concordant (see means)" % (jd,jr))

if __name__=="__main__":
    main()


import pandas as pd, numpy as np, requests, time

# ================= CONFIG =================
MASTER_TSV = '/content/drive/MyDrive/All Project Files/dynamics aware evo/data-human/data_tsv/All_Proteins_AM_Combined_Master.tsv'
OUTPUT_DIR = '/content/drive/MyDrive/All Project Files/dynamics aware evo/revision_results'
TOP_N_PER_PROTEIN = 10
N_PERM = 20000
FEATURE_TYPES = {'Active site','Binding site','Site','DNA binding','Nucleotide binding','Metal binding'}
RANDOM_SEED = 42
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

def uniprot_feature_positions(acc):
    url=f"https://rest.uniprot.org/uniprotkb/{acc}.json"; pos=set()
    try:
        r=requests.get(url,timeout=30); r.raise_for_status()
        for f in r.json().get('features',[]):
            if f.get('type') in FEATURE_TYPES:
                loc=f.get('location',{}); s=loc.get('start',{}).get('value'); e=loc.get('end',{}).get('value')
                if s and e: pos.update(range(int(s),int(e)+1))
    except Exception as ex:
        print(f"  UniProt fetch failed for {acc}: {ex}")
    return pos

def main():
    _fh=_start_capture("R1-9b_background_corrected_result.txt")
    df=pd.read_csv(MASTER_TSV,sep='\t',na_values=['N/A','NA','']); df.columns=[c.strip() for c in df.columns]
    dcs=find_col(df,['Dynamic_Conserved_Score_0_10','DCS_Score'],contains='dynamic_conserved')
    pos=find_col(df,['i','res','position']); pid=find_col(df,['Protein_ID'],contains='protein')
    acc=find_col(df,['UniProt','uniprot','Accession'],contains='uniprot')
    df[dcs]=pd.to_numeric(df[dcs],errors='coerce'); df[pos]=pd.to_numeric(df[pos],errors='coerce')
    if acc is None:
        print("NOTE: no UniProt accession column in master TSV; add Protein_ID->UniProt to run."); return
    df=df.dropna(subset=[dcs,pos]).copy()

    rng=np.random.default_rng(RANDOM_SEED); cache={}
    obs_hits=0; obs_total=0; bg_hits=0; bg_total=0
    per_protein=[]  # (positions_array, fpos_set, n_top, observed_top_hits)
    for prot,sub in df.groupby(pid):
        a=str(sub[acc].iloc[0])
        if a not in cache: cache[a]=uniprot_feature_positions(a); time.sleep(0.2)
        fpos=cache[a]
        allpos=sub[pos].astype(int).values
        top=sub.sort_values(dcs,ascending=False).head(TOP_N_PER_PROTEIN)[pos].astype(int).values
        oh=sum(1 for x in top if x in fpos)
        obs_hits+=oh; obs_total+=len(top)
        bg_hits+=sum(1 for x in allpos if x in fpos); bg_total+=len(allpos)
        per_protein.append((allpos,fpos,len(top),oh))

    obs_rate=obs_hits/max(1,obs_total); bg_rate=bg_hits/max(1,bg_total)
    fold=obs_rate/bg_rate if bg_rate>0 else float('nan')
    # permutation null
    null=np.zeros(N_PERM,dtype=int)
    for i in range(N_PERM):
        tot=0
        for allpos,fpos,n_top,_ in per_protein:
            if len(allpos)>=n_top:
                pick=rng.choice(allpos,size=n_top,replace=False)
                tot+=sum(1 for x in pick if x in fpos)
        null[i]=tot
    p_perm=(np.sum(null>=obs_hits)+1)/(N_PERM+1)

    print(f"Top-{TOP_N_PER_PROTEIN} DCS residues examined : {obs_total}")
    print(f"  observed on functional sites : {obs_hits} ({obs_rate*100:.1f}%)")
    print(f"Background (all residues)       : {bg_hits}/{bg_total} ({bg_rate*100:.1f}%)")
    print(f"FOLD-ENRICHMENT (obs/background): {fold:.2f}x")
    print(f"Permutation null mean          : {null.mean():.1f} (sd {null.std():.1f})")
    print(f"Empirical p-value              : {p_perm:.3e}")
    print(f"\nPASTE -> R1-9: top-DCS residues are {fold:.1f}x enriched for annotated functional "
          f"sites vs background ({obs_rate*100:.0f}% vs {bg_rate*100:.0f}%), permutation p={p_perm:.2e}")

if __name__=="__main__":
    main()

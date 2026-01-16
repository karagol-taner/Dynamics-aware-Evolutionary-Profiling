import os
import glob
import numpy as np
import pandas as pd

# ---------- FOLDERS ----------
freq_dir   = "/content/drive/MyDrive/All Project Files/dynamics aware evo/EVcouplings/align_frequencies"
enrich_dir = "/content/drive/MyDrive/All Project Files/dynamics aware evo/EVcouplings/enrichments"
rmsf_dir   = "/content/drive/MyDrive/MD/ATLAS_MDs_Alpha/0_protein_tsv_files/with_ID/RMSF"
neq_dir    = "/content/drive/MyDrive/MD/ATLAS_MDs_Alpha/0_protein_tsv_files/with_ID/Neq"

combined_dir = "/content/drive/MyDrive/All Project Files/dynamics aware evo/combined_files"
os.makedirs(combined_dir, exist_ok=True)
# -----------------------------------------------------------


def find_first(patterns):
    """Return first existing file matching any of the glob patterns, or None."""
    if isinstance(patterns, str):
        patterns = [patterns]
    for p in patterns:
        matches = glob.glob(p)
        if matches:
            return matches[0]
    return None


def needleman_wunsch_map(seq1, seq2, match=2, mismatch=-3, gap=-1):
    """
    Needleman-Wunsch Global Alignment.
    Returns mapping: seq1_index (0-based) -> seq2_index (0-based).
    """
    n, m = len(seq1), len(seq2)
    s1 = list(seq1)
    s2 = list(seq2)

    score = np.zeros((n + 1, m + 1))

    # Initialize edges
    for i in range(n + 1): score[i][0] = i * gap
    for j in range(m + 1): score[0][j] = j * gap

    # Fill Matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_score = score[i-1][j-1] + (match if s1[i-1] == s2[j-1] else mismatch)
            del_score   = score[i-1][j] + gap
            ins_score   = score[i][j-1] + gap
            score[i][j] = max(match_score, del_score, ins_score)

    # Traceback
    mapping = {}
    i, j = n, m
    
    while i > 0 or j > 0:
        curr = score[i][j]
        if i > 0 and j > 0:
            is_match = (match if s1[i-1] == s2[j-1] else mismatch)
            if abs(curr - (score[i-1][j-1] + is_match)) < 1e-9:
                mapping[i-1] = j-1
                i -= 1
                j -= 1
                continue
        if i > 0 and abs(curr - (score[i-1][j] + gap)) < 1e-9:
            i -= 1
            continue
        if j > 0 and abs(curr - (score[i][j-1] + gap)) < 1e-9:
            j -= 1
            continue

    return mapping


def build_md_aligned_rows(full_seq, backbone_df, rmsf_df=None, neq_df=None):
    if rmsf_df is None and neq_df is None:
        df = backbone_df.copy()
        cols_to_add = ["MD_aligned", "RMSF_R1", "RMSF_R2", "RMSF_R3", "Neq_R1", "Neq_R2", "Neq_R3"]
        for c in cols_to_add:
            df[c] = np.nan
        return df

    if rmsf_df is not None:
        md_seq_df = rmsf_df
    else:
        md_seq_df = neq_df

    md_seq = "".join(md_seq_df["seq"].astype(str)).replace(" ", "")
    full_seq_str = "".join(full_seq)

    mapping = needleman_wunsch_map(md_seq, full_seq_str, match=2, mismatch=-3, gap=-1)

    backbone_idx = backbone_df.set_index("i")
    backbone_indices = list(backbone_idx.index)

    rows = []
    used_ev_positions = set()

    def get_vals(df_source, idx, prefix):
        if df_source is None or idx >= len(df_source):
            return (np.nan, np.nan, np.nan)
        row = df_source.iloc[idx]
        return (
            row.get(f"{prefix}_R1", np.nan),
            row.get(f"{prefix}_R2", np.nan),
            row.get(f"{prefix}_R3", np.nan),
        )

    for md_idx in range(len(md_seq_df)):
        md_res = md_seq_df.iloc[md_idx]["seq"]
        ev_idx0 = mapping.get(md_idx, None)

        if ev_idx0 is not None:
            i_ev = ev_idx0 + 1
            used_ev_positions.add(i_ev)
            if i_ev in backbone_idx.index:
                b = backbone_idx.loc[i_ev]
                i_val = i_ev
                A_i_val = b["A_i"]
                cons_val = b["conservation"]
                enr_val = b["enrichment"]
            else:
                i_val = i_ev
                A_i_val = np.nan
                cons_val = np.nan
                enr_val = np.nan
        else:
            i_val = np.nan
            A_i_val = np.nan
            cons_val = np.nan
            enr_val = np.nan

        rmsf1, rmsf2, rmsf3 = get_vals(rmsf_df, md_idx, "RMSF")
        neq1, neq2, neq3 = get_vals(neq_df, md_idx, "Neq")

        rows.append({
            "i": i_val,
            "A_i": A_i_val,
            "conservation": cons_val,
            "enrichment": enr_val,
            "MD_aligned": md_res,
            "RMSF_R1": rmsf1,
            "RMSF_R2": rmsf2,
            "RMSF_R3": rmsf3,
            "Neq_R1": neq1,
            "Neq_R2": neq2,
            "Neq_R3": neq3,
        })

    for i_ev in backbone_indices:
        if i_ev in used_ev_positions:
            continue
        b = backbone_idx.loc[i_ev]
        rows.append({
            "i": i_ev,
            "A_i": b["A_i"],
            "conservation": b["conservation"],
            "enrichment": b["enrichment"],
            "MD_aligned": np.nan,
            "RMSF_R1": np.nan,
            "RMSF_R2": np.nan,
            "RMSF_R3": np.nan,
            "Neq_R1": np.nan,
            "Neq_R2": np.nan,
            "Neq_R3": np.nan,
        })

    return pd.DataFrame(rows, columns=[
        "i", "A_i", "conservation", "enrichment",
        "MD_aligned", "RMSF_R1", "RMSF_R2", "RMSF_R3",
        "Neq_R1", "Neq_R2", "Neq_R3"
    ])


# ===================== MAIN LOOP =====================

for pid in range(1, 94):
    protein_tag = f"protein{pid}"
    print(f"\n=== Processing {protein_tag} ===")

    # 1. Frequency
    freq_path = find_first([
        os.path.join(freq_dir, f"{protein_tag}_*.csv"),
        os.path.join(freq_dir, f"{protein_tag}_*.tsv"),
    ])
    if freq_path is None:
        print("  No frequency file → skipping.")
        continue

    print(f"  Frequency: {os.path.basename(freq_path)}")
    if freq_path.endswith(".tsv"):
        freq = pd.read_csv(freq_path, sep="\t")
    else:
        freq = pd.read_csv(freq_path)

    if not {"i", "A_i", "conservation"}.issubset(freq.columns):
        print("  Frequency missing required columns → skipping.")
        continue

    backbone = freq[["i", "A_i", "conservation"]].copy()

    # 2. Enrichment
    enrich_path = find_first([
        os.path.join(enrich_dir, f"{protein_tag}_*_enrichment.csv"),
        os.path.join(enrich_dir, f"{protein_tag}_*enrichment*.csv"),
    ])
    if enrich_path:
        print(f"  Enrichment: {os.path.basename(enrich_path)}")
        enrich = pd.read_csv(enrich_path)
        if {"i", "enrichment"}.issubset(enrich.columns):
            enrich = enrich[["i", "enrichment"]]
            backbone = backbone.merge(enrich, on="i", how="left")
        else:
            backbone["enrichment"] = np.nan
    else:
        backbone["enrichment"] = np.nan

    full_seq = "".join(backbone["A_i"].astype(str))

    # 3. RMSF & Neq
    rmsf_df = None
    rmsf_path = find_first([
        os.path.join(rmsf_dir, f"{protein_tag}_*.csv"),
        os.path.join(rmsf_dir, f"{protein_tag}_*.tsv"),
    ])
    if rmsf_path:
        print(f"  RMSF: {os.path.basename(rmsf_path)}")
        rmsf_df = pd.read_csv(rmsf_path, sep="\t")
        if "seq" not in rmsf_df.columns: rmsf_df = None

    neq_df = None
    neq_path = find_first([
        os.path.join(neq_dir, f"{protein_tag}_*.csv"),
        os.path.join(neq_dir, f"{protein_tag}_*.tsv"),
    ])
    if neq_path:
        print(f"  Neq: {os.path.basename(neq_path)}")
        neq_df = pd.read_csv(neq_path, sep="\t")
        if "seq" not in neq_df.columns: neq_df = None

    # 4. Build Table
    combined = build_md_aligned_rows(full_seq, backbone, rmsf_df=rmsf_df, neq_df=neq_df)

    # ---- NEW FIX: Format 'i' to remove decimals (1.0 -> 1) ----
    # We apply a lambda: if it's a number, make it int->str; if it's empty, keep it NaN
    combined["i"] = combined["i"].apply(lambda x: str(int(x)) if pd.notnull(x) else np.nan)

    # 5. Fill remaining N/A (this covers 'enrichment' and the 'i' NaNs)
    combined = combined.fillna("N/A")

    out_path = os.path.join(combined_dir, f"{protein_tag}_combined.csv")
    combined.to_csv(out_path, index=False)
    print(f"  → Saved: {out_path}")

print("\nALL DONE.")

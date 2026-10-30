"""Data processing, stratified splitting, and length profiling for Zebrafish repeats."""

import os
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any

import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def reverse_complement(seq: str) -> str:
    """Compute the reverse complement of a DNA sequence."""
    return seq.translate(COMPLEMENT)[::-1]


def assign_length_bins(
    df: pd.DataFrame,
    seq_col: str = "Sequence",
    bins: List[int] = None,
    labels: List[str] = None,
) -> pd.DataFrame:
    """Assign sequence length and length range bins to dataframe."""
    if bins is None:
        bins = [0, 50, 100, 200, 500, 1000, 2000, 1000000]
    if labels is None:
        labels = ["<50", "50-100", "100-200", "200-500", "500-1000", "1000-2000", ">2000"]

    df = df.copy()
    df["seq_length"] = df[seq_col].astype(str).str.len()
    df["length_range"] = pd.cut(df["seq_length"], bins=bins, labels=labels, right=False)
    return df


def prepare_stratified_splits(
    raw_csv: str,
    train_csv: str,
    val_csv: str,
    val_ratio: float = 0.10,
    random_seed: int = 42,
    bins: List[int] = None,
    labels: List[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load, deduplicate, bin by length, and produce leakage-free stratified splits."""
    print(f"Loading raw dataset from {raw_csv}...")
    df = pd.read_csv(raw_csv)
    df = df.dropna(subset=["Sequence", "repeat_class"])
    df["Sequence"] = df["Sequence"].astype(str).str.strip().str.upper()
    df["repeat_class"] = df["repeat_class"].astype(str).str.strip()

    # Drop exact duplicate sequences
    initial_count = len(df)
    df = df.drop_duplicates(subset=["Sequence"]).reset_index(drop=True)
    print(f"Loaded {len(df)} unique sequences (dropped {initial_count - len(df)} duplicates).")

    # Add length bins
    df = assign_length_bins(df, seq_col="Sequence", bins=bins, labels=labels)

    # Composite stratification label: class + length_range
    df["strat_key"] = df["repeat_class"].astype(str) + "__" + df["length_range"].astype(str)

    # Merge any stratum with count < 2 into the most common stratum of that class
    for cls_name, cls_group in df.groupby("repeat_class"):
        counts = cls_group["strat_key"].value_counts()
        most_common_key = counts.index[0]
        rare_keys = counts[counts < 2].index.tolist()
        for rk in rare_keys:
            df.loc[df["strat_key"] == rk, "strat_key"] = most_common_key

    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        random_state=random_seed,
        stratify=df["strat_key"],
    )

    train_df = train_df.copy().reset_index(drop=True)
    val_df = val_df.copy().reset_index(drop=True)

    # Leakage check: ensure no val sequence or its reverse complement is in train
    train_seqs = set(train_df["Sequence"].tolist())
    train_rc_seqs = {reverse_complement(s) for s in train_seqs}

    val_leak_mask = val_df["Sequence"].isin(train_seqs) | val_df["Sequence"].isin(train_rc_seqs)
    if val_leak_mask.any():
        n_leak = val_leak_mask.sum()
        print(f"Warning: Found {n_leak} leaking sequences in validation set! Removing them from val...")
        val_df = val_df[~val_leak_mask].reset_index(drop=True)

    print(f"Final splits: Train = {len(train_df)} rows, Val = {len(val_df)} rows.")

    # Save to disk
    Path(train_csv).parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    print(f"Saved: {train_csv} and {val_csv}")

    return train_df, val_df


def profile_length_distributions(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_csv: str,
    tokenizer_id: str = "HuggingFaceBio/Carbon-X",
) -> pd.DataFrame:
    """Compute nucleotide length and token length distributions across splits."""
    print(f"Profiling length distribution and token counts using {tokenizer_id}...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)

    records = []
    for split_name, df in [("train", train_df), ("val", val_df)]:
        seq_lens = df["seq_length"].values
        sample_size = min(len(df), 2000)
        sample_seqs = df["Sequence"].sample(n=sample_size, random_state=42).tolist()
        token_lens = [len(tokenizer.encode(f"<dna>{s}</dna>", add_special_tokens=False)) for s in sample_seqs]

        records.append({
            "split": split_name,
            "total_samples": len(df),
            "seq_len_min": int(np.min(seq_lens)),
            "seq_len_25pct": int(np.percentile(seq_lens, 25)),
            "seq_len_median": int(np.median(seq_lens)),
            "seq_len_75pct": int(np.percentile(seq_lens, 75)),
            "seq_len_95pct": int(np.percentile(seq_lens, 95)),
            "seq_len_99pct": int(np.percentile(seq_lens, 99)),
            "seq_len_max": int(np.max(seq_lens)),
            "token_len_min": int(np.min(token_lens)),
            "token_len_median": int(np.median(token_lens)),
            "token_len_95pct": int(np.percentile(token_lens, 95)),
            "token_len_max": int(np.max(token_lens)),
        })

    summary_df = pd.DataFrame(records)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_csv, index=False)
    print(f"Saved length distribution summary to {output_csv}:")
    print(summary_df.to_string())
    return summary_df


if __name__ == "__main__":
    with open("configs/qlora.yaml", "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    train_df, val_df = prepare_stratified_splits(
        raw_csv=data_cfg["raw_csv"],
        train_csv=data_cfg["train_csv"],
        val_csv=data_cfg["val_csv"],
        val_ratio=1.0 - float(data_cfg["split_ratio"]),
        random_seed=data_cfg["random_seed"],
        bins=data_cfg["length_bins"],
        labels=data_cfg["length_labels"],
    )

    profile_length_distributions(
        train_df=train_df,
        val_df=val_df,
        output_csv="results/length_distribution.csv",
        tokenizer_id=config["model"]["model_id"],
    )

import pytest
import pandas as pd
from src.data import assign_length_bins, reverse_complement, prepare_stratified_splits


def test_reverse_complement():
    assert reverse_complement("ATGC") == "GCAT"
    assert reverse_complement("AATTGGCC") == "GGCCAATT"


def test_assign_length_bins():
    df = pd.DataFrame({"Sequence": ["A" * 30, "A" * 75, "A" * 150, "A" * 350, "A" * 800, "A" * 1500, "A" * 3000]})
    df = assign_length_bins(df)
    assert df["length_range"].tolist() == ["<50", "50-100", "100-200", "200-500", "500-1000", "1000-2000", ">2000"]


def test_no_leakage_and_stratification(tmp_path):
    # Create synthetic dataset with multiple classes and lengths
    records = []
    classes = ["DNA", "LINE", "LTR"]
    for c in classes:
        for length in [40, 80, 150, 400]:
            for i in range(10):
                records.append({
                    "Sequence": f"{c}_{length}_{i}_" + "A" * length,
                    "repeat_class": c
                })
    raw_csv = tmp_path / "raw.csv"
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    pd.DataFrame(records).to_csv(raw_csv, index=False)

    train_df, val_df = prepare_stratified_splits(
        raw_csv=str(raw_csv),
        train_csv=str(train_csv),
        val_csv=str(val_csv),
        val_ratio=0.10,
        random_seed=42,
    )

    # Check zero overlap
    train_seqs = set(train_df["Sequence"])
    val_seqs = set(val_df["Sequence"])
    assert len(train_seqs.intersection(val_seqs)) == 0

    # Check reverse complement non-overlap
    val_rc = {reverse_complement(s) for s in val_seqs}
    assert len(train_seqs.intersection(val_rc)) == 0

    # Check all classes present in both splits
    assert set(train_df["repeat_class"]) == set(classes)
    assert set(val_df["repeat_class"]) == set(classes)

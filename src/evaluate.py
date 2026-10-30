"""Evaluation script with overall metrics and breakdown across sequence length ranges."""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
)

from src.prompts import get_class_prompt, build_chat_messages, format_chat_prompt
from src.normalize import normalize_label
from src.model import load_model_for_evaluation
from src.data import assign_length_bins


def compute_metrics(
    predictions_df: pd.DataFrame,
    true_col: str = "true_label",
    pred_col: str = "pred_label",
) -> Dict[str, Any]:
    """Calculate accuracy, macro/weighted precision, recall, F1, MCC, and invalid rate."""
    y_true = predictions_df[true_col].astype(str).tolist()
    y_pred = predictions_df[pred_col].astype(str).tolist()
    total = len(y_true)

    invalid_mask = [p == "INVALID" for p in y_pred]
    n_invalid = sum(invalid_mask)
    invalid_rate = round(n_invalid / total, 4) if total > 0 else 0.0

    labels = sorted(list(set(y_true)))

    acc = round(float(accuracy_score(y_true, y_pred)), 4)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    mcc = round(float(matthews_corrcoef(y_true, y_pred)), 4)

    class_rep = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    return {
        "total_samples": total,
        "valid_predictions": total - n_invalid,
        "invalid_predictions": n_invalid,
        "invalid_rate": invalid_rate,
        "accuracy": acc,
        "precision_macro": round(float(p_macro), 4),
        "recall_macro": round(float(r_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "precision_weighted": round(float(p_weighted), 4),
        "recall_weighted": round(float(r_weighted), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "mcc": mcc,
        "per_class": class_rep,
    }


def compute_length_breakdown_metrics(
    predictions_df: pd.DataFrame,
    bins: List[int] = None,
    labels: List[str] = None,
    output_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Compute benchmark performance metrics across sequence length ranges."""
    df = predictions_df.copy()
    if "length_range" not in df.columns:
        df = assign_length_bins(df, seq_col="Sequence", bins=bins, labels=labels)

    records = []
    for length_bucket, group in df.groupby("length_range", observed=False):
        if len(group) == 0:
            continue
        m = compute_metrics(group)
        records.append({
            "length_range": str(length_bucket),
            "sample_count": len(group),
            "accuracy": m["accuracy"],
            "f1_macro": m["f1_macro"],
            "f1_weighted": m["f1_weighted"],
            "invalid_rate": m["invalid_rate"],
        })

    summary_df = pd.DataFrame(records)
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_csv, index=False)
        print(f"Length range breakdown saved to {output_csv}:")
        print(summary_df.to_string())

    return summary_df


def plot_confusion_matrix(
    predictions_df: pd.DataFrame,
    output_png: Path,
    true_col: str = "true_label",
    pred_col: str = "pred_label",
    title: str = "Repeat Classification Confusion Matrix",
) -> None:
    """Plot and save confusion matrix heatmap."""
    y_true = predictions_df[true_col].astype(str).tolist()
    y_pred = predictions_df[pred_col].astype(str).tolist()
    labels = sorted(list(set(y_true)))

    pred_labels = list(labels)
    if "INVALID" in y_pred and "INVALID" not in pred_labels:
        pred_labels.append("INVALID")

    cm = confusion_matrix(y_true, y_pred, labels=pred_labels)[: len(labels)]
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=pred_labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={"label": "Normalized Recall"},
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Class", fontsize=12)
    ax.set_ylabel("True Class", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300)
    plt.close(fig)
    print(f"Confusion matrix saved to {output_png}")


def run_evaluation(
    model: Any,
    tokenizer: Any,
    test_csv: str,
    output_csv: str,
    allowed_classes: List[str],
    species: str = "Danio_rerio (zebrafish)",
    limit: Optional[int] = None,
    eval_only: bool = False,
):
    """Run greedy prediction and evaluate predictions streaming to CSV."""
    df = pd.read_csv(test_csv)
    df = df.dropna(subset=["Sequence", "repeat_class"])
    df = assign_length_bins(df)

    if limit is not None and limit > 0:
        df = df.head(limit)

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    completed_ids = set()
    if out_path.exists() and not eval_only:
        existing_df = pd.read_csv(out_path)
        if "sample_id" in existing_df.columns:
            completed_ids = set(existing_df["sample_id"].tolist())

    if not eval_only and len(completed_ids) < len(df):
        if not out_path.exists():
            header = "sample_id,Sequence,seq_length,length_range,true_label,raw_output,pred_label,match_status,latency_sec\n"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(header)

        pbar = tqdm(total=len(df), initial=len(completed_ids), desc=f"Evaluating {Path(test_csv).name}")
        for idx, row in df.iterrows():
            if idx in completed_ids:
                continue

            seq = str(row["Sequence"]).strip()
            true_label = str(row["repeat_class"]).strip()
            seq_len = row["seq_length"]
            length_range = row["length_range"]

            user_prompt = get_class_prompt(seq, allowed_classes, species=species)
            messages = build_chat_messages(user_prompt)
            prompt_str = format_chat_prompt(tokenizer, messages, add_generation_prompt=True)

            device = getattr(model, "device", next(model.parameters()).device)
            inputs = {k: v.to(device) for k, v in tokenizer(prompt_str, return_tensors="pt").items()}

            t0 = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=16,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            input_len = inputs["input_ids"].shape[1]
            gen_ids = outputs[0][input_len:]
            raw_output = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            latency = round(time.time() - t0, 3)

            pred_label, match_status = normalize_label(raw_output, allowed_classes)

            escaped_seq = seq.replace(",", "")
            escaped_raw = raw_output.replace('"', '""').replace("\n", " ")
            line = f'{idx},"{escaped_seq}",{seq_len},"{length_range}","{true_label}","{escaped_raw}","{pred_label}","{match_status}",{latency}\n'

            with open(out_path, "a", encoding="utf-8") as f:
                f.write(line)

            pbar.update(1)
        pbar.close()

    results_df = pd.read_csv(out_path)
    metrics = compute_metrics(results_df)

    metrics_json = out_path.parent / f"metrics_{out_path.stem}.json"
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_json}")

    breakdown_csv = out_path.parent / f"length_breakdown_{out_path.stem}.csv"
    compute_length_breakdown_metrics(results_df, output_csv=str(breakdown_csv))

    cm_png = out_path.parent / f"confusion_matrix_{out_path.stem}.png"
    plot_confusion_matrix(results_df, cm_png, title=f"Confusion Matrix: {out_path.stem}")

    return metrics

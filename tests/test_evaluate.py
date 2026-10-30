import pytest
import pandas as pd
from pathlib import Path
from src.evaluate import compute_metrics, compute_length_breakdown_metrics, plot_confusion_matrix


def test_compute_metrics():
    df = pd.DataFrame({
        "true_label": ["DNA", "LTR", "LINE", "Satellite"],
        "pred_label": ["DNA", "LTR", "DNA", "INVALID"],
    })
    m = compute_metrics(df)
    assert m["total_samples"] == 4
    assert m["invalid_predictions"] == 1
    assert m["invalid_rate"] == 0.25
    assert m["accuracy"] == 0.5


def test_compute_length_breakdown_metrics(tmp_path):
    df = pd.DataFrame({
        "Sequence": ["A" * 30, "A" * 40, "A" * 70, "A" * 150, "A" * 600],
        "true_label": ["DNA", "DNA", "LTR", "LINE", "Satellite"],
        "pred_label": ["DNA", "DNA", "LTR", "LINE", "Satellite"],
    })
    out_csv = tmp_path / "breakdown.csv"
    res = compute_length_breakdown_metrics(df, output_csv=str(out_csv))
    assert out_csv.exists()
    assert len(res) == 4  # '<50', '50-100', '100-200', '500-1000'
    # All were perfect predictions
    assert (res["accuracy"] == 1.0).all()


def test_plot_confusion_matrix(tmp_path):
    df = pd.DataFrame({
        "true_label": ["DNA", "LTR", "LINE"],
        "pred_label": ["DNA", "LINE", "LINE"],
    })
    png_path = tmp_path / "cm.png"
    plot_confusion_matrix(df, output_png=png_path)
    assert png_path.exists()
    assert png_path.stat().st_size > 0

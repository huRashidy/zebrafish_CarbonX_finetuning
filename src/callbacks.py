import csv
import time
from pathlib import Path
import torch
from transformers import TrainerCallback


class CSVTrackingCallback(TrainerCallback):
    """Callback to stream training step metrics and evaluation metrics to CSV files."""

    def __init__(self, train_csv: str = "results/train_log.csv", eval_csv: str = "results/eval_log.csv"):
        self.train_csv = Path(train_csv)
        self.eval_csv = Path(eval_csv)
        self.train_csv.parent.mkdir(parents=True, exist_ok=True)
        self.eval_csv.parent.mkdir(parents=True, exist_ok=True)

        # Initialize train_log.csv if not exists
        if not self.train_csv.exists():
            with open(self.train_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "step",
                    "epoch",
                    "loss",
                    "learning_rate",
                    "grad_norm",
                    "gpu0_mem_mb",
                    "gpu1_mem_mb",
                    "timestamp",
                ])

        # Initialize eval_log.csv if not exists
        if not self.eval_csv.exists():
            with open(self.eval_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "step",
                    "epoch",
                    "eval_loss",
                    "eval_runtime",
                    "eval_samples_per_second",
                    "timestamp",
                ])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or state is None:
            return

        # Check if this is a training log (contains 'loss')
        if "loss" in logs:
            step = state.global_step
            epoch = round(state.epoch, 4) if state.epoch is not None else 0.0
            loss = logs.get("loss", "")
            lr = logs.get("learning_rate", "")
            grad_norm = logs.get("grad_norm", "")

            gpu0_mb = 0.0
            gpu1_mb = 0.0
            if torch.cuda.is_available():
                if torch.cuda.device_count() > 0:
                    gpu0_mb = round(torch.cuda.memory_allocated(0) / (1024 * 1024), 2)
                if torch.cuda.device_count() > 1:
                    gpu1_mb = round(torch.cuda.memory_allocated(1) / (1024 * 1024), 2)

            with open(self.train_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    step,
                    epoch,
                    loss,
                    lr,
                    grad_norm,
                    gpu0_mb,
                    gpu1_mb,
                    round(time.time(), 2),
                ])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or state is None:
            return

        step = state.global_step
        epoch = round(state.epoch, 4) if state.epoch is not None else 0.0
        eval_loss = metrics.get("eval_loss", "")
        eval_runtime = metrics.get("eval_runtime", "")
        eval_samples_per_sec = metrics.get("eval_samples_per_second", "")

        with open(self.eval_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                step,
                epoch,
                eval_loss,
                eval_runtime,
                eval_samples_per_sec,
                round(time.time(), 2),
            ])

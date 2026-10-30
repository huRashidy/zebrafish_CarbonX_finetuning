"""QLoRA fine-tuning script for Carbon-X on Zebrafish repeat sequences."""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List

# Ensure local modules are accessible
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import torch
import pandas as pd
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

from src.prompts import format_sft_dataset_row
from src.model import load_base_model_and_tokenizer, setup_single_gpu_peft_model
from src.callbacks import CSVTrackingCallback


class ShardedSFTTrainer(SFTTrainer):
    """SFTTrainer subclass that computes native Causal LM loss across sharded GPUs."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
        loss = loss.to(self.args.device)
        return (loss, outputs) if return_outputs else loss


def prepare_hf_dataset(
    csv_path: str,
    allowed_classes: List[str],
    species: str = "Danio_rerio (zebrafish)",
    limit: int = None,
) -> Dataset:
    """Load split CSV and convert to HF Dataset formatted for chat SFT."""
    df = pd.read_csv(csv_path)
    if limit is not None and limit > 0:
        df = df.head(limit)

    records = [
        format_sft_dataset_row(row, allowed_classes=allowed_classes, species=species)
        for _, row in df.iterrows()
    ]
    return Dataset.from_list(records)


def train(
    config_path: str = "configs/qlora.yaml",
    smoke_test: bool = False,
    override_epochs: int = None,
):
    """Execute QLoRA fine-tuning with 2-GPU sharding and single-GPU LoRA allocation."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    lora_cfg = config["lora"]
    data_cfg = config["data"]
    training_cfg = config["training"]

    print("=" * 60)
    print(f"Starting Carbon-X Zebrafish QLoRA Fine-Tuning")
    print(f"Base model sharded on 2 GPUs; LoRA adapter on GPU {lora_cfg.get('target_gpu', 1)}")
    if smoke_test:
        print("[SMOKE TEST MODE ENABLED - running on small subset]")
    print("=" * 60)

    # 1. Load Tokenizer & Base Model
    tokenizer, base_model = load_base_model_and_tokenizer(model_cfg)

    # 2. Setup targeted LoRA on single GPU
    peft_model = setup_single_gpu_peft_model(base_model, lora_cfg)

    # 3. Load Data & Prepare Datasets
    train_csv = data_cfg["train_csv"]
    val_csv = data_cfg["val_csv"]

    train_df = pd.read_csv(train_csv)
    allowed_classes = sorted(train_df["repeat_class"].dropna().unique().tolist())
    print(f"\nAllowed classes ({len(allowed_classes)}): {allowed_classes}")

    train_limit = 32 if smoke_test else None
    val_limit = 16 if smoke_test else None

    print(f"Preparing datasets (smoke_test limit: {train_limit})...")
    train_dataset = prepare_hf_dataset(train_csv, allowed_classes, limit=train_limit)
    val_dataset = prepare_hf_dataset(val_csv, allowed_classes, limit=val_limit)
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # 4. Training configuration
    output_dir = "outputs/smoke_test" if smoke_test else training_cfg["output_dir"]
    train_log_csv = "results/smoke_train_log.csv" if smoke_test else training_cfg.get("train_log_csv", "results/train_log.csv")
    eval_log_csv = "results/smoke_eval_log.csv" if smoke_test else training_cfg.get("eval_log_csv", "results/eval_log.csv")

    epochs = 1 if smoke_test else (override_epochs or int(training_cfg["num_train_epochs"]))
    logging_steps = 2 if smoke_test else int(training_cfg["logging_steps"])
    eval_steps = 5 if smoke_test else int(training_cfg["eval_steps"])
    save_steps = 5 if smoke_test else int(training_cfg["save_steps"])
    max_steps = 10 if smoke_test else -1
    warmup_steps = 2 if smoke_test else int(training_cfg.get("warmup_steps", 10))

    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=int(training_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(training_cfg["gradient_accumulation_steps"]),
        learning_rate=float(training_cfg["learning_rate"]),
        lr_scheduler_type=training_cfg["lr_scheduler_type"],
        warmup_steps=warmup_steps,
        num_train_epochs=epochs,
        max_steps=max_steps,
        logging_steps=logging_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=int(training_cfg["save_total_limit"]),
        fp16=bool(training_cfg["fp16"]),
        bf16=bool(training_cfg["bf16"]),
        gradient_checkpointing=bool(training_cfg["gradient_checkpointing"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=training_cfg["optim"],
        report_to="none",
        load_best_model_at_end=not smoke_test,
        metric_for_best_model="eval_loss",
        max_length=int(training_cfg["max_length"]),
        assistant_only_loss=True,
    )

    csv_callback = CSVTrackingCallback(
        train_csv=train_log_csv,
        eval_csv=eval_log_csv,
    )

    trainer = ShardedSFTTrainer(
        model=peft_model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        callbacks=[csv_callback],
    )

    print("\nStarting training loop...")
    trainer.train()

    print(f"\nTraining complete! Saving final model adapter and tokenizer to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Carbon-X Zebrafish QLoRA Fine-Tuning")
    parser.add_argument("--config", type=str, default="configs/qlora.yaml", help="Path to YAML config")
    parser.add_argument("--smoke_test", action="store_true", help="Run quick 10-step smoke test")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    args = parser.parse_args()

    train(
        config_path=args.config,
        smoke_test=args.smoke_test,
        override_epochs=args.epochs,
    )

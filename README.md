# Carbon-X Zebrafish Repeat Fine-Tuning (QLoRA)

This project fine-tunes `HuggingFaceBio/Carbon-X` on *Danio rerio* (zebrafish) repetitive elements annotated by RepeatMasker using QLoRA (TRL `SFTTrainer` + PEFT + bitsandbytes).

## Architecture Highlights
- **Hardware:** 2 x NVIDIA GeForce RTX 2080 Ti (11GB VRAM each).
- **Base Model Sharding:** Base model is sharded across GPU 0 and GPU 1 using 4-bit NF4 quantization.
- **LoRA Adapter Placement:** LoRA trainable adapters and optimizer states are allocated exclusively on GPU 1.
- **Sequence Preservation:** No sequence truncation; full sequences grouped by length with evaluation reported across sequence length ranges.
- **Tracking:** Complete CSV logging for training losses, GPU memory footprint, and evaluation metrics.
- **Cross-Species Evaluation:** Benchmarked against the human repeat dataset in `/home/hussin/repeat_benchmark`.

## Workflow
1. `src/data.py`: Prepares stratified train/val splits and profiles length distributions.
2. `src/prompts.py`: Standardizes chat templates with `<dna>` tags.
3. `src/model.py`: Sharded model loader with targeted single-GPU LoRA allocation.
4. `src/train.py`: SFTTrainer execution loop with CSV tracking.
5. `src/evaluate.py`: Generates validation and cross-species benchmark metrics by length range.

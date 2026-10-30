# Carbon-X Zebrafish Repeat Fine-Tuning (QLoRA)

This project fine-tunes **`HuggingFaceBio/Carbon-X`** (14.8B genomic foundation model) on *Danio rerio* (zebrafish) repetitive elements annotated by RepeatMasker using **QLoRA** (TRL `SFTTrainer` + PEFT + bitsandbytes 4-bit NF4).

---

## Key Highlights & Architecture

- **Multi-GPU Sharding & Targeted LoRA Placement:**
  - Base model (40 layers, 14.8B parameters) sharded in 4-bit NF4 across **2 × NVIDIA GeForce RTX 2080 Ti (11 GB each)**.
  - GPU 0: Layers 0–17 (18 layers, ~6.0 GB VRAM).
  - GPU 1: Layers 18–39 (22 layers, ~6.8 GB VRAM).
  - LoRA trainable parameters ($A$ and $B$, rank=16, alpha=32) are allocated **strictly on GPU 1** (targeting layers 18–39), keeping all backward gradients and optimizer states on a single GPU.
- **Strict Sequence Preservation (No Cropping):**
  - Full genomic sequences preserved up to maximum length (max 6,213 bp, ~1,038 tokens).
  - Evaluated and tracked across **7 discrete sequence length ranges**: `<50`, `50–100`, `100–200`, `200–500`, `500–1000`, `1000–2000`, and `>2000` bp.
- **Deduplication & Stratification:**
  - 6,830 unique sequences (both exact and reverse-complement deduplicated) split 90% train (6,147 samples) and 10% validation (682 samples), stratified jointly across class and length buckets with 0% data leakage.
- **Cross-Species Transfer Benchmark:**
  - Fine-tuned adapter evaluated on the 2,000-sample human repeat benchmark (`/home/hussin/repeat_benchmark/data/repeat_class_dataset_2000.csv`).

---

## Prompt Template

Every sequence is formatted with official Hugging Face Bio `<dna>` framing and single-turn chat formatting with thinking disabled (`enable_thinking=False`):

```python
f"The following DNA sequence is a repetitive element from Danio_rerio (zebrafish) genome, "
f"annotated by RepeatMasker.\n\n"
f"Sequence: <dna>{sequence.strip()}</dna>\n\n"
f"Which repeat class does this sequence belong to?\n"
f"Choose exactly one from the following allowed classes:\n"
f"{classes_str}\n\n"
f"Answer with only the class name from the list, nothing else."
```

---

## Training Dynamics & Loss Convergence

Training ran for 1 full epoch (385 optimizer steps, batch size 1 × gradient accumulation 16) with evaluation every 50 steps:

| Step | Epoch | Train Loss | Validation Loss | Gradient Norm | Learning Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **10** | 0.03 | 17.34 | — | 27.12 | 1.80e-4 |
| **50** | 0.13 | 3.74 | 0.2366 | 10.31 | 1.95e-4 |
| **100** | 0.26 | 2.54 | 0.1793 | 5.47 | 1.74e-4 |
| **150** | 0.39 | 2.28 | 0.1647 | 4.94 | 1.40e-4 |
| **200** | 0.52 | 2.03 | 0.1379 | 9.75 | 9.87e-5 |
| **250** | 0.65 | 1.98 | 0.1209 | 12.06 | 5.82e-5 |
| **300** | 0.78 | 1.27 | 0.1132 | 5.00 | 2.49e-5 |
| **350** | 0.91 | 1.34 | **0.1078** | 8.50 | 4.51e-6 |
| **385** | 1.00 | 1.68 | **0.1077** | 5.34 | 1.26e-7 |

- **Total Training Duration:** 2 hours, 56 minutes.
- **Best Adapter Saved:** `outputs/carbonx-zf-qlora` (135 MB adapter weights + tokenizer).

---

## Validation Results (*Danio rerio*, N = 682)

- **Accuracy:** **`73.31%`**
- **Macro F1:** **`0.7581`**
- **Weighted F1:** **`0.7321`**
- **MCC:** **`0.7047`**
- **Invalid Rate:** **`0.00%`** (0 / 682 invalid outputs)

### Scaling Law across Sequence Length Ranges

| Length Range (bp) | Sample Count | Accuracy | Macro F1 | Weighted F1 | Invalid Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`< 50`** | 141 | 65.96% | 0.4434 | 0.6573 | 0.0% |
| **`50–100`** | 196 | 69.90% | 0.6090 | 0.6954 | 0.0% |
| **`100–200`** | 159 | 72.96% | 0.7498 | 0.7281 | 0.0% |
| **`200–500`** | 138 | 81.16% | 0.8469 | 0.8043 | 0.0% |
| **`500–1000`** | 34 | 82.35% | 0.8186 | 0.8168 | 0.0% |
| **`1000–2000`** | 10 | 100.00% | 1.0000 | 1.0000 | 0.0% |
| **`> 2000`** | 4 | 100.00% | 1.0000 | 1.0000 | 0.0% |

---

## Cross-Species Generalization to Human Repeats (N = 2,000)

Evaluating the fine-tuned zebrafish adapter on human repetitive elements (`repeat_benchmark`):

| Model & Evaluation Scope | Candidate Classes | Accuracy | Macro F1 | Weighted F1 | Matthews Corr (MCC) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Base Carbon-X (Zero-Shot)** | 15 human classes | 4.90% | 0.0355 | 0.0305 | +0.0348 |
| **Fine-Tuned Carbon-X (QLoRA)** | 15 human classes | **29.05%** | **0.3176** | **0.2691** | **+0.2017** |
| **Base Carbon-X (Zero-Shot)** | 11 shared classes | 5.35% | 0.0516 | 0.0358 | +0.0426 |
| **Fine-Tuned Carbon-X (QLoRA)** | 11 shared classes | **31.71%** | **0.4483** | **0.3033** | **+0.2218** |

- **Macro F1 Improvement:** **`8.7×` increase** over zero-shot base Carbon-X on shared repeat classes.
- **Accuracy Improvement:** **`6.0×` increase** on cross-species evaluation.

---

## Reproduction & Execution

### 1. Training
```bash
bash scripts/run_train.sh
```

### 2. Validation Evaluation & Length Breakdown
```bash
bash scripts/run_eval.sh
```

### 3. Human Cross-Species Benchmark
```bash
cd /home/hussin/repeat_benchmark
bash scripts/run_qlora_benchmark.sh
```

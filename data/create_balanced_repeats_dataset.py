#!/usr/bin/env python3
"""
Generate a balanced dataset of 8,000 repeat sequences sampled from all valid repeat classes
in /home/hussin/NTv3/data/dr_repeats_train.csv and extracted from
/home/hussin/NTv3/data/Danio_rerio_merged.fa.

Output format:
CSV with header "Sequence,repeat_class"
"""

import sys
import random
import csv
from collections import defaultdict
from pyfaidx import Fasta

TOTAL_TARGET = 8000
CSV_INPUT = "/home/hussin/NTv3/data/dr_repeats_train.csv"
FASTA_INPUT = "/home/hussin/NTv3/data/Danio_rerio_merged.fa"
OUTPUT_CSV = "/home/hussin/NTv3/data/dr_repeats_8000_balanced.csv"
RANDOM_SEED = 42

COMPLEMENT = str.maketrans('ATCGatcgNn', 'TAGCtagcNn')

def revcomp(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]

def main():
    print(f"Loading reference FASTA index: {FASTA_INPUT}")
    fa = Fasta(FASTA_INPUT, sequence_always_upper=True)
    available_chroms = set(fa.keys())
    print(f"Available chromosomes in FASTA ({len(available_chroms)}): {sorted(list(available_chroms))}")

    # Reservoir sampling per class
    # Classes to ignore: unknown, ending with '?', and ARTEFACT (only 6 instances)
    # First pass: collect indices or reservoir sample candidates
    # To ensure balance:
    # 8000 // 11 = 727, with 3 classes getting 728 (727 * 8 + 728 * 3 = 8000)
    
    # Let's inspect candidate counts per valid class first
    print("\nScanning dr_repeats_train.csv for valid candidates...")
    
    # We will store lines grouped by class using reservoir sampling
    # or two-pass sampling for exact reproducibility and minimal RAM
    # Since we want ~728 samples per class, reservoir sampling with reservoir size ~1000 per class takes trivial RAM!
    
    # We will keep a larger reservoir (e.g. 1500 per class) to filter out sequences containing invalid characters if any
    RESERVOIR_SIZE = 1500
    reservoirs = defaultdict(list)
    counts = defaultdict(int)
    
    random.seed(RANDOM_SEED)

    with open(CSV_INPUT, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        geno_name_idx = header.index("genoName")
        start_idx = header.index("genoStart")
        end_idx = header.index("genoEnd")
        strand_idx = header.index("strand")
        class_idx = header.index("repClass")
        
        for line_no, line in enumerate(f, start=2):
            parts = line.strip().split("\t")
            if len(parts) <= class_idx:
                continue
            
            rep_class = parts[class_idx]
            if rep_class.lower() == "unknown" or rep_class.endswith("?") or rep_class == "ARTEFACT":
                continue
            
            chrom = parts[geno_name_idx].replace("chr", "")
            if chrom not in available_chroms:
                continue
            
            try:
                start = int(parts[start_idx])
                end = int(parts[end_idx])
            except ValueError:
                continue
            
            if start >= end:
                continue
                
            strand = parts[strand_idx]
            counts[rep_class] += 1
            
            # Reservoir sampling for this class
            n = counts[rep_class]
            res = reservoirs[rep_class]
            item = (chrom, start, end, strand)
            if len(res) < RESERVOIR_SIZE:
                res.append(item)
            else:
                j = random.randint(0, n - 1)
                if j < RESERVOIR_SIZE:
                    res[j] = item

    classes = sorted(reservoirs.keys())
    print(f"\nDiscovered {len(classes)} target classes:")
    for c in classes:
        print(f"  - {c:<16}: {counts[c]:>8} total occurrences in genome")

    # Allocate exact quota per class
    num_classes = len(classes)
    base_quota = TOTAL_TARGET // num_classes
    remainder = TOTAL_TARGET % num_classes
    
    quotas = {}
    for i, c in enumerate(classes):
        quotas[c] = base_quota + (1 if i < remainder else 0)

    print("\nClass Quotas for 8000 total:")
    for c in classes:
        print(f"  - {c:<16}: {quotas[c]}")
    assert sum(quotas.values()) == TOTAL_TARGET

    print("\nExtracting sequences and generating dataset...")
    records = []
    
    for c in classes:
        res = reservoirs[c]
        random.shuffle(res)
        needed = quotas[c]
        extracted = 0
        
        for chrom, start, end, strand in res:
            if extracted >= needed:
                break
                
            # Pyfaidx slice [start:end] (0-based start, non-inclusive end)
            chrom_len = len(fa[chrom])
            if end > chrom_len:
                continue
                
            seq = str(fa[chrom][start:end]).upper()
            if not seq:
                continue
                
            # Discard if contains mostly 'N'
            if seq.count('N') / len(seq) > 0.2:
                continue
                
            if strand == '-':
                seq = revcomp(seq)
                
            records.append((seq, c))
            extracted += 1
            
        print(f"Extracted {extracted}/{needed} sequences for class '{c}'")
        if extracted < needed:
            raise RuntimeError(f"Could not extract sufficient sequences for class {c}: got {extracted}, need {needed}")

    # Shuffle the final 8000 records
    random.shuffle(records)
    assert len(records) == TOTAL_TARGET

    print(f"\nWriting dataset to {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Sequence", "repeat_class"])
        for seq, c in records:
            writer.writerow([seq, c])

    print(f"Successfully generated {OUTPUT_CSV} with {len(records)} rows!")

if __name__ == "__main__":
    main()

"""Output parsing and label normalization for repeat benchmark."""

import re
import difflib
from typing import List, Optional, Tuple


def clean_raw_text(text: str) -> str:
    """Strip markup, thought tags, code fences, and punctuation."""
    if not text:
        return ""
    # Remove thinking tags if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove dna tags if echoed
    text = re.sub(r"</?dna>", "", text, flags=re.IGNORECASE)
    # Remove markdown bold/italics/backticks
    text = text.replace("*", "").replace("`", "").replace('"', "").replace("'", "")
    # Remove leading/trailing whitespace
    text = text.strip()
    # Take first line if multiple lines
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if lines:
        text = lines[0]
    # Remove common prefix phrases
    prefix_patterns = [
        r"^(the\s+)?repeat\s+(class|family)\s+is[:\s]*",
        r"^(the\s+)?answer\s+is[:\s]*",
        r"^class[:\s]*",
        r"^family[:\s]*",
        r"^prediction[:\s]*",
    ]
    for pattern in prefix_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
    # Strip terminal punctuation
    text = text.rstrip(".!,;:")
    return text.strip()


def normalize_label(
    raw_output: str,
    allowed_labels: List[str],
    fuzzy_cutoff: float = 0.85,
) -> Tuple[str, str]:
    """Normalize model output to a known label.

    Returns:
        (predicted_label, match_status) where match_status is one of:
        'exact', 'case_insensitive', 'substring', 'fuzzy', 'invalid'
    """
    cleaned = clean_raw_text(raw_output)
    if not cleaned:
        return ("INVALID", "empty")

    label_map = {lbl.lower(): lbl for lbl in allowed_labels}

    # 1. Exact match (case-sensitive)
    if cleaned in allowed_labels:
        return (cleaned, "exact")

    # 2. Case-insensitive match
    if cleaned.lower() in label_map:
        return (label_map[cleaned.lower()], "case_insensitive")

    # 3. Substring search in cleaned line
    # Look for exact word matches in the cleaned text
    words = re.findall(r"[\w-]+", cleaned)
    found_matches = []
    for word in words:
        if word.lower() in label_map:
            found_matches.append(label_map[word.lower()])
    if len(set(found_matches)) == 1:
        return (found_matches[0], "substring")

    # 4. Fuzzy match using SequenceMatcher
    close_matches = difflib.get_close_matches(
        cleaned.lower(), list(label_map.keys()), n=1, cutoff=fuzzy_cutoff
    )
    if close_matches:
        matched_key = close_matches[0]
        return (label_map[matched_key], "fuzzy")

    return ("INVALID", "invalid")

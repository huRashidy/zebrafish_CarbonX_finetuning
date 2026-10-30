"""Prompt templates and message formatting for Carbon-X fine-tuning and inference.

Rules:
1. DNA sequences are enclosed in <dna>...</dna> tags.
2. The sequence and task are placed in the user turn.
3. Chat template uses enable_thinking=False.
4. Assistant turn contains only the class name.
"""

from typing import List, Dict, Any, Optional


def get_class_prompt(
    sequence: str,
    allowed_classes: List[str],
    species: str = "Danio_rerio (zebrafish)",
) -> str:
    """Format prompt for repeat class prediction."""
    classes_str = ", ".join(sorted(allowed_classes))
    return (
        f"The following DNA sequence is a repetitive element from {species} genome, "
        f"annotated by RepeatMasker.\n\n"
        f"Sequence: <dna>{sequence.strip()}</dna>\n\n"
        f"Which repeat class does this sequence belong to?\n"
        f"Choose exactly one from the following allowed classes:\n"
        f"{classes_str}\n\n"
        f"Answer with only the class name from the list, nothing else."
    )


def build_chat_messages(
    user_prompt: str,
    assistant_response: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Create a conversational message list for training or inference."""
    messages = [{"role": "user", "content": user_prompt}]
    if assistant_response is not None:
        messages.append({"role": "assistant", "content": str(assistant_response).strip()})
    return messages


def format_chat_prompt(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = False,
) -> str:
    """Apply the model's chat template with thinking disabled."""
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )


def format_sft_dataset_row(
    row: Dict[str, Any],
    allowed_classes: List[str],
    species: str = "Danio_rerio (zebrafish)",
) -> Dict[str, Any]:
    """Format a dataframe row into a chat messages dictionary for SFTTrainer."""
    seq = str(row["Sequence"]).strip()
    label = str(row["repeat_class"]).strip()
    user_prompt = get_class_prompt(seq, allowed_classes, species=species)
    messages = build_chat_messages(user_prompt, assistant_response=label)
    return {"messages": messages}

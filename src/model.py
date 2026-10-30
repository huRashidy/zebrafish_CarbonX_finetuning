"""Model loading, 2-GPU sharding, and single-GPU LoRA adapter allocation for Carbon-X."""

import re
from typing import Dict, Any, Tuple, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)


def get_layer_device_mapping(model: Any) -> Dict[int, Any]:
    """Extract which device each transformer layer is mapped to from hf_device_map."""
    layer_map = {}
    if not hasattr(model, "hf_device_map"):
        return layer_map

    pattern = re.compile(r"layers\.(\d+)")
    for component, device in model.hf_device_map.items():
        match = pattern.search(component)
        if match:
            layer_idx = int(match.group(1))
            layer_map[layer_idx] = device

    return dict(sorted(layer_map.items()))


def load_base_model_and_tokenizer(
    model_cfg: Dict[str, Any],
) -> Tuple[Any, Any]:
    """Load Carbon-X base model in 4-bit NF4 sharded across 2 GPUs."""
    model_id = model_cfg.get("model_id", "HuggingFaceBio/Carbon-X")
    print(f"\n[1/3] Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    n_gpus = torch.cuda.device_count()
    print(f"[2/3] Detected {n_gpus} CUDA GPU(s):")
    for i in range(n_gpus):
        print(f"      GPU {i}: {torch.cuda.get_device_name(i)}")

    max_mem_cfg = model_cfg.get("max_memory", {})
    if n_gpus >= 2:
        max_memory = {
            0: max_mem_cfg.get(0, "9.5GiB"),
            1: max_mem_cfg.get(1, "10.5GiB"),
        }
    else:
        max_memory = {0: "10.5GiB"}
    max_memory["cpu"] = max_mem_cfg.get("cpu", "32GiB")

    quant = model_cfg.get("quantization", "4bit")
    if quant == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif quant == "8bit":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
    else:
        raise ValueError(f"Unsupported quantization: {quant}. Use '4bit' or '8bit'.")

    torch_dtype = getattr(torch, model_cfg.get("torch_dtype", "float16"))
    attn_impl = model_cfg.get("attn_implementation", "sdpa")

    print(f"[3/3] Loading {model_id} with {quant} quantization across GPUs...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=model_cfg.get("device_map", "auto"),
        max_memory=max_memory,
        torch_dtype=torch_dtype,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )

    if hasattr(model, "hf_device_map"):
        devices = set(model.hf_device_map.values())
        print(f"      Model sharded across devices: {devices}")
        layer_mapping = get_layer_device_mapping(model)
        gpu0_layers = [idx for idx, dev in layer_mapping.items() if dev == 0 or dev == "cuda:0"]
        gpu1_layers = [idx for idx, dev in layer_mapping.items() if dev == 1 or dev == "cuda:1"]
        print(f"      GPU 0 transformer layers: {len(gpu0_layers)} layers ({min(gpu0_layers, default=-1)} to {max(gpu0_layers, default=-1)})")
        print(f"      GPU 1 transformer layers: {len(gpu1_layers)} layers ({min(gpu1_layers, default=-1)} to {max(gpu1_layers, default=-1)})")

    return tokenizer, model


def setup_single_gpu_peft_model(
    model: Any,
    lora_cfg: Dict[str, Any],
) -> Any:
    """Attach LoRA adapters strictly to transformer layers residing on target_gpu."""
    print("\nPreparing base model for k-bit training with gradient checkpointing...")
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    layer_mapping = get_layer_device_mapping(model)
    target_gpu = lora_cfg.get("target_gpu", 1)

    target_layers = []
    for layer_idx, dev in layer_mapping.items():
        if dev == target_gpu or dev == f"cuda:{target_gpu}":
            target_layers.append(layer_idx)

    print(f"\nConfiguring LoRA targeted to GPU {target_gpu}:")
    print(f"  Target layers to transform ({len(target_layers)} layers): {target_layers}")

    target_modules = lora_cfg.get("target_modules", "all-linear")
    if target_modules == "all-linear":
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    peft_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        target_modules=target_modules,
        layers_to_transform=target_layers if target_layers else None,
        layers_pattern="layers",
        bias=lora_cfg.get("bias", "none"),
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(model, peft_config)

    # Verify device allocation of trainable parameters
    trainable_devices = set()
    trainable_param_count = 0
    total_param_count = 0
    for name, param in peft_model.named_parameters():
        total_param_count += param.numel()
        if param.requires_grad:
            trainable_param_count += param.numel()
            trainable_devices.add(str(param.device))

    pct_trainable = 100 * trainable_param_count / total_param_count if total_param_count > 0 else 0
    print(f"  Trainable parameters: {trainable_param_count:,} / {total_param_count:,} ({pct_trainable:.3f}%)")
    print(f"  LoRA adapter parameter device(s): {trainable_devices}")

    return peft_model


def load_model_for_evaluation(
    model_cfg: Dict[str, Any],
    adapter_path: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Load Carbon-X (and optionally merge/load PEFT adapter) for inference."""
    tokenizer, model = load_base_model_and_tokenizer(model_cfg)
    if adapter_path and adapter_path != "none":
        print(f"Loading LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        print("Adapter successfully loaded onto base model.")
    model.eval()
    return tokenizer, model

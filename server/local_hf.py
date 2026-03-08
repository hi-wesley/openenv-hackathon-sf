from __future__ import annotations

from functools import lru_cache
from typing import Tuple

from dfa_agent_env.config import EnvConfig, get_config


def _lazy_imports():
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return torch, snapshot_download, AutoModelForCausalLM, AutoTokenizer


def resolve_device(preference: str) -> str:
    torch, _, _, _ = _lazy_imports()
    normalized = (preference or "auto").strip().lower()
    if normalized != "auto":
        return normalized
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@lru_cache(maxsize=2)
def load_local_model(
    model_id: str,
    device_preference: str,
    local_files_only: bool,
) -> Tuple[object, object, str]:
    torch, snapshot_download, AutoModelForCausalLM, AutoTokenizer = _lazy_imports()
    device = resolve_device(device_preference)
    model_path = snapshot_download(
        repo_id=model_id,
        local_files_only=local_files_only,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        local_files_only=True,
    )
    if getattr(model, "generation_config", None) is not None and tokenizer.eos_token_id is not None:
        model.generation_config.pad_token_id = tokenizer.eos_token_id
    if device in {"cuda", "mps"}:
        model = model.to(device)
    model.eval()
    return tokenizer, model, device


def generate_chat_text(
    *,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_new_tokens: int | None = None,
    config: EnvConfig | None = None,
) -> str:
    cfg = config or get_config()
    torch, _, _, _ = _lazy_imports()
    tokenizer, model, device = load_local_model(
        cfg.local_model_id,
        cfg.local_model_device,
        cfg.local_model_local_files_only,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt_text = f"{system_prompt}\n\n{user_prompt}"
    inputs = tokenizer(prompt_text, return_tensors="pt")
    if device in {"cuda", "mps"}:
        inputs = {key: value.to(device) for key, value in inputs.items()}
    do_sample = temperature > 0.0
    generate_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens or cfg.local_model_max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
    with torch.no_grad():
        outputs = model.generate(**generate_kwargs)
    prompt_length = inputs["input_ids"].shape[1]
    return tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True).strip()

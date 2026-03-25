"""
Local fine-tuned SQL model — loads the LoRA checkpoint directly using HuggingFace.

This is a singleton that loads the model ONCE on first call and reuses it.
It replaces the OpenRouter API call for SQL generation, so that:
  - The fine-tuned Qwen 2.5-7B model handles NL → SQL (fast, local, free)
  - OpenRouter Qwen 72B still handles intent classification + answer synthesis

Checkpoint path is read from LOCAL_SQL_MODEL env var, or falls back to the
default: finetune/checkpoints/text_to_sql/checkpoint-100
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# ─── Configured paths ───────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent
_DEFAULT_CHECKPOINT = str(
    _REPO_ROOT / "finetune" / "checkpoints" / "text_to_sql" / "checkpoint-100"
)
CHECKPOINT_PATH = os.environ.get("LOCAL_SQL_MODEL", _DEFAULT_CHECKPOINT)
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# ─── Singleton state ─────────────────────────────────────────────────────────
_model = None
_tokenizer = None
_device = None
_available: bool | None = None   # None = not yet attempted, False = failed, True = ready


def _load_model() -> bool:
    """
    Attempt to load the fine-tuned model. Returns True on success.
    Errors are caught and logged so the caller can fall back gracefully.
    """
    global _model, _tokenizer, _device, _available

    if _available is not None:
        return _available

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        checkpoint = Path(CHECKPOINT_PATH)
        if not checkpoint.exists():
            logger.warning(
                "Local SQL model checkpoint not found at %s — falling back to API",
                CHECKPOINT_PATH,
            )
            _available = False
            return False

        logger.info("Loading local fine-tuned SQL model from %s ...", CHECKPOINT_PATH)

        # Detect device
        if torch.cuda.is_available():
            _device = "cuda"
        else:
            _device = "cpu"
            logger.warning(
                "CUDA not available — running local model on CPU (will be slow)"
            )

        bnb_config = None
        if _device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=bnb_config,
            device_map="auto" if _device == "cuda" else None,
            torch_dtype=torch.float32 if _device == "cpu" else None,
        )

        _model = PeftModel.from_pretrained(base, CHECKPOINT_PATH)
        _model.eval()

        _tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH)
        _tokenizer.pad_token = _tokenizer.eos_token

        _available = True
        logger.info("Local SQL model loaded successfully on %s", _device)
        return True

    except Exception as exc:
        logger.warning("Could not load local SQL model: %s — falling back to API", exc)
        _available = False
        return False


def is_available() -> bool:
    """Return True if the local model is (or can be) loaded."""
    return _load_model()


def generate_sql(prompt: str, max_new_tokens: int = 256) -> str:
    """
    Run the local fine-tuned model on a prompt and return the generated SQL string.

    Args:
        prompt: The full prompt string (system + user query already composed)
        max_new_tokens: Max tokens to generate

    Returns:
        Raw model output string (caller should clean/validate)

    Raises:
        RuntimeError: If model is unavailable (caller should fall back to API)
    """
    import torch

    if not _load_model():
        raise RuntimeError("Local SQL model unavailable")

    inputs = _tokenizer(prompt, return_tensors="pt")
    if _device == "cuda":
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,           # greedy — SQL generation should be deterministic
            temperature=1.0,           # ignored when do_sample=False
            pad_token_id=_tokenizer.eos_token_id,
        )

    # Decode only newly generated tokens (skip the prompt)
    new_ids = output_ids[0][inputs["input_ids"].shape[-1]:]
    return _tokenizer.decode(new_ids, skip_special_tokens=True).strip()

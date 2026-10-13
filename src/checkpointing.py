"""Checkpoint loading and saving for the two DynaDEx training phases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from .modeling_dynadex import DynaDExModel


MAPPER_FILENAME = "schema_mapper.pt"
CONFIG_FILENAME = "dynadex_config.json"


def load_mapper(model: DynaDExModel, checkpoint: Path) -> None:
    """Load the schema mapper associated with an adapter checkpoint."""
    if model.schema_mapper is None:
        return
    mapper_path = checkpoint / MAPPER_FILENAME
    if not mapper_path.is_file():
        raise FileNotFoundError(f"Missing schema mapper checkpoint: {mapper_path}")
    state = torch.load(mapper_path, map_location="cpu", weights_only=True)
    model.schema_mapper.load_state_dict(state, strict=True)


def save_bundle(
    model: DynaDExModel,
    output_dir: Path,
    metadata: dict[str, Any],
) -> None:
    """Save the active LoRA adapter, schema mapper, and non-sensitive metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.language_model.save_pretrained(output_dir)
    if model.schema_mapper is not None:
        torch.save(model.schema_mapper.state_dict(), output_dir / MAPPER_FILENAME)
    (output_dir / CONFIG_FILENAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_phase1_model(
    base_model: str,
    checkpoint: Path,
    latent_length: int,
    use_schema_mapper: bool,
    torch_dtype: torch.dtype,
    trainable_adapter: bool = False,
) -> DynaDExModel:
    """Load the Phase 1 adapter without changing its source-domain policy."""
    backbone = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    language_model = PeftModel.from_pretrained(
        backbone,
        checkpoint,
        is_trainable=trainable_adapter,
    )
    model = DynaDExModel(language_model, latent_length, use_schema_mapper)
    load_mapper(model, checkpoint)
    return model


def build_phase2_model(
    base_model: str,
    phase1_checkpoint: Path,
    latent_length: int,
    use_schema_mapper: bool,
    torch_dtype: torch.dtype,
) -> DynaDExModel:
    """Freeze the merged source model and attach a new target-domain LoRA adapter."""
    backbone = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    source_adapter = PeftModel.from_pretrained(backbone, phase1_checkpoint)
    source_model = source_adapter.merge_and_unload()
    source_model.requires_grad_(False)
    model = DynaDExModel(source_model, latent_length, use_schema_mapper)
    load_mapper(model, phase1_checkpoint)
    model.attach_lora()
    if model.schema_mapper is not None:
        model.schema_mapper.requires_grad_(True)
    return model


def load_phase2_model(
    base_model: str,
    phase1_checkpoint: Path,
    phase2_checkpoint: Path,
    latent_length: int,
    use_schema_mapper: bool,
    torch_dtype: torch.dtype,
) -> DynaDExModel:
    """Load a frozen source model followed by its target-domain adapter."""
    backbone = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    source_adapter = PeftModel.from_pretrained(backbone, phase1_checkpoint)
    source_model = source_adapter.merge_and_unload()
    language_model = PeftModel.from_pretrained(source_model, phase2_checkpoint)
    model = DynaDExModel(language_model, latent_length, use_schema_mapper)
    load_mapper(model, phase2_checkpoint)
    return model

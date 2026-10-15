"""Checkpoint loading and saving for the two DynaDEx training phases."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from pathlib import Path, PureWindowsPath
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from .data import SCHEMA_ENCODING
from .modeling_dynadex import DynaDExModel

MAPPER_FILENAME = "schema_mapper.pt"
CONFIG_FILENAME = "dynadex_config.json"


def public_model_id(value: str) -> str:
    """Keep public model identifiers while omitting local filesystem paths."""
    return (
        "<local-model>"
        if Path(value).is_absolute() or PureWindowsPath(value).is_absolute()
        else value
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_bundle(
    checkpoint: Path, latent_length: int, use_schema_mapper: bool, phase: int
) -> dict[str, Any]:
    config_path = checkpoint / CONFIG_FILENAME
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("phase") != phase:
        raise ValueError(f"Expected a Phase {phase} checkpoint")
    if (
        config.get("latent_length") != latent_length
        or config.get("schema_mapper") != use_schema_mapper
    ):
        raise ValueError("Checkpoint and requested schema mapper configuration disagree")
    if use_schema_mapper and config.get("schema_encoding") != SCHEMA_ENCODING:
        raise ValueError(
            "Checkpoint schema encoding is incompatible with label/definition tokenization"
        )
    return config


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
    adapter_path = output_dir / "adapter_config.json"
    if adapter_path.is_file():
        adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
        if isinstance(adapter.get("base_model_name_or_path"), str):
            adapter["base_model_name_or_path"] = public_model_id(adapter["base_model_name_or_path"])
        adapter_path.write_text(json.dumps(adapter, indent=2) + "\n", encoding="utf-8")
    metadata = {
        **metadata,
        "schema_encoding": SCHEMA_ENCODING,
        "versions": {
            name: version(name) for name in ("torch", "transformers", "peft", "accelerate")
        },
    }
    metadata["base_model"] = public_model_id(str(metadata["base_model"]))
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
    validate_bundle(checkpoint, latent_length, use_schema_mapper, 1)
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
    validate_bundle(phase1_checkpoint, latent_length, use_schema_mapper, 1)
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
    validate_bundle(phase1_checkpoint, latent_length, use_schema_mapper, 1)
    validate_bundle(phase2_checkpoint, latent_length, use_schema_mapper, 2)
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

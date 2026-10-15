"""Schema-conditioned causal language model used by DynaDEx."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM

LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def lora_config() -> LoraConfig:
    """Return the LoRA configuration shared by both training phases."""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=list(LORA_TARGET_MODULES),
    )


class SchemaMapper(nn.Module):
    """Map a pooled schema representation to a latent prefix sequence."""

    def __init__(self, hidden_size: int, latent_length: int = 10) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_length = latent_length
        self.network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size * latent_length),
        )

    def forward(self, pooled_schema: torch.Tensor) -> torch.Tensor:
        batch_size = pooled_schema.shape[0]
        mapped = self.network(pooled_schema)
        return mapped.view(batch_size, self.latent_length, self.hidden_size)


class DynaDExModel(nn.Module):
    """Combine a causal LLM with continuous schema-conditioned context."""

    accepts_loss_kwargs = False

    def __init__(
        self,
        language_model: nn.Module,
        latent_length: int = 10,
        use_schema_mapper: bool = True,
    ) -> None:
        super().__init__()
        self.language_model = language_model
        self.hidden_size = int(language_model.config.hidden_size)
        if latent_length <= 0:
            raise ValueError("Latent length must be positive")
        self.schema_mapper = (
            SchemaMapper(self.hidden_size, latent_length) if use_schema_mapper else None
        )
        self.language_model.config.use_cache = False
        if self.schema_mapper is not None:
            embedding = language_model.get_input_embeddings().weight
            self.schema_mapper.to(device=embedding.device, dtype=embedding.dtype)

    @classmethod
    def from_pretrained(
        cls,
        base_model: str,
        latent_length: int = 10,
        use_schema_mapper: bool = True,
        add_lora: bool = False,
        torch_dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True,
    ) -> DynaDExModel:
        """Load a backbone and optionally attach a trainable LoRA adapter."""
        language_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        if add_lora:
            language_model = get_peft_model(language_model, lora_config())
        model = cls(language_model, latent_length, use_schema_mapper)
        if model.schema_mapper is not None:
            embedding = model.language_model.get_input_embeddings().weight
            model.schema_mapper.to(device=embedding.device, dtype=embedding.dtype)
        return model

    def attach_lora(self) -> None:
        """Attach the target-domain LoRA adapter to a frozen source model."""
        self.language_model = get_peft_model(self.language_model, lora_config())

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        self.language_model.gradient_checkpointing_enable(**kwargs)
        self.language_model.enable_input_require_grads()

    def _latent_prefix(
        self,
        schema_input_ids: torch.Tensor,
        schema_attention_mask: torch.Tensor,
        schema_role_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.schema_mapper is None:
            raise RuntimeError("The schema mapper is disabled")
        embedding_layer = self.language_model.get_input_embeddings()
        schema_input_ids = schema_input_ids.to(embedding_layer.weight.device)
        schema_attention_mask = schema_attention_mask.to(embedding_layer.weight.device)
        schema_role_mask = schema_role_mask.to(embedding_layer.weight.device)

        with torch.no_grad():
            token_embeddings = embedding_layer(schema_input_ids)
            token_embeddings = token_embeddings.float()
            token_mask = schema_attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
            token_count = token_mask.sum(dim=2).clamp_min(1.0)
            role_embeddings = (token_embeddings * token_mask).sum(dim=2) / token_count
            role_mask = schema_role_mask.unsqueeze(-1).to(role_embeddings.dtype)
            role_count = role_mask.sum(dim=1).clamp_min(1.0)
            pooled_schema = (role_embeddings * role_mask).sum(dim=1) / role_count

        mapper_parameter = next(self.schema_mapper.parameters())
        pooled_schema = pooled_schema.to(
            device=mapper_parameter.device,
            dtype=mapper_parameter.dtype,
        )
        return self.schema_mapper(pooled_schema)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        schema_input_ids: torch.Tensor | None = None,
        schema_attention_mask: torch.Tensor | None = None,
        schema_role_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run teacher-forced training with an optional latent schema prefix."""
        if self.schema_mapper is None:
            return self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
                return_dict=True,
                **kwargs,
            )
        if schema_input_ids is None or schema_attention_mask is None or schema_role_mask is None:
            raise ValueError("Schema tensors are required when the schema mapper is enabled")

        embedding_layer = self.language_model.get_input_embeddings()
        input_ids = input_ids.to(embedding_layer.weight.device)
        attention_mask = attention_mask.to(embedding_layer.weight.device)
        if labels is not None:
            labels = labels.to(embedding_layer.weight.device)

        latent_prefix = self._latent_prefix(
            schema_input_ids, schema_attention_mask, schema_role_mask
        ).to(embedding_layer.weight.dtype)
        text_embeddings = embedding_layer(input_ids)
        combined_embeddings = torch.cat((latent_prefix, text_embeddings), dim=1)

        batch_size, prefix_length = latent_prefix.shape[:2]
        prefix_mask = torch.ones(
            (batch_size, prefix_length),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        combined_mask = torch.cat((prefix_mask, attention_mask), dim=1)
        combined_labels = None
        if labels is not None:
            prefix_labels = torch.full(
                (batch_size, prefix_length),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            combined_labels = torch.cat((prefix_labels, labels), dim=1)

        return self.language_model(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_mask,
            labels=combined_labels,
            use_cache=False,
            return_dict=True,
            **kwargs,
        )

    def _generation_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        schema_input_ids: torch.Tensor,
        schema_attention_mask: torch.Tensor,
        schema_role_mask: torch.Tensor,
        pad_token_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Insert latent vectors immediately before each unpadded prompt."""
        embedding_layer = self.language_model.get_input_embeddings()
        device = embedding_layer.weight.device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        text_embeddings = embedding_layer(input_ids)
        latent_prefix = self._latent_prefix(
            schema_input_ids, schema_attention_mask, schema_role_mask
        ).to(device=device, dtype=text_embeddings.dtype)

        rows: list[torch.Tensor] = []
        id_rows: list[torch.Tensor] = []
        placeholder_id = int(getattr(self.language_model.config, "bos_token_id", None) or 0)
        for row_index in range(input_ids.shape[0]):
            valid = attention_mask[row_index].bool()
            prompt_embeddings = text_embeddings[row_index][valid]
            prompt_ids = input_ids[row_index][valid]
            rows.append(torch.cat((latent_prefix[row_index], prompt_embeddings), dim=0))
            latent_ids = torch.full(
                (latent_prefix.shape[1],),
                placeholder_id,
                dtype=input_ids.dtype,
                device=device,
            )
            id_rows.append(torch.cat((latent_ids, prompt_ids), dim=0))

        maximum_length = max(row.shape[0] for row in rows)
        pad_ids = torch.full(
            (input_ids.shape[0], maximum_length),
            pad_token_id,
            dtype=input_ids.dtype,
            device=device,
        )
        combined_mask = torch.zeros(
            (input_ids.shape[0], maximum_length),
            dtype=attention_mask.dtype,
            device=device,
        )
        pad_embedding = embedding_layer(
            torch.tensor([pad_token_id], dtype=input_ids.dtype, device=device)
        )[0]
        combined_embeddings = (
            pad_embedding.view(1, 1, -1).expand(input_ids.shape[0], maximum_length, -1).clone()
        )
        for row_index, (row, row_ids) in enumerate(zip(rows, id_rows, strict=False)):
            start = maximum_length - row.shape[0]
            combined_embeddings[row_index, start:] = row
            pad_ids[row_index, start:] = row_ids
            combined_mask[row_index, start:] = 1
        return pad_ids, combined_embeddings, combined_mask

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        schema_input_ids: torch.Tensor | None,
        schema_attention_mask: torch.Tensor | None,
        schema_role_mask: torch.Tensor | None,
        pad_token_id: int,
        **generation_kwargs: Any,
    ) -> torch.Tensor:
        """Generate only continuation tokens for a batch of prompts."""
        if self.schema_mapper is None:
            device = self.language_model.get_input_embeddings().weight.device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            output_ids = self.language_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=pad_token_id,
                **generation_kwargs,
            )
            return output_ids[:, input_ids.shape[1] :]
        if schema_input_ids is None or schema_attention_mask is None or schema_role_mask is None:
            raise ValueError("Schema tensors are required when the schema mapper is enabled")
        combined_ids, combined_embeddings, combined_mask = self._generation_inputs(
            input_ids,
            attention_mask,
            schema_input_ids,
            schema_attention_mask,
            schema_role_mask,
            pad_token_id,
        )
        output_ids = self.language_model.generate(
            input_ids=combined_ids,
            inputs_embeds=combined_embeddings,
            attention_mask=combined_mask,
            pad_token_id=pad_token_id,
            **generation_kwargs,
        )
        return output_ids[:, combined_ids.shape[1] :]

"""Shared training helpers, including Gradient-Regularized Optimization."""

from __future__ import annotations

from typing import Any

import torch
from transformers import Trainer, set_seed


def configure_reproducibility(seed: int) -> None:
    """Set framework seeds and enable TF32 on supported CUDA devices."""
    set_seed(seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def parse_dtype(name: str) -> torch.dtype:
    """Map a command-line precision name to a PyTorch dtype."""
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[name]


class GROTrainer(Trainer):
    """Trainer implementing the two-gradient GRO update used in Phase 2."""

    def __init__(
        self,
        *args: Any,
        gro_rho: float = 0.05,
        gro_alpha: float = 0.20,
        gro_gamma: float = 1e-12,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if gro_rho < 0:
            raise ValueError("gro_rho must be non-negative")
        if not 0 <= gro_alpha <= 1:
            raise ValueError("gro_alpha must be in [0, 1]")
        if gro_gamma <= 0:
            raise ValueError("gro_gamma must be positive")
        self.gro_rho = gro_rho
        self.gro_alpha = gro_alpha
        self.gro_gamma = gro_gamma
        self.model_accepts_loss_kwargs = False
        if self.args.world_size != 1 or self.args.n_gpu > 1:
            raise ValueError(
                "GRO uses the single-GPU training configuration specified in the paper"
            )

    def _batch_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: int | None,
    ) -> torch.Tensor:
        return self.compute_loss(model, inputs)

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        accumulation_steps = max(
            1,
            int(
                getattr(
                    self,
                    "current_gradient_accumulation_steps",
                    self.args.gradient_accumulation_steps,
                )
            ),
        )
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]

        accumulated = [
            parameter.grad.detach().clone() if parameter.grad is not None else None
            for parameter in trainable
        ]
        for parameter in trainable:
            parameter.grad = None

        rng_before = torch.random.get_rng_state()
        cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        with self.compute_loss_context_manager():
            loss = self._batch_loss(model, inputs, num_items_in_batch)
        if self.args.n_gpu > 1:
            loss = loss.mean()
        # Obtain the unnormalized batch gradient; normalize once after mixing.
        backward_scale = self.accelerator.gradient_accumulation_steps
        self.accelerator.backward(loss * backward_scale)
        rng_after = torch.random.get_rng_state()
        cuda_after = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        active = [parameter for parameter in trainable if parameter.grad is not None]
        original_gradients = [parameter.grad.detach().clone() for parameter in active]
        if not active:
            for parameter, previous in zip(trainable, accumulated, strict=False):
                parameter.grad = previous
            return loss.detach() / accumulation_steps

        scaler = self.accelerator.scaler
        loss_scale = scaler.get_scale() if scaler is not None else 1.0
        gradient_norm = torch.linalg.vector_norm(
            torch.stack(
                [(gradient.float() / loss_scale).norm(2) for gradient in original_gradients]
            ),
            ord=2,
        )
        scale = self.gro_rho / (gradient_norm + self.gro_gamma)
        original_parameters = [parameter.detach().clone() for parameter in active]
        try:
            with torch.no_grad():
                for parameter, gradient in zip(active, original_gradients, strict=False):
                    parameter.add_(
                        (gradient.float() / loss_scale).to(parameter.dtype), alpha=float(scale)
                    )
                    parameter.grad = None
            # Reuse the same dropout realization for both evaluations of the batch.
            torch.random.set_rng_state(rng_before)
            if cuda_before is not None:
                torch.cuda.set_rng_state_all(cuda_before)
            with self.compute_loss_context_manager():
                perturbed_loss = self._batch_loss(model, inputs, num_items_in_batch)
            if self.args.n_gpu > 1:
                perturbed_loss = perturbed_loss.mean()
            self.accelerator.backward(perturbed_loss * backward_scale)
            lookahead_gradients = [
                parameter.grad.detach().clone()
                if parameter.grad is not None
                else torch.zeros_like(original)
                for parameter, original in zip(active, original_gradients, strict=False)
            ]
        finally:
            with torch.no_grad():
                for parameter, original_parameter in zip(active, original_parameters, strict=False):
                    parameter.copy_(original_parameter)
            torch.random.set_rng_state(rng_after)
            if cuda_after is not None:
                torch.cuda.set_rng_state_all(cuda_after)

        for parameter, original, lookahead in zip(
            active, original_gradients, lookahead_gradients, strict=False
        ):
            parameter.grad = (
                (1.0 - self.gro_alpha) * original + self.gro_alpha * lookahead
            ) / accumulation_steps

        for parameter, previous in zip(trainable, accumulated, strict=False):
            if previous is None:
                continue
            if parameter.grad is None:
                parameter.grad = previous
            else:
                parameter.grad.add_(previous)
        return loss.detach() / accumulation_steps

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

    def _batch_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: int | None,
    ) -> torch.Tensor:
        if num_items_in_batch is None:
            return self.compute_loss(model, inputs)
        return self.compute_loss(
            model,
            inputs,
            num_items_in_batch=num_items_in_batch,
        )

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        accumulation_steps = max(1, int(self.args.gradient_accumulation_steps))
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]

        accumulated = [
            parameter.grad.detach().clone() if parameter.grad is not None else None
            for parameter in trainable
        ]
        for parameter in trainable:
            parameter.grad = None

        with self.compute_loss_context_manager():
            loss = self._batch_loss(model, inputs, num_items_in_batch)
        if self.args.n_gpu > 1:
            loss = loss.mean()
        self.accelerator.backward(loss / accumulation_steps)

        active = [parameter for parameter in trainable if parameter.grad is not None]
        original_gradients = [parameter.grad.detach().clone() for parameter in active]
        if not active:
            return loss.detach()

        gradient_norm = torch.linalg.vector_norm(
            torch.stack([gradient.norm(2) for gradient in original_gradients]),
            ord=2,
        )
        scale = self.gro_rho / (gradient_norm + self.gro_gamma)
        with torch.no_grad():
            for parameter, gradient in zip(active, original_gradients):
                parameter.add_(gradient, alpha=float(scale))
                parameter.grad = None

        try:
            with self.compute_loss_context_manager():
                perturbed_loss = self._batch_loss(model, inputs, num_items_in_batch)
            if self.args.n_gpu > 1:
                perturbed_loss = perturbed_loss.mean()
            self.accelerator.backward(perturbed_loss / accumulation_steps)
            lookahead_gradients = [
                parameter.grad.detach().clone()
                if parameter.grad is not None
                else torch.zeros_like(original)
                for parameter, original in zip(active, original_gradients)
            ]
        finally:
            with torch.no_grad():
                for parameter, gradient in zip(active, original_gradients):
                    parameter.sub_(gradient, alpha=float(scale))

        dot_product = sum(
            (original * lookahead).sum()
            for original, lookahead in zip(original_gradients, lookahead_gradients)
        )
        if float(dot_product) < 0:
            original_norm_sq = sum((gradient * gradient).sum() for gradient in original_gradients)
            projection = dot_product / (original_norm_sq + self.gro_gamma)
            lookahead_gradients = [
                lookahead - projection * original
                for original, lookahead in zip(original_gradients, lookahead_gradients)
            ]

        for parameter, original, lookahead in zip(
            active, original_gradients, lookahead_gradients
        ):
            parameter.grad = (
                (1.0 - self.gro_alpha) * original + self.gro_alpha * lookahead
            )

        for parameter, previous in zip(trainable, accumulated):
            if previous is None:
                continue
            if parameter.grad is None:
                parameter.grad = previous
            else:
                parameter.grad.add_(previous)
        return loss.detach()

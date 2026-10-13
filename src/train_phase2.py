"""Run budget-constrained target adaptation with RDA and GRO."""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoTokenizer, Trainer, TrainingArguments

from .checkpointing import build_phase2_model, save_bundle
from .data import DynaDExCollator, DynaDExDataset
from .training_common import GROTrainer, configure_reproducibility, parse_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--phase1-checkpoint", type=Path, default=Path("outputs/phase1"))
    parser.add_argument("--train-file", type=Path, default=Path("data/processed/target/train_ins.jsonl"))
    parser.add_argument("--dev-file", type=Path, default=Path("data/processed/target/dev_ins.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase2"))
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--per-device-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--maximum-length", type=int, default=2048)
    parser.add_argument("--maximum-train-samples", type=int, default=2000)
    parser.add_argument("--latent-length", type=int, default=10)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.20)
    parser.add_argument("--gamma", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--without-gro", action="store_true")
    parser.add_argument("--without-schema-mapper", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--dataloader-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_reproducibility(args.seed)
    dtype = parse_dtype(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    use_schema_mapper = not args.without_schema_mapper
    model = build_phase2_model(
        args.base_model,
        args.phase1_checkpoint,
        args.latent_length,
        use_schema_mapper,
        dtype,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    maximum_samples = args.maximum_train_samples if args.maximum_train_samples > 0 else None
    train_dataset = DynaDExDataset(
        args.train_file,
        tokenizer,
        maximum_length=args.maximum_length,
        maximum_samples=maximum_samples,
        seed=args.seed,
        use_schema_mapper=use_schema_mapper,
    )
    dev_dataset = DynaDExDataset(
        args.dev_file,
        tokenizer,
        maximum_length=args.maximum_length,
        seed=args.seed,
        use_schema_mapper=use_schema_mapper,
    )
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=args.dtype == "bfloat16",
        fp16=args.dtype == "float16",
        tf32=True,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_workers,
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer_class = Trainer if args.without_gro else GROTrainer
    trainer_kwargs = {}
    if not args.without_gro:
        trainer_kwargs.update(
            gro_rho=args.rho,
            gro_alpha=args.alpha,
            gro_gamma=args.gamma,
        )
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=DynaDExCollator(tokenizer, use_schema_mapper),
        **trainer_kwargs,
    )
    trainer.train()
    save_bundle(
        model,
        args.output_dir,
        {
            "phase": 2,
            "base_model": args.base_model,
            "latent_length": args.latent_length,
            "schema_mapper": use_schema_mapper,
            "gro": not args.without_gro,
            "rho": None if args.without_gro else args.rho,
            "alpha": None if args.without_gro else args.alpha,
            "gamma": None if args.without_gro else args.gamma,
            "maximum_train_samples": maximum_samples,
            "seed": args.seed,
        },
    )


if __name__ == "__main__":
    main()

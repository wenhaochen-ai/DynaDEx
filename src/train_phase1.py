"""Train the source-domain DynaDEx policy on CPB1.0."""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import AutoTokenizer, Trainer, TrainingArguments

from .checkpointing import file_sha256, save_bundle
from .data import DynaDExCollator, DynaDExDataset
from .modeling_dynadex import DynaDExModel
from .training_common import configure_reproducibility, parse_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen3-8B")
    parser.add_argument(
        "--train-file", type=Path, default=Path("data/processed/cpb1/cpb1_train_ins.jsonl")
    )
    parser.add_argument(
        "--dev-file", type=Path, default=Path("data/processed/cpb1/cpb1_dev_ins.jsonl")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase1"))
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--per-device-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--maximum-length", type=int, default=2048)
    parser.add_argument("--latent-length", type=int, default=10)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
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
    model = DynaDExModel.from_pretrained(
        args.base_model,
        latent_length=args.latent_length,
        use_schema_mapper=use_schema_mapper,
        add_lora=True,
        torch_dtype=dtype,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    train_dataset = DynaDExDataset(
        args.train_file,
        tokenizer,
        maximum_length=args.maximum_length,
        seed=args.seed,
        use_schema_mapper=use_schema_mapper,
        latent_length=args.latent_length,
    )
    dev_dataset = DynaDExDataset(
        args.dev_file,
        tokenizer,
        maximum_length=args.maximum_length,
        seed=args.seed,
        use_schema_mapper=use_schema_mapper,
        latent_length=args.latent_length,
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
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=DynaDExCollator(tokenizer, use_schema_mapper),
    )
    trainer.train()
    save_bundle(
        model,
        args.output_dir,
        {
            "phase": 1,
            "base_model": args.base_model,
            "latent_length": args.latent_length,
            "schema_mapper": use_schema_mapper,
            "seed": args.seed,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "maximum_length": args.maximum_length,
            "train_file_sha256": file_sha256(args.train_file),
            "dev_file_sha256": file_sha256(args.dev_file),
        },
    )


if __name__ == "__main__":
    main()

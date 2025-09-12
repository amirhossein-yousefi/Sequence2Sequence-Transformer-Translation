from __future__ import annotations

import argparse
import os
from typing import Optional
import numpy as np
import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from src.config import TrainConfig
from src.data import load_and_prepare_dataset, tokenize_dataset
from src.model import load_model_and_tokenizer, get_data_collator
from src.metrics import build_compute_metrics
from src.predict import sample_predictions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MarianMT on translation datasets.")

    # Languages & dataset
    p.add_argument("--src_lang", type=str, default="en")
    p.add_argument("--tgt_lang", type=str, default="es")
    p.add_argument("--dataset_name", type=str, default="opus_books")
    p.add_argument("--dataset_config", type=str, default=None)

    # Model
    p.add_argument("--model_name", type=str, default=None)

    # Tokenization
    p.add_argument("--max_source_len", type=int, default=128)
    p.add_argument("--max_target_len", type=int, default=128)

    # Optimization
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)

    # Trainer / generation
    p.add_argument("--generation_max_length", type=int, default=128)
    p.add_argument("--evaluation_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    p.add_argument("--save_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--metric_for_best_model", type=str, default="bleu")

    # Mixed precision
    p.add_argument("--fp16", action="store_true", default=torch.cuda.is_available())
    p.add_argument("--bf16", action="store_true", default=False)

    # IO
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--logging_dir", type=str, default=None)

    # Workflow toggles
    p.add_argument("--do_train", action="store_true", default=True)
    p.add_argument("--do_eval", action="store_true", default=True)
    p.add_argument("--do_predict", action="store_true", default=True)

    return p.parse_args()


def main(ns: Optional[argparse.Namespace] = None):
    args = ns or parse_args()

    cfg = TrainConfig(
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        max_source_len=args.max_source_len,
        max_target_len=args.max_target_len,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        generation_max_length=args.generation_max_length,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        metric_for_best_model=args.metric_for_best_model,
        fp16=args.fp16,
        bf16=args.bf16,
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        do_train=args.do_train,
        do_eval=args.do_eval,
        do_predict=args.do_predict,
    ).resolve()

    set_seed(cfg.seed)

    # 1) Dataset
    dataset = load_and_prepare_dataset(cfg)
    print(dataset)

    # 2) Tokenizer & Model
    model, tokenizer = load_model_and_tokenizer(cfg)

    # 3) Tokenization
    tokenized = tokenize_dataset(dataset, tokenizer, cfg)

    # 4) Data collator
    data_collator = get_data_collator(tokenizer, model)

    # 5) Metrics
    compute_metrics = build_compute_metrics(tokenizer)

    # 6) Training args & Trainer
    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir,
        eval_strategy=cfg.evaluation_strategy,
        save_strategy=cfg.save_strategy,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        weight_decay=cfg.weight_decay,
        num_train_epochs=cfg.num_epochs,
        predict_with_generate=cfg.predict_with_generate,
        generation_max_length=cfg.generation_max_length,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        fp16=cfg.fp16 and not cfg.bf16,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model=cfg.metric_for_best_model,
        greater_is_better=True,
        report_to=cfg.report_to,
        logging_dir=cfg.logging_dir,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"] if cfg.do_train else None,
        eval_dataset=tokenized["validation"] if cfg.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 7) Train
    if cfg.do_train:
        train_result = trainer.train()
        print(train_result)
        trainer.save_model(cfg.output_dir)  # saves tokenizer too

    # 8) Validation metrics (best checkpoint)
    if cfg.do_eval:
        val_metrics = trainer.evaluate()
        print("Validation:", val_metrics)

    # 9) Test set evaluation
    if cfg.do_eval and "test" in tokenized:
        test_metrics = trainer.evaluate(eval_dataset=tokenized["test"], metric_key_prefix="test")
        print("Test:", test_metrics)

    # 10) Qualitative check
    if cfg.do_predict and "test" in tokenized:
        lines = sample_predictions(
            trainer=trainer,
            tokenizer=tokenizer,
            raw_test=dataset["test"],
            tokenized_test=tokenized["test"],
            cfg=cfg,
            num_samples=5,
        )
        print("\n".join("\n" + ln for ln in lines))


if __name__ == "__main__":
    main()

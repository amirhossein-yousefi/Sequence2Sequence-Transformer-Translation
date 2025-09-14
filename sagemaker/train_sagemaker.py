import argparse
import json
import os
from dataclasses import asdict

from transformers import TrainingArguments, Trainer
import torch

# Import the repo's modules
from src.config import TrainConfig
from src.model import load_model_and_tokenizer, get_data_collator
from src.data import load_and_prepare_dataset, tokenize_dataset
from src.metrics import build_compute_metrics
from src.predict import sample_predictions


def parse_args():
    p = argparse.ArgumentParser(description="SageMaker training for Seq2Seq translation")
    # High-level task knobs
    p.add_argument("--src_lang", type=str, default="en")
    p.add_argument("--tgt_lang", type=str, default="es")
    p.add_argument("--model_name", type=str, default=None, help="HF model id; defaults to MarianMT derived from src/tgt")
    p.add_argument("--dataset_name", type=str, default="Helsinki-NLP/opus_books")
    p.add_argument("--dataset_config", type=str, default=None, help="Dataset config name if applicable (e.g., 'de-en')")

    # Trainer hyperparameters
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--evaluation_strategy", type=str, default="epoch")
    p.add_argument("--save_strategy", type=str, default="epoch")
    p.add_argument("--save_total_limit", type=int, default=1)
    p.add_argument("--predict_with_generate", type=bool, default=True)
    p.add_argument("--generation_max_length", type=int, default=128)
    p.add_argument("--report_to", type=str, default="tensorboard")
    p.add_argument("--max_source_len", type=int, default=192)
    p.add_argument("--max_target_len", type=int, default=192)

    # SageMaker/paths
    p.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    p.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    return p.parse_args()


def main():
    args = parse_args()

    # Build repo TrainConfig, resolving defaults
    cfg = TrainConfig(
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        batch_size=args.per_device_train_batch_size,
        num_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        predict_with_generate=args.predict_with_generate,
        generation_max_length=args.generation_max_length,
        report_to=args.report_to,
        max_source_len=args.max_source_len,
        max_target_len=args.max_target_len,
        output_dir=args.model_dir,  # ensure it's /opt/ml/model
    ).resolve()

    # Ensure output dirs exist
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_data_dir, exist_ok=True)

    # 1) Model & tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg)

    # 2) Data: load and tokenize
    raw_ds = load_and_prepare_dataset(cfg)                       # ensures train/validation/test
    tok_ds = tokenize_dataset(raw_ds, tokenizer, cfg)            # tokenized datasets

    # 3) Trainer Arguments
    training_args = TrainingArguments(
        output_dir=args.model_dir,  # very important: save to /opt/ml/model for SageMaker
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        predict_with_generate=args.predict_with_generate,
        generation_max_length=args.generation_max_length,
        report_to=args.report_to,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok_ds["train"],
        eval_dataset=tok_ds["validation"],
        tokenizer=tokenizer,
        data_collator=get_data_collator(tokenizer, model),
        compute_metrics=build_compute_metrics(tokenizer),
    )

    # 4) Train / evaluate
    train_out = trainer.train()
    eval_metrics = trainer.evaluate()

    # Save artifacts explicitly (Trainer also saves to output_dir)
    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)

    # Write metrics to /opt/ml/output/data/metrics.json for convenience
    metrics_path = os.path.join(args.output_data_dir, "metrics.json")
    payload = {
        "train": train_out.metrics if hasattr(train_out, "metrics") else {},
        "eval": eval_metrics,
        "config": asdict(cfg),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # (Optional) qualitative samples to logs
    try:
        for line in sample_predictions(trainer, tokenizer, raw_ds["test"], tok_ds["test"], cfg, num_samples=3):
            print(line)
    except Exception as e:
        print(f"[WARN] Unable to print sample predictions: {e}")

    print("[OK] Training complete. Artifacts saved in:", args.model_dir)


if __name__ == "__main__":
    main()
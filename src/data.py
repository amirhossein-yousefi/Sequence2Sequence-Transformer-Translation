from __future__ import annotations

from typing import Callable, Dict
from datasets import load_dataset, DatasetDict
from transformers import PreTrainedTokenizerBase
from src.config import TrainConfig


def load_and_prepare_dataset(cfg: TrainConfig) -> DatasetDict:
    """
    Loads dataset and ensures we have train/validation/test splits.
    Defaults to opus_books which has a 'translation' field with src/tgt keys.
    """
    ds = load_dataset(cfg.dataset_name, cfg.dataset_config)

    # If only 'train' exists, create validation and test from it (90/5/5).
    if "train" in ds and set(ds.keys()) == {"train"}:
        tmp = ds["train"].train_test_split(test_size=0.10, seed=cfg.seed)
        valid_test = tmp["test"].train_test_split(test_size=0.50, seed=cfg.seed)
        ds = DatasetDict(
            train=tmp["train"],
            validation=valid_test["train"],
            test=valid_test["test"],
        )
    else:
        # Ensure 'validation' exists; if not, carve from train.
        if "validation" not in ds:
            tmp = ds["train"].train_test_split(test_size=0.10, seed=cfg.seed)
            ds = DatasetDict(
                train=tmp["train"],
                validation=tmp["test"],
                test=ds.get("test", tmp["test"]),
            )

    return ds


def build_preprocess_fn(tokenizer: PreTrainedTokenizerBase, cfg: TrainConfig) -> Callable[[Dict], Dict]:
    """
    Returns a batched mapping function that tokenizes src/tgt text.
    Uses the modern `text_target=` API for target side.
    """
    def preprocess(batch: Dict) -> Dict:
        src_texts = [ex[cfg.src_lang] for ex in batch["translation"]]
        tgt_texts = [ex[cfg.tgt_lang] for ex in batch["translation"]]

        model_inputs = tokenizer(
            src_texts,
            max_length=cfg.max_source_len,
            truncation=True,
        )
        labels = tokenizer(
            text_target=tgt_texts,
            max_length=cfg.max_target_len,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess


def tokenize_dataset(ds: DatasetDict, tokenizer: PreTrainedTokenizerBase, cfg: TrainConfig) -> DatasetDict:
    """
    Applies preprocessing to all splits and removes original columns.
    """
    preprocess = build_preprocess_fn(tokenizer, cfg)
    cols_to_remove = ds["train"].column_names
    tokenized = ds.map(
        preprocess,
        batched=True,
        remove_columns=cols_to_remove,
        desc="Tokenizing dataset",
    )
    return tokenized

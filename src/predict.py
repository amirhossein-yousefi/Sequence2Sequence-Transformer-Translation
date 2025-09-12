from __future__ import annotations

from typing import List
from datasets import Dataset
from transformers import PreTrainedTokenizerBase, Trainer
from src.config import TrainConfig


def sample_predictions(
    trainer: Trainer,
    tokenizer: PreTrainedTokenizerBase,
    raw_test: Dataset,
    tokenized_test: Dataset,
    cfg: TrainConfig,
    num_samples: int = 5,
) -> List[str]:
    """
    Runs generation on a few examples and returns pretty-printed strings.
    """
    k = min(num_samples, len(raw_test))
    raw_subset = raw_test.select(range(k))
    tokenized_subset = tokenized_test.select(range(k))
    pred_output = trainer.predict(tokenized_subset)
    decoded = tokenizer.batch_decode(pred_output.predictions, skip_special_tokens=True)

    lines = []
    for i, hyp in enumerate(decoded):
        src = raw_subset[i]["translation"][cfg.src_lang]
        ref = raw_subset[i]["translation"][cfg.tgt_lang]
        lines.append(f"[{i}] SRC: {src}\n    REF: {ref}\n    HYP: {hyp}")
    return lines

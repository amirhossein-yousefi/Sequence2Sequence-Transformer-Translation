from __future__ import annotations

from typing import Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)
from src.config import TrainConfig


def load_model_and_tokenizer(cfg: TrainConfig) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name)
    return model, tokenizer


def get_data_collator(tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel) -> DataCollatorForSeq2Seq:
    return DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

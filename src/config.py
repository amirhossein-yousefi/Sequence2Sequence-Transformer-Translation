from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, List
import torch


@dataclass
class TrainConfig:
    """
    Configuration for training/evaluation/prediction.
    Values mirror your original script defaults.
    """
    # Languages & model
    src_lang: str = "en"
    tgt_lang: str = "es"
    model_name: Optional[str] = None  # if None, uses Helsinki-NLP/opus-mt-{src}-{tgt}

    # Dataset
    dataset_name: str = "opus_books"
    dataset_config: Optional[str] = None  # if None, uses "{src}-{tgt}"

    # Tokenization
    max_source_len: int = 128
    max_target_len: int = 128

    # Optimization
    batch_size: int = 8
    num_epochs: int = 3
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    seed: int = 42

    # Generation / Trainer
    predict_with_generate: bool = True
    generation_max_length: int = 128
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_steps: int = 50
    save_total_limit: int = 2
    metric_for_best_model: str = "bleu"
    greater_is_better: bool = True
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])

    # Mixed precision
    fp16: bool = field(default_factory=lambda: bool(torch.cuda.is_available()))
    bf16: bool = False  # set True if you know your hardware supports it

    # IO
    output_dir: Optional[str] = None
    logging_dir: Optional[str] = None

    # Workflow toggles
    do_train: bool = True
    do_eval: bool = True
    do_predict: bool = True

    def resolve(self) -> "TrainConfig":
        """Compute derived/default fields."""
        if self.model_name is None:
            self.model_name = f"Helsinki-NLP/opus-mt-{self.src_lang}-{self.tgt_lang}"
        if self.dataset_config is None:
            self.dataset_config = f"{self.src_lang}-{self.tgt_lang}"
        if self.output_dir is None:
            self.output_dir = f"outputs/mt_{self.src_lang}_{self.tgt_lang}_marian"
        if self.logging_dir is None:
            self.logging_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logging_dir, exist_ok=True)
        # Trainings logs without accidental third-party uploads
        os.environ.setdefault("WANDB_DISABLED", "true")
        return self

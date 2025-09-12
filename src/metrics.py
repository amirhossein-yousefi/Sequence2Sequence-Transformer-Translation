from __future__ import annotations

from typing import Callable, Dict, Tuple
import numpy as np
import evaluate
from transformers import PreTrainedTokenizerBase


def load_eval_metrics() -> Tuple[evaluate.EvaluationModule, evaluate.EvaluationModule]:
    bleu = evaluate.load("sacrebleu")
    chrf = evaluate.load("chrf")
    return bleu, chrf


def _postprocess_text(preds, labels):
    preds = [p.strip() for p in preds]
    labels = [l.strip() for l in labels]
    return preds, labels


def build_compute_metrics(tokenizer: PreTrainedTokenizerBase) -> Callable[[Tuple[np.ndarray, np.ndarray]], Dict]:
    """
    Returns a `compute_metrics` function compatible with HuggingFace Trainer.
    """
    bleu_metric, chrf_metric = load_eval_metrics()

    def compute(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict:
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = _postprocess_text(decoded_preds, decoded_labels)

        bleu = bleu_metric.compute(predictions=decoded_preds,
                                   references=[[l] for l in decoded_labels])["score"]
        chrf = chrf_metric.compute(predictions=decoded_preds,
                                   references=decoded_labels)["score"]

        # Optional: generation length (excluding padding)
        gen_lens = [np.count_nonzero(p != tokenizer.pad_token_id) for p in preds]
        return {"bleu": bleu, "chrf": chrf, "gen_len": float(np.mean(gen_lens))}

    return compute

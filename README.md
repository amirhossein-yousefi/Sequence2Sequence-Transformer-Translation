
# Sequence2Sequence‑Transformer‑Translation

> A lean, modern baseline for neural machine translation (NMT) built on 🤗 Transformers, 🤗 Datasets, and 🤗 Evaluate — with sane defaults, strong metrics (BLEU & chrF), and a clean, hackable code layout.

<p align="center">
  <img alt="transformer-diagram" src="https://user-images.githubusercontent.com/000000/placeholder.png" width="1" height="1">
</p>

<div align="center">

**Python** `3.13+` • **Transformers** `4.42+` • **Datasets** `3.0+` • **Evaluate** `0.4.2+` • **TensorBoard** logging • **fp16** auto when CUDA is available

</div>

---

## 🚀 Model on Hugging Face

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Speech--Intensity--Whisper-yellow.svg)](https://huggingface.co/Amirhossein75/speech-intensity-whisper)

<p align="center">
  <a href="https://huggingface.co/Amirhossein75/speech-intensity-whisper">
    <img src="https://img.shields.io/badge/🤗%20View%20on%20Hugging%20Face-blueviolet?style=for-the-badge" alt="Hugging Face Repo">
  </a>
</p>

---


## ✨ What’s inside

- **Encoder–decoder Seq2Seq** model via pretrained MarianMT checkpoints (e.g., `Helsinki-NLP/opus-mt-en-es`), easily swappable for any supported language pair.  
- **Batteries‑included data pipeline**: loads the OPUS Books dataset and ensures **train/validation/test** splits are present.  
- **Tokenization done right**: source/target are tokenized with `text_target=` to avoid label leaks and special‑token headaches.  
- **Solid evaluation**: **sacreBLEU** and **chrF** via 🤗 Evaluate, including average generated length.  
- **Trainer‑first ergonomics**: uses Hugging Face **Trainer** & **TrainingArguments** for robust, reproducible training.  
- **Clean module layout** you can understand at a glance and modify safely.

```
.
├── src/
│   ├── __init__.py
│   ├── config.py        # TrainConfig dataclass (defaults, device, dirs)
│   ├── data.py          # dataset loading & tokenization utilities
│   ├── metrics.py       # BLEU & chrF metrics for Trainer
│   ├── model.py         # model/tokenizer/data collator loaders
│   ├── predict.py       # pretty printed sample predictions
│   └── train.py         # end-to-end training pipeline (HF Trainer)
├── requirements.txt
├── pyproject.toml
└── uv.lock              # optional: reproducible installs via uv
```

> **Defaults at a glance**
>
> - Language pair: **English → Spanish** (`en`→`es`)  
> - Dataset: **OPUS Books** (`Helsinki-NLP/opus_books`)  
> - Base model: **MarianMT** (`Helsinki-NLP/opus-mt-en-es`) if no model is specified  
> - Metrics: **sacreBLEU** and **chrF**  
> - Mixed precision: **fp16** automatically enabled when CUDA is available  
> - Logging: **TensorBoard** logs written under `outputs/.../logs`

---

## 🚀 Quickstart

### 0) Clone & enter the project

```bash
git clone https://github.com/amirhossein-yousefi/Sequence2Sequence-Transformer-Translation.git
cd Sequence2Sequence-Transformer-Translation
```

### 1) Create an environment & install deps

**Option A — pip (recommended):**
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Option B — uv (fast, reproducible):**
```bash
# Requires uv: https://github.com/astral-sh/uv
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

> The `pyproject.toml` pins runtime requirements and marks this as a Python **3.13+** project.

### 2) Train (two ways)

#### A) From Python (most flexible)

```python
from transformers import TrainingArguments, Trainer
from src.config import TrainConfig
from src.model import load_model_and_tokenizer, get_data_collator
from src.data import load_and_prepare_dataset, tokenize_dataset
from src.metrics import build_compute_metrics
from src.predict import sample_predictions

# 1) Configure & resolve defaults
cfg = TrainConfig().resolve()   # en->es, opus_books, MarianMT, fp16 if CUDA

# 2) Model & tokenizer
model, tokenizer = load_model_and_tokenizer(cfg)

# 3) Data
raw_ds = load_and_prepare_dataset(cfg)                       # ensures train/valid/test
tok_ds = tokenize_dataset(raw_ds, tokenizer, cfg)            # applies text->ids

# 4) Trainer
args = TrainingArguments(
    output_dir=cfg.output_dir,
    per_device_train_batch_size=cfg.batch_size,
    per_device_eval_batch_size=cfg.batch_size,
    num_train_epochs=cfg.num_epochs,
    learning_rate=cfg.learning_rate,
    weight_decay=cfg.weight_decay,
    gradient_accumulation_steps=cfg.gradient_accumulation_steps,
    logging_steps=cfg.logging_steps,
    evaluation_strategy=cfg.evaluation_strategy,
    save_strategy=cfg.save_strategy,
    save_total_limit=cfg.save_total_limit,
    predict_with_generate=cfg.predict_with_generate,
    generation_max_length=cfg.generation_max_length,
    report_to=cfg.report_to,
    fp16=cfg.fp16,
    bf16=cfg.bf16,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tok_ds["train"],
    eval_dataset=tok_ds["validation"],
    tokenizer=tokenizer,
    data_collator=get_data_collator(tokenizer, model),
    compute_metrics=build_compute_metrics(tokenizer),
)

# 5) Train / evaluate / predict
trainer.train()
metrics = trainer.evaluate()
print(metrics)

# A few qualitative samples
for line in sample_predictions(trainer, tokenizer, raw_ds["test"], tok_ds["test"], cfg, num_samples=5):
    print(line)

# (Optional) Save final artifacts
trainer.save_model(cfg.output_dir + "/final")
tokenizer.save_pretrained(cfg.output_dir + "/final")
```

#### B) As a script

`src/train.py` wires the exact same pipeline. You can run it directly or via module:

```bash
python -m src.train
# or
python src/train.py
```

> Tip: If you see an `ImportError: No module named 'src'`, run from the repository root or set `PYTHONPATH=.`, e.g. `PYTHONPATH=. python -m src.train`.

---

## 🔧 Customizing

### Change the language pair
Edit `src/config.py` or override in your script:
```python
cfg = TrainConfig(src_lang="de", tgt_lang="en").resolve()
```
By default the code derives the MarianMT checkpoint as:
```python
cfg.model_name = f"Helsinki-NLP/opus-mt-{cfg.src_lang}-{cfg.tgt_lang}"
```

### Use a specific model
Swap in any compatible seq2seq checkpoint from the Hub (T5, mBART, other OPUS‑MT variants, etc.):
```python
cfg = TrainConfig(model_name="Helsinki-NLP/opus-mt-tc-big-en-es").resolve()
```

### Pick a different dataset
Point to another Hugging Face dataset that exposes a `translation` column with your `src_lang`/`tgt_lang` keys:
```python
cfg = TrainConfig(dataset_name="wmt14", dataset_config="de-en").resolve()
```

### Tweak lengths & optimization
```python
cfg = TrainConfig(
    max_source_len=192,
    max_target_len=192,
    batch_size=16,
    num_epochs=5,
    learning_rate=3e-5,
).resolve()
```

---

## 📊 Metrics & logging

- **BLEU** via **sacreBLEU** and **chrF** are computed inside the Trainer loop and reported in eval logs.  
- Average generated length (`gen_len`) is included for sanity checking.  
- Logs land under `outputs/.../logs` for easy **TensorBoard** inspection:  
  ```bash
  tensorboard --logdir outputs
  ```

---

### 📉 Loss Curve

The following plot shows the training loss progression:

![Training Loss Curve](assets/train_loss.svg)


The following plot shows the validation loss progression:

![Training Loss Curve](assets/eval_loss.svg)

*(SVG file generated during training(by tensorboard logs) and stored under `assets/`)*

## 🖥️ Training Hardware & Environment

- **Device:** Laptop (Windows, WDDM driver model)  
- **GPU:** NVIDIA GeForce **RTX 3080 Ti Laptop GPU** (16 GB VRAM)  
- **Driver:** **576.52**  
- **CUDA (driver):** **12.9**  
- **PyTorch:** **2.8.0+cu129**  
- **CUDA available:** ✅ 


## 📊 Training Logs & Metrics

- **Total FLOPs (training):** `4,945,267,757,416,448`  
- **Training runtime:** `2,449.291` seconds  
- **Logging:** TensorBoard-compatible logs in `mt_en_es_marian/logs`  

You can monitor training live with:

```bash
tensorboard --logdir mt_en_es_marian/logs
```


## 🧠 Background & design notes

- The project uses the **Transformer** encoder–decoder architecture popularized by *Attention Is All You Need* (Vaswani et al., 2017).  
- We start from a strong **pretrained MT model (MarianMT / OPUS‑MT)** and **fine‑tune on OPUS Books**, a lightweight, parallel text dataset great for tutorials and fast iteration.  
- The training loop is implemented with the Hugging Face **Trainer** API for correctness and minimal boilerplate, while remaining fully swappable if you want to write a custom loop.  
- Defaults aim for **reproducibility** (`seed=42`), **sensible generation** (`predict_with_generate=True`), and **mixed precision** out of the box.

---

## 🗂️ Repository layout

- `src/config.py` — `TrainConfig` dataclass centralizes all knobs (languages, dataset, lengths, optimization, logging, IO) and resolves derived paths; enables `fp16` automatically if CUDA is present.  
- `src/data.py` — `load_and_prepare_dataset` ensures `train/validation/test` availability (splitting when necessary) and `tokenize_dataset` builds a batched `preprocess` mapper using `text_target=` for labels.  
- `src/model.py` — `load_model_and_tokenizer` loads the checkpoint and tokenizer; `get_data_collator` returns a `DataCollatorForSeq2Seq` for dynamic padding and label shifting.  
- `src/metrics.py` — wraps 🤗 Evaluate to compute **sacreBLEU** and **chrF**, with robust post‑processing.  
- `src/predict.py` — `sample_predictions` prints `SRC/REF/HYP` triplets for a quick qualitative check.  
- `src/train.py` — end‑to‑end glue around **Trainer** / **TrainingArguments** for training and evaluation.

---

## 🧪 Repro tips

- Fix the RNG seed in `TrainConfig` (default `42`).  
- Keep `save_total_limit` small to avoid disk bloat.  
- Always compare both **BLEU** *and* **chrF**; chrF often correlates better with perceived quality for morphologically rich languages.  
- When switching language pairs, verify the dataset has the exact keys (`cfg.src_lang`, `cfg.tgt_lang`) under its `translation` column.

---

## ❓ FAQ

**Q: Do I need a GPU?**  
A: No, but it helps a lot. fp16 will be enabled automatically when CUDA is available.

**Q: Can I push models to the Hub?**  
A: Yes — call `trainer.push_to_hub()` once authenticated, or use the `huggingface_hub` CLI.

**Q: My dataset has only a `train` split.**  
A: The loader will carve out **validation**/**test** (90/5/5) from `train` automatically.

**Q: How do I evaluate on my own files?**  
A: Create a small 🤗 Datasets `Dataset` with a `translation` column and your language keys, then run `trainer.predict(...)` or the helper in `src/predict.py`.

---

## 📝 Requirements

- Python **3.13+**  
- `transformers>=4.42.0`, `datasets>=3.0.0`, `evaluate>=0.4.2`, `sacrebleu>=2.4.2`, `sentencepiece>=0.1.99`, `accelerate>=0.32.0`, `numpy>=1.24`, `tensorboard`  
(see `requirements.txt` / `pyproject.toml`)

---

## 📚 References & acknowledgments

- Vaswani et al., **Attention Is All You Need** (NeurIPS 2017).  
- **Helsinki‑NLP OPUS‑MT** MarianMT checkpoints (e.g., `Helsinki-NLP/opus-mt-en-es`).  
- **OPUS Books** dataset (`Helsinki-NLP/opus_books`).  
- Hugging Face **Transformers** — **Trainer**, **Translation task guide**.  
- Hugging Face **Evaluate** — **sacreBLEU** and **chrF** metrics.

---

## 🙌 Contributing

Issues and PRs are welcome — especially new language‑pair recipes, evaluation scripts, and docs fixes.

---

## 🔒 License

If you plan to use this code in your own work or redistribute it, please add a license file to the repository.


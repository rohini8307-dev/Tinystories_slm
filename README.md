**TinyStories → Children Stories → Simple Stories**

- This project implements a minimal GPT-style language model trained fully from scratch, following a progressive training pipeline:
        TinyStories → Children Stories → Simple Stories

- The same model architecture is reused across all stages, with weights carried forward instead of training separate models.

## Model Overview

* Architecture: GPT-style decoder-only transformer
* Training: Fully from scratch (no pretrained weights)
* Tokenizer: GPT-2 tokenizer (`tiktoken`)
* Training style: Continual learning

The model learns progressively:

1. Basic language structure (TinyStories)
2. Richer narrative patterns (Children Stories)
3. Cleaner, simpler storytelling (Simple Stories)

## Requirements

* Python 3.8+
* PyTorch 2.0+
* tiktoken
* numpy
* tqdm

## Stage 0 — Data Preprocessing

All datasets are converted into binary token files using the GPT-2 tokenizer.

Input text files:

* TinyStoriesV2-GPT4-train.txt
* TinyStoriesV2-GPT4-val.txt
* Children stories text
* Simple stories text

Output:

* train.bin, val.bin
* train_ch.bin, val_ch.bin
* train_simp.bin, val_simp.bin

## Stage 1 — TinyStories Base Training

Train the GPT model from scratch.

```bash
python train_tiny.py
```

### Configuration

* Batch size: 32
* Learning rate: 3e-4
* Max steps: 200,000
* Layers: 8
* Attention heads: 8
* Embedding size: 336
* Context length: 256

### Output

* `tinystories_28M_final.pt`

This model learns:

* Grammar
* Sentence flow
* Very short story structure

## Stage 2 — Children Stories Continual Training

The TinyStories model is loaded and further trained, not reinitialized.

```bash
python train_children.py
```

### Key Features

* Loads `tinystories_28M_final.pt`
* Lower learning rate for stability
* Early stopping based on validation loss
* Saves best checkpoint only

### Output

* `children_stories_fin.pt`

Improvements:

* Longer coherence
* Better narrative continuity
* Richer vocabulary usage

---

## Stage 3 — Simple Stories Final Refinement

Further continual training starting from the ChildrenStories model.

```bash
python train_simp.py
```

### Key Features

* Loads `children_stories_fin.pt`
* Crash-safe checkpointing
* Resume training automatically if interrupted
* Early stopping enabled

### Output

* `simple_stories_ckpt.pt`

Focus:

* Cleaner language
* Simple, fluent storytelling
* Better prompt alignment

## Model Architecture

* Vocabulary size: 50,257 (GPT-2)
* Block size: 256
* Transformer layers: 8
* Attention heads: 8
* Embedding dimension: 336
* Dropout: 0.1
* Weight tying between embedding & output head

Causal self-attention ensures the model cannot see future tokens.


## Device Support

The project automatically detects and uses:
- GPU (CUDA) — if available
- CPU — fallback if CUDA is not available

## Notes

- Uses causal self-attention to prevent attending to future tokens
- Implements learning rate warmup + cosine decay
- Applies gradient clipping for stable training
- Mixed precision used in TinyStories training
- Demonstrates true continual learning, not isolated fine-tuning

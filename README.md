# CS-4063 NLP — Assignment 3: Transformer + RAG

**Student ID:** i232518  
**Course:** CS-4063 Natural Language Processing — FAST NUCES  

---

## Overview

A three-stage NLP pipeline built entirely from scratch in PyTorch (no pretrained models, no `nn.Transformer`, `nn.MultiheadAttention`, or `nn.TransformerEncoder`):

1. **Part A — Encoder-Only Transformer** — Multi-task sentiment + length-bucket classification; produces CLS review embeddings  
2. **Part B — Retrieval Module** — Cosine similarity search over stored training embeddings (top-K RAG retrieval)  
3. **Part C — Decoder-Only Transformer** — Causal GPT-style explanation generation conditioned on review, labels, and retrieved context

---

---

## Setup & Requirements

```bash
pip install torch numpy matplotlib scikit-learn
```

Python 3.8+ and PyTorch 1.12+ are required. A CUDA or MPS GPU is strongly recommended — on CPU, reduce `NUM_EPOCHS` and `BATCH_SIZE` in Cell 0.

---

## Running the Notebook

1. Place the three `.gz` dataset files in the same directory as the notebook (or update `DATA_FILES` paths in Cell 0).
2. Open the notebook and run **Kernel → Restart & Run All**.
3. All outputs are saved automatically to `models/` and `results/`.

> The notebook is fully self-contained and runs end-to-end without any manual steps.

---

## Configuration

All hyperparameters are in **Cell 0** (`## 0. Imports & Global Configuration`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `REVIEWS_PER_CAT` | 12,000 | Reviews loaded per category (10k–15k) |
| `MAX_SEQ_LEN` | 128 | Encoder input token length |
| `VOCAB_SIZE` | 20,000 | Vocabulary size (train data only) |
| `EMBED_DIM` | 128 | Model dimension (d_model) |
| `NUM_HEADS` | 4 | Attention heads |
| `FF_DIM` | 256 | Feed-forward hidden dimension |
| `NUM_ENC_LAYERS` | 2 | Encoder transformer blocks |
| `NUM_DEC_LAYERS` | 2 | Decoder transformer blocks |
| `DROPOUT` | 0.1 | Dropout rate |
| `BATCH_SIZE` | 64 | Training batch size |
| `NUM_EPOCHS` | 8 | Encoder training epochs |
| `DEC_EPOCHS` | 6 | Decoder training epochs |
| `LR` | 3e-4 | Learning rate (AdamW + cosine schedule) |
| `TOP_K` | 3 | Retrieved neighbours for RAG |

---

## Dataset

Amazon Reviews dataset from [https://nijianmo.github.io/amazon/index.html](https://nijianmo.github.io/amazon/index.html).

- **Categories:** Sports & Outdoors, Beauty, Cell Phones & Accessories
- **Per category:** 12,000 reviews (stratified across star ratings 1–5)
- **Total:** 36,000 reviews
- **Split:** 70% train / 15% val / 15% test

**Labels:**
- Sentiment: ratings 1–2 → Negative, 3 → Neutral, 4–5 → Positive
- Length bucket: ≤30 tokens → Short, 31–80 → Medium, >80 → Long

---

## Implementation Notes

- All attention mechanisms implemented from scratch: `ScaledDotProductAttention`, `MultiHeadAttention`, `EncoderBlock`, `DecoderBlock`
- Encoder uses a **learnable CLS token** for sequence aggregation
- Decoder uses a **causal lower-triangular mask** to prevent attending to future tokens
- Decoder uses **weight tying** between the embedding matrix and the LM head
- Retrieval uses **L2-normalised dot product** (equivalent to cosine similarity) for efficient search
- Vocabulary is built **exclusively from training data**

---

## Commit History

Commits follow this progression:

1. Project scaffolding & data loading
2. Preprocessing pipeline & vocabulary
3. Positional encoding & attention mechanism
4. Encoder block & full encoder model
5. Encoder training loop & evaluation
6. Embedding extraction & retrieval module
7. Decoder block & full decoder model
8. Decoder dataset construction & training
9. Qualitative evaluation & RAG ablation
10. Report & final cleanup

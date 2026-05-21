# CLAUDE.md — Project Context for Ceng7811

## Project Goal

Implement and experiment with the method from the paper:
**"ARC-NLP at PAN 2023: Hierarchical Long Text Classification for Trigger Detection"**
(Sahin, Kucukkaya, Toraman — CLEF 2023)

Task: Multi-label trigger detection on fanfiction documents (PAN CLEF 2023 dataset).
Each document may carry one or more of 32 trigger warning labels (e.g., violence, death, sexual-assault).

---

## Paper Method Summary

The paper proposes **Hierarchical Recurrence over Transformer-based LM** — a 4-stage pipeline:

### Stage 1 — Segmentation
- Clean text: strip HTML tags, URLs, English stop words, lowercase
- Split each document into **200-word segments** with **50-word overlap** between consecutive segments
- Each segment inherits the document's label(s)

### Stage 2 — Tokenization
- Fine-tune **RoBERTa-base** (HuggingFace) on the segmented documents
- Hyperparams: lr=1e-5, epochs=3, batch=8
- Tokenizer max length: **256 tokens** per segment

### Stage 3 — Feature Extraction
- Feed each segment through the fine-tuned RoBERTa
- Extract the **CLS token embedding** from the last hidden layer → segment feature vector (768-dim)

### Stage 4 — Model Training
- For each of the 32 trigger classes, train a **separate single-layer LSTM**
  - Hidden size: 100, batch: 8
  - Input: sequence of CLS embeddings for all segments of a document
  - Output: final hidden state h₀ → document vector
  - Classifier: 2 FC layers (sizes 100), ReLU activation
  - Loss: **Binary Cross-Entropy (BCE)**
  - Optimizer: SGD, lr=0.01
  - Train up to 10 epochs, save best by validation F1
- **Class imbalance**: For classes 15–32 (minority classes from `kidnapping` to `animal-cruelty`), apply positive class weight = `neg_count / pos_count` in BCE loss
- Combine 32 binary predictions → final multi-label output

### Paper Results (Validation Set)
| Model | F1-macro | F1-micro |
|---|---|---|
| BERT (baseline) | 0.047 | 0.461 |
| RoBERTa-Segment | 0.187 | 0.696 |
| TF-IDF + XGBoost | 0.258 | 0.727 |
| **Hierarchical LSTM + RoBERTa (paper)** | **0.372** | **0.736** |

---

## Dataset — PAN CLEF 2023 Trigger Detection

**Source**: `pan23-trigger-detection/` directory + preprocessed outputs

| Split | Raw Docs | After Preprocessing |
|---|---|---|
| Train | 307,102 | 307,021 |
| Validation | 17,104 | 17,099 |
| Test | 17,040 | 17,039 (unlabeled) |

- Documents: 50–6,000 words (avg ~2,350 words, median ~2,122)
- **32 trigger labels**, heavily imbalanced: `pornographic-content` appears in 77.5% of train docs; `animal-cruelty` in 0.05%
- Average labels per document: 1.45 (level2), 1.28 (level1)
- Test set has **no labels** (competition blind test)

### Raw File Structure
```
pan23-trigger-detection/
  pan23-trigger-detection-train/
    works.jsonl       # {id, text} records
    labels.jsonl      # {id, labels} records
  pan23-trigger-detection-validation/
    works.jsonl
    labels.jsonl
  pan23-trigger-detection-test/
    works.jsonl       # no labels.jsonl
```

---

## Preprocessing (Already Done)

Script: `dataProcess.py`
Outputs:
- `pan23_hierarchical_preprocessed.csv` — flattened CSV with pipe-separated label strings
- `pan23_hierarchical_preprocessed.jsonl` — one JSON record per row
- `pan23_preprocessing_stats.txt` — label frequency stats

### Label Hierarchy (Custom, defined in dataProcess.py)
The preprocessing adds a two-level hierarchy:
- **level2_labels**: original 32 PAN23 labels (normalized)
- **level1_labels**: 8 parent categories mapped from level2:
  - Violence & Physical Harm
  - Sexual Content / Sexual Violence
  - Mental Health & Self-Harm
  - Discrimination & Hate
  - Abuse & Interpersonal Harm
  - Death / Loss
  - Substance Use / Addiction
  - Animals
  - Other Sensitive Content (catch-all)

### Preprocessed Record Schema
```json
{
  "id": "...",
  "split": "train|validation|test",
  "text": "...",
  "word_count": 2350,
  "level1_labels": ["Violence & Physical Harm", ...],
  "level2_labels": ["violence", "blood", ...],
  "n_level1_labels": 1,
  "n_level2_labels": 2,
  "level1_labels_str": "Violence & Physical Harm|...",
  "level2_labels_str": "violence|blood|..."
}
```

---

## What Needs to Be Implemented

In order of the paper's pipeline:

1. **Segmentation** — split preprocessed text into 200-word segments with 50-word overlap (paper does HTML/URL/stopword cleaning first; our preprocessing already did whitespace normalization but NOT stopword removal or HTML stripping from raw — check if needed)
2. **RoBERTa fine-tuning** — on segments with document labels (multi-label BCE)
3. **CLS embedding extraction** — run fine-tuned RoBERTa over all segments of each doc, collect CLS vectors
4. **32 LSTM classifiers** — one per trigger class, trained on per-document CLS embedding sequences
5. **Evaluation** — F1-macro and F1-micro on validation set (primary metrics per PAN CLEF 2023)

---

## Key Implementation Notes

- The paper trains **one LSTM per label** (32 total), not a single multi-label LSTM
- Positive class weights are only applied to classes 15–32 (ordered by frequency — see Table 2 in paper): kidnapping, mental-illness, dissection, eating-disorder, abduction, body-hatred, childbirth, racism, sexism, miscarriage, transphobia, abortion, fat-phobia, animal-death, ableism, classism, misogyny, animal-cruelty
- The LSTM sees a **variable-length sequence** of CLS embeddings (one per segment) for each document; documents with more words produce more segments
- RoBERTa and LSTM are trained **separately** — not end-to-end
- Metrics: multi-label F1-macro and F1-micro (sklearn `f1_score` with `average='macro'` and `average='micro'`)

---

## Environment & Tools

- Python, PyTorch, HuggingFace Transformers
- Primary preprocessed data: `pan23_hierarchical_preprocessed.jsonl`
- Visualizations already generated: `document_length_distribution.png`, `level1_category_distribution.png`, `top15_level2_labels.png`

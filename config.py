from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH        = "pan23_hierarchical_preprocessed.jsonl"
ROBERTA_CKPT_DIR = "models/roberta_finetuned"
LSTM_CKPT_DIR    = "models/lstm"
EMBED_DIR        = "embeddings"

# ── RoBERTa (paper §4.2) ───────────────────────────────────────────────────────
ROBERTA_MODEL  = "roberta-base"
MAX_TOKENS     = 256    # tokenizer max length
ROBERTA_LR     = 1e-5
ROBERTA_EPOCHS = 3
ROBERTA_BATCH  = 8
EMBED_DIM      = 768    # roberta-base CLS embedding dimension

# ── Segmentation (paper §4.1) ──────────────────────────────────────────────────
SEGMENT_SIZE    = 200   # words per segment
SEGMENT_OVERLAP = 50    # word overlap between consecutive segments

# ── LSTM (paper §4.4) ──────────────────────────────────────────────────────────
LSTM_HIDDEN = 100
LSTM_LR     = 0.01      # SGD learning rate
LSTM_BATCH  = 8
LSTM_EPOCHS = 10        # max epochs; best checkpoint saved by val F1

# ── Label taxonomy ─────────────────────────────────────────────────────────────
# 32 labels ordered by frequency rank (Table 2 of paper).
# Labels at index >= MINORITY_START_IDX get pos_weight = neg_count/pos_count in BCE.
LABEL_ORDER = [
    # Classes 1-14 (no positive class weight)
    "pornographic-content",
    "violence",
    "death",
    "sexual-assault",
    "abuse",
    "blood",
    "suicide",
    "pregnancy",
    "child-abuse",
    "incest",
    "underage",
    "homophobia",
    "self-harm",
    "dying",
    # Classes 15-32 (positive class weight applied)
    "kidnapping",
    "mental-illness",
    "dissection",
    "eating-disorders",
    "abduction",
    "body-hatred",
    "childbirth",
    "racism",
    "sexism",
    "miscarriages",
    "transphobia",
    "abortion",
    "fat-phobia",
    "animal-death",
    "ableism",
    "classism",
    "misogyny",
    "animal-cruelty",
]
MINORITY_START_IDX = 14   # 0-indexed; LABEL_ORDER[14] = "kidnapping" = class 15
NUM_LABELS = len(LABEL_ORDER)  # 32

# Ensure output dirs exist when config is imported
for _d in [ROBERTA_CKPT_DIR, LSTM_CKPT_DIR, EMBED_DIR]:
    Path(_d).mkdir(parents=True, exist_ok=True)

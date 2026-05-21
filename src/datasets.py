"""
PyTorch Dataset classes for both training stages.

SegmentDataset        — flat segment-level dataset for RoBERTa fine-tuning (Stage 2).
EmbeddingSequenceDataset — doc-level dataset of precomputed CLS embeddings for LSTM (Stage 4).
"""

import json
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizerFast

from config import LABEL_ORDER, MAX_TOKENS, NUM_LABELS, SEGMENT_OVERLAP, SEGMENT_SIZE
from src.text_utils import clean_text, segment_text


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_label_index(label_list: List[str]) -> Dict[str, int]:
    return {lbl: idx for idx, lbl in enumerate(label_list)}


def _labels_to_vector(labels: List[str], label_index: Dict[str, int]) -> torch.Tensor:
    vec = torch.zeros(NUM_LABELS, dtype=torch.float32)
    for lbl in labels:
        idx = label_index.get(lbl)
        if idx is not None:
            vec[idx] = 1.0
        else:
            # Attempt fuzzy match (handles minor normalization drift)
            for canonical, i in label_index.items():
                if lbl.replace("-", "") == canonical.replace("-", ""):
                    vec[i] = 1.0
                    break
    return vec


# ── SegmentDataset ─────────────────────────────────────────────────────────────

class SegmentDataset(Dataset):
    """Flat segment-level dataset for RoBERTa fine-tuning.

    Each item is one 200-word segment; it inherits the parent document's
    full 32-dim binary label vector (paper §4.1, §4.2).
    """

    def __init__(self, jsonl_path: str, tokenizer: RobertaTokenizerFast, split: str = "train"):
        self._tokenizer = tokenizer
        self._label_index = _build_label_index(LABEL_ORDER)
        self._items: List[Tuple[Dict, torch.Tensor]] = []

        unmapped: set = set()
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("split") != split:
                    continue
                raw_labels: List[str] = rec.get("level2_labels", [])
                if not raw_labels and split != "test":
                    continue

                # Track unmapped labels for debugging
                for lbl in raw_labels:
                    if lbl not in self._label_index:
                        fuzzy_matched = any(
                            lbl.replace("-", "") == c.replace("-", "")
                            for c in self._label_index
                        )
                        if not fuzzy_matched:
                            unmapped.add(lbl)

                label_vec = _labels_to_vector(raw_labels, self._label_index)
                cleaned = clean_text(rec["text"])
                for seg in segment_text(cleaned, SEGMENT_SIZE, SEGMENT_OVERLAP):
                    self._items.append((seg, label_vec))

        if unmapped:
            print(f"[SegmentDataset/{split}] WARNING: unmapped labels: {unmapped}")
        print(f"[SegmentDataset/{split}] {len(self._items):,} segments loaded.")

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict:
        seg_text, label_vec = self._items[idx]
        encoding = self._tokenizer(
            seg_text,
            max_length=MAX_TOKENS,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         label_vec,
        }


# ── EmbeddingSequenceDataset ───────────────────────────────────────────────────

class EmbeddingSequenceDataset(Dataset):
    """Doc-level dataset of precomputed CLS embedding sequences for LSTM training.

    Expects a .pt file structured as:
        List[{"doc_id": str, "embeddings": Tensor[n_segs, 768], "label": Tensor[32]}]
    """

    def __init__(self, pt_path: str):
        self._data = torch.load(pt_path, weights_only=False)
        print(f"[EmbeddingSequenceDataset] {len(self._data):,} documents loaded from {pt_path}")

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        item = self._data[idx]
        embeddings = item["embeddings"].float()   # [n_segs, 768]
        label      = item["label"].float()        # [32]
        seq_len    = embeddings.size(0)
        return embeddings, label, seq_len


def collate_embedding_sequences(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad embedding sequences and sort by length (required for pack_padded_sequence).

    Returns:
        padded_seqs : [batch, max_len, 768]
        labels      : [batch, 32]
        lengths     : [batch]  (sorted descending)
    """
    embeddings, labels, lengths = zip(*batch)
    lengths_t = torch.tensor(lengths, dtype=torch.long)

    # Sort descending by length
    sorted_idx = torch.argsort(lengths_t, descending=True)
    lengths_t  = lengths_t[sorted_idx]
    labels_t   = torch.stack([labels[i] for i in sorted_idx])

    max_len = int(lengths_t[0].item())
    embed_dim = embeddings[0].size(-1)
    padded = torch.zeros(len(batch), max_len, embed_dim)
    for j, orig_idx in enumerate(sorted_idx.tolist()):
        n = lengths[orig_idx]
        padded[j, :n, :] = embeddings[orig_idx]

    return padded, labels_t, lengths_t

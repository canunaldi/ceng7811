"""
Stage 3 — Extract CLS embeddings from fine-tuned RoBERTa (paper §4.3).

For each document in every split (train / val / test):
  1. clean_text → segment_text → tokenize
  2. Forward through fine-tuned RoBERTa encoder
  3. Extract CLS token embedding from the LAST hidden layer:
       outputs.last_hidden_state[:, 0, :]  (NOT pooler_output)
  4. Collect per-document sequence of CLS vectors

Output files (float16 to save disk space, ~7 GB for train):
  embeddings/train.pt
  embeddings/val.pt
  embeddings/test.pt

Each file is a List[dict] where every dict has:
  {"doc_id": str, "embeddings": Tensor[n_segs, 768], "label": Tensor[32]}

Usage:
    python step3_extract_embeddings.py
"""

import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaModel, RobertaTokenizerFast
from tqdm import tqdm

import config
from src.text_utils import clean_text, segment_text
from src.datasets import _build_label_index, _labels_to_vector

EXTRACT_BATCH = 32   # segments per forward pass (larger = faster, more memory)


# ── Device ─────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Per-document segment dataset (internal helper) ────────────────────────────

class _DocSegmentDataset(Dataset):
    def __init__(self, segments: List[str], tokenizer: RobertaTokenizerFast):
        self._segments  = segments
        self._tokenizer = tokenizer

    def __len__(self):
        return len(self._segments)

    def __getitem__(self, idx):
        enc = self._tokenizer(
            self._segments[idx],
            max_length=config.MAX_TOKENS,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)


# ── Extraction ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_doc_embeddings(
    doc_text: str,
    encoder: RobertaModel,
    tokenizer: RobertaTokenizerFast,
    device: torch.device,
) -> torch.Tensor:
    """Return CLS embeddings for all segments of one document: Tensor[n_segs, 768]."""
    cleaned  = clean_text(doc_text)
    segments = segment_text(cleaned, config.SEGMENT_SIZE, config.SEGMENT_OVERLAP)

    seg_dataset = _DocSegmentDataset(segments, tokenizer)
    loader = DataLoader(seg_dataset, batch_size=EXTRACT_BATCH, shuffle=False)

    cls_list = []
    for input_ids, attention_mask in loader:
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
        # CLS token = first token of last hidden layer (paper §4.3)
        cls = outputs.last_hidden_state[:, 0, :].cpu()
        cls_list.append(cls)

    return torch.cat(cls_list, dim=0)   # [n_segs, 768]


def process_split(
    split: str,
    jsonl_path: str,
    encoder: RobertaModel,
    tokenizer: RobertaTokenizerFast,
    label_index: Dict[str, int],
    device: torch.device,
) -> List[dict]:
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    split_lines = [json.loads(l) for l in lines if json.loads(l).get("split") == split]
    print(f"\n[{split}] {len(split_lines):,} documents")

    for rec in tqdm(split_lines, desc=f"  {split}", unit="doc"):
        raw_labels: List[str] = rec.get("level2_labels", [])
        label_vec = _labels_to_vector(raw_labels, label_index)
        embeddings = extract_doc_embeddings(rec["text"], encoder, tokenizer, device)
        records.append({
            "doc_id":     str(rec["id"]),
            "embeddings": embeddings.half(),   # float16 to save ~7 GB
            "label":      label_vec,
        })

    return records


def main():
    device = get_device()
    print(f"Device: {device}")

    print("Loading fine-tuned RoBERTa encoder …")
    encoder   = RobertaModel.from_pretrained(config.ROBERTA_CKPT_DIR).to(device).eval()
    tokenizer = RobertaTokenizerFast.from_pretrained(config.ROBERTA_CKPT_DIR)

    label_index = _build_label_index(config.LABEL_ORDER)

    for split in ["train", "val", "test"]:
        records   = process_split(split, config.DATA_PATH, encoder, tokenizer, label_index, device)
        out_path  = Path(config.EMBED_DIR) / f"{split}.pt"
        torch.save(records, str(out_path))
        print(f"  Saved {len(records):,} docs → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

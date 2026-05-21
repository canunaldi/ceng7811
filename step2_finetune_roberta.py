"""
Stage 2 — Fine-tune RoBERTa on segmented documents (paper §4.2).

Each 200-word segment is treated as an independent training example and
inherits the parent document's full 32-dim multi-label vector. The model
is trained for exactly 3 epochs with AdamW (lr=1e-5) and BCEWithLogitsLoss.

Usage:
    python step2_finetune_roberta.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast
from tqdm import tqdm

import config
from src.datasets import SegmentDataset


# ── Device ─────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model(device: torch.device) -> RobertaForSequenceClassification:
    model = RobertaForSequenceClassification.from_pretrained(
        config.ROBERTA_MODEL,
        num_labels=config.NUM_LABELS,
        problem_type="multi_label_classification",
    )
    return model.to(device)


# ── Training ───────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="  train", leave=False):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    device = get_device()
    print(f"Device: {device}")

    print("Loading tokenizer …")
    tokenizer = RobertaTokenizerFast.from_pretrained(config.ROBERTA_MODEL)

    print(f"Building SegmentDataset (train, subset_frac={config.ROBERTA_SUBSET_FRAC}) — this may take a few minutes …")
    dataset = SegmentDataset(
        config.DATA_PATH,
        tokenizer,
        split="train",
        subset_frac=config.ROBERTA_SUBSET_FRAC,
        subset_seed=config.ROBERTA_SUBSET_SEED,
    )
    loader  = DataLoader(
        dataset,
        batch_size=config.ROBERTA_BATCH,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    print("Building model …")
    model   = build_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.ROBERTA_LR)
    loss_fn   = nn.BCEWithLogitsLoss()

    for epoch in range(1, config.ROBERTA_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.ROBERTA_EPOCHS}")
        avg_loss = train_epoch(model, loader, optimizer, loss_fn, device)
        print(f"  avg loss: {avg_loss:.4f}")

    print(f"\nSaving model to {config.ROBERTA_CKPT_DIR} …")
    model.save_pretrained(config.ROBERTA_CKPT_DIR)
    tokenizer.save_pretrained(config.ROBERTA_CKPT_DIR)
    print("Done.")


if __name__ == "__main__":
    main()

"""
Stage 4 — Train 32 binary LSTM classifiers, one per trigger label (paper §4.4).

Architecture per classifier:
  LSTM(input=768, hidden=100, num_layers=1, batch_first=True)
  → h_n[-1]  (document vector, "h₀" in Figure 1)
  → Linear(100, 100) → ReLU → Linear(100, 1)

Training:
  - Optimizer: SGD, lr=0.01  (paper specifies SGD explicitly)
  - Loss: BCEWithLogitsLoss
  - Positive class weight (neg_count / pos_count) applied for classes 15–32
  - Up to 10 epochs; checkpoint saved when validation binary-F1 improves

Usage (full run):
    python step4_train_lstm.py

Usage (grouped, for limited-session environments like Colab):
    python step4_train_lstm.py --start 0  --end 8    # group 1: labels 1-8
    python step4_train_lstm.py --start 8  --end 16   # group 2: labels 9-16
    python step4_train_lstm.py --start 16 --end 24   # group 3: labels 17-24
    python step4_train_lstm.py --start 24 --end 32   # group 4: labels 25-32

Optional epoch override:
    python step4_train_lstm.py --start 0 --end 8 --epochs 5

Checkpoints are saved per-label, so completed groups survive disconnections.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

import config
from src.datasets import EmbeddingSequenceDataset, collate_embedding_sequences


# ── Device ─────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Model ──────────────────────────────────────────────────────────────────────

class LSTMClassifier(nn.Module):
    """Single-layer LSTM with two-FC classifier head (paper §4.4).

    Input : packed sequence of CLS embeddings [n_segs, 768] per document
    Output: scalar logit for binary classification
    """

    def __init__(self, input_size: int = 768, hidden_size: int = 100):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=1, batch_first=True)
        self.fc1  = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(hidden_size, 1)

    def forward(self, packed_input: nn.utils.rnn.PackedSequence) -> torch.Tensor:
        _, (h_n, _) = self.lstm(packed_input)   # h_n: [1, batch, 100]
        doc_vec = h_n[-1]                        # [batch, 100]
        return self.fc2(self.relu(self.fc1(doc_vec))).squeeze(-1)   # [batch]


# ── Training helpers ────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, loss_fn, label_idx, device):
    model.train()
    total_loss = 0.0
    for padded_seqs, labels, lengths in loader:
        padded_seqs = padded_seqs.to(device)
        labels      = labels[:, label_idx].to(device)

        packed = nn.utils.rnn.pack_padded_sequence(
            padded_seqs, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        optimizer.zero_grad()
        logits = model(packed)
        loss   = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, label_idx, device):
    model.eval()
    all_preds, all_targets = [], []
    for padded_seqs, labels, lengths in loader:
        padded_seqs = padded_seqs.to(device)
        packed = nn.utils.rnn.pack_padded_sequence(
            padded_seqs, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        logits = model(packed)
        preds  = (torch.sigmoid(logits) > 0.5).int().cpu().tolist()
        targets = labels[:, label_idx].int().cpu().tolist()
        all_preds.extend(preds)
        all_targets.extend(targets)
    return f1_score(all_targets, all_preds, zero_division=0)


# ── Main ────────────────────────────────────────────────────────────────────────

def compute_pos_weight(train_dataset: EmbeddingSequenceDataset, label_idx: int) -> float:
    """pos_weight = neg_count / pos_count (paper §4.4 example: 1000/20 = 50)."""
    pos = sum(
        1 for item in train_dataset._data if item["label"][label_idx].item() == 1.0
    )
    neg = len(train_dataset) - pos
    if pos == 0:
        return 1.0
    return neg / pos


def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM classifiers for trigger detection.")
    parser.add_argument("--start",  type=int, default=0,                   help="First label index (inclusive, 0-based). Default: 0")
    parser.add_argument("--end",    type=int, default=config.NUM_LABELS,   help=f"Last label index (exclusive, 0-based). Default: {config.NUM_LABELS}")
    parser.add_argument("--epochs", type=int, default=config.LSTM_EPOCHS,  help=f"Max epochs per classifier. Default: {config.LSTM_EPOCHS}")
    return parser.parse_args()


def main():
    args   = parse_args()
    start  = max(0, args.start)
    end    = min(config.NUM_LABELS, args.end)
    epochs = args.epochs

    device = get_device()
    print(f"Device: {device}")
    print(f"Training labels [{start}, {end}) — {end - start} classifiers, {epochs} epochs each")

    print("Loading embedding datasets …")
    train_ds = EmbeddingSequenceDataset(f"{config.EMBED_DIR}/train.pt")
    val_ds   = EmbeddingSequenceDataset(f"{config.EMBED_DIR}/val.pt")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.LSTM_BATCH,
        shuffle=True,
        collate_fn=collate_embedding_sequences,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.LSTM_BATCH,
        shuffle=False,
        collate_fn=collate_embedding_sequences,
        num_workers=0,
    )

    ckpt_dir = Path(config.LSTM_CKPT_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    label_slice = list(enumerate(config.LABEL_ORDER))[start:end]

    for label_idx, label_name in label_slice:
        print(f"\n[{label_idx+1:02d}/32] {label_name}")

        ckpt_path = ckpt_dir / f"{label_idx:02d}_{label_name}.pt"
        if ckpt_path.exists():
            print(f"  checkpoint already exists — skipping (delete to retrain)")
            continue

        # Positive class weight for minority classes (classes 15–32, 0-indexed 14–31)
        if label_idx >= config.MINORITY_START_IDX:
            pw = compute_pos_weight(train_ds, label_idx)
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw).to(device))
            print(f"  pos_weight = {pw:.2f}")
        else:
            loss_fn = nn.BCEWithLogitsLoss()

        model     = LSTMClassifier(config.EMBED_DIM, config.LSTM_HIDDEN).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=config.LSTM_LR)

        best_f1 = 0.0

        for epoch in range(1, epochs + 1):
            avg_loss = train_epoch(model, train_loader, optimizer, loss_fn, label_idx, device)
            val_f1   = evaluate(model, val_loader, label_idx, device)
            print(f"  epoch {epoch:2d}/{epochs}  loss={avg_loss:.4f}  val_f1={val_f1:.4f}", end="")

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), str(ckpt_path))
                print("  ✓ saved", end="")
            print()

        print(f"  best val F1 = {best_f1:.4f}  →  {ckpt_path.name}")

    total_ckpts = len(list(ckpt_dir.glob("*.pt")))
    print(f"\nGroup [{start}, {end}) complete. Total checkpoints saved: {total_ckpts}/32")


if __name__ == "__main__":
    main()

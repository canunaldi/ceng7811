"""
Stage 5 — Evaluate all 32 LSTM classifiers on the validation set (paper §5.2).

Loads precomputed val embeddings and the 32 best LSTM checkpoints, runs
inference, then prints:
  - Per-class binary F1
  - Overall multi-label F1-macro and F1-micro

Paper targets: F1-macro = 0.372, F1-micro = 0.736

Usage:
    python step5_evaluate.py
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

import config
from src.datasets import EmbeddingSequenceDataset, collate_embedding_sequences
from step4_train_lstm import LSTMClassifier


# ── Device ─────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Inference ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_label(
    model: LSTMClassifier,
    loader: DataLoader,
    label_idx: int,
    device: torch.device,
    threshold: float = 0.5,
):
    model.eval()
    preds, targets, scores = [], [], []
    for padded_seqs, labels, lengths in loader:
        padded_seqs = padded_seqs.to(device)
        packed = nn.utils.rnn.pack_padded_sequence(
            padded_seqs, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        logits = model(packed)
        prob   = torch.sigmoid(logits).cpu()
        pred   = (prob > threshold).int().tolist()
        target = labels[:, label_idx].int().tolist()
        preds.extend(pred)
        targets.extend(target)
        scores.extend(prob.tolist())
    return preds, targets, scores


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    device = get_device()
    print(f"Device: {device}\n")

    print("Loading validation embeddings …")
    val_ds = EmbeddingSequenceDataset(f"{config.EMBED_DIR}/val.pt")
    val_loader = DataLoader(
        val_ds,
        batch_size=config.LSTM_BATCH,
        shuffle=False,
        collate_fn=collate_embedding_sequences,
        num_workers=0,
    )

    ckpt_dir = Path(config.LSTM_CKPT_DIR)
    all_preds   = []   # list of 32 lists, one per label
    all_targets = []

    # Collect ground-truth once (all 32 label columns)
    print("Collecting ground-truth labels …")
    gt_matrix = []
    for _, labels, _ in val_loader:
        gt_matrix.append(labels.int())
    import torch as _t
    gt_matrix = _t.cat(gt_matrix, dim=0).numpy()   # [n_val, 32]

    pred_matrix = _t.zeros_like(_t.tensor(gt_matrix)).numpy()

    print("\nRunning 32 classifiers …")
    header = f"{'Label':<22}  {'Pos%':>5}  {'F1':>6}  {'P':>6}  {'R':>6}"
    print(header)
    print("-" * len(header))

    for label_idx, label_name in enumerate(config.LABEL_ORDER):
        ckpt_path = ckpt_dir / f"{label_idx:02d}_{label_name}.pt"
        if not ckpt_path.exists():
            print(f"  [{label_idx+1:02d}] {label_name:<20} — checkpoint not found, skipping")
            continue

        model = LSTMClassifier(config.EMBED_DIM, config.LSTM_HIDDEN).to(device)
        model.load_state_dict(torch.load(str(ckpt_path), map_location=device, weights_only=True))

        preds, targets, _ = predict_label(model, val_loader, label_idx, device)
        pred_matrix[:, label_idx] = preds

        f1_bin = f1_score(targets, preds, zero_division=0)
        pos_pct = sum(targets) / len(targets) * 100
        from sklearn.metrics import precision_score, recall_score
        prec = precision_score(targets, preds, zero_division=0)
        rec  = recall_score(targets, preds, zero_division=0)
        print(f"  {label_name:<22}  {pos_pct:5.1f}%  {f1_bin:6.4f}  {prec:6.4f}  {rec:6.4f}")

    print("\n" + "=" * 55)
    f1_macro = f1_score(gt_matrix, pred_matrix, average="macro", zero_division=0)
    f1_micro = f1_score(gt_matrix, pred_matrix, average="micro", zero_division=0)
    print(f"  F1-macro : {f1_macro:.4f}   (paper: 0.3720)")
    print(f"  F1-micro : {f1_micro:.4f}   (paper: 0.7360)")
    print("=" * 55)


if __name__ == "__main__":
    main()

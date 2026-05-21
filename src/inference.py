"""
TriggerDetector — reusable inference pipeline for the full paper method.

Loads the fine-tuned RoBERTa encoder and all 32 LSTM classifiers once at
construction time, then provides a predict() method for arbitrary input text.

Used by both step5_evaluate.py (batch evaluation) and app.py (Gradio UI).
"""

from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizerFast

import config
from src.text_utils import clean_text, segment_text


# Import LSTMClassifier without triggering the training main()
from step4_train_lstm import LSTMClassifier


def _get_device(device: Optional[torch.device] = None) -> torch.device:
    if device is not None:
        return device
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TriggerDetector:
    """End-to-end trigger detection: raw text → label scores.

    Models are loaded once at construction. Subsequent calls to predict()
    are fast (seconds for typical fanfiction length).
    """

    def __init__(
        self,
        roberta_path: str = config.ROBERTA_CKPT_DIR,
        lstm_dir: str = config.LSTM_CKPT_DIR,
        device: Optional[torch.device] = None,
    ):
        self.device = _get_device(device)
        print(f"[TriggerDetector] device = {self.device}")

        print("[TriggerDetector] Loading RoBERTa encoder …")
        self.tokenizer = RobertaTokenizerFast.from_pretrained(roberta_path)
        self.encoder   = RobertaModel.from_pretrained(roberta_path).to(self.device).eval()

        print("[TriggerDetector] Loading 32 LSTM classifiers …")
        self.lstms: List[Optional[LSTMClassifier]] = []
        lstm_path = Path(lstm_dir)
        for label_idx, label_name in enumerate(config.LABEL_ORDER):
            ckpt = lstm_path / f"{label_idx:02d}_{label_name}.pt"
            if ckpt.exists():
                model = LSTMClassifier(config.EMBED_DIM, config.LSTM_HIDDEN).to(self.device)
                model.load_state_dict(
                    torch.load(str(ckpt), map_location=self.device, weights_only=True)
                )
                model.eval()
                self.lstms.append(model)
            else:
                print(f"  WARNING: checkpoint not found for {label_name} — will predict 0")
                self.lstms.append(None)

        n_loaded = sum(1 for m in self.lstms if m is not None)
        print(f"[TriggerDetector] Ready — {n_loaded}/32 classifiers loaded.")

    @torch.no_grad()
    def _extract_cls_embeddings(self, text: str) -> torch.Tensor:
        """Clean → segment → tokenize → RoBERTa → CLS embeddings [n_segs, 768]."""
        cleaned  = clean_text(text)
        segments = segment_text(cleaned, config.SEGMENT_SIZE, config.SEGMENT_OVERLAP)

        cls_list = []
        for seg in segments:
            enc = self.tokenizer(
                seg,
                max_length=config.MAX_TOKENS,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids      = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls = outputs.last_hidden_state[:, 0, :].cpu()  # [1, 768]
            cls_list.append(cls)

        return torch.cat(cls_list, dim=0)   # [n_segs, 768]

    @torch.no_grad()
    def predict(self, text: str, threshold: float = 0.5) -> Dict:
        """Detect trigger warnings in the given text.

        Args:
            text:      Raw input text (may contain HTML — cleaned internally).
            threshold: Sigmoid score threshold for positive prediction.

        Returns a dict with:
            "labels"     : List[str]  — labels predicted as present (score > threshold)
            "scores"     : Dict[str, float] — score for detected labels only
            "all_scores" : Dict[str, float] — sigmoid score for all 32 labels
            "n_segments" : int — number of segments the text was split into
        """
        embeddings = self._extract_cls_embeddings(text)   # [n_segs, 768]
        n_segs = embeddings.size(0)

        # Pack the single document as a batch of size 1
        lengths = torch.tensor([n_segs], dtype=torch.long)
        padded  = embeddings.unsqueeze(0).to(self.device)  # [1, n_segs, 768]
        packed  = nn.utils.rnn.pack_padded_sequence(
            padded, lengths.cpu(), batch_first=True, enforce_sorted=True
        )

        all_scores: Dict[str, float] = {}
        for label_idx, label_name in enumerate(config.LABEL_ORDER):
            model = self.lstms[label_idx]
            if model is None:
                all_scores[label_name] = 0.0
                continue
            logit = model(packed)
            score = torch.sigmoid(logit).item()
            all_scores[label_name] = round(score, 4)

        detected = {k: v for k, v in all_scores.items() if v >= threshold}
        return {
            "labels":     sorted(detected.keys(), key=lambda k: -detected[k]),
            "scores":     detected,
            "all_scores": all_scores,
            "n_segments": n_segs,
        }

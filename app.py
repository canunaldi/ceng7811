"""
Gradio UI — real-time trigger warning detector.

Models are loaded once at startup; subsequent predictions take a few seconds.

Usage:
    python app.py
Then open http://localhost:7860 in your browser.
"""

import gradio as gr
import pandas as pd

import config
from src.inference import TriggerDetector

# ── Load models once ───────────────────────────────────────────────────────────
print("Loading models (this may take ~30 seconds) …")
detector = TriggerDetector(
    roberta_path=config.ROBERTA_CKPT_DIR,
    lstm_dir=config.LSTM_CKPT_DIR,
)
print("Models ready.\n")


# ── Level-1 category mapping (for display grouping) ───────────────────────────
_LEVEL1 = {
    "violence": "Violence & Physical Harm",
    "blood": "Violence & Physical Harm",
    "kidnapping": "Violence & Physical Harm",
    "dissection": "Violence & Physical Harm",
    "sexual-assault": "Sexual Content / Sexual Violence",
    "pornographic-content": "Sexual Content / Sexual Violence",
    "underage": "Sexual Content / Sexual Violence",
    "incest": "Sexual Content / Sexual Violence",
    "suicide": "Mental Health & Self-Harm",
    "self-harm": "Mental Health & Self-Harm",
    "eating-disorders": "Mental Health & Self-Harm",
    "mental-illness": "Mental Health & Self-Harm",
    "body-hatred": "Mental Health & Self-Harm",
    "abuse": "Abuse & Interpersonal Harm",
    "child-abuse": "Abuse & Interpersonal Harm",
    "abduction": "Abuse & Interpersonal Harm",
    "death": "Death / Loss",
    "dying": "Death / Loss",
    "animal-death": "Animals",
    "animal-cruelty": "Animals",
    "racism": "Discrimination & Hate",
    "sexism": "Discrimination & Hate",
    "homophobia": "Discrimination & Hate",
    "transphobia": "Discrimination & Hate",
    "ableism": "Discrimination & Hate",
    "classism": "Discrimination & Hate",
    "misogyny": "Discrimination & Hate",
    "fat-phobia": "Discrimination & Hate",
}

def _category(label: str) -> str:
    return _LEVEL1.get(label, "Other Sensitive Content")


# ── Inference callback ─────────────────────────────────────────────────────────

def analyze(text: str, threshold: float):
    text = text.strip()
    if not text:
        return (
            "Please enter some text.",
            pd.DataFrame(columns=["Label", "Category", "Confidence"]),
            pd.DataFrame(columns=["Label", "Score"]),
        )

    result = detector.predict(text, threshold=threshold)

    # ── Detected labels table ──────────────────────────────────────────────────
    if result["labels"]:
        detected_rows = [
            {
                "Label":      lbl,
                "Category":   _category(lbl),
                "Confidence": f"{result['scores'][lbl]:.1%}",
            }
            for lbl in result["labels"]
        ]
        summary = (
            f"**{len(result['labels'])} trigger warning(s) detected**  "
            f"| Text split into **{result['n_segments']} segments**"
        )
    else:
        detected_rows = []
        summary = (
            f"No trigger warnings detected above threshold {threshold:.0%}.  "
            f"| Text split into **{result['n_segments']} segments**"
        )

    detected_df = pd.DataFrame(
        detected_rows if detected_rows else [{"Label": "—", "Category": "—", "Confidence": "—"}]
    )

    # ── All-scores table (sorted by score) ────────────────────────────────────
    all_rows = sorted(result["all_scores"].items(), key=lambda x: -x[1])
    all_df = pd.DataFrame([{"Label": k, "Score": f"{v:.1%}"} for k, v in all_rows])

    return summary, detected_df, all_df


# ── Gradio layout ──────────────────────────────────────────────────────────────

DESCRIPTION = """
# Trigger Warning Detector
**Method**: Hierarchical Recurrence over Transformer-based Language Model
*(ARC-NLP @ PAN CLEF 2023 — Sahin, Kucukkaya, Toraman)*

Paste fanfiction text below. The model splits it into 200-word segments, extracts
RoBERTa embeddings for each segment, then runs 32 binary LSTM classifiers to detect
trigger warnings.
"""

EXAMPLES = [
    ["She whispered softly as the sun set over the horizon, golden light spilling across the quiet garden.", 0.5],
    ["The blood pooled beneath him as the fight escalated — no one was getting out alive.", 0.5],
]

with gr.Blocks(title="Trigger Warning Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                lines=14,
                placeholder="Paste your fanfiction text here…",
                label="Input Text",
            )
            with gr.Row():
                threshold_slider = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                    label="Detection Threshold",
                    info="Labels with confidence above this value are flagged as detected.",
                )
                analyze_btn = gr.Button("Detect Triggers", variant="primary", scale=0)

        with gr.Column(scale=3):
            summary_md = gr.Markdown("Results will appear here after analysis.")
            with gr.Tabs():
                with gr.Tab("Detected Warnings"):
                    detected_table = gr.Dataframe(
                        headers=["Label", "Category", "Confidence"],
                        label="Detected Trigger Warnings",
                        interactive=False,
                        wrap=True,
                    )
                with gr.Tab("All Scores"):
                    all_scores_table = gr.Dataframe(
                        headers=["Label", "Score"],
                        label="Confidence Score for All 32 Labels",
                        interactive=False,
                        wrap=True,
                    )

    analyze_btn.click(
        fn=analyze,
        inputs=[text_input, threshold_slider],
        outputs=[summary_md, detected_table, all_scores_table],
    )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[text_input, threshold_slider],
        outputs=[summary_md, detected_table, all_scores_table],
        fn=analyze,
        cache_examples=False,
        label="Try an example",
    )

    gr.Markdown(
        "_Trained on PAN CLEF 2023 Trigger Detection dataset (307k fanfiction works from AO3)._"
    )

if __name__ == "__main__":
    demo.launch(share=False)

import json
import re
from pathlib import Path
from collections import Counter

import pandas as pd
from tqdm.auto import tqdm

# enable tqdm for pandas apply/map
tqdm.pandas()


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path("pan23-trigger-detection")

OUTPUT_CSV = "pan23_hierarchical_preprocessed.csv"
OUTPUT_JSONL = "pan23_hierarchical_preprocessed.jsonl"
STATS_TXT = "pan23_preprocessing_stats.txt"

MIN_WORDS = 30
MAX_WORDS = 7000
KEEP_TEST = True


# ============================================================
# LABEL HIERARCHY
# ============================================================

PARENT_MAP = {
    "violence": "Violence & Physical Harm",
    "graphic-violence": "Violence & Physical Harm",
    "torture": "Violence & Physical Harm",
    "blood": "Violence & Physical Harm",
    "gore": "Violence & Physical Harm",
    "kidnapping": "Violence & Physical Harm",

    "rape": "Sexual Content / Sexual Violence",
    "sexual-assault": "Sexual Content / Sexual Violence",
    "non-con": "Sexual Content / Sexual Violence",
    "dub-con": "Sexual Content / Sexual Violence",
    "explicit-sexual-content": "Sexual Content / Sexual Violence",

    "suicide": "Mental Health & Self-Harm",
    "suicidal-thoughts": "Mental Health & Self-Harm",
    "self-harm": "Mental Health & Self-Harm",
    "depression": "Mental Health & Self-Harm",
    "eating-disorders": "Mental Health & Self-Harm",
    "panic-attacks": "Mental Health & Self-Harm",

    "racism": "Discrimination & Hate",
    "sexism": "Discrimination & Hate",
    "homophobia": "Discrimination & Hate",
    "transphobia": "Discrimination & Hate",
    "ableism": "Discrimination & Hate",

    "abuse": "Abuse & Interpersonal Harm",
    "child-abuse": "Abuse & Interpersonal Harm",
    "domestic-violence": "Abuse & Interpersonal Harm",
    "emotional-abuse": "Abuse & Interpersonal Harm",
    "manipulation": "Abuse & Interpersonal Harm",
    "gaslighting": "Abuse & Interpersonal Harm",

    "death": "Death / Loss",
    "major-character-death": "Death / Loss",
    "grief": "Death / Loss",

    "drug-use": "Substance Use / Addiction",
    "alcohol-abuse": "Substance Use / Addiction",
    "addiction": "Substance Use / Addiction",

    "animal-death": "Animals",
    "animal-cruelty": "Animals",
}


# ============================================================
# HELPERS
# ============================================================

def count_lines(path: Path) -> int:
    print(f"Counting lines in {path} ...")
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def load_jsonl(path: Path, desc: str = None):
    total = count_lines(path)
    rows = []
    print(f"Reading {path} ({total:,} lines)")
    with path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, total=total, desc=desc or f"Loading {path.name}", unit="lines"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    print(f"Finished reading {path.name}: {len(rows):,} records")
    return rows


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def word_count(text: str) -> int:
    return len(text.split())


def normalize_label(label: str) -> str:
    label = label.strip().lower()
    label = label.replace("_", "-")
    label = re.sub(r"[ /]+", "-", label)
    label = re.sub(r"[^a-z0-9\-]", "", label)
    label = re.sub(r"-{2,}", "-", label).strip("-")
    return label


def ensure_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def infer_text_field(record: dict) -> str:
    for key in ["text", "work_text", "body", "content"]:
        if key in record and isinstance(record[key], str):
            return record[key]
    raise KeyError(f"Could not find text field in record keys: {list(record.keys())}")


def infer_id_field(record: dict):
    for key in ["id", "doc_id", "work_id"]:
        if key in record:
            return record[key]
    raise KeyError(f"Could not find id field in record keys: {list(record.keys())}")


def infer_label_field(record: dict):
    for key in ["labels", "label", "tags"]:
        if key in record:
            return ensure_list(record[key])
    raise KeyError(f"Could not find label field in record keys: {list(record.keys())}")


def get_parent_labels(child_labels):
    return sorted({PARENT_MAP.get(lbl, "Other Sensitive Content") for lbl in child_labels})


def deduplicate_labels(labels):
    return sorted(set(labels))


def format_multilabel_column(labels):
    return "|".join(labels) if labels else ""


def read_split(split_dir: Path, split_name: str) -> pd.DataFrame:
    print(f"\n{'='*70}")
    print(f"Processing split: {split_name}")
    print(f"Directory: {split_dir}")
    print(f"{'='*70}")

    works_path = split_dir / "works.jsonl"
    labels_path = split_dir / "labels.jsonl"

    print("Loading works...")
    works = load_jsonl(works_path, desc=f"{split_name}: works")

    print(f"Extracting ids/text for {split_name} works...")
    works_df = pd.DataFrame(
        {
            "id": [infer_id_field(r) for r in tqdm(works, desc=f"{split_name}: extract work ids")],
            "text": [infer_text_field(r) for r in tqdm(works, desc=f"{split_name}: extract work text")],
        }
    )

    print(f"Normalizing whitespace for {split_name}...")
    works_df["text"] = works_df["text"].progress_map(normalize_whitespace)

    print(f"Counting words for {split_name}...")
    works_df["word_count"] = works_df["text"].progress_map(word_count)
    works_df["split"] = split_name

    print(f"{split_name} works loaded: {len(works_df):,}")

    if labels_path.exists():
        print("Loading labels...")
        labels = load_jsonl(labels_path, desc=f"{split_name}: labels")

        print(f"Extracting and normalizing labels for {split_name}...")
        labels_df = pd.DataFrame(
            {
                "id": [infer_id_field(r) for r in tqdm(labels, desc=f"{split_name}: extract label ids")],
                "flat_labels": [
                    sorted(set(normalize_label(x) for x in infer_label_field(r)))
                    for r in tqdm(labels, desc=f"{split_name}: normalize labels")
                ],
            }
        )

        print(f"Merging works and labels for {split_name}...")
        df = works_df.merge(labels_df, on="id", how="left")
    else:
        print(f"No labels.jsonl found for {split_name}; creating empty labels.")
        df = works_df.copy()
        df["flat_labels"] = [[] for _ in range(len(df))]

    print(f"Finished split {split_name}: {len(df):,} rows")
    return df


# ============================================================
# MAIN
# ============================================================

def main():
    print("\nStarting PAN23 preprocessing pipeline...\n")

    split_candidates = {
        "train": DATASET_ROOT / "pan23-trigger-detection-train",
        "validation": DATASET_ROOT / "pan23-trigger-detection-validation",
        "test": DATASET_ROOT / "pan23-trigger-detection-test",
    }

    print("Checking dataset structure...")
    missing = [name for name, path in split_candidates.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing split folders: {missing}\n"
            f"Expected structure like:\n"
            f"{DATASET_ROOT}/train/works.jsonl\n"
            f"{DATASET_ROOT}/train/labels.jsonl\n"
            f"{DATASET_ROOT}/validation/works.jsonl\n"
            f"{DATASET_ROOT}/validation/labels.jsonl\n"
            f"{DATASET_ROOT}/test/works.jsonl"
        )
    print("Dataset structure looks good.\n")

    # Load splits
    dfs = []
    for split_name, split_path in split_candidates.items():
        dfs.append(read_split(split_path, split_name))

    print("\nCombining splits...")
    df = pd.concat(dfs, ignore_index=True)
    print(f"Combined rows: {len(df):,}")

    print("Dropping duplicate ids within the same split...")
    before = len(df)
    df = df.drop_duplicates(subset=["split", "id"], keep="first").reset_index(drop=True)
    print(f"Removed {before - len(df):,} duplicate split+id rows")

    print("Deduplicating flat labels...")
    df["flat_labels"] = df["flat_labels"].progress_apply(deduplicate_labels)

    print("Formatting label strings...")
    df["flat_labels_str"] = df["flat_labels"].progress_apply(format_multilabel_column)

    print("Dropping exact duplicate text+labels+split rows...")
    before = len(df)
    df = df.drop_duplicates(subset=["text", "flat_labels_str", "split"], keep="first").reset_index(drop=True)
    print(f"Removed {before - len(df):,} duplicate content rows")

    print(f"Filtering rows by word count [{MIN_WORDS}, {MAX_WORDS}]...")
    before_len = len(df)
    df = df[(df["word_count"] >= MIN_WORDS) & (df["word_count"] <= MAX_WORDS)].copy()
    removed_by_length = before_len - len(df)
    print(f"Removed {removed_by_length:,} rows by length filtering")
    print(f"Remaining rows: {len(df):,}")

    if KEEP_TEST:
        print("Keeping unlabeled test rows, dropping unlabeled train/validation rows...")
        before = len(df)
        df = df[(df["split"] == "test") | (df["flat_labels"].map(len) > 0)].copy()
        print(f"Removed {before - len(df):,} unlabeled non-test rows")
    else:
        print("Dropping all unlabeled rows...")
        before = len(df)
        df = df[df["flat_labels"].map(len) > 0].copy()
        print(f"Removed {before - len(df):,} unlabeled rows")

    print("Creating hierarchical labels...")
    df["level2_labels"] = df["flat_labels"]
    df["level1_labels"] = df["level2_labels"].progress_apply(get_parent_labels)

    print("Computing label counts per sample...")
    df["n_level2_labels"] = df["level2_labels"].progress_map(len)
    df["n_level1_labels"] = df["level1_labels"].progress_map(len)

    print("Formatting hierarchy labels for CSV...")
    df["level1_labels_str"] = df["level1_labels"].progress_apply(format_multilabel_column)
    df["level2_labels_str"] = df["level2_labels"].progress_apply(format_multilabel_column)

    final_cols = [
        "id",
        "split",
        "text",
        "word_count",
        "level1_labels",
        "level2_labels",
        "n_level1_labels",
        "n_level2_labels",
        "level1_labels_str",
        "level2_labels_str",
    ]
    df = df[final_cols].copy()

    print("\nSaving CSV...")
    csv_df = df.copy()
    csv_df["level1_labels"] = csv_df["level1_labels"].progress_apply(json.dumps)
    csv_df["level2_labels"] = csv_df["level2_labels"].progress_apply(json.dumps)
    csv_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Saved CSV: {OUTPUT_CSV}")

    print("Saving JSONL...")
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Writing JSONL", unit="rows"):
            rec = row.to_dict()
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved JSONL: {OUTPUT_JSONL}")

    print("Computing statistics...")
    level2_counter = Counter()
    level1_counter = Counter()

    labeled_df = df[df["split"] != "test"].copy()

    for labels in tqdm(labeled_df["level2_labels"], desc="Counting level2 labels"):
        level2_counter.update(labels)

    for labels in tqdm(labeled_df["level1_labels"], desc="Counting level1 labels"):
        level1_counter.update(labels)

    split_counts = df["split"].value_counts().to_dict()

    stats_lines = []
    stats_lines.append("PAN23 Trigger Detection preprocessing stats")
    stats_lines.append("=" * 50)
    stats_lines.append(f"Total rows after preprocessing: {len(df)}")
    stats_lines.append(f"Rows removed by length filtering: {removed_by_length}")
    stats_lines.append("")
    stats_lines.append("Split counts:")
    for split, count in split_counts.items():
        stats_lines.append(f"  {split}: {count}")
    stats_lines.append("")
    stats_lines.append(f"Average word count: {df['word_count'].mean():.2f}")
    stats_lines.append(f"Median word count: {df['word_count'].median():.2f}")
    stats_lines.append(f"Average number of level2 labels: {labeled_df['n_level2_labels'].mean():.2f}")
    stats_lines.append(f"Average number of level1 labels: {labeled_df['n_level1_labels'].mean():.2f}")
    stats_lines.append("")
    stats_lines.append("Top 15 level2 labels:")
    for label, count in level2_counter.most_common(15):
        stats_lines.append(f"  {label}: {count}")
    stats_lines.append("")
    stats_lines.append("Top 15 level1 labels:")
    for label, count in level1_counter.most_common(15):
        stats_lines.append(f"  {label}: {count}")

    print(f"Saving stats: {STATS_TXT}")
    with open(STATS_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(stats_lines))

    print("\nDone.")
    print(f"Final rows: {len(df):,}")
    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_JSONL}")
    print(f"Saved: {STATS_TXT}")

    print("\nPreview:")
    print(df.head(3)[["id", "split", "word_count", "level1_labels", "level2_labels"]])


if __name__ == "__main__":
    main()
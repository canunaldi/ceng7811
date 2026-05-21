import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1. Top 15 Level 2 labels
# -----------------------------
level2_counts = {
    "pornographic-content": 251265,
    "sexual-assault": 33060,
    "violence": 30723,
    "abuse": 23410,
    "death": 21936,
    "blood": 15937,
    "pregnancy": 14404,
    "incest": 14239,
    "underage": 9398,
    "suicide": 8661,
    "dying": 7913,
    "child-abuse": 7578,
    "self-harm": 5530,
    "homophobia": 5219,
    "kidnapping": 4730,
}

df_level2 = pd.DataFrame({
    "label": list(level2_counts.keys()),
    "count": list(level2_counts.values())
}).sort_values("count", ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(df_level2["label"], df_level2["count"])
plt.xlabel("Count")
plt.ylabel("Level 2 Label")
plt.title("Top 15 Level 2 Labels")
plt.tight_layout()
plt.savefig("top15_level2_labels.png", dpi=300, bbox_inches="tight")
plt.show()


# -----------------------------
# 2. Level 1 category counts
# -----------------------------
level1_counts = {
    "Other Sensitive Content": 270671,
    "Violence & Physical Harm": 44509,
    "Sexual Content / Sexual Violence": 33060,
    "Abuse & Interpersonal Harm": 23617,
    "Death / Loss": 21936,
    "Mental Health & Self-Harm": 13210,
    "Discrimination & Hate": 6580,
    "Animals": 353,
}

df_level1 = pd.DataFrame({
    "label": list(level1_counts.keys()),
    "count": list(level1_counts.values())
}).sort_values("count", ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(df_level1["label"], df_level1["count"])
plt.xlabel("Count")
plt.ylabel("Level 1 Category")
plt.title("Level 1 Category Distribution")
plt.tight_layout()
plt.savefig("level1_category_distribution.png", dpi=300, bbox_inches="tight")
plt.show()


# -----------------------------
# 3. Word count histogram
# -----------------------------
# This requires your preprocessed CSV file
csv_path = "pan23_hierarchical_preprocessed.csv"

try:
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(10, 6))
    plt.hist(df["word_count"].dropna(), bins=50)
    plt.xlabel("Word Count")
    plt.ylabel("Number of Documents")
    plt.title("Document Length Distribution")
    plt.tight_layout()
    plt.savefig("document_length_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()

except FileNotFoundError:
    print(f"CSV file not found: {csv_path}")
    print("Skipping histogram. Put the preprocessed CSV in the same folder to generate it.")
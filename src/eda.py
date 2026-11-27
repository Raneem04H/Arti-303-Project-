
from pathlib import Path
from collections import Counter
import re

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "Data" / "processed" / "english_tickets.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda"


def run_eda():
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Dataset shape:", df.shape)

    # 1. Support category distribution
    category_counts = df["queue"].value_counts()
    print("\nCategory Distribution:\n", category_counts)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=category_counts.values,
                y=category_counts.index, color="steelblue")
    plt.title("Distribution of Customer Support Tickets")
    plt.xlabel("Number of Tickets")
    plt.ylabel("Support Category")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "category_distribution.png")
    plt.close()

    # 2. Ticket type distribution
    type_counts = df["type"].value_counts()

    plt.figure(figsize=(9, 5))
    sns.barplot(x=type_counts.index,
                y=type_counts.values, color="mediumpurple")
    plt.title("Distribution of Ticket Types")
    plt.xlabel("Ticket Type")
    plt.ylabel("Number of Tickets")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "type_distribution.png")
    plt.close()

    # 3. Ticket priority distribution
    priority_counts = df["priority"].value_counts()

    plt.figure(figsize=(9, 5))
    sns.barplot(x=priority_counts.index,
                y=priority_counts.values, color="coral")
    plt.title("Distribution of Ticket Priorities")
    plt.xlabel("Priority Level")
    plt.ylabel("Number of Tickets")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "priority_distribution.png")
    plt.close()

    # Prepare ticket text for analysis
    df["ticket_text"] = (
        df["subject"].fillna("").astype(str) + " " +
        df["body"].fillna("").astype(str)
    ).str.strip()

    df["word_count"] = df["ticket_text"].str.split().str.len()

    # 4. Ticket text length distribution
    print("\nText Length Statistics:\n",
          df["word_count"].describe())

    plt.figure(figsize=(10, 5))
    sns.histplot(data=df, x="word_count",
                 bins=40, color="teal")
    plt.title("Distribution of Ticket Text Lengths")
    plt.xlabel("Number of Words per Ticket")
    plt.ylabel("Number of Tickets")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "text_length_distribution.png")
    plt.close()

    # 5. Average text length by category
    avg_length = (
        df.groupby("queue")["word_count"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(x=avg_length.values,
                y=avg_length.index, color="mediumseagreen")
    plt.title("Average Ticket Text Length by Support Category")
    plt.xlabel("Average Number of Words")
    plt.ylabel("Support Category")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "average_length_by_category.png")
    plt.close()

    # 6. Most common words
    all_text = " ".join(df["ticket_text"])
    words = re.findall(r"\b[a-zA-Z]+\b", all_text.lower())

    filtered_words = [
        word for word in words
        if word not in ENGLISH_STOP_WORDS and len(word) > 2
    ]

    common_words = Counter(filtered_words).most_common(15)
    word_labels = [word for word, count in common_words]
    word_values = [count for word, count in common_words]

    plt.figure(figsize=(11, 6))
    sns.barplot(x=word_values,
                y=word_labels, color="slateblue")
    plt.title("Top 15 Most Common Words in Customer Tickets")
    plt.xlabel("Word Frequency")
    plt.ylabel("Word")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "common_words.png")
    plt.close()

    # 7. Missing values analysis
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[
        missing_counts > 0
    ].sort_values(ascending=False)

    print("\nMissing Values:\n", missing_counts)

    if not missing_counts.empty:
        plt.figure(figsize=(11, 6))
        sns.barplot(x=missing_counts.values,
                    y=missing_counts.index, color="indianred")
        plt.title("Missing Values in the English Ticket Dataset")
        plt.xlabel("Number of Missing Values")
        plt.ylabel("Column")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "missing_values.png")
        plt.close()

    print("\nEDA completed. Figures saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    run_eda()

"""
STEP 2: Data Collection and Preprocessing
==========================================
Run this AFTER downloading your datasets.

HOW TO GET THE DATA (do this before running this script):
---------------------------------------------------------
1. Enron Email Dataset:
   - Go to: https://www.kaggle.com/datasets/wcukierski/enron-email-dataset
   - Download emails.csv
   - Upload it to your Google Colab session (or put it in Google Drive)

2. PhishingCorpus:
   - Go to: https://github.com/victoriadrake/phishing-corpus
   - Download the repository as a ZIP
   - The phishing emails are in the /corpus folder as .txt files
   - Upload the folder to Colab

   ALTERNATIVELY - easier option:
   - Use this Kaggle dataset which has both already combined:
   - https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset
   - Download phishing_email.csv (has 'text' and 'label' columns, 0=legit, 1=phishing)
   - This is the easiest option and what the comments below assume

After downloading, set the FILE_PATH variable below to point to your file.
"""

import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIG - change this path to wherever your file is
# ============================================================
# Option A - if using the combined Kaggle dataset (recommended):
FILE_PATH = "phishing_email.csv"   # change to your actual path

# Option B - if you have separate Enron + PhishingCorpus files,
# set these instead and use the loader at the bottom of the file.
ENRON_PATH   = "emails.csv"
PHISHING_DIR = "corpus/"

SAVE_DIR = "processed_data/"
RANDOM_SEED = 42
# ============================================================


def clean_text(text):
    """
    Clean a raw email string:
    - Remove HTML tags
    - Remove URLs
    - Remove email headers (lines starting with From:, To:, Subject: etc.)
    - Remove extra whitespace
    - Keep only printable ASCII
    """
    if not isinstance(text, str):
        return ""

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', ' ', text)

    # Remove email header lines
    text = re.sub(r'^(From|To|Subject|Cc|Bcc|Date|Message-ID|Content-Type|'
                  r'Content-Transfer-Encoding|MIME-Version|X-\w+):.*$',
                  '', text, flags=re.MULTILINE | re.IGNORECASE)

    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\-\']', ' ', text)

    # Collapse multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def load_combined_dataset(file_path):
    """
    Load the combined phishing dataset from Kaggle.
    Expected columns: 'text' (email body) and 'label' (0=legit, 1=phishing)
    OR 'Email Text' and 'Email Type' depending on the exact dataset version.
    """
    print(f"Loading dataset from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Columns found: {list(df.columns)}")
    print(f"Shape: {df.shape}")

    # Handle different column name variations
    if 'text' in df.columns and 'label' in df.columns:
        df = df[['text', 'label']].copy()
        df.columns = ['text', 'label']

    elif 'Email Text' in df.columns and 'Email Type' in df.columns:
        df = df[['Email Text', 'Email Type']].copy()
        df.columns = ['text', 'label']
        # Convert text labels to numeric
        df['label'] = df['label'].map({'Safe Email': 0, 'Phishing Email': 1})

    elif 'body' in df.columns and 'label' in df.columns:
        df = df[['body', 'label']].copy()
        df.columns = ['text', 'label']

    else:
        raise ValueError(
            f"Could not find expected columns. Found: {list(df.columns)}\n"
            "Expected either: 'text'+'label' OR 'Email Text'+'Email Type'"
        )

    return df


def load_separate_datasets(enron_path, phishing_dir):
    """
    Alternative loader if you have the Enron CSV and
    PhishingCorpus folder separately.
    """
    print("Loading Enron dataset...")
    enron_df = pd.read_csv(enron_path)

    # Enron CSV has a 'message' column with the full email
    if 'message' in enron_df.columns:
        legit = pd.DataFrame({
            'text': enron_df['message'].dropna().sample(
                n=min(10000, len(enron_df)), random_state=RANDOM_SEED
            ),
            'label': 0
        })
    else:
        raise ValueError(f"Enron CSV columns: {list(enron_df.columns)}")

    print("Loading PhishingCorpus...")
    phishing_texts = []
    for fname in os.listdir(phishing_dir):
        fpath = os.path.join(phishing_dir, fname)
        try:
            with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                phishing_texts.append(f.read())
        except Exception:
            continue

    phishing = pd.DataFrame({
        'text': phishing_texts,
        'label': 1
    })

    print(f"Loaded {len(legit)} legitimate and {len(phishing)} phishing emails")
    return pd.concat([legit, phishing], ignore_index=True)


def preprocess(df):
    """
    Full preprocessing pipeline:
    1. Drop nulls
    2. Clean text
    3. Drop short/empty texts
    4. Balance classes
    5. Train/val/test split
    """
    print("\n--- Starting preprocessing ---")
    print(f"Initial shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}")

    # Step 1: Drop nulls
    df = df.dropna(subset=['text', 'label'])
    df['label'] = df['label'].astype(int)

    # Step 2: Clean text
    print("\nCleaning text...")
    df['text'] = df['text'].apply(clean_text)

    # Step 3: Drop very short texts (less than 20 characters after cleaning)
    df = df[df['text'].str.len() >= 20]
    print(f"After removing short texts: {df.shape}")

    # Step 4: Drop duplicates
    df = df.drop_duplicates(subset=['text'])
    print(f"After removing duplicates: {df.shape}")

    # Step 5: Balance classes using undersampling
    min_class = df['label'].value_counts().min()
    print(f"\nBalancing classes to {min_class} samples each...")
    df_balanced = pd.concat([
        df[df['label'] == 0].sample(n=min_class, random_state=RANDOM_SEED),
        df[df['label'] == 1].sample(n=min_class, random_state=RANDOM_SEED)
    ]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    print(f"Balanced dataset shape: {df_balanced.shape}")
    print(f"Label distribution after balancing:\n{df_balanced['label'].value_counts()}")

    # Step 6: Split into train (70%), val (10%), test (20%)
    train_df, temp_df = train_test_split(
        df_balanced, test_size=0.30,
        stratify=df_balanced['label'],
        random_state=RANDOM_SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.667,   # 0.667 of 30% = 20% of total
        stratify=temp_df['label'],
        random_state=RANDOM_SEED
    )

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)} ({len(train_df)/len(df_balanced)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} ({len(val_df)/len(df_balanced)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} ({len(test_df)/len(df_balanced)*100:.1f}%)")

    return train_df, val_df, test_df


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Try loading the combined dataset first
    if os.path.exists(FILE_PATH):
        df = load_combined_dataset(FILE_PATH)
    elif os.path.exists(ENRON_PATH) and os.path.exists(PHISHING_DIR):
        df = load_separate_datasets(ENRON_PATH, PHISHING_DIR)
    else:
        print("ERROR: Could not find dataset files.")
        print(f"  Looking for: {FILE_PATH}")
        print(f"  Or:          {ENRON_PATH} + {PHISHING_DIR}")
        print("\nPlease download the dataset from Kaggle and update FILE_PATH.")
        return

    train_df, val_df, test_df = preprocess(df)

    # Save to CSV
    train_df.to_csv(f"{SAVE_DIR}train.csv", index=False)
    val_df.to_csv(f"{SAVE_DIR}val.csv", index=False)
    test_df.to_csv(f"{SAVE_DIR}test.csv", index=False)

    print(f"\nSaved to {SAVE_DIR}")
    print("Files: train.csv, val.csv, test.csv")
    print("\nPreprocessing complete. Run step3_train_transformers.py next.")


if __name__ == "__main__":
    main()

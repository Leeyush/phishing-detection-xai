"""
STEP 3: Fine-tune BERT and RoBERTa
====================================
Run this after step2_preprocess.py has created processed_data/

IMPORTANT - Google Colab setup:
- Make sure you have a GPU runtime: Runtime > Change runtime type > T4 GPU
- This script fine-tunes BERT first, then RoBERTa
- Each model takes roughly 20-40 minutes on a T4 GPU
- Models are saved to checkpoints/ folder

Run in Colab with:
    !python step3_train_transformers.py
OR copy-paste each section into separate Colab cells.
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score
)

# ============================================================
# CONFIG
# ============================================================
DATA_DIR      = "processed_data/"
CHECKPOINT_DIR = "checkpoints/"
MAX_LEN       = 512
BATCH_SIZE    = 16
EPOCHS        = 4
LEARNING_RATE = 2e-5
WEIGHT_DECAY  = 0.01
WARMUP_RATIO  = 0.1
DROPOUT       = 0.1
RANDOM_SEED   = 42

MODELS_TO_TRAIN = [
    ("bert-base-uncased",  "bert_phishing"),
    ("roberta-base",       "roberta_phishing"),
]
# ============================================================

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Check GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cpu":
    print("WARNING: No GPU detected. Training will be very slow.")
    print("In Colab: Runtime > Change runtime type > T4 GPU")


class EmailDataset(Dataset):
    """PyTorch dataset that tokenises emails on the fly."""

    def __init__(self, dataframe, tokenizer, max_len):
        self.texts  = dataframe['text'].tolist()
        self.labels = dataframe['label'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids':      encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels':         torch.tensor(self.labels[idx], dtype=torch.long)
        }


def compute_metrics(eval_pred):
    """
    Called by the Trainer after each evaluation step.
    Returns accuracy, precision, recall, F1 and AUC-ROC.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Get probabilities for AUC
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]

    acc = accuracy_score(labels, predictions)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    try:
        auc = roc_auc_score(labels, probs)
    except Exception:
        auc = 0.0

    return {
        'accuracy':  round(acc, 4),
        'precision': round(prec, 4),
        'recall':    round(rec, 4),
        'f1':        round(f1, 4),
        'auc_roc':   round(auc, 4),
    }


def train_model(model_name, save_name, train_df, val_df):
    """Fine-tune a single model and save it."""
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")

    save_path = os.path.join(CHECKPOINT_DIR, save_name)
    os.makedirs(save_path, exist_ok=True)

    # Load tokeniser and model
    print("Loading tokeniser and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        hidden_dropout_prob=DROPOUT,          # for BERT
        attention_probs_dropout_prob=DROPOUT,  # for BERT
    )

    # Build datasets
    train_dataset = EmailDataset(train_df, tokenizer, MAX_LEN)
    val_dataset   = EmailDataset(val_df,   tokenizer, MAX_LEN)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=save_path,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=os.path.join(save_path, "logs"),
        logging_steps=50,
        seed=RANDOM_SEED,
        fp16=(device == "cuda"),   # Use mixed precision on GPU
        report_to="none",          # Disable wandb/tensorboard
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model and tokeniser
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to: {save_path}")

    return trainer, tokenizer


def evaluate_on_test(trainer, tokenizer, test_df, model_name):
    """Run the trained model on the test set and print full results."""
    print(f"\nEvaluating {model_name} on test set...")

    test_dataset = EmailDataset(test_df, tokenizer, MAX_LEN)
    results = trainer.evaluate(test_dataset)

    print(f"\n--- Test Results: {model_name} ---")
    for k, v in results.items():
        if not k.startswith('eval_runtime'):
            print(f"  {k}: {v}")

    return results


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Load preprocessed data
    print("Loading preprocessed data...")
    train_df = pd.read_csv(f"{DATA_DIR}train.csv")
    val_df   = pd.read_csv(f"{DATA_DIR}val.csv")
    test_df  = pd.read_csv(f"{DATA_DIR}test.csv")

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    all_results = {}

    for model_name, save_name in MODELS_TO_TRAIN:
        trainer, tokenizer = train_model(
            model_name, save_name, train_df, val_df
        )
        results = evaluate_on_test(trainer, tokenizer, test_df, model_name)
        all_results[model_name] = results

    # Print summary table
    print("\n\n--- SUMMARY TABLE (copy these into your dissertation) ---")
    print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} "
          f"{'Recall':>8} {'F1':>8} {'AUC-ROC':>10}")
    print("-" * 75)
    for model, res in all_results.items():
        short_name = model.replace('-base', '').replace('-uncased', '')
        print(
            f"{short_name:<25} "
            f"{res.get('eval_accuracy',0):>10.4f} "
            f"{res.get('eval_precision',0):>10.4f} "
            f"{res.get('eval_recall',0):>8.4f} "
            f"{res.get('eval_f1',0):>8.4f} "
            f"{res.get('eval_auc_roc',0):>10.4f}"
        )

    print("\nDone. Run step4_baselines.py next.")


if __name__ == "__main__":
    main()

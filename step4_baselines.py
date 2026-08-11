"""
STEP 4: Train SVM and Random Forest Baselines
===============================================
Run this after step3_train_transformers.py

These are the classical ML baselines that your transformer
models are compared against in Table 3 of your dissertation.

This script:
- Trains TF-IDF + SVM
- Trains TF-IDF + Random Forest
- Evaluates both on the test set
- Prints results to add to your dissertation tables
- Saves both models to baselines/ folder
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# CONFIG
# ============================================================
DATA_DIR    = "processed_data/"
SAVE_DIR    = "baselines/"
RANDOM_SEED = 42
# ============================================================


def build_tfidf_svm():
    """
    TF-IDF vectoriser + Linear SVM with calibration
    (calibration is needed to get probability scores for AUC-ROC)
    """
    svm = CalibratedClassifierCV(
        LinearSVC(max_iter=2000, random_state=RANDOM_SEED)
    )
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),   # unigrams and bigrams
            sublinear_tf=True,    # apply log normalisation
            min_df=2,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{2,}',  # tokens of 2+ characters
        )),
        ('clf', svm)
    ])
    return pipeline


def build_tfidf_rf():
    """TF-IDF vectoriser + Random Forest"""
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        random_state=RANDOM_SEED,
        n_jobs=-1   # use all CPU cores
    )
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            strip_accents='unicode',
            analyzer='word',
            token_pattern=r'\w{2,}',
        )),
        ('clf', rf)
    ])
    return pipeline


def tune_and_train(pipeline, param_grid, train_texts, train_labels, model_name):
    """
    Run grid search cross-validation on the training set,
    then refit the best model.
    """
    print(f"\nRunning grid search for {model_name}...")
    print(f"  Parameter grid: {param_grid}")

    gs = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,               # 3-fold CV (faster than 5-fold)
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    gs.fit(train_texts, train_labels)

    print(f"  Best params: {gs.best_params_}")
    print(f"  Best CV F1:  {gs.best_score_:.4f}")

    return gs.best_estimator_


def evaluate(model, texts, labels, model_name):
    """Evaluate a trained model and return metrics dict."""
    predictions = model.predict(texts)
    probs = model.predict_proba(texts)[:, 1]

    acc  = accuracy_score(labels, predictions)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    auc  = roc_auc_score(labels, probs)
    cm   = confusion_matrix(labels, predictions)

    print(f"\n--- {model_name} Test Results ---")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TP={cm[1,1]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TN={cm[0,0]}")
    print(f"\n  Full Classification Report:")
    print(classification_report(labels, predictions,
                                target_names=['Legitimate', 'Phishing']))

    return {
        'accuracy':  acc,
        'precision': prec,
        'recall':    rec,
        'f1':        f1,
        'auc_roc':   auc,
        'tp': int(cm[1,1]),
        'tn': int(cm[0,0]),
        'fp': int(cm[0,1]),
        'fn': int(cm[1,0]),
    }


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load data
    print("Loading preprocessed data...")
    train_df = pd.read_csv(f"{DATA_DIR}train.csv")
    test_df  = pd.read_csv(f"{DATA_DIR}test.csv")

    train_texts  = train_df['text'].tolist()
    train_labels = train_df['label'].tolist()
    test_texts   = test_df['text'].tolist()
    test_labels  = test_df['label'].tolist()

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    all_results = {}

    # ── SVM ──────────────────────────────────────────────────
    print("\n" + "="*50)
    print("Training SVM baseline")
    print("="*50)

    svm_param_grid = {
        'clf__base_estimator__C': [0.1, 1.0, 10.0]
    }
    svm_model = tune_and_train(
        build_tfidf_svm(),
        svm_param_grid,
        train_texts, train_labels,
        "SVM"
    )
    all_results['SVM'] = evaluate(
        svm_model, test_texts, test_labels, "SVM (TF-IDF)"
    )
    with open(f"{SAVE_DIR}svm_model.pkl", 'wb') as f:
        pickle.dump(svm_model, f)
    print("SVM saved.")

    # ── Random Forest ────────────────────────────────────────
    print("\n" + "="*50)
    print("Training Random Forest baseline")
    print("="*50)

    rf_param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth':    [None, 50]
    }
    rf_model = tune_and_train(
        build_tfidf_rf(),
        rf_param_grid,
        train_texts, train_labels,
        "Random Forest"
    )
    all_results['Random Forest'] = evaluate(
        rf_model, test_texts, test_labels, "Random Forest (TF-IDF)"
    )
    with open(f"{SAVE_DIR}rf_model.pkl", 'wb') as f:
        pickle.dump(rf_model, f)
    print("Random Forest saved.")

    # ── Summary ───────────────────────────────────────────────
    print("\n\n--- BASELINE RESULTS (add to Table 3 in your dissertation) ---")
    print(f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} "
          f"{'Recall':>8} {'F1':>8} {'AUC-ROC':>10}")
    print("-" * 70)
    for name, res in all_results.items():
        print(
            f"{name:<20} "
            f"{res['accuracy']:>10.4f} "
            f"{res['precision']:>10.4f} "
            f"{res['recall']:>8.4f} "
            f"{res['f1']:>8.4f} "
            f"{res['auc_roc']:>10.4f}"
        )

    print("\n--- CONFUSION MATRIX SUMMARY (for Table 4) ---")
    print(f"{'Model':<20} {'TP':>6} {'TN':>6} {'FP':>6} {'FN':>6}")
    print("-" * 45)
    for name, res in all_results.items():
        print(
            f"{name:<20} "
            f"{res['tp']:>6} "
            f"{res['tn']:>6} "
            f"{res['fp']:>6} "
            f"{res['fn']:>6}"
        )

    print("\nDone. Run step5_xai.py next.")


if __name__ == "__main__":
    main()

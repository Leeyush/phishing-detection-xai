# Setup & Running Guide

This pipeline was built and tested on **Google Colab** (free T4 GPU tier), and the steps below assume that environment. It will also run on any machine with a CUDA-capable GPU and Python 3.10+.

## 1. Get the dataset

The training data is not included in this repo (the combined CSV is ~100MB). Download it from Kaggle:

- [Phishing Email Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset) — a single `phishing_email.csv` with `text` and `label` columns (0 = legitimate, 1 = phishing), already combining Enron and PhishingCorpus.

Place `phishing_email.csv` in the project root (or update `FILE_PATH` in `src/step2_preprocess.py`).

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Enable a GPU (Colab)

`Runtime > Change runtime type > Hardware accelerator > T4 GPU > Save`

## 4. Run the pipeline in order

```bash
# Stage 1 — clean, balance, and split the data (train/val/test = 70/10/20)
python src/step2_preprocess.py
# → writes processed_data/{train,val,test}.csv

# Stage 2 — fine-tune BERT and RoBERTa (~20–40 min each on a T4)
python src/step3_train_transformers.py
# → writes checkpoints/bert_phishing/ and checkpoints/roberta_phishing/

# Stage 3 — train the classical ML baselines (SVM, Random Forest)
python src/step4_baselines.py
# → writes baselines/svm_model.pkl and baselines/rf_model.pkl

# Stage 4 — generate SHAP / LIME explanations and stability analysis
python src/step5_xai.py
# → writes figures/figure2_shap_tokens.png, figure3_lime_bar.png, figure4_lime_stability.png
```

Alternatively, open `notebooks/PhishingDetection_Code.ipynb` in Colab and run the cells top to bottom — it wraps the same four scripts with setup/install cells and inline result printing.

## Expected runtime (T4 GPU)

| Stage | Script | Approx. time |
|---|---|---|
| Preprocessing | `step2_preprocess.py` | ~5 min |
| Fine-tuning (BERT + RoBERTa) | `step3_train_transformers.py` | ~40–80 min |
| Baselines | `step4_baselines.py` | ~10 min |
| XAI (SHAP + LIME) | `step5_xai.py` | ~20–30 min |

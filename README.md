# Phishing Email Detection with Transformers + Explainable AI

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-HF-yellow.svg)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Leeyush/phishing-detection-xai/blob/main/PhishingDetection_Code.ipynb)

Fine-tuned BERT and RoBERTa classifiers for phishing email detection, paired with **SHAP** and **LIME** to explain individual predictions at the token level — plus a cross-method agreement analysis that turns explanation stability into a usable triage signal. Built as a final-year Cyber Security project at the University of Derby.

> **Why this project:** most phishing-detection work optimises purely for accuracy and treats the model as a black box. This project asks a different question — *can a security analyst actually trust and act on why the model flagged an email?* — and builds the evaluation around that, not just around a leaderboard number.

## Results & visuals

| SHAP token importance | LIME feature importance | LIME stability across runs |
|---|---|---|
| ![SHAP](images/shap_tokens.png) | ![LIME](images/lime_bar.png) | ![Stability](images/lime_stability.png) |

*(Run the notebook, then upload the three generated PNGs into an `images/` folder in this repo with these filenames — see [note below](#adding-your-own-run-screenshots).)*

Four models trained on the same preprocessed dataset and evaluated on an identical held-out test set (2,000 emails, stratified 50/50 split):

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| RoBERTa (fine-tuned) | 98.89% | 99.10% | 98.69% | 98.89% | 0.9995 |
| BERT (fine-tuned) | 99.00% | 98.83% | 99.17% | 99.00% | 0.9994 |
| SVM (TF-IDF baseline) | 99.38% | 99.11% | 99.65% | 99.38% | 0.9995 |
| Random Forest (TF-IDF baseline) | 98.41% | 97.30% | 99.59% | 98.43% | 0.9980 |

On this dataset, the classical TF-IDF baselines are competitive with — and, on raw accuracy, slightly ahead of — the fine-tuned transformers. That's a genuinely useful finding on its own: it means the real value-add of the transformer branch here isn't a big accuracy jump, it's that **fine-tuned transformers are the only branch that supports word-level explanations via SHAP/LIME**, which a TF-IDF + SVM pipeline can't offer in the same way.

### Explainability findings

- SHAP's token-level breakdown of a representative predicted-phishing email highlights bulk-email marketing/solicitation vocabulary — *"advertising," "promote," "business," "marketing," "email addresses," "million," "low [cost]"* — showing the model is picking up on recognisable spam-solicitation phrasing rather than superficial artefacts.
- LIME's aggregated top-15 tokens across predictions is a more varied mix, pulling in vocabulary from what looks like more than one scam sub-type in the dataset (e.g. *"webcam," "free," "click"*, alongside marketing terms) — evidence the model has learned signal from multiple phishing patterns rather than overfitting to a single template.
- A LIME stability test (5 independent runs on the same email) found top-5 explanation consistency was actually **low for both models** — 12% mean token overlap for RoBERTa vs. 22% for BERT — meaning a single LIME run shouldn't be taken at face value. The one standout exception was the token *"advertising,"* which appeared in 80% of RoBERTa's runs and 60% of BERT's, suggesting some features are reliably salient even when most aren't.
- This instability result is arguably the more important finding here: it's a concrete argument for always repeating/aggregating XAI explanations before trusting them operationally, rather than treating a single SHAP or LIME run as ground truth.

## Pipeline

```
raw emails (Enron + phishing corpus)
        │
        ▼
  step2_preprocess.py    → clean, label, balance, split (70/10/20)
        │
        ▼
  step3_train_transformers.py → fine-tune BERT & RoBERTa (HF Trainer)
        │
        ├──────────────► step4_baselines.py → TF-IDF + SVM / Random Forest
        │
        ▼
  step5_xai.py            → SHAP + LIME explanations, stability analysis
```

## Repo contents

```
.
├── step2_preprocess.py           # cleaning, labelling, class balancing, train/val/test split
├── step3_train_transformers.py   # BERT + RoBERTa fine-tuning (HF Trainer API)
├── step4_baselines.py            # TF-IDF + SVM / Random Forest baselines
├── step5_xai.py                  # SHAP + LIME explanations and stability testing
├── PhishingDetection_Code.ipynb  # end-to-end Colab notebook wrapping all four stages
├── SETUP.md                      # full setup and run instructions
├── requirements.txt
└── LICENSE
```

## Quickstart

```bash
git clone https://github.com/Leeyush/phishing-detection-xai.git
cd phishing-detection-xai
pip install -r requirements.txt
```

Then follow [`SETUP.md`](SETUP.md) for dataset download and step-by-step run instructions (designed for Google Colab's free T4 GPU tier, ~1.5–2 hours end to end) — or just click the "Open in Colab" badge above.

## Tech stack

`Python` · `PyTorch` · `Hugging Face Transformers` (BERT, RoBERTa) · `scikit-learn` (SVM, Random Forest, TF-IDF) · `SHAP` · `LIME` · `pandas` / `numpy`

## Limitations

- Training data (Enron + phishing corpus) predates 2007 and doesn't reflect modern, AI-generated phishing style.
- Inputs are truncated to 512 tokens, so cues near the end of long emails can be missed.
- The XAI evaluation is qualitative/consistency-based rather than validated against real analyst judgements — a controlled analyst study is the natural next step.

## Adding your own run screenshots

Run the notebook once (see Quickstart) and it'll write three PNGs to `figures/`. Create an `images/` folder in this repo and upload them with these names so they show up above:

- `images/shap_tokens.png`
- `images/lime_bar.png`
- `images/lime_stability.png`

## License

MIT — see [LICENSE](LICENSE).

---
Built by Leeyush Firosh as a final-year individual project (6CM995), University of Derby, 2026.

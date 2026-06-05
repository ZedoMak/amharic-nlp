# Model Comparison Report

## Objective

Evaluate simple baseline models for Amharic tweet sentiment classification and choose a practical model for the current project baseline.

The task is a 4-class sentiment classification problem with these labels:

- `mixed`
- `negative`
- `neutral`
- `positive`

## Dataset

The dataset is downloaded by `download_data.py` and preprocessed by `src/prepare_data.py`.

| Split / artifact              |  Rows | Notes                                       |
| ----------------------------- | ----: | ------------------------------------------- |
| `data/train.csv`              | 7,511 | Raw training data                           |
| `data/test.csv`               |   939 | Raw test data                               |
| `data/preprocessed_train.csv` | 6,282 | Rows remaining after cleaning and filtering |

Preprocessed label distribution:

| Label      |  Rows |
| ---------- | ----: |
| `neutral`  | 2,990 |
| `negative` | 1,492 |
| `positive` | 1,289 |
| `mixed`    |   511 |

The dataset is imbalanced, especially because `neutral` is the largest class and `mixed` is the smallest class. Because of that, accuracy alone is not enough for model selection.

## Preprocessing

The preprocessing pipeline is implemented in `src/preprocess.py` and `src/prepare_data.py`.

Current cleaning steps:

- Convert text to lowercase.
- Remove retweet markers.
- Remove user mentions.
- Remove hashtag symbols while keeping hashtag text.
- Normalize whitespace.
- Drop rows with missing sentiment labels.
- Drop rows where cleaned text has 10 characters or fewer.

## Evaluation Setup

All models were evaluated with the same setup:

- Data: `data/preprocessed_train.csv`
- Split: stratified 80 percent train / 20 percent validation
- Random seed: `42`
- Validation size: 1,257 examples
- Features: TF-IDF
- TF-IDF settings:
  - `max_features=10000`
  - `ngram_range=(1, 2)`
  - `min_df=2`

Candidate models:

- Logistic Regression with balanced class weights
- Multinomial Naive Bayes
- Linear SVM with balanced class weights

## Results

| Model               | Accuracy | Macro F1 | Weighted F1 |
| ------------------- | -------: | -------: | ----------: |
| Logistic Regression |   0.4495 |   0.3892 |      0.4599 |
| Naive Bayes         |   0.5036 |   0.2522 |      0.3879 |
| Linear SVM          |   0.4455 |   0.3590 |      0.4444 |

Per-class F1 scores:

| Model               |  Mixed | Negative | Neutral | Positive |
| ------------------- | -----: | -------: | ------: | -------: |
| Logistic Regression | 0.1245 |   0.4240 |  0.5251 |   0.4833 |
| Naive Bayes         | 0.0000 |   0.1369 |  0.6520 |   0.2201 |
| Linear SVM          | 0.0804 |   0.3651 |  0.5462 |   0.4444 |

## Interpretation

Naive Bayes has the highest raw accuracy, but its macro F1 is much lower than the other models. It gets most of its accuracy by predicting the majority class well, especially `neutral`, while failing to identify the `mixed` class at all on this validation split.

Logistic Regression has lower accuracy than Naive Bayes, but it has the best macro F1 and weighted F1. This means it gives the best overall balance across classes and handles minority labels better than the other candidates.

Linear SVM is competitive with Logistic Regression, but it has slightly weaker macro and weighted F1. The current interactive prediction script also expects probability estimates for confidence output, which Logistic Regression supports directly while `LinearSVC` does not.

## Decision

Use Logistic Regression with TF-IDF as the current project baseline.

Decision rationale:

- Best macro F1 among evaluated models.
- Best weighted F1 among evaluated models.
- Better minority-class behavior than Naive Bayes.
- Supports probability estimates for the confidence score shown in `src/predict.py`.
- Already matches the saved artifacts used by the project:
  - `models/logistic_model.pkl`
  - `models/tfidf.pkl`

## Reproduce the Comparison

Run:

```bash
python src/compare_models.py
```

This reproduces the accuracy comparison printed by the existing comparison script.

## Recommended Next Steps If You Clone This Repo

- Extend `src/compare_models.py` to print macro F1, weighted F1, per-class precision, recall, and F1.
- Add a confusion matrix to inspect common label confusions.
- Improve URL cleaning, for example with a pattern such as `https?://\S+`.
- Try character n-gram TF-IDF features, which often work well for morphologically rich languages.
- Tune Logistic Regression hyperparameters such as `C`, solver, and n-gram range.
- Use cross-validation for more stable model selection.
- Consider transformer-based Amharic or multilingual models after the classical ML baseline is solid.

# Amharic NLP Sentiment Classifier

This project builds a simple sentiment classifier for Amharic tweets. It downloads a labeled dataset, cleans the text, trains TF-IDF based machine learning models, compares baseline classifiers, and provides an interactive prediction script.

The current baseline is a Logistic Regression classifier trained on word-level unigram and bigram TF-IDF features.

## Project Structure

```text
.
|-- download_data.py              # Download train/test CSV files
|-- explore.py                    # Inspect dataset shape, labels, and missing values
|-- data/                         # Local data files, ignored by git
|-- models/                       # Saved model artifacts
|   |-- logistic_model.pkl
|   `-- tfidf.pkl
|-- reports/
|   `-- model_comparison.md       # Evaluation results and model decision
`-- src/
    |-- compare_models.py         # Compare baseline classifiers
    |-- predict.py                # Interactive sentiment prediction
    |-- prepare_data.py           # Clean and save preprocessed training data
    |-- preprocess.py             # Text cleaning helper
    `-- train.py                  # Train and save the Logistic Regression model
```

## Labels

The classifier predicts one of four sentiment labels:

- `mixed`
- `negative`
- `neutral`
- `positive`

## Setup

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install the required Python packages:

```bash
pip install pandas scikit-learn joblib
```

## Data Preparation

Download the raw dataset:

```bash
python download_data.py
```

Inspect the raw data:

```bash
python explore.py
```

Create the preprocessed training file:

```bash
python src/prepare_data.py
```

This writes:

```text
data/preprocessed_train.csv
```

The `data/` directory is ignored by git, so run the download and preprocessing steps again when setting up the project on a new machine.

## Training

Train the current baseline model:

```bash
python src/train.py
```

This saves:

```text
models/logistic_model.pkl
models/tfidf.pkl
```

## Model Comparison

Compare the baseline candidate models:

```bash
python src/compare_models.py
```

Current validation results:

| Model               | Accuracy | Macro F1 | Weighted F1 |
| ------------------- | -------: | -------: | ----------: |
| Logistic Regression |   0.4495 |   0.3892 |      0.4599 |
| Naive Bayes         |   0.5036 |   0.2522 |      0.3879 |
| Linear SVM          |   0.4455 |   0.3590 |      0.4444 |

Naive Bayes has the highest accuracy, but Logistic Regression is the better baseline because it has stronger macro and weighted F1 scores and performs better across minority classes.

See [reports/model_comparison.md](reports/model_comparison.md) for the full evaluation and decision-making notes.

## Prediction

Run the interactive predictor:

```bash
python src/predict.py
```

Type an Amharic text sample and press Enter. The script prints the predicted sentiment label and confidence score. Type `quit` to exit.

## Current Baseline

The selected baseline is:

- Feature extraction: TF-IDF
- Features: word unigrams and bigrams
- Model: Logistic Regression
- Class handling: balanced class weights
- Saved artifacts: `models/logistic_model.pkl` and `models/tfidf.pkl`

## Limitations

- The dataset is imbalanced, with `neutral` as the largest class and `mixed` as the smallest class.
- Accuracy can be misleading, so macro F1 and per-class behavior should be used for model selection.
- The current preprocessing is intentionally simple and can be improved.
- The confidence score comes from the model probabilities and should not be treated as a calibrated production confidence score.

## Future Improvements To Make to It If cloned

- Add a `requirements.txt` file for easier environment setup.
- Extend `src/compare_models.py` to report macro F1, weighted F1, and per-class metrics directly.
- Add a confusion matrix report.
- Improve URL and punctuation cleanup.
- Try character n-gram TF-IDF features.
- Tune Logistic Regression hyperparameters.
- Evaluate multilingual or Amharic-focused transformer models after the baseline is stable.

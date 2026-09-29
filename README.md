# IMDB Review Sentiment Classifier

A classical machine-learning pipeline that labels movie reviews as positive or negative, packaged with a Tkinter desktop app that returns a prediction, a confidence score, and the words that drove it.

## Dataset and Problem

The project uses the IMDB Movie Reviews dataset: 50,000 English reviews, each labeled `positive` or `negative`, split evenly between the two classes (25,000 each). Reviews average about 230 words, though length varies widely (4 to 2,470 words), and 58.4% contain HTML markup such as `<br />`.

The task is binary sentiment classification: given the raw text of a review, predict whether the reviewer's opinion is favorable or unfavorable. Practical difficulties include noisy text, long and skewed review lengths, negation ("not good"), and mixed opinions within a single review.

## Approach

1. **Exploration:** checked missing values, duplicates, class balance, length distributions, HTML prevalence, and the most frequent words per class.
2. **Cleaning:** removed 418 duplicate reviews, leaving 49,582. There were no missing values or conflicting labels.
3. **Text normalization:** unescaped HTML entities, stripped tags and URLs, lowercased, merged contractions ("don't" to "dont"), shortened elongated letters ("sooooo" to "soo"), and removed punctuation and symbols.
4. **Feature engineering:** built two text representations (TF-IDF and binary bag-of-words, both with 1-2 grams and 100,000 features) plus nine numeric features: word count, character count, average word length, exclamation and question counts, uppercase ratio, HTML tag count, lexical diversity, and negation count.
5. **Modeling:** compared three classifiers (Logistic Regression, Linear SVM, Multinomial Naive Bayes) across three feature sets using 5-fold cross-validation, tuned the best configuration, and evaluated once on a held-out test set.

## Design Decisions

- **Stratified 80/20 split (seed 42):** preserves the class balance in both sets (50.2% positive) and makes results reproducible. All fitting, including the vectorizers and scaler, used training data only, which avoids leakage into the test set.
- **Duplicate removal:** duplicated reviews across the train and test sets would inflate accuracy.
- **Contraction merging:** turning "don't" into a single token keeps negation visible to the model instead of splitting it into "don" and "t".
- **Unigrams and bigrams:** bigrams capture short phrases such as "not good" and "well worth" that single words miss.
- **TF-IDF with sublinear term frequency:** damps the effect of words repeated many times in long reviews. `min_df=5` and `max_df=0.95` remove very rare and near-universal terms.
- **Binary bag-of-words as a baseline:** tests whether TF-IDF weighting adds value over simple word presence.
- **Log transform, winsorizing, and min-max scaling for numeric features:** counts were heavily skewed (for example, skewness of 29.5 for exclamation marks), so they were compressed and clipped at the 1st and 99th percentiles before scaling to a 0-1 range that matches the TF-IDF values.
- **Linear models:** high-dimensional sparse text data is close to linearly separable, and linear models train quickly and expose per-word weights, which enables the explanation feature in the app.
- **Probability calibration:** Linear SVM does not output probabilities, so it was wrapped in `CalibratedClassifierCV` to give the interface a meaningful confidence score.

## Implementation

| File | Purpose |
| --- | --- |
| `nlp.ipynb` | Exploration, cleaning, feature engineering, model comparison, tuning, evaluation, and saving of artifacts |
| `model.py` | Inference module: repeats the training-time cleaning and feature steps on a single review and returns the label, probability, and top contributing words |
| `NLP_GUI.py` | Tkinter interface with a text box, sample reviews, a probability bar, and lists of words pushing the prediction positive or negative |
| `artifacts/` | Saved vectorizers, scaler, preprocessing metadata, and the final model |

The same `clean_text` and `extract_features` functions are reused at inference, along with the saved scaling and clipping parameters, so a review typed into the app is processed exactly as training data was. Word-level explanations are computed by multiplying each feature's model coefficient by its value in the review.

## Results

Cross-validated F1 (5-fold, training set only):

| Model | TF-IDF | TF-IDF + numeric | Binary BoW |
| --- | --- | --- | --- |
| Linear SVM | 0.9143 | **0.9146** | 0.8943 |
| Logistic Regression | 0.9133 | 0.9122 | 0.9005 |
| Multinomial Naive Bayes | 0.8871 | 0.8842 | 0.8797 |

The best configuration was Linear SVM on TF-IDF plus numeric features, with `C=0.3` chosen by grid search (CV F1 0.9137). On the held-out test set of 9,917 reviews:

| Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- |
| 0.9133 | 0.9097 | 0.9184 | 0.9140 | 0.9713 |

## Key Findings

- **Feature representation mattered more than the classifier.** Moving from binary word presence to TF-IDF improved F1 by about two points for the linear models, while the gap between Linear SVM and Logistic Regression was about a tenth of a point.
- **Linear models beat Naive Bayes** by roughly 2.5 to 3 points of F1.
- **Engineered numeric features added almost nothing.** Their correlation with the label was weak (the strongest were negation count at -0.17 and question count at -0.17), and they raised the best F1 by only 0.0003, which is within cross-validation noise. Most of the signal comes from the words themselves.
- **Results were stable and did not overfit.** Cross-validation standard deviations were under 0.3 points, and test performance (F1 0.914) matches cross-validation (0.9137). Errors are balanced across classes, with 0.92 recall and 0.91 precision on both sides.
- **Remaining errors** likely come from sarcasm, mixed reviews, and long texts where sentiment shifts, which a bag-of-n-grams representation cannot fully capture. Sequence models or pretrained transformers would be natural next steps.

## Limitations

Hyperparameter tuning was run only on the best configuration from the comparison, not on every model. The test set was used for a single final evaluation, so reported test metrics are unbiased, but the model has only been evaluated on IMDB reviews and may not transfer to other domains.

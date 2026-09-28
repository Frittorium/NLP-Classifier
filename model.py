# model.py
import os, re, html
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def clean_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = text.lower()
    text = re.sub(r"n't\b", "nt", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"[^a-z0-9' ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_features(raw: pd.Series, clean: pd.Series, negations) -> pd.DataFrame:
    tokens = clean.str.split()
    n_words = tokens.str.len()
    f = pd.DataFrame(index=raw.index)
    f["n_words"] = n_words
    f["n_chars"] = clean.str.len()
    f["avg_word_len"] = f["n_chars"] / n_words.clip(lower=1)
    f["n_exclaim"] = raw.str.count("!")
    f["n_question"] = raw.str.count(r"\?")
    letters = raw.str.count(r"[A-Za-z]").clip(lower=1)
    f["upper_ratio"] = raw.str.count(r"[A-Z]") / letters
    f["n_html_tags"] = raw.str.count(r"<[^>]+>")
    f["lexical_diversity"] = tokens.map(lambda t: len(set(t)) / max(len(t), 1))
    f["n_negations"] = tokens.map(lambda t: sum(w in negations for w in t))
    return f


class SentimentModel:
    def __init__(self, art_dir: str = ART):
        self.model = joblib.load(os.path.join(art_dir, "final_model.joblib"))
        self.info = joblib.load(os.path.join(art_dir, "final_model_info.joblib"))
        self.meta = joblib.load(os.path.join(art_dir, "preprocess_meta.joblib"))
        self.uses_numeric = self.info["uses_numeric"]
        vec_file = "bow_vectorizer.joblib" if self.info["features"] == "bow" else "tfidf_vectorizer.joblib"
        self.vec = joblib.load(os.path.join(art_dir, vec_file))
        self.scaler = joblib.load(os.path.join(art_dir, "num_scaler.joblib")) if self.uses_numeric else None
        self.names = self.vec.get_feature_names_out()
        self.negations = set(self.meta["negations"])

    def _vectorize(self, raw: str):
        s_raw = pd.Series([raw])
        s_clean = s_raw.map(clean_text)
        if s_clean.iloc[0] == "":
            raise ValueError("The review has no usable text after cleaning.")
        X = self.vec.transform(s_clean)
        if self.uses_numeric:
            f = extract_features(s_raw, s_clean, self.negations)
            for c in self.meta["log_features"]:
                f[c] = np.log1p(f[c])
            f = f.clip(lower=pd.Series(self.meta["clip_low"]),
                       upper=pd.Series(self.meta["clip_high"]), axis=1)
            s = np.clip(self.scaler.transform(f[self.meta["num_features"]]), 0, 1)
            X = hstack([X, csr_matrix(s, dtype=np.float32)]).tocsr()
        return X

    def _coef(self):
        m = self.model
        if hasattr(m, "calibrated_classifiers_"):          # calibrated LinearSVC
            coefs = []
            for c in m.calibrated_classifiers_:
                est = getattr(c, "estimator", None) or getattr(c, "base_estimator", None)
                coefs.append(est.coef_.ravel())
            return np.mean(coefs, axis=0)
        if hasattr(m, "coef_"):               
            return m.coef_.ravel()
        if hasattr(m, "feature_log_prob_"):
            return m.feature_log_prob_[1] - m.feature_log_prob_[0]
        return None

    def _explain(self, X, k: int = 5):
        coef = self._coef()
        if coef is None:
            return [], []
        row = X.tocsr()[0]
        n_text = len(self.names)
        contrib = [(self.names[i], float(coef[i] * v))
                   for i, v in zip(row.indices, row.data) if i < n_text]
        pos = sorted((c for c in contrib if c[1] > 0), key=lambda c: -c[1])[:k]
        neg = sorted((c for c in contrib if c[1] < 0), key=lambda c: c[1])[:k]
        return pos, neg

    def predict(self, raw: str) -> dict:
        X = self._vectorize(raw)
        p_pos = float(self.model.predict_proba(X)[0, 1])
        pos, neg = self._explain(X)
        return {"label": "Positive" if p_pos >= 0.5 else "Negative",
                "p_pos": p_pos,
                "confidence": max(p_pos, 1 - p_pos),
                "pos_words": pos, "neg_words": neg}

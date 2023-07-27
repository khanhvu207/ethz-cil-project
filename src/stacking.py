import os

import numpy as np
import pandas as pd
import scipy
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.model_selection import cross_val_score

models = [
    "bert_attention/bert_attention",
    "albert-base_attention/albert-base_attention",
    "albert-large_attention/albert-large_attention",
    "roberta-base_attention/roberta-base_attention",
    "roberta-large_attention/roberta-large_attention",
    "timelm_attention/timelm_attention",
]

val_preds = []
test_preds = []
val_labels = None
for preds_path in models:
    val_full_path = os.path.join("outputs", preds_path, "val_pred.csv")
    test_full_path = os.path.join("outputs", preds_path, "test_pred.csv")
    val_df = pd.read_csv(val_full_path)
    test_df = pd.read_csv(test_full_path)
    val_probs = val_df["pred_probability"].to_numpy("float")
    test_probs = test_df["pred_probability"].to_numpy("float")
    labels = val_df["label"].to_numpy("int")
    val_preds.append(val_probs)
    test_preds.append(test_probs)
    if val_labels is None:
        val_labels = labels
    else:
        assert (val_labels == labels).all(), "Labels are not matched!"
    print("Loaded", preds_path)

def main(**args):
    X = np.asarray(val_preds).T
    X_test = np.asarray(test_preds).T
    y = np.asarray(val_labels)
    # model = LogisticRegression(C=5.0, random_state=42) # 0.9088
    model = RidgeClassifier(alpha=70, random_state=42) # 0.9093
    scores = cross_val_score(model, X, y, cv=5)
    print("Cross-validation score:", np.mean(scores))
    model = model.fit(X, y)
    
    stacked_pred = model.predict(X_test)
    stacked_pred = stacked_pred.astype(int) * 2 - 1
    print("Test prediction statistics")
    print("Number of negative predictions:", (stacked_pred == -1).sum())
    print("Number of positive predictions:", (stacked_pred == 1).sum())
    submission = pd.DataFrame()
    submission["Id"] = np.arange(1, stacked_pred.shape[0] + 1)
    submission["Prediction"] = stacked_pred
    submission.to_csv("outputs/stacked_submission.csv", index=False)

if __name__ == "__main__":
    import fire

    fire.Fire(main)

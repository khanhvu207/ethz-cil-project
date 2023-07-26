import os

import numpy as np
import pandas as pd
import scipy

models = [
   "bert_attention/bert_attention",
   "roberta-base_attention/roberta-base_attention", 
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


def cost_function(x):
    t = x[0]
    coeffs = x[1:]
    last = 1 - sum(coeffs)
    coeffs = np.append(coeffs, [last])
    assert len(coeffs) == len(
        val_preds
    ), "The number of coefficients is not equal to the number of models"

    stacked_pred = np.zeros_like(val_preds[0])
    for i in range(len(val_preds)):
        stacked_pred += val_preds[i] * coeffs[i]

    stacked_pred = (stacked_pred > t).astype(int)
    acc = (stacked_pred == val_labels).astype(float).mean()
    return -acc


def main(**args):
    init = [0.5]
    for i in range(len(val_preds) - 1):
        init.append(1.0 / len(val_preds))

    if "algo" not in args.keys():
        algo = "Powell"
    else:
        algo = args["algo"]
    print(f"Searching for mixing coefficients with {algo}")
    result = scipy.optimize.minimize(cost_function, init, method=algo)
    print("Best accuracy:", -cost_function(result.x))

    t = result.x[0]
    coeffs = result.x[1:]
    last = 1 - sum(coeffs)
    coeffs = np.append(coeffs, [last])
    print("Optimal decision threshold:", t)
    print("Optimal mixing coefficients:", coeffs)

    stacked_pred = np.zeros_like(test_preds[0])
    for i in range(len(test_preds)):
        stacked_pred += test_preds[i] * coeffs[i]

    stacked_pred = (stacked_pred > t).astype(int) * 2 - 1
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

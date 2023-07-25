import os
import pandas as pd
import numpy as np

def train_val_split(csv_name, ratio=0.9):
    df = pd.read_csv(f"data/{csv_name}_train.csv")
    tweets = df["tweet"].apply(str)
    indices = np.arange(len(df))
    rng = np.random.default_rng(42)
    rng.shuffle(indices)
    train_indices = indices[:int(ratio * len(indices))]
    val_indices = indices[int(ratio * len(indices)):]
    sorted_train_indices = train_indices[np.argsort(np.array([len(tweets[i]) for i in train_indices]))]
    sorted_val_indices = val_indices[np.argsort(np.array([len(tweets[i]) for i in val_indices]))]
    train_df = df.iloc[sorted_train_indices]
    val_df = df.iloc[sorted_val_indices]
    test_df = pd.read_csv(f"data/{csv_name}_test.csv")
    train_df.to_csv(f"data/train/{csv_name}.csv", index=False)
    val_df.to_csv(f"data/val/{csv_name}.csv", index=False)
    test_df.to_csv(f"data/test/{csv_name}.csv", index=False)
    print(f"{csv_name} splitted! Train: {len(train_df)}, Val: {len(train_df)}")


if __name__ == "__main__":
    datasets = [
        "raw",
        "cleaned",
        "hashtag",
        "cleaned_lemmatized", 
        "cleaned_spelling_corrected",
        "cleaned_stopword_removed"
    ]
    for name in datasets:
        train_val_split(name)
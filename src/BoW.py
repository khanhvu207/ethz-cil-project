# # Sentiment Classification Project

import numpy as np
import pandas as pd


def work(dataset_type, filename, ngram, method, model_type):
    if dataset_type == "old":
        tweets = []
        labels = []

        def load_tweets(filename, label):
            with open(filename, "r", encoding="utf-8") as f:
                for line in f:
                    tweets.append(line.rstrip())
                    labels.append(label)

        load_tweets("data/train_neg_full.txt", 0)
        load_tweets("data/train_pos_full.txt", 1)

        # Convert to NumPy array to facilitate indexing
        tweets = np.array(tweets)
        labels = np.array(labels)

        print(f"{len(tweets)} tweets loaded")
    elif dataset_type == "new":  # csv file have different data format
        train_df = pd.read_csv(filename)
        tweets = train_df["tweet"].values
        # temporary work around for empty tweet (emptied because of some cleaning)
        tweets = np.array(["I" if tweet == "" else tweet for tweet in tweets])
        labels = train_df["label"].values
        print(f"{len(tweets)} tweets loaded")
    # # Build validation set
    # We use 90% of tweets for training, and 10% for validation

    np.random.seed(1)  # Reproducibility!

    shuffled_indices = np.random.permutation(len(tweets))
    split_idx = int(0.9 * len(tweets))
    train_indices = shuffled_indices[:split_idx]
    val_indices = shuffled_indices[split_idx:]

    len(train_indices), len(val_indices)

    # # Bag-of-words baseline

    from sklearn.feature_extraction.text import (CountVectorizer,
                                                 TfidfVectorizer)

    # We only keep the 5000 most frequent words, both to reduce the computational cost and reduce overfitting
    if method == "count":
        vectorizer = CountVectorizer(ngram_range=(ngram, ngram), max_features=5000)
    elif method == "tf-idf":
        vectorizer = TfidfVectorizer(ngram_range=(ngram, ngram), max_features=5000)
    # Important: we call fit_transform on the training set, and only transform on the validation set
    X_train = vectorizer.fit_transform(tweets[train_indices])
    X_val = vectorizer.transform(tweets[val_indices])

    Y_train = labels[train_indices]
    Y_val = labels[val_indices]

    # Now we train a logistic classifier...

    if model_type == "linear":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(C=1e5, max_iter=100)
    elif model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        # if use default parameter will take forever to run
        model = RandomForestClassifier(
            n_estimators=100, max_depth=50, max_features=50, n_jobs=-1
        )
    elif model_type == "XGBoost":
        from xgboost import XGBClassifier

        model = XGBClassifier(n_estimators=100, learning_rate=0.1)
    elif model_type == "Ridge":
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=1.0)

    model.fit(X_train, Y_train)

    # this Ridge API cannot do integer prediction
    if model_type in ["Ridge"]:
        Y_train_pred = (model.predict(X_train) > 0.5).astype(np.int64)
        Y_val_pred = (model.predict(X_val) > 0.5).astype(np.int64)
    else:
        Y_train_pred = model.predict(X_train)
        Y_val_pred = model.predict(X_val)

    train_accuracy = (Y_train_pred == Y_train).mean()
    val_accuracy = (Y_val_pred == Y_val).mean()

    print(f"Accuracy (training set): {train_accuracy:.05f}")
    print(f"Accuracy (validation set): {val_accuracy:.05f}")

    # # Model interpretation
    # breakpoint()
    if model_type == "linear":
        model_features = model.coef_[0]
    elif model_type == "Ridge":
        model_features = model.coef_
    elif model_type in ["XGBoost", "random_forest"]:
        model_features = model.feature_importances_

    sorted_features = np.argsort(model_features)
    top_neg = sorted_features[:10]
    top_pos = sorted_features[-10:]

    mapping = vectorizer.get_feature_names_out()

    print("---- Top 10 negative words")
    for i in top_neg:
        print(mapping[i], model_features[i])
    print()

    print("---- Top 10 positive words")
    for i in top_pos:
        print(mapping[i], model_features[i])
    print()
    return val_accuracy


dataset_type = "new"
results = []
filenames = [
    "data/raw_train.csv",
    "data/cleaned_train.csv",
    "data/cleaned_stopword_removed_train.csv",
    "data/cleaned_lemmatized_train.csv",
]
ngrams = [1, 2]

table_header_filenames = [
    "raw",
    "cleaned",
    "cleaned_stopword_removed",
    "cleaned_lemmatized",
]
methods = ["count", "tf-idf"]
model_types = ["linear", "random_forest", "XGBoost", "Ridge"]
for filename, table_header_filename in zip(filenames, table_header_filenames):
    for ngram in ngrams:
        for method in methods:
            for model_type in model_types:
                print(
                    f"Working on {filename} dataset, {method} method, {model_type} model, ngram={ngram}"
                )
                acc = work(dataset_type, filename, ngram, method, model_type)
                results.append(
                    {
                        "File": table_header_filename,
                        "Ngram": ngram,
                        "Method": method,
                        "Model Type": model_type,
                        "Accuracy": acc,
                    }
                )

# NOTE: generated latex table have some grammar problem, just give it to chatgpt, and it will be corrected
# Create a DataFrame with the results
results_df = pd.DataFrame(results)

# Define the LaTeX table format
table_format = r"""
\begin{{table}}[htbp]
\centering
\caption{{Model Accuracy}}
\begin{{tabular}}{{cccc}}
\hline
Filename & Method & Ngram & Model Type & Accuracy \\
\hline
{0}
\hline
\end{{tabular}}
\label{{tab:model_accuracy}}
\end{{table}}
"""

# Generate the LaTeX table content
table_content = results_df.to_latex(index=False, escape=False)

# Combine the table format and content
latex_table = table_format.format(table_content)
print(latex_table)

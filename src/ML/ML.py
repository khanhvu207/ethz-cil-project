# # Sentiment Classification Project
# # ML Baseline

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load GloVe word embeddings
def load_glove_embeddings(path):
    embedding_dict = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embedding_dict[word] = vector
    return embedding_dict

# Function to convert tweets to GloVe embeddings
def tweet_to_glove_embedding(tweet, embedding_dict):
    words = tweet.split()
    embeddings = [embedding_dict[word] for word in words if word in embedding_dict]
    if len(embeddings) == 0:
        return np.zeros(len(embedding_dict[next(iter(embedding_dict))]))
    return np.mean(embeddings, axis=0)


def work(train_filename, val_filename, ngram, embedding_method, model_type):

    train_path = train_filename
    val_path = val_filename

    train_df = pd.read_csv(train_path)
    train_tweets = train_df["tweet"].values
    # temporary work around for empty tweet (emptied because of some cleaning)
    train_tweets = np.array(["I" if tweet == "" else tweet for tweet in train_tweets])
    train_labels = train_df["label"].values

    val_df = pd.read_csv(val_path)
    val_tweets = val_df["tweet"].values
    val_tweets = np.array(["I" if tweet == "" else tweet for tweet in val_tweets])
    val_labels = val_df["label"].values

    # We only keep the 5000 most frequent words, both to reduce the computational cost and reduce overfitting
    if embedding_method == 'count':
        vectorizer = CountVectorizer(
            ngram_range=(ngram, ngram), max_features=5000)
    elif embedding_method == 'tf-idf':
        vectorizer = TfidfVectorizer(
            ngram_range=(ngram, ngram), max_features=5000)
    
    if embedding_method=='count' or embedding_method=='tf-idf':
        # Important: we call fit_transform on the training set, and only transform on the validation set
        X_train = vectorizer.fit_transform(train_tweets)
        X_val = vectorizer.transform(val_tweets)

    if embedding_method == 'glove':
        glove_path = 'embedding/glove.twitter.27B.100d.txt'  # Change this to the path where you downloaded the GloVe embeddings
        glove_embeddings = load_glove_embeddings(glove_path)
        X_train = np.array([tweet_to_glove_embedding(tweet, glove_embeddings) for tweet in train_tweets])
        X_val = np.array([tweet_to_glove_embedding(tweet, glove_embeddings) for tweet in val_tweets])

    Y_train = train_labels
    Y_val = val_labels

    if model_type == 'linear':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(C=1e5, max_iter=100)
    elif model_type == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        # if use default parameter will take forever to run
        if embedding_method == 'glove':
            # glove is more complex
            model = RandomForestClassifier(
            n_estimators=100, max_depth=10, max_features=50, n_jobs=-1)
        else:
            model = RandomForestClassifier(
            n_estimators=100, max_depth=50, max_features=50, n_jobs=-1)
    elif model_type == 'XGBoost':
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=100, learning_rate=0.1)
    elif model_type == 'Ridge':
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)

    model.fit(X_train, Y_train) 

    # this Ridge API cannot do integer prediction
    if model_type in ['Ridge']:
        Y_train_pred = (model.predict(X_train) > 0.5).astype(np.int64)
        Y_val_pred = (model.predict(X_val) > 0.5).astype(np.int64)
    else:
        Y_train_pred = model.predict(X_train)
        Y_val_pred = model.predict(X_val)

    train_accuracy = (Y_train_pred == Y_train).mean()
    val_accuracy = (Y_val_pred == Y_val).mean()

    print(f'Accuracy (training set): {train_accuracy:.05f}')
    print(f'Accuracy (validation set): {val_accuracy:.05f}')

    # Model interpretation
    if embedding_method != 'glove':
        if model_type == 'linear':
            model_features = model.coef_[0]
        elif model_type == 'Ridge':
            model_features = model.coef_
        elif model_type in ['XGBoost', 'random_forest']:
            model_features = model.feature_importances_

        sorted_features = np.argsort(model_features)
        top_neg = sorted_features[:10]
        top_pos = sorted_features[-10:]

        mapping = vectorizer.get_feature_names_out()

        print('---- Top 10 negative words')
        for i in top_neg:
            print(mapping[i], model_features[i])
        print()

        print('---- Top 10 positive words')
        for i in top_pos:
            print(mapping[i], model_features[i])
        print()
    return val_accuracy

results = []
train_filenames = ['../data/train/raw.csv', '../data/train/cleaned_stopword_removed.csv']
val_filenames = ['../data/val/raw.csv', '../data/val/cleaned_stopword_removed.csv']
ngrams = [1, 2]
embedding_methods = ['count', 'tf-idf', 'glove']
model_types = ['linear', 'random_forest', 'XGBoost', 'Ridge']
for train_filename, val_filename in zip(train_filenames, val_filenames):
    for ngram in ngrams:
        for embedding_method in embedding_methods:
            for model_type in model_types:
                print(
                    f'Train on {train_filename} dataset, val on {val_filename} dataset, {embedding_method} embedding_method, {model_type} model, ngram={ngram}')
                acc = work(train_filename, val_filename, ngram, embedding_method, model_type)
                results.append({'File': train_filename, 'Ngram': ngram, 'Method': embedding_method,
                                'Model Type': model_type, 'Accuracy': acc})
print(results)

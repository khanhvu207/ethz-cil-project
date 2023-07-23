import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,Dense, Activation, MaxPool1D, LSTM, Conv1D, Flatten
from tensorflow.keras.optimizers import Adam


from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string('train_path', 'data/raw_train.csv', 'path to the training dataset')
flags.DEFINE_string('val_path', 'data/raw_train.csv', 'path to the validation dataset')
flags.DEFINE_string('embedding_path', 'embedding/glove.twitter.27B.100d.txt', 'path to the embedding file')
flags.DEFINE_boolean('partial', True, 'use partial data as testing')

flags.DEFINE_integer('num_words', 10000, 'number of words for tokenizer')
flags.DEFINE_integer('maxlen', 140, 'maximum number of words to keep in one tweet')
flags.DEFINE_integer('emb_dim', 100, 'dim of the embedding')
flags.DEFINE_string('model', 'CNN', 'model type to train')
flags.DEFINE_float('learning_rate', 1e-4, 'learning rate')
flags.DEFINE_integer('epoch', int(1e4), 'number of epochs for training')
flags.DEFINE_integer('batch_size', 256, 'number of samples per batch')

flags.DEFINE_string('save_dir', '/results/', 'the directory to save the training history')


# load and pad data
def load_csv_and_embedding(train_path, validation_path, embedding_path, emb_dim, num_words = 10000, maxlen = 140, partial = True):
    train_df = pd.read_csv(train_path)
    train_tweets = train_df["tweet"].values
    # temporary work around for empty tweet (emptied because of some cleaning)
    train_tweets = np.array(["I" if tweet == "" else tweet for tweet in train_tweets])
    train_labels = train_df["label"].values

    val_df = pd.read_csv(validation_path)
    val_tweets = val_df["tweet"].values
    val_tweets = np.array(["I" if tweet == "" else tweet for tweet in val_tweets])
    val_labels = val_df["label"].values


    np.random.seed(1)  # Reproducibility!
    # take partial data for testing purpose
    if partial == True:
        train_shuffled_indices = np.random.permutation(len(train_tweets))
        train_partial_idx = int(0.005 * len(train_tweets))
        
        train_partial_indices = train_shuffled_indices[:train_partial_idx]
        
        train_tweets = train_tweets[train_partial_indices]
        train_labels = train_labels[train_partial_indices]

        val_shuffled_indices = np.random.permutation(len(val_tweets))
        val_partial_idx = int(0.0005*len(val_tweets))

        val_partial_indices = val_shuffled_indices[:val_partial_idx]

        val_tweets = val_tweets[val_partial_indices]
        val_labels = val_labels[val_partial_indices]
    
    x_train = train_tweets
    x_val = val_tweets
    
    y_train = train_labels
    y_val = val_labels
    
    print("x_train shape", x_train.shape)
    print("y_train shape", y_train.shape)
    print("x_val shape", x_val.shape)
    print("y_val shape", y_val.shape)
    tokenizer = Tokenizer(num_words)
    tokenizer.fit_on_texts(train_tweets)
    
    x_train = tokenizer.texts_to_sequences(x_train) 
    x_train = pad_sequences(x_train, padding='post', maxlen=maxlen)
    x_val = tokenizer.texts_to_sequences(x_val) 
    x_val = pad_sequences(x_val, padding='post', maxlen=maxlen)
    
    # load embedding 
    vocab = len(tokenizer.word_index)+1
    emb_mat = np.zeros((vocab, emb_dim))
    #Initializing a zero matrix for each word, they will be compared to have their final embedding
    with open(embedding_path) as f:
      for line in f:
        word, *emb = line.split() 
        if word in tokenizer.word_index:
            ind=tokenizer.word_index[word]
            emb_mat[ind]=np.array(emb,dtype="float32")[:emb_dim]
            
    print("embeddding mat shape", emb_mat.shape)
    
    return x_train, y_train, x_val, y_val, emb_mat


class RNN_classifier():
	def __init__(self, emb_mat, maxlen, learning_rate = 1e-4, trainable=False):
		self.emb_mat = emb_mat
		self.vocab = emb_mat.shape[0]
		self.emb_dim = emb_mat.shape[1]
		self.maxlen = maxlen
		self.trainable = trainable

		self.model = self.create_model(learning_rate)

	def create_model(self, learning_rate):
		model= Sequential()
		model.add(Embedding(input_dim=self.vocab, output_dim=self.emb_dim,weights=[emb_mat], input_length=self.maxlen, trainable=self.trainable))
		model.add(MaxPool1D())
		model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences = False))
		model.add(Dense(16,activation="relu"))
		model.add(Dense(1, activation='sigmoid'))
		optimizer = Adam(learning_rate)
		model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])

		return model

class CNN_classifier():
	def __init__(self, emb_mat, maxlen, learning_rate = 1e-4, trainable=False):
		self.emb_mat = emb_mat
		self.vocab = emb_mat.shape[0]
		self.emb_dim = emb_mat.shape[1]
		self.maxlen = maxlen
		self.trainable = trainable

		self.model = self.create_model(learning_rate)

	def create_model(self, learning_rate):
		model= Sequential()
		model.add(Embedding(input_dim=self.vocab, output_dim=self.emb_dim, input_length=self.maxlen))
		model.add(Conv1D(64, 5, activation='relu'))
		model.add(MaxPool1D(5))
		model.add(Conv1D(128, 5, activation='relu'))
		model.add(MaxPool1D(5))
		model.add(Dense(16,activation="relu"))
		model.add(Flatten())
		model.add(Dense(1, activation='sigmoid'))
		optimizer = Adam(learning_rate)
		model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])

		return model


def main(_):
	x_train, y_train, x_val, y_val, emb_mat = load_csv_and_embedding(FLAGS.train_path, FLAGS.val_path, FLAGS.embedding_path, FLAGS.emb_dim, FLAGS.num_words, FLAGS.maxlen, FLAGS.partial)

	if 	FLAGS.model == 'CNN':
		model = CNN_classifier(emb_mat, FLAGS.maxlen).model

	elif FLAGS.model == 'RNN':
		model = RNN_classifier(emb_mat, FLAGS.maxlen).model

	print(model.summary()) 

	history = model.fit(x_train, y_train, epochs = FLAGS.epoch, verbose=True, batch_size=FLAGS.batch_size)

	#save the history 
	hist_df = pd.DataFrame(history.history)

	#save to json:
	hist_csv_file = FLAGS.model+'_history.csv'
	with open(hist_csv_file, mode='w') as f:
		hist_df.to_csv(f)

	#showing evaluation results
	test_score = model.evaluate(x_val, y_val)

	print(test_score)

if __name__ == '__main__':
	app.run(main)



import os
import numpy as np
import pickle
from gensim.models import KeyedVectors
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed, Dense
from sklearn.model_selection import train_test_split

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Define model paths
models = {
    "fasttext": "../models/restricted/fasttext_300.vec",
    "efnilex": "../models/restricted/efnilex_600.vec",
    "elmo": "../models/restricted/elmo_1024.vec",
    "hubert_x2": "../models/restricted/hubert.vec",
    "huspacy": "../models/restricted/huspacy.vec",
    "roberta_x2": "../models/restricted/roberta.vec",
    "hubert_de": "../models/restricted/hubert_decontex.vec",
    "roberta_de": "../models/restricted/roberta_decontex.vec",
    "roberta_agg": "../models/restricted/roberta_aggregate.vec",
    "hubert_agg": "../models/restricted/hubert_aggregate.vec"
}

train_file = '../datasets/nerkor_merged/nerkor_train.conllup'
valid_file = '../datasets/nerkor_merged/nerkor_val.conllup'
test_file = '../datasets/nerkor_merged/nerkor_test.conllup'

logging_folder = "../logs/pos"

lstm_units_list = [8, 16, 32, 64]

def read_conll_file(file_path):
    sentences = []
    pos_labels = []
    sentence_tokens = []
    sentence_tags = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('#'):
                continue
            line = line.strip()
            if not line:
                if sentence_tokens:
                    sentences.append(sentence_tokens)
                    pos_labels.append(sentence_tags)
                    sentence_tokens = []
                    sentence_tags = []
                continue
            parts = line.split('\t')
            token = parts[0]
            tag = parts[2]
            sentence_tokens.append(token)
            sentence_tags.append(tag)
        
        if sentence_tokens:
            sentences.append(sentence_tokens)
            pos_labels.append(sentence_tags)
    
    return sentences, pos_labels

def sentences_to_sequences(sentences, word_index):
    sequences = []
    for sentence in sentences:
        seq = []
        for token in sentence:
            token_lower = token#.lower()
            seq.append(word_index.get(token_lower, 0))
        sequences.append(seq)
    return sequences

def labels_to_sequences(ner_labels, label_index):
    sequences = []
    for sent in ner_labels:
        seq = [label_index[label] for label in sent]
        sequences.append(seq)
    return sequences

train_sentences, train_pos_labels = read_conll_file(train_file)
valid_sentences, valid_pos_labels = read_conll_file(valid_file)
test_sentences, test_pos_labels = read_conll_file(test_file)

for model_name, model_path in models.items():
    w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=False)
    embedding_dim = w2v_model.vector_size

    word_index = {}
    for sentence in train_sentences:
        for token in sentence:
            if token not in word_index:
                word_index[token] = len(word_index) + 1

    vocab_size = len(word_index) + 1

    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, idx in word_index.items():
        if word in w2v_model:
            embedding_matrix[idx] = w2v_model[word]
        else:
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))

    X_train = sentences_to_sequences(train_sentences, word_index)
    X_valid = sentences_to_sequences(valid_sentences, word_index)
    X_test = sentences_to_sequences(test_sentences, word_index)

    label_set = set()
    for dataset in [train_pos_labels, valid_pos_labels, test_pos_labels]:
        for sent in dataset:
            label_set.update(sent)
    label_index = {label: idx for idx, label in enumerate(sorted(label_set))}
    num_tags = len(label_index)

    Y_train = labels_to_sequences(train_pos_labels, label_index)
    Y_valid = labels_to_sequences(valid_pos_labels, label_index)
    Y_test  = labels_to_sequences(test_pos_labels, label_index)

    max_len = max(max(len(s) for s in X_train), max(len(s) for s in X_valid), max(len(s) for s in X_test))

    X_train_pad = pad_sequences(X_train, maxlen=max_len, padding='post', truncating='post')
    X_valid_pad = pad_sequences(X_valid, maxlen=max_len, padding='post', truncating='post')
    X_test_pad  = pad_sequences(X_test,  maxlen=max_len, padding='post', truncating='post')

    Y_train_pad = pad_sequences(Y_train, maxlen=max_len, padding='post', truncating='post')
    Y_valid_pad = pad_sequences(Y_valid, maxlen=max_len, padding='post', truncating='post')
    Y_test_pad  = pad_sequences(Y_test,  maxlen=max_len, padding='post', truncating='post')

    Y_train_pad = np.array([to_categorical(seq, num_classes=num_tags) for seq in Y_train_pad])
    Y_valid_pad = np.array([to_categorical(seq, num_classes=num_tags) for seq in Y_valid_pad])
    Y_test_pad  = np.array([to_categorical(seq, num_classes=num_tags) for seq in Y_test_pad])

    def build_lstm(vocab_size, embedding_dim, embedding_matrix, max_len, num_tags, lstm_units):
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size,
                            output_dim=embedding_dim,
                            weights=[embedding_matrix],
                            input_length=max_len,
                            trainable=False))
        model.add(Bidirectional(LSTM(units=lstm_units, return_sequences=True, dropout=0.5)))
        model.add(TimeDistributed(Dense(num_tags, activation='softmax')))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    for lstm_units in lstm_units_list:
        print(f"Training model: {model_name}, LSTM units: {lstm_units}")
        model = build_lstm(vocab_size, embedding_dim, embedding_matrix, max_len, num_tags, lstm_units)
        history = model.fit(X_train_pad, Y_train_pad,
                            batch_size=32,
                            epochs=5,
                            validation_data=(X_valid_pad, Y_valid_pad))

        val_accuracies = history.history['val_accuracy']
        with open(f"{logging_folder}/val/{model_name}_val_{lstm_units}.pkl", "wb") as f:
            pickle.dump(val_accuracies, f)

        test_loss, test_accuracy = model.evaluate(X_test_pad, Y_test_pad)
        test_results_path = f"{logging_folder}/test/{model_name}_test_{lstm_units}.pkl"
        with open(test_results_path, "wb") as f:
            pickle.dump(test_accuracy, f)
        print(f"Saved test accuracy to {test_results_path}")

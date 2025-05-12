import os
import pickle
import numpy as np
from gensim.models import KeyedVectors
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed, Dense
from tensorflow.keras.layers import Layer, Lambda
import tensorflow.keras.backend as K

# Disable GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Define model configurations
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

logging_folder = "../logs/ner"

lstm_units_list = [1, 2, 4, 8, 16, 32, 64]

# Load dataset function
def read_conll_file(file_path):
    sentences, ner_labels = [], []
    sentence_tokens, sentence_tags = [], []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith('#'):
                continue
            line = line.strip()
            if not line:
                if sentence_tokens:
                    sentences.append(sentence_tokens)
                    ner_labels.append(sentence_tags)
                    sentence_tokens, sentence_tags = [], []
                continue
            parts = line.split('\t')
            token, tag = parts[0], parts[5]
            sentence_tokens.append(token)
            sentence_tags.append(tag)
        if sentence_tokens:
            sentences.append(sentence_tokens)
            ner_labels.append(sentence_tags)
    return sentences, ner_labels

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

# Load dataset
train_sentences, train_ner_labels = read_conll_file(train_file)
valid_sentences, valid_ner_labels = read_conll_file(valid_file)
test_sentences, test_ner_labels = read_conll_file(test_file)

# Create label index
label_set = set(tag for sent in train_ner_labels for tag in sent)
label_index = {label: idx for idx, label in enumerate(sorted(label_set))}
num_tags = len(label_index)

def build_lstm(vocab_size, embedding_dim, embedding_matrix, max_len, num_tags, lstm_units):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix],
                  input_length=max_len, trainable=False),
        Bidirectional(LSTM(units=lstm_units, return_sequences=True, dropout=0.5)),
        TimeDistributed(Dense(num_tags, activation='softmax'))
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[masked_accuracy])
    return model

def masked_accuracy(y_true, y_pred):
    y_true_labels = K.argmax(y_true, axis=-1)
    y_pred_labels = K.argmax(y_pred, axis=-1)

    mask = K.cast(K.not_equal(y_true_labels, 0), dtype='float32')  # mask: 1 if not PAD, 0 if PAD
    matches = K.cast(K.equal(y_true_labels, y_pred_labels), dtype='float32')

    accuracy = K.sum(matches * mask) / K.sum(mask)
    return accuracy

# Iterate over models and LSTM units
for model_name, model_path in models.items():
    print(f"Loading embeddings for {model_name} from {model_path}")
    w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=False)
    embedding_dim = w2v_model.vector_size
    
    # Build Vocabulary & Create an Embedding Matrix
    word_index = {}
    for sentence in train_sentences:
        for token in sentence:
            token_lower = token#.lower()  # normalize token if required
            if token_lower not in word_index:
                word_index[token_lower] = len(word_index) + 1  # Start indices from 1

    vocab_size = len(word_index) + 1  # +1 for padding index (0)

    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, idx in word_index.items():
        if word in w2v_model:
            embedding_matrix[idx] = w2v_model[word]
        else:
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
    
    # Convert tokens and NER labels to sequences of indices.
    X_train = sentences_to_sequences(train_sentences, word_index)
    X_valid = sentences_to_sequences(valid_sentences, word_index)
    X_test = sentences_to_sequences(test_sentences, word_index)

    label_set = set()
    for dataset in [train_ner_labels, valid_ner_labels, test_ner_labels]:
        for sent in dataset:
            label_set.update(sent)
    label_index = {label: idx for idx, label in enumerate(sorted(label_set))}
    num_tags = len(label_index)

    Y_train = labels_to_sequences(train_ner_labels, label_index)
    Y_valid = labels_to_sequences(valid_ner_labels, label_index)
    Y_test  = labels_to_sequences(test_ner_labels,  label_index)
    
    # Pad sequences to the same length
    max_len = max(max(len(s) for s in X_train),
                max(len(s) for s in X_valid),
                max(len(s) for s in X_test))

    X_train_pad = pad_sequences(X_train, maxlen=max_len, padding='post', truncating='post')
    X_valid_pad = pad_sequences(X_valid, maxlen=max_len, padding='post', truncating='post')
    X_test_pad  = pad_sequences(X_test,  maxlen=max_len, padding='post', truncating='post')

    Y_train_pad = pad_sequences(Y_train, maxlen=max_len, padding='post', truncating='post')
    Y_valid_pad = pad_sequences(Y_valid, maxlen=max_len, padding='post', truncating='post')
    Y_test_pad  = pad_sequences(Y_test,  maxlen=max_len, padding='post', truncating='post')

    # Convert label sequences into one‑hot encoding.
    Y_train_pad = np.array([to_categorical(seq, num_classes=num_tags) for seq in Y_train_pad])
    Y_valid_pad = np.array([to_categorical(seq, num_classes=num_tags) for seq in Y_valid_pad])
    Y_test_pad  = np.array([to_categorical(seq, num_classes=num_tags) for seq in Y_test_pad])
    
    for lstm_units in lstm_units_list:
        print(f"Training model {model_name} with {lstm_units} LSTM units")
        model = build_lstm(vocab_size, embedding_dim, embedding_matrix, 100, num_tags, lstm_units)
        history = model.fit(X_train_pad, Y_train_pad, batch_size=32, epochs=5, validation_data=(X_valid_pad, Y_valid_pad))
        
        log_path = f"{logging_folder}/val/{model_name}_val_{lstm_units}.pkl"
        with open(log_path, "wb") as f:
            pickle.dump(history.history['val_accuracy'], f)
        print(f"Saved validation accuracies to {log_path}")

        test_loss, test_accuracy = model.evaluate(X_test_pad, Y_test_pad)
        test_results_path = f"{logging_folder}/test/{model_name}_test_{lstm_units}.pkl"
        with open(test_results_path, "wb") as f:
            pickle.dump(test_accuracy, f)
        print(f"Saved test accuracy to {test_results_path}")


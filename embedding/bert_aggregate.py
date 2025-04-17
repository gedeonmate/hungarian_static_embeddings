import pickle
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from gensim.models import KeyedVectors
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model_path = "../models/original/hubert-base-cc"
output_file = "hubert_aggregate.vec"

vocab_path = "../vocabs/final_vocab.pkl"
train_corpus = "../datasets/train_sentences.txt"

batch_size = 32  # Adjust based on your hardware/memory

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=512, truncation=True)
model = AutoModel.from_pretrained(model_path)
model.to("cuda")  # Move model to GPU
model.eval()  # Disable dropout and switch to eval mode

# Load vocabulary.
def load_vocab(vocab_path):
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    return set(vocab)

vocab = load_vocab(vocab_path)

# Read sentences and build a mapping from each vocab word to sentence indices.
sentences = []
word_to_indices = {}

with open(train_corpus, "r", encoding="utf-8") as file:
    for idx, line in enumerate(tqdm(file, desc="Reading File")):
        sentence = line.strip()
        sentences.append(sentence)
        for word in sentence.split():
            if word not in vocab:
                continue
            word_to_indices.setdefault(word, []).append(idx)

# Batch process sentences to compute embeddings.
all_embeddings = []  # List to hold embeddings for each sentence

for i in tqdm(range(0, len(sentences), batch_size), desc="Processing embeddings"):
    batch_sentences = sentences[i : i + batch_size]
    # Use the tokenizer's batch method. Special tokens are added automatically.
    encoded = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")
    encoded = {key: tensor.to("cuda") for key, tensor in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded)
    # Assume the CLS token embedding is the first token.
    batch_embeddings = outputs[0][:, 0, :].cpu().numpy()
    all_embeddings.extend(batch_embeddings)

# For each word, average the embeddings from all sentences where it appears.
vector_size = model.config.hidden_size
keyed_vectors = KeyedVectors(vector_size=vector_size)

for word, indices in tqdm(word_to_indices.items(), desc="Writing vectors"):
    # Stack embeddings for the relevant sentence indices and compute the mean.
    avg_embedding = np.mean(np.stack([all_embeddings[idx] for idx in indices]), axis=0)
    keyed_vectors.add_vector(word, avg_embedding)

# Save the aggregated vectors
keyed_vectors.save_word2vec_format(output_file, binary=False)

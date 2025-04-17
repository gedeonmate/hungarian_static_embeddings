from gensim.models import KeyedVectors
from gensim.models import fasttext
from gensim.test.utils import datapath
from elmoformanylangs import Embedder
import pickle
from tqdm import tqdm
import numpy as np
import logging

fasttext_bin_file = "../models/original/fasttext_300.bin"
efnilex_file = "../models/original/efnilex_600.w2v" 
huspacy_file = "../models/original/hu_vectors_web_lg/floret/floret_vectors.vec"
elmo_file = '../models/original/elmo'
hubert_file = "../models/original/hubert2static.vec"
roberta_file = "../models/original/roberta2static.vec"

output_folder = "../models/restricted"
vocab_file = "../vocabs/final_vocab.pkl"

# Function to restrict vocabulary with progress tracking
def restrict_vocabulary(embeddings, allowed_words):
    # Get the intersection of allowed words and the embeddings' vocabulary
    filtered_words = [word for word in allowed_words if word in embeddings.key_to_index]
    
    # Create a new KeyedVectors instance
    new_kv = KeyedVectors(vector_size=embeddings.vector_size)
    
    # Add vectors for each word with a progress bar
    for word in tqdm(filtered_words, desc="Filtering vocabulary"):
        new_kv.add_vector(word, embeddings[word])
    
    return new_kv

print("Loading vocabulary")
with open(vocab_file, "rb") as file:
    narrow_vocab = pickle.load(file)

# Load fasttext embeddings
print("Processing fasttext")
fasttext = fasttext.load_facebook_vectors(datapath(fasttext_bin_file))

# Restrict vocabulary for the embeddings and save it
restricted_fasttext = restrict_vocabulary(fasttext, narrow_vocab)
restricted_fasttext.save_word2vec_format(f"{output_folder}/fasttext_300.vec", binary=False)

# Load efnilex embeddings
print("Processing efnilex")
efnilex = KeyedVectors.load_word2vec_format(efnilex_file, binary=False)

# Restrict vocabulary for the embeddings and save it
restricted_efnilex = restrict_vocabulary(efnilex, narrow_vocab)
restricted_efnilex.save_word2vec_format(f"{output_folder}/efnilex_600.vec", binary=False)

# Load huSpacy embeddings
print("Processing huspacy")
huspacy = KeyedVectors.load_word2vec_format(huspacy_file, binary=False)

# Restrict vocabulary for the embeddings and save it
restricted_huspacy = restrict_vocabulary(huspacy, narrow_vocab)
restricted_huspacy.save_word2vec_format(f"{output_folder}/huspacy.vec", binary=False)

print("Processing elmo")

# Set logging level to suppress INFO and WARNING logs
logging.basicConfig(level=logging.ERROR)

elmo = Embedder(elmo_file)
embedding_dim = 1024 

# Create a KeyedVectors instance
kv = KeyedVectors(vector_size=embedding_dim)

# Add each word and its embedding to KeyedVectors
for word in tqdm(narrow_vocab, desc="Filtering vocabulary"):
    emb = elmo.sents2elmo([[word]])[0][0]
    kv.add_vector(word, emb)

# Save the embeddings to a file in Word2Vec format
output_file = f"{output_folder}/elmo_1024.vec"
kv.save_word2vec_format(output_file, binary=False)

# Load huBERT embeddings
print("Processing HuBERT")
hubert = KeyedVectors.load_word2vec_format(hubert_file, binary=False)

# Restrict vocabulary for the embeddings and save it
restricted_hubert = restrict_vocabulary(hubert, narrow_vocab)
restricted_hubert.save_word2vec_format(f"{output_folder}/hubert.vec", binary=False)

# Load XLM-R embeddings
print("Processing XLM-R")
roberta = KeyedVectors.load_word2vec_format(roberta_file, binary=False)

# Restrict vocabulary for the embeddings and save it
restricted_roberta = restrict_vocabulary(roberta, narrow_vocab)
restricted_roberta.save_word2vec_format(f"{output_folder}/roberta.vec", binary=False)
print(f"FINISHED PROCESSING")


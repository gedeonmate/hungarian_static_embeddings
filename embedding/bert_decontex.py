import torch
import time
import logging
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from gensim.models import KeyedVectors
import os
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#MODEL_NAME = "../models/original/hubert-base-cc"
MODEL_NAME = "../models/original/xlm-roberta-base"
OUTPUT_FILE = "roberta_decontex.vec"

vocab_path = "../vocabs/final_vocab.pkl"

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()  # Set to inference mode

with open(vocab_path, "rb") as f:
    vocab = list(pickle.load(f))

# Initialize word vectors
VECTOR_SIZE = model.config.hidden_size
kw = KeyedVectors(vector_size=VECTOR_SIZE)

# Batch embedding function
def embed_batch(words):
    encoded = tokenizer(words, padding=True, truncation=True, return_tensors="pt")
    encoded = {key: tensor.to(device) for key, tensor in encoded.items()}
    
    with torch.no_grad():
        outputs = model(**encoded)
    
    return outputs[0][:, 0, :].cpu().numpy()  # CLS token embeddings

# Process words in batches
BATCH_SIZE = 1000
PRINT_INTERVAL = 5
num_batches = (len(vocab) + BATCH_SIZE - 1) // BATCH_SIZE  # Round up division

start_time = time.time()
progress = 0

for i in tqdm(range(num_batches), desc="Embedding words"):
    batch_words = vocab[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
    batch_words = [word for word in batch_words if "#" not in word]  # Filter unwanted words

    try:
        embeddings = embed_batch(batch_words)
        kw.add_vectors(batch_words, embeddings)
    except Exception as e:
        logging.warning(f"Failed to process words: {batch_words}\nError: {e}")

    # Progress update
    if (i / num_batches * 100) - progress * PRINT_INTERVAL > PRINT_INTERVAL:
        progress += 1
        elapsed_time = time.time() - start_time
        logging.info(f"{progress * PRINT_INTERVAL}% Done. Time elapsed: {elapsed_time:.2f} sec")
        start_time = time.time()

# Save in `.vec` format
kw.save_word2vec_format(OUTPUT_FILE, binary=False) 
logging.info(f"Word embeddings saved successfully to {OUTPUT_FILE}!")

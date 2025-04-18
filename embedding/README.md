# 🧠 Static Embedding Extraction

This folder contains code for extracting static word embeddings from **huBERT** and **XLM-RoBERTa** models, as well as for applying vocabulary restriction to ensure comparability between embeddings.

## 📄 Contents

- `bert_aggregate.py`: Extracts static embeddings from huBERT/XLM-RoBERTa using the method referenced in the paper as *aggregate*.
- `bert_decontex.py`: Extracts static embeddings from huBERT/XLM-RoBERTa using the method referenced in the paper as *decontextualized*.
- `restrict_emb.py`: Aligns the vocabularies across embeddings to their shared subset.

## 💾 Pre-extracted Embeddings

The **extracted embeddings** themselves are hosted in a Hugging Face collection:

👉 [Hugging Face Collection (link coming soon)](#)

The collection includes:
- Static embeddings from huBERT and XLM-RoBERTa
- Embeddings produced via the X2Static method
- The shared vocabulary used across embedding models

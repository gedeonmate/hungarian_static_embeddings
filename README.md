# 🧠 Hungarian Static Word Embeddings

This repository contains the code and resources for the paper: **A Comparative Analysis of Static Word Embeddings for Hungarian** by *Máté Gedeon*. The paper is available here: https://arxiv.org/abs/2505.07809

## 📝 Abstract
The paper presents a comprehensive analysis of various static word embeddings for the Hungarian language, including traditional models such as Word2Vec, FastText, as well as static embeddings derived from BERT-based models using different extraction methods. We evaluate these embeddings on both intrinsic and extrinsic tasks to provide a holistic view of their performance. For intrinsic evaluation, we employ a word analogy task, which assesses the embeddings’ ability to capture semantic and syntactic relationships. Our results indicate that traditional static embeddings, particularly FastText, excel in this task, achieving high accuracy and mean reciprocal rank (MRR) scores. Among the BERT-based models, the X2Static method for extracting static embeddings demonstrates superior performance compared to decontextualized and aggregate methods, approaching the effectiveness of traditional static embeddings. For extrinsic evaluation, we utilize a bidirectional LSTM model to perform Named Entity Recognition (NER) and Part-of- Speech (POS) tagging tasks. The results reveal that embeddings derived from dynamic models, especially those extracted using the X2Static method, outperform purely static embeddings. Notably, ELMo embeddings achieve the highest accuracy in both NER and POS tagging tasks, underscoring the benefits of contextualized representations even when used in a static form. Our findings highlight the continued relevance of static word embeddings in NLP applications and the potential of advanced extraction methods to enhance the utility of BERT-based models. This piece of research contributes to the understanding of embedding performance in the Hungarian language and provides valuable insights for future developments in the field. The training scripts, evaluation codes, restricted vocabulary, and extracted embeddings will be made publicly available to support further research and reproducibility.

---

## 📁 Repository Structure

### `datasets/`
Includes the processing script of NerKor, which was used for POS and NER training.

### `embedding/`
Includes the codes used to extract static embeddings from huBERT and XLM-RoBERTa. Also includes the code used to restrict the embeddings to the vocabulary of their intersection.

### `extrinsic/`
Includes the training codes used for POS and NER. It also includes a jupyter notebook processing and visualizing the results.

### `intrinsic/`
Contains a jupyter notebook showcasing the analogy task with results.

---

## 💾 Datasets and Embeddings
The embeddings extracted from huBERT and XLM-RoBERTa, the corpus used for the X2Static training, and the used vocabulary can be found on Hugging Face.
The models can be found here: https://huggingface.co/gedeonmate/static_hungarian_bert
The dataset and used vocabulary can be found here: https://huggingface.co/datasets/gedeonmate/hun_stat_dataset




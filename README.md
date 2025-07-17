# 📘 Extractive Text Summarization using Topic Models and Seq2Seq Networks

A deep learning-based extractive summarization model that combines **LDA topic modeling** and **Seq2Seq attention networks** to extract context-rich, relevant sentences from large documents.

> 🔗 [View on GitHub](https://github.com/MagbulBasha2/Textsummarization)

---

## 📑 Table of Contents

- [🧠 Problem Statement](#-problem-statement)
- [📚 Literature Review](#-literature-review)
- [🧰 Methodology & Architecture](#-methodology--architecture)
- [🗃️ Dataset Description](#️-dataset-description)
- [⚙️ Preprocessing](#️-preprocessing)
- [🧩 Feature Engineering](#-feature-engineering)
  - [🔹 LDA Topic Modeling](#-lda-topic-modeling)
  - [🔹 FastText Word Embeddings](#-fasttext-word-embeddings)
- [🧠 Model Details](#-model-details)
  - [🔹 BiLSTM + Seq2Seq with Attention](#-bilstm--seq2seq-with-attention)
  - [🔹 Sentence Scoring](#-sentence-scoring)
- [📈 Evaluation & Metrics](#-evaluation--metrics)
- [🧪 Results](#-results)
- [🛠️ How to Run](#️-how-to-run)
- [🚀 Future Work](#-future-work)
- [👨‍💻 Contributors](#-contributors)

---

## 🧠 Problem Statement

Existing extractive summarization models fail to capture long-range dependencies and document-level semantics. They either require large datasets or ignore global topics, leading to incomplete or redundant summaries.

Our goal is to build a **hybrid model** that leverages:
- Topic distributions from **LDA**
- Semantic richness from **FastText embeddings**
- Context capturing via **BiLSTM + Seq2Seq with attention**

---

## 📚 Literature Review

Models like TextRank, SummaRuNNer, and BERTSum offer extractive solutions but:
- Ignore topic coherence.
- Are heavy (BERT) or shallow (TextRank).
- Lack diverse sentence selection.

We address this gap by integrating topic modeling into the deep network pipeline.

---

## 🧰 Methodology & Architecture

Our approach combines **two BiLSTM encoders**, one for:
- Sentence content (FastText)
- Sentence topic (LDA)

These are passed through **two parallel Seq2Seq models** with attention. Sentence saliency scores are then computed using **MLPs** and fused using a weighted score.

### 🔧 Architecture Diagram:

![Architecture](https://github.com/MagbulBasha2/Textsummarization/blob/main/architecture.png.png)


---

## 🗃️ Dataset Description

### 1. **CNN/DailyMail Dataset**
- 300,000+ news articles
- Human-written bullet-point highlights
- Used for training and ROUGE evaluation

### 2. **BBC News Summary Dataset**
- ~2,200 articles
- Categorized by topic
- Used for topic-specific evaluations

---

## ⚙️ Preprocessing

We apply:
- Sentence Tokenization
- Word Tokenization
- Stopword Removal
- Lemmatization (using spaCy)

Each document is converted into cleaned sentences for scoring.

---

## 🧩 Feature Engineering

### 🔹 LDA Topic Modeling
- Extracts latent topics per word (`t_wj`) and document (`t_D`)
- Combined into topic vectors for sentence-level topic distribution

### 🔹 FastText Word Embeddings
- Pre-trained 300D vectors
- Captures semantic similarity
- Input for sentence content encoder

---

## 🧠 Model Details

### 🔹 BiLSTM + Seq2Seq with Attention
Two separate BiLSTM layers:
- **Topic BiLSTM** → Sentence Topic Embedding `E_Ti`
- **Content BiLSTM** → Sentence Content Embedding `E_wi`

Each goes into a **Seq2Seq attention network**, then through an MLP with sigmoid activation.

### 🔹 Sentence Scoring

Each sentence is scored on:
- **SCS**: Sentence Content Score
- **STS**: Sentence Topic Score
- **SNS**: Sentence Novelty Score
- **SPS**: Sentence Position Score

These are fused into:

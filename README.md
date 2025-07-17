# 📝 Extractive Text Summarization using Topic Models and Seq2Seq Networks

This project implements a novel extractive text summarization framework that combines **Latent Dirichlet Allocation (LDA)** for topic modeling and **Sequence-to-Sequence (Seq2Seq)** networks with attention mechanisms to identify and extract the most informative sentences from a document.

> 🔗 [Project Repository on GitHub](https://github.com/MagbulBasha2/Textsummarization)

---

## 📌 Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Results](#results)
- [Future Work](#future-work)
- [Contributors](#contributors)

---

## 📚 Project Overview

Traditional extractive summarization techniques often miss semantic richness and fail to capture document-level context. To address these issues, our model combines **topic modeling** and **deep learning** to generate high-quality summaries from news articles.

The summarizer works in multiple stages:
1. Preprocesses the input article.
2. Extracts **topic vectors** using LDA.
3. Embeds words using **FastText embeddings**.
4. Encodes content and topic info via **BiLSTM** layers.
5. Scores each sentence using four metrics: **STS**, **SCS**, **SNS**, and **SPS**.
6. Selects top-ranked sentences to form the extractive summary.

---

## ⭐ Key Features

- **Dual Representation**: Captures both global topics and local semantics.
- **BiLSTM Encoders**: Separate encoders for topic and content embeddings.
- **Seq2Seq + Attention**: Calculates attention-based saliency scores.
- **Multiple Scoring Metrics**:
  - STS: Sentence Topic Score
  - SCS: Sentence Content Score
  - SNS: Sentence Novelty Score
  - SPS: Sentence Position Score
- **ROUGE Evaluation**: Benchmarked against popular algorithms.

---

## 🧠 Architecture

![ROUGE Score Comparison](https://github.com/MagbulBasha2/Textsummarization/blob/main/rouge_score_comparison.png)

---

## 🛠️ Technologies Used

- Python 3.x
- PyTorch
- Gensim
- NLTK
- spaCy
- FastText (300D)
- Matplotlib, Seaborn (for plotting)
- ROUGE metric (for evaluation)

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/MagbulBasha2/Textsummarization.git
   cd Textsummarization

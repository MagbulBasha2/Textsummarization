# 📘 Extractive Text Summarization using Topic Models and Seq2Seq Networks


This project implements a hybrid extractive summarization pipeline that combines topic modeling (LDA), word embeddings (FastText), deep sentence encoding (BiLSTM), and a Seq2Seq model with attention for sentence saliency prediction. It also benchmarks classic extractive methods (TextRank, LexRank, LSA, Luhn, KL-Sum, SumBasic) and evaluates summaries using ROUGE, BLEU, METEOR, SacreBLEU, and custom metrics (coherence, redundancy, coverage, relevance).

---

## Table of Contents

- [Project Structure](#project-structure)
- [Setup & Requirements](#setup--requirements)
- [Pipeline Overview](#pipeline-overview)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Topic Modeling with LDA](#2-topic-modeling-with-lda)
  - [3. Word Embeddings with FastText](#3-word-embeddings-with-fasttext)
  - [4. Sentence Embeddings with BiLSTM](#4-sentence-embeddings-with-bilstm)
  - [5. Dataset Preparation for Model Training](#5-dataset-preparation-for-model-training)
  - [6. Seq2Seq Model with Attention](#6-seq2seq-model-with-attention)
  - [7. Model Training Loop](#7-model-training-loop)
  - [8. Inference: Saliency Score Generation](#8-inference-saliency-score-generation)
  - [9. Hybrid Scoring and Summary Generation](#9-hybrid-scoring-and-summary-generation)
  - [10. Evaluation and Visualization](#10-evaluation-and-visualization)
- [References](#references)
- [License](#license)

---

## Project Structure

```
minorproject/
├── project2.0.ipynb         # Main Jupyter notebook (all code)
├── train.csv                # Training data
├── validation.csv           # Validation data
├── test.csv                 # Test data
├── fasttext_model.kv        # Pre-trained FastText model (KeyedVectors)
├── lda_dictionary.dict      # LDA dictionary
├── lda_model.model          # LDA model
├── checkpoints/             # Model checkpoints
├── summary_evaluation_results.csv
├── evaluation_scores_only.csv
├── average_evaluation_scores_tabular.csv
├── rouge_score_comparison.png
├── comparison_plot.png
└── README.md                # (This file)
```

---

## Setup & Requirements

1. **Clone the repository and navigate to the project folder.**

2. **Install dependencies:**
    ```bash
    pip install pandas numpy torch tqdm spacy gensim prettytable rouge-score tabulate nltk sumy sacrebleu matplotlib sentence-transformers scikit-learn
    python -m spacy download en_core_web_sm
    ```

3. **Prepare Data:**
    - Place `train.csv`, `validation.csv`, and `test.csv` in the project folder.
    - Download or train a FastText model and save as `fasttext_model.kv`.

4. **Run the Notebook:**
    - Open `project2.0.ipynb` in VS Code or Jupyter.
    - Run all cells sequentially.

---

## Pipeline Overview

Below is a step-by-step explanation of the training and inference pipeline (excluding label generation):

---

### 1. Data Preprocessing

**Goal:**  
Prepare raw articles for modeling by splitting them into sentences and words, lemmatizing, and removing stopwords.

**How it works:**
- Each article is processed using spaCy (`en_core_web_sm`), which splits the text into sentences.
- Each sentence is tokenized into words, lemmatized (converted to their base form), and filtered to remove stopwords and non-alphabetic tokens.
- Sentences with fewer than 4 words are discarded to avoid noise.
- The result:  
  - `sentence_tokenized`: List of cleaned sentences per article.  
  - `word_tokenized`: List of lists of cleaned words per sentence.

**Why:**  
This step ensures that only meaningful, normalized text is used for downstream topic modeling and embedding, improving model quality and reducing noise.

---

### 2. Topic Modeling with LDA

**Goal:**  
Capture the main topics present in each sentence and document.

**How it works:**
- All sentences from the training set are used to build a Gensim `Dictionary` (mapping words to IDs).
- Each sentence is converted to a Bag-of-Words (BoW) vector using this dictionary.
- An LDA (Latent Dirichlet Allocation) model is trained on these BoW vectors to learn a fixed number of topics (e.g., 10).
- For each document:
  - The BoW of all its sentences is aggregated to get a document-level BoW.
  - The LDA model infers a topic distribution for the document (`topic_vector_D`).
- For each sentence:
  - The LDA model infers a topic distribution for the sentence (`word_topic_vectors`).
- The final topic vector for each sentence is the sum of its own topic vector and the document topic vector (`final_topic_vectors`).

**Why:**  
Topic vectors provide a high-level semantic representation of each sentence and document, helping the model understand what each sentence is about in the context of the whole article.

---

### 3. Word Embeddings with FastText

**Goal:**  
Represent each word in a sentence as a dense vector capturing its semantic meaning.

**How it works:**
- A pre-trained FastText model is loaded.
- For each word in each sentence, its FastText embedding is retrieved.
- If a word is not in the FastText vocabulary, a zero vector is used.
- The result:  
  - `word_embeddings`: For each document, a list of lists of word vectors (one list per sentence).

**Why:**  
Word embeddings allow the model to capture the nuanced meaning of words and their relationships, which is crucial for understanding sentence content.

---

### 4. Sentence Embeddings with BiLSTM

**Goal:**  
Encode each sentence into a fixed-size vector that summarizes its content or topic.

**How it works:**
- Two separate BiLSTM encoders are defined:
  - One for topic vectors (`topic_bilstm`)
  - One for content (word) embeddings (`word_bilstm`)
- For each sentence:
  - The sequence of topic vectors (or word embeddings) is passed through the corresponding BiLSTM.
  - The final hidden states from both directions are concatenated to form the sentence embedding.
- The result:  
  - `sentence_topic_embeddings`: List of topic-based sentence embeddings per document.
  - `sentence_content_embeddings`: List of content-based sentence embeddings per document.

**Why:**  
BiLSTMs can capture the sequential context within sentences, producing richer sentence representations for both topic and content.

---

### 5. Dataset Preparation for Model Training

**Goal:**  
Prepare the data for PyTorch model training.

**How it works:**
- A custom `SentenceSaliencyDataset` is created, which returns:
  - Content embeddings (per sentence)
  - Topic embeddings (per sentence)
  - Binary labels (per sentence, from label generation)
- A `collate_fn` pads sequences in each batch to the same length for efficient batch processing.

**Why:**  
This structure allows the model to process batches of documents with variable numbers of sentences, which is common in real-world text.

---

### 6. Seq2Seq Model with Attention

**Goal:**  
Predict the importance (saliency) of each sentence using both content and topic information.

**How it works:**
- The model consists of:
  - A BiLSTM encoder and decoder (with attention) for each modality (content and topic).
  - An attention mechanism that computes a context vector for each decoder time step.
  - A multi-layer perceptron (MLP) that combines the context and decoder output to predict a saliency score (probability) for each sentence.
- Two models are trained:
  - One for content (`content_seq2seq`)
  - One for topic (`topic_seq2seq`)
- Both models are trained using Binary Cross-Entropy Loss (BCELoss) to match the predicted scores to the binary labels.

**Why:**  
The attention mechanism allows the model to focus on relevant parts of the input when making predictions, improving its ability to identify important sentences.

---

### 7. Model Training Loop

**Goal:**  
Optimize the Seq2Seq models to accurately predict sentence saliency.

**How it works:**
- For each epoch:
  - The models are set to training mode.
  - For each batch:
    - Content and topic embeddings, and labels, are moved to the device (CPU/GPU).
    - Forward pass: Models output saliency scores for each sentence.
    - Loss is computed and backpropagated.
    - Gradients are clipped to prevent exploding gradients.
    - Optimizers update the model weights.
  - After each epoch, validation loss is computed.
  - Learning rate schedulers adjust the learning rate at specified milestones.
  - Early stopping is used: If validation loss does not improve for several epochs, training stops.
  - Progress is logged to a CSV and PrettyTable for monitoring.

**Why:**  
This standard supervised training loop ensures the models learn to distinguish important from unimportant sentences, while preventing overfitting and allowing for recovery from checkpoints.

---

### 8. Inference: Saliency Score Generation

**Goal:**  
Generate saliency scores for each sentence in the dataset using the trained models.

**How it works:**
- After training, the best model checkpoints are loaded.
- For each document:
  - The trained models predict a saliency score for each sentence (content and topic separately).
- These scores are stored in the DataFrame for further use.

**Why:**  
These scores are used to rank sentences for summary generation and for further hybrid scoring.

---

### 9. Hybrid Scoring and Summary Generation

**Goal:**  
Combine multiple signals to select the best sentences for the summary.

**How it works:**
- For each sentence, four scores are computed:
  - Content saliency score (from model)
  - Topic saliency score (from model)
  - Sentence novelty score (SNS): Measures how different a sentence is from previous ones (using cosine similarity).
  - Sentence position score (SPS): Gives higher scores to earlier sentences.
- The final sentence score (FSS) is a weighted sum of these four scores.
- The top-N sentences (by FSS) are selected as the extractive summary.

**Why:**  
Combining multiple signals (content, topic, novelty, position) leads to more informative and less redundant summaries.

---

### 10. Evaluation and Visualization

**Goal:**  
Assess the quality of generated summaries.

**How it works:**
- Summaries are compared to reference summaries using:
  - ROUGE (n-gram overlap)
  - BLEU, METEOR, SacreBLEU (machine translation metrics)
  - Custom metrics: coherence, redundancy, coverage, relevance (using Sentence-BERT)
- Baseline methods (TextRank, LexRank, LSA, Luhn, KL-Sum, SumBasic) are also evaluated for comparison.
- Results are visualized in tables and bar plots.

**Why:**  
Comprehensive evaluation ensures the model’s performance is robust and competitive with standard baselines.

---

## References

- [spaCy](https://spacy.io/)
- [Gensim LDA](https://radimrehurek.com/gensim/)
- [FastText](https://fasttext.cc/)
- [Sumy](https://github.com/miso-belica/sumy)
- [NLTK](https://www.nltk.org/)
- [ROUGE Score](https://github.com/google-research/text-metrics)
- [Sentence Transformers](https://www.sbert.net/)

---

## License

This project is for academic/research use only. Please cite appropriately if used in publications.

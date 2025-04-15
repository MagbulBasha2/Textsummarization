import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, render_template_string, redirect, url_for
import spacy
from gensim import corpora
from gensim.models import LdaModel
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import os
from werkzeug.utils import secure_filename
import re


# Initialize Flask app
app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'docx', 'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Setup device for torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load spaCy model (with only tokenizer and sentencizer)
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
nlp.add_pipe("sentencizer")

# Load saved models
dictionary = corpora.Dictionary.load("lda_dictionary.dict")                # Gensim dictionary
lda_model = LdaModel.load("lda_model.model")                               # Trained LDA model
ft_model = KeyedVectors.load("fasttext_model.kv")                          # FastText model

# Define BiLSTM encoder model for content/topic encoding
class BiLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BiLSTMEncoder, self).__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True, batch_first=True)

    def forward(self, x, lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, (h_fwd, h_bwd) = self.bilstm(packed_input)
        sentence_embedding = torch.cat((h_fwd[-1], h_bwd[-1]), dim=-1)
        return sentence_embedding

# Define Seq2Seq model with attention mechanism
# Model configuration
hidden_dim = 128           # LSTM hidden units
embedding_dim = 2 * hidden_dim  # 256 - this is the dimension used in the model

# Define Seq2Seq model with attention mechanism
class Seq2SeqWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=2, bidirectional=True,
                               batch_first=True, dropout=dropout)
        self.decoder = nn.LSTM(input_dim, hidden_dim, num_layers=2, bidirectional=True,
                               batch_first=True, dropout=dropout)
        self.attention = nn.Linear(2 * hidden_dim * 2, 1)
        self.mlp_fc1 = nn.Linear(2 * hidden_dim * 2, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.mlp_fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

    def forward(self, embeddings, lengths):
        # Pack the sequence for variable length processing
        packed = nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        enc_out, _ = self.encoder(packed)
        enc_out, _ = nn.utils.rnn.pad_packed_sequence(enc_out, batch_first=True)
        dec_out, _ = self.decoder(packed)
        dec_out, _ = nn.utils.rnn.pad_packed_sequence(dec_out, batch_first=True)

        # Compute attention weights for each time step
        attn_weights = []
        for i in range(dec_out.size(1)):
            dec_i = dec_out[:, i, :].unsqueeze(1)
            attn_input = torch.cat((dec_i.repeat(1, enc_out.size(1), 1), enc_out), dim=-1)
            attn_score = self.attention(attn_input).squeeze(-1)
            attn_weights.append(torch.softmax(attn_score, dim=-1))
        attn_weights = torch.stack(attn_weights, dim=1)
        context = torch.bmm(attn_weights, enc_out)

        # Concatenate context with decoder outputs then apply the MLP
        mlp_input = torch.cat((context, dec_out), dim=-1)
        B, T, F = mlp_input.shape
        mlp_input = mlp_input.view(B * T, F)
        mlp_out = self.relu(self.bn(self.mlp_fc1(mlp_input)))
        mlp_out = self.dropout(mlp_out)
        scores = self.sigmoid(self.mlp_fc2(mlp_out)).view(B, T)
        return scores

# Load pretrained models with correct dropout values
content_seq2seq = Seq2SeqWithAttention(embedding_dim, hidden_dim, dropout=0.5).to(device)
topic_seq2seq = Seq2SeqWithAttention(embedding_dim, hidden_dim, dropout=0.3).to(device)
# Load content_seq2seq checkpoint
checkpoint = torch.load("checkpoints/content_seq2seq.pth", map_location=device)
content_seq2seq.load_state_dict(checkpoint['model_state_dict'])

# Load topic_seq2seq checkpoint
checkpoint = torch.load("checkpoints/topic_seq2seq.pth", map_location=device)
topic_seq2seq.load_state_dict(checkpoint['model_state_dict'])

content_seq2seq.eval()
topic_seq2seq.eval()
# -------- TEXT PROCESSING & FEATURE EXTRACTION -------- #
def preprocess(article):
    doc = nlp(article)
    original = [sent.text.strip() for sent in doc.sents]
    filtered, processed = [], []
    for sent in doc.sents:
        words = [token.lemma_.lower() for token in sent if not token.is_stop and token.is_alpha]
        if len(words) >= 4:
            filtered.append(sent.text.strip())
            processed.append(words)
    return filtered, processed

def get_document_topic_vector(bow):
    topic_distribution = lda_model.get_document_topics(bow, minimum_probability=0)
    return np.array([prob for _, prob in sorted(topic_distribution)])

def get_word_topic_vector(sentence):
    bow = dictionary.doc2bow(sentence)
    topic_distribution = lda_model.get_document_topics(bow, minimum_probability=0)
    return np.array([prob for _, prob in sorted(topic_distribution)])

def compute_final_topic_vectors(doc_topic, word_vectors):
    return [word + doc_topic for word in word_vectors]

def get_word_embeddings(word_sentences):
    embeddings = []
    for sentence in word_sentences:
        sent_embed = []
        for word in sentence:
            if word in ft_model:
                sent_embed.append(ft_model[word])
            else:
                sent_embed.append(np.zeros(ft_model.vector_size))
        embeddings.append(sent_embed)
    return embeddings

# -------- SCORING -------- #
def calculate_sns(embeddings):
    sns_scores = []
    for i in range(len(embeddings)):
        if i == 0:
            sns_scores.append(1.0)
        else:
            current = embeddings[i].reshape(1, -1)
            previous = np.array(embeddings[:i])
            sims = cosine_similarity(current, previous)[0]
            max_sim = np.max(sims) if len(sims) > 0 else 0
            sns_scores.append(1 - max_sim)
    return sns_scores

def calculate_sps(total, pos, paragraph_positions):
    """
    Enhanced sentence position score that considers both global position and
    paragraph-level position for better document representation
    
    Args:
        total: Total number of sentences
        pos: Current sentence position
        paragraph_positions: List of tuples containing (start, end) indices for paragraphs
    
    Returns:
        Position score between 0-1 with higher values for strategically important positions
    """
    # Find which paragraph this position belongs to
    paragraph_idx = next((i for i, (start, end) in enumerate(paragraph_positions) if start <= pos < end), 0)
    start, end = paragraph_positions[paragraph_idx]
    paragraph_size = end - start
    
    # Calculate within-paragraph position (beginning/end sentences are important)
    if paragraph_size <= 1:
        para_pos_score = 1.0
    else:
        relative_pos = (pos - start) / (paragraph_size - 1)
        # U-shaped function giving importance to beginning and end of paragraph
        para_pos_score = 1.0 - min(relative_pos, 1.0 - relative_pos) * 0.8
    
    # Global position importance (first paragraphs and last paragraphs get higher weights)
    global_para_importance = 1.0
    if len(paragraph_positions) > 1:
        relative_para_pos = paragraph_idx / (len(paragraph_positions) - 1)
        # U-shaped function giving importance to beginning and end paragraphs
        global_para_importance = 1.0 - min(relative_para_pos, 1.0 - relative_para_pos) * 0.6
    
    # Global position score (traditional)
    global_pos_score = 1 - (pos / total)
    
    # Combine scores with emphasis on paragraph structure
    return 0.4 * global_pos_score + 0.4 * para_pos_score + 0.2 * global_para_importance

def calculate_fss(scs, sts, sns, sps, alpha=0.1, beta=0.2, gamma=0.1, delta=0.6):
    """
    Calculate Final Fusion Score with balanced weights
    - scs: Content score (model predicted)
    - sts: Topic score (model predicted)
    - sns: Sentence novelty score
    - sps: Position score (enhanced)
    """
    return alpha * scs + beta * sts + gamma * sns + delta * sps

# -------- FILE HANDLING -------- #
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(file_path):
    # Get the file extension
    extension = file_path.rsplit('.', 1)[1].lower()
    
    if extension == 'txt':
        # Read text file
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif extension == 'docx':
        try:
            import docx
            doc = docx.Document(file_path)
            return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        except ImportError:
            return "Error: python-docx library not installed. Cannot process DOCX files."
    elif extension == 'pdf':
        try:
            import PyPDF2
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page_num in range(len(reader.pages)):
                    text += reader.pages[page_num].extract_text()
                return text
        except ImportError:
            return "Error: PyPDF2 library not installed. Cannot process PDF files."
    else:
        return "Unsupported file format."

def summarize_document(document):
    # --- Paragraph Segmentation ---
    paragraphs = [p.strip() for p in document.split('\n\n') if p.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [p.strip() for p in document.split('\n') if p.strip()]

    all_filtered = []       # Cleaned sentences
    all_processed = []      # Tokenized/lemmatized words per sentence
    paragraph_sentence_indices = []  # (start, end) indices for sentences per paragraph
    total_filtered_sentences = 0

    # --- Preprocess Each Paragraph ---
    for paragraph in paragraphs:
        filtered, processed = preprocess(paragraph)
        all_filtered.extend(filtered)
        all_processed.extend(processed)
        paragraph_sentence_indices.append((total_filtered_sentences, total_filtered_sentences + len(filtered)))
        total_filtered_sentences += len(filtered)

    # --- Determine Summary Length ---
    if total_filtered_sentences <= 3:
        total_summary_length = total_filtered_sentences
    elif total_filtered_sentences <= 10:
        total_summary_length = max(3, total_filtered_sentences // 2)
    elif total_filtered_sentences <= 20:
        total_summary_length = max(5, total_filtered_sentences // 3)
    else:
        total_summary_length = min(max(7, total_filtered_sentences // 4), 10)

    if total_summary_length >= total_filtered_sentences:
        return " ".join(all_filtered)

    # --- Topic Modeling and Embedding ---
    doc_bow = dictionary.doc2bow([word for sent in all_processed for word in sent])
    doc_topic_vector = get_document_topic_vector(doc_bow)
    word_topic_vectors = [get_word_topic_vector(sent) for sent in all_processed]
    final_topic_vectors = compute_final_topic_vectors(doc_topic_vector, word_topic_vectors)
    word_embeddings = get_word_embeddings(all_processed)

    # --- Define input dimensions for BiLSTM encoders ---
    topic_input_dim = len(doc_topic_vector)  # Number of topics in LDA model
    word_input_dim = ft_model.vector_size    # FastText embedding dimension

    # --- BiLSTM Sentence Encodings ---
    topic_bilstm = BiLSTMEncoder(topic_input_dim, hidden_dim).to(device)
    word_bilstm = BiLSTMEncoder(word_input_dim, hidden_dim).to(device)

    doc_E_Ti, doc_E_Wi = [], []
    for i in range(len(all_filtered)):
        # Fix warning by converting list to numpy array first
        topic_tensor = torch.tensor(np.array(final_topic_vectors[i], dtype=np.float32), 
                                  dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        # Convert word embeddings list to numpy array first
        word_embeddings_array = np.array(word_embeddings[i], dtype=np.float32)
        word_tensor = torch.tensor(word_embeddings_array, 
                                 dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            E_Ti = topic_bilstm(topic_tensor, [1]).squeeze(0).cpu().numpy()
            E_Wi = word_bilstm(word_tensor, [len(word_embeddings[i])]).squeeze(0).cpu().numpy()
        doc_E_Ti.append(E_Ti)
        doc_E_Wi.append(E_Wi)

    # --- Sentence Scoring ---
    content_tensor = torch.tensor(np.array(doc_E_Wi), dtype=torch.float32).unsqueeze(0).to(device)
    topic_tensor = torch.tensor(np.array(doc_E_Ti), dtype=torch.float32).unsqueeze(0).to(device)
    lengths = [len(all_filtered)]

    with torch.no_grad():
        content_scores = content_seq2seq(content_tensor, lengths).squeeze(0).cpu().numpy()
        topic_scores = topic_seq2seq(topic_tensor, lengths).squeeze(0).cpu().numpy()

    sns_scores = calculate_sns(doc_E_Wi)
    sps_scores = [calculate_sps(len(all_filtered), i, paragraph_sentence_indices) for i in range(len(all_filtered))]

    doc_topic_mean = np.mean(final_topic_vectors, axis=0)
    topic_rep_scores = [cosine_similarity([final_topic_vectors[i]], [doc_topic_mean])[0][0]
                        for i in range(len(all_filtered))]

    # --- Final Sentence Scores ---
    fss_scores = [
        calculate_fss(content_scores[i], topic_scores[i], sns_scores[i], sps_scores[i],
                      alpha=0.3, beta=0.3, gamma=0.2, delta=0.2)
        for i in range(len(all_filtered))
    ]

    # --- Summary Sentence Allocation Per Paragraph ---
    summary_allocation = {}
    paragraph_importance = []

    for start_idx, end_idx in paragraph_sentence_indices:
        if start_idx >= len(all_filtered):
            paragraph_importance.append(0)
            continue
        para_sentences = list(range(start_idx, min(end_idx, len(all_filtered))))
        if not para_sentences:
            paragraph_importance.append(0)
            continue
        max_score = max(fss_scores[i] for i in para_sentences)
        para_length_factor = min(1.0, (end_idx - start_idx) / 5)
        paragraph_importance.append(max_score * para_length_factor)

    total_importance = sum(paragraph_importance)
    remaining = total_summary_length

    if total_importance == 0:
        for i in range(len(paragraph_importance)):
            summary_allocation[i] = 1
    else:
        for i, importance in enumerate(paragraph_importance):
            if importance > 0 and remaining > 0:
                allocation = max(1, int(round((importance / total_importance) * total_summary_length)))
                summary_allocation[i] = min(allocation, remaining)
                remaining -= summary_allocation[i]
                if remaining <= 0:
                    break

        # Allocate remaining sentences if any
        if remaining > 0:
            sorted_paras = sorted(range(len(paragraph_importance)), key=lambda i: paragraph_importance[i], reverse=True)
            for p_idx in sorted_paras:
                if remaining <= 0:
                    break
                if p_idx in summary_allocation:
                    start, end = paragraph_sentence_indices[p_idx]
                    max_alloc = min(end - start, len(all_filtered) - start) - summary_allocation[p_idx]
                    if max_alloc > 0:
                        extra = min(remaining, max_alloc)
                        summary_allocation[p_idx] += extra
                        remaining -= extra

    # --- Select Top Sentences ---
    selected_indices = []
    for p_idx, alloc in summary_allocation.items():
        if p_idx >= len(paragraph_sentence_indices) or alloc <= 0:
            continue
        start_idx, end_idx = paragraph_sentence_indices[p_idx]
        para_sentences = list(range(start_idx, min(end_idx, len(all_filtered))))
        top_sentences = sorted(para_sentences, key=lambda i: fss_scores[i], reverse=True)[:alloc]
        selected_indices.extend(top_sentences)

    # --- Fill in Extra Sentences If Needed ---
    if len(selected_indices) < total_summary_length:
        remaining = total_summary_length - len(selected_indices)
        mask = np.ones(len(all_filtered), dtype=bool)
        mask[selected_indices] = False
        candidates = np.where(mask)[0]
        additional = sorted(candidates, key=lambda i: fss_scores[i], reverse=True)[:remaining]
        selected_indices.extend(additional)

    # --- Ensure Proper Ending (Conclusion) for Long Documents ---
    if total_filtered_sentences > 8 and len(selected_indices) < total_filtered_sentences:
        tail_range = list(range(max(0, len(all_filtered) - 3), len(all_filtered)))
        tail_candidates = [
            (i, fss_scores[i]) for i in tail_range if i not in selected_indices
        ]
        if tail_candidates:
            best_tail_idx = max(tail_candidates, key=lambda x: x[1])[0]
            if best_tail_idx not in selected_indices:
                if len(selected_indices) >= total_summary_length:
                    # Remove lowest scoring sentence to maintain limit
                    lowest_idx = min(selected_indices, key=lambda i: fss_scores[i])
                    selected_indices.remove(lowest_idx)
                selected_indices.append(best_tail_idx)

    # --- Final Summary with Sorted Sentences ---
    selected_indices = sorted(set(selected_indices))
    summary_sentences = [all_filtered[i] for i in selected_indices]

    return " ".join(summary_sentences)
# -------- FLASK ROUTES -------- #
@app.route('/', methods=['GET', 'POST'])
def index():
    summary = ""
    filename = ""
    input_text = ""
    error = ""
    input_method = "file"

    if request.method == 'POST':
        # Get input method from form
        input_method = request.form.get('input_method', 'file')

        # ---------- FILE UPLOAD MODE ----------
        if input_method == 'file':
            file = request.files.get('file')

            if not file:
                error = "No file part found."
            elif file.filename == '':
                error = "No file selected."
            elif allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                document = extract_text_from_file(filepath)
                if document.startswith("Error:"):
                    error = document
                else:
                    summary = summarize_document(document)
            else:
                allowed = ', '.join(ALLOWED_EXTENSIONS)
                error = f"Invalid file type. Allowed types: {allowed}"

        # ---------- TEXT INPUT MODE ----------
        elif input_method == 'text':
            input_text = request.form.get('input_text', '').strip()

            if not input_text:
                error = "No text provided."
            else:
                summary = summarize_document(input_text)




    # HTML frontend template
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Document Summarizer</title>
    <style>
        /* Theme Variables */
        :root {
            --bg-color: #ffffff;
            --text-color: #000000;
            --card-bg: #f9f9f9;
            --input-bg: #ffffff;
            --placeholder-color: #666666;
        }

        .dark-theme {
            --bg-color: #1e1e1e;
            --text-color: #f0f0f0;
            --card-bg: #2c2c2c;
            --input-bg: #333333;
            --placeholder-color: #aaaaaa;
        }

        body {
            font-family: Arial, sans-serif;
            margin: 40px;
            line-height: 1.6;
            background-color: var(--bg-color);
            color: var(--text-color);
            transition: background-color 0.3s, color 0.3s;
        }

        .container {
            max-width: 800px;
            margin: 0 auto;
        }

        .input-form {
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: var(--card-bg);
        }

        .form-group {
            margin-bottom: 15px;
        }

        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }

input[type="file"] {
    display: block;
    margin-bottom: 10px;
    background-color: var(--input-bg);
    color: var(--text-color);
    border: 1px solid var(--text-color); /* << Add this */
}

        input[type="file"]::file-selector-button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 5px 10px;
            cursor: pointer;
        }

textarea {
    width: 100%;
    height: 200px;
    padding: 8px;
    font-family: Arial, sans-serif;
    background-color: var(--input-bg);
    color: var(--text-color);
    border: 1px solid var(--text-color); /* << Add this */
}

        textarea::placeholder {
            color: var(--placeholder-color);
            opacity: 1;
        }

        button {
            padding: 10px 20px;
            background: #4CAF50;
            color: white;
            border: none;
            font-size: 16px;
            cursor: pointer;
            border-radius: 4px;
            margin-right: 10px;
        }

        button:hover {
            background: #45a049;
        }

        .summary {
            margin-top: 30px;
            padding: 15px;
            background-color: var(--card-bg);
            border-left: 4px solid #4CAF50;
        }

        .error {
            color: #D8000C;
            background-color: #FFD2D2;
            padding: 10px;
            margin-top: 20px;
            border-radius: 4px;
        }

        .processed-file {
            margin-top: 20px;
            font-style: italic;
            color: #666;
        }

        .footer {
            margin-top: 40px;
            text-align: center;
            font-size: 12px;
            color: #777;
        }

        .file-info {
            margin-top: 20px;
            font-size: 14px;
            color: #666;
        }

        .tabs {
            display: flex;
            border-bottom: 1px solid #ddd;
            margin-bottom: 20px;
        }

.tab {
    padding: 10px 15px;
    cursor: pointer;
    background-color: var(--card-bg);
    color: var(--text-color); /* << Add this */
    border: 1px solid #ddd;
    border-bottom: none;
    margin-right: 5px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}

        .tab.active {
            background-color: #4CAF50;
            color: white;
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        .theme-toggle {
            float: right;
            font-size: 14px;
            margin-top: -30px;
        }
    </style>

    <script>
        function switchTab(methodName) {
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            document.getElementById(methodName + '-tab').classList.add('active');

            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            document.getElementById(methodName + '-content').classList.add('active');

            document.getElementById('input-method').value = methodName;
        }

        window.onload = function() {
            const activeMethod = "{{ input_method }}";
            switchTab(activeMethod);

            // Restore theme
            if (localStorage.getItem('theme') === 'dark') {
                document.body.classList.add('dark-theme');
            }
        }

        function toggleTheme() {
            document.body.classList.toggle('dark-theme');
            localStorage.setItem('theme', document.body.classList.contains('dark-theme') ? 'dark' : 'light');
        }

        function copySummary() {
            const text = document.getElementById("summary-text").innerText;
            navigator.clipboard.writeText(text).then(function() {
                alert("Summary copied to clipboard!");
            });
        }

        function downloadSummary() {
            const text = document.getElementById("summary-text").innerText;
            const blob = new Blob([text], { type: "text/plain" });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "summary.txt";
            a.click();
            window.URL.revokeObjectURL(url);
        }
    </script>
</head>
<body>
    <div class="container">
        <h2>Extractive Document Summarizer</h2>
        <div class="theme-toggle">
            <button onclick="toggleTheme()">Toggle Theme</button>
        </div>
        <p>Upload a document or paste text to generate a concise summary.</p>

        <div class="input-form">
            <form method="post" enctype="multipart/form-data">
                <input type="hidden" id="input-method" name="input_method" value="{{ input_method }}">

                <div class="tabs">
                    <div id="file-tab" class="tab" onclick="switchTab('file')">Upload File</div>
                    <div id="text-tab" class="tab" onclick="switchTab('text')">Enter Text</div>
                </div>

                <div id="file-content" class="tab-content">
                    <div class="form-group">
                        <label for="file">Select a document:</label>
                        <input type="file" name="file" id="file" accept=".txt,.docx,.pdf">
                        <small>Supported formats: TXT, DOCX, PDF (max 16MB)</small>
                    </div>
                </div>

                <div id="text-content" class="tab-content">
                    <div class="form-group">
                        <label for="input_text">Paste your text:</label>
                        <textarea name="input_text" id="input_text" placeholder="Enter your text here...">{{ input_text }}</textarea>
                    </div>
                </div>

                <button type="submit">Generate Summary</button>
            </form>
        </div>

        {% if error %}
            <div class="error">{{ error }}</div>
        {% endif %}

        {% if filename %}
            <div class="file-info">Processed file: {{ filename }}</div>
        {% endif %}

        {% if summary %}
            <div class="summary">
                <h3>Summary:</h3>
                <p id="summary-text">{{ summary }}</p>
                <button onclick="copySummary()">Copy Summary</button>
                <button onclick="downloadSummary()">Download Summary</button>
            </div>
        {% endif %}
        <div class="footer">
            <p>Deep Learning-based Extractive Summarization System</p>
        </div>
    </div>
</body>
</html>
 """
    return render_template_string(html_template, summary=summary, filename=filename, error=error, input_text=input_text, input_method=input_method)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, render_template, request, jsonify
import regex as reg
import numpy as np
import math
from collections import Counter
import nltk
import spacy
from nltk.util import ngrams
from nltk import word_tokenize, pos_tag, sent_tokenize
from nltk.corpus import wordnet as wn, stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from gensim.models import Word2Vec
import re
import os
import heapq
import fitz  # PyMuPDF
from transformers import pipeline

def chunk_text(text, max_words=150):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i+max_words])

# --- FIX for SSL Certificate Error ---
# This line disables SSL certificate verification for Hugging Face Hub downloads.
# It's a workaround for network environments with certificate issues.
os.environ["HF_HUB_DISABLE_CERT_CHECK"] = "1"


# --- 1. Load Pre-trained Models ---
# These models are downloaded once when the app starts.
# FIX: Explicitly set the framework to "pt" (PyTorch) to avoid Keras 3 compatibility issues.
print("Loading pre-trained models...")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6", framework="pt")
question_answerer = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", framework="pt")
question_generator = pipeline("text2text-generation", model="valhalla/t5-small-qg-hl", framework="pt")
print("Models loaded successfully.")


# Load other models once at startup
try:
    nlp_spacy = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading 'en_core_web_sm' model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp_spacy = spacy.load("en_core_web_sm")

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')


app = Flask(__name__)

# FIX: Added a dictionary for POS tag descriptions to make the app self-contained
POS_TAG_DESCRIPTIONS = {
    "CC": "Coordinating conjunction", "CD": "Cardinal number", "DT": "Determiner",
    "EX": "Existential there", "FW": "Foreign word", "IN": "Preposition or subordinating conjunction",
    "JJ": "Adjective", "JJR": "Adjective, comparative", "JJS": "Adjective, superlative",
    "LS": "List item marker", "MD": "Modal", "NN": "Noun, singular or mass",
    "NNS": "Noun, plural", "NNP": "Proper noun, singular", "NNPS": "Proper noun, plural",
    "PDT": "Predeterminer", "POS": "Possessive ending", "PRP": "Personal pronoun",
    "PRP$": "Possessive pronoun", "RB": "Adverb", "RBR": "Adverb, comparative",
    "RBS": "Adverb, superlative", "RP": "Particle", "SYM": "Symbol", "TO": "to",
    "UH": "Interjection", "VB": "Verb, base form", "VBD": "Verb, past tense",
    "VBG": "Verb, gerund or present participle", "VBN": "Verb, past participle",
    "VBP": "Verb, non-3rd person singular present", "VBZ": "Verb, 3rd person singular present",
    "WDT": "Wh-determiner", "WP": "Wh-pronoun", "WP$": "Possessive wh-pronoun",
    "WRB": "Wh-adverb", ".": "Punctuation", ",": "Punctuation", ":": "Punctuation",
    "(": "Punctuation", ")": "Punctuation", "``": "Opening quotation mark", "''": "Closing quotation mark",
}


# ---------- Pages ----------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/journey")
def journey():
    return render_template("modules.html")

@app.route("/modules")
def modules():
    return render_template("modules_overview.html")

@app.route("/modules/1-nlp")
def module_nlp():
    return render_template("module_nlp.html")

@app.route("/modules/2-text-regex")
def module_text_regex():
    return render_template("module_text_regex.html")

@app.route("/modules/3-edit-distance")
def module_edit_distance():
    return render_template("module_edit_distance.html")

@app.route("/modules/4-ngram-pos-ner")
def module_ngram_pos_ner():
    return render_template("module_ngram_pos_ner.html")

@app.route("/modules/5-wsd")
def module_wsd():
    return render_template("module_wsd.html")
    
@app.route("/setup-python-nlp")
def setup_python_nlp():
    return render_template("setup_python_nlp.html")

@app.route("/modules/6-sparse-vector")
def module_sparse_vector():
    return render_template("module_sparse_vector.html")

@app.route("/modules/7-dense-vector")
def module_dense_vector():
    return render_template("module_dense_vector.html")

@app.route("/modules/8-sparse-app")
def module_sparse_app():
    return render_template("module_sparse_app.html")

@app.route("/modules/9-dense-app")
def module_dense_app():
    return render_template("module_dense_app.html")

@app.route("/pdf-study-helper")
def pdf_study_helper():
    return render_template("pdf_study_helper.html")


# --- Preprocessing function for TF-IDF ---
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
def preprocess_text_for_api(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# ---------- APIs ----------

# --- PDF Study Helper APIs ---
@app.route('/api/analyze-pdf-advanced', methods=['POST'])
def analyze_pdf_advanced():
    try:
        if 'pdf' not in request.files:
            return jsonify({'error': 'No PDF file provided'}), 400
        
        file = request.files['pdf']
        
        # 1. Extract text from PDF
        doc = fitz.open(stream=file.read(), filetype="pdf")
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()

        # Limit text size to avoid overwhelming the models
        text_for_analysis = full_text[:10000]

        # 2. Abstractive Summarization
 # 2. Abstractive Summarization (with chunking)
        summaries = []
        for chunk in chunk_text(text_for_analysis, max_words=150):  # smaller chunks ≈ safe
            try:
                result = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
                summaries.append(result[0]['summary_text'])
            except Exception as e:
                summaries.append(f"[Error summarizing chunk: {str(e)}]")

        summary = " ".join(summaries)
       

        # 3. Key Term Extraction (TF-IDF is still great for this)
        vectorizer_key = TfidfVectorizer(max_features=15, stop_words='english')
        vectorizer_key.fit_transform([preprocess_text_for_api(text_for_analysis)])
        key_terms = vectorizer_key.get_feature_names_out().tolist()

        # 4. Intelligent Question Generation
        first_passage = " ".join(text_for_analysis.split()[:300]) # Use first 300 words
        generated_qg = question_generator(first_passage)
        
        # FIX: Handle both single dict and list of dicts from the pipeline
        if isinstance(generated_qg, dict):
            generated_qg = [generated_qg] # Wrap single dict in a list
            
        questions = [res.get('generated_text', '') for res in generated_qg]


        return jsonify({
            'full_text': full_text,
            'summary': summary,
            'key_terms': key_terms,
            'questions': questions[:5] # Return top 5 questions
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/ask-document-advanced', methods=['POST'])
def ask_document_advanced():
    try:
        data = request.get_json()
        context = data['context']
        query = data['query']

        # Use the powerful question-answering model
        result = question_answerer(question=query, context=context)
        
        return jsonify({'answer': result['answer']})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# --- Module Lab APIs ---

@app.route("/api/regex/find", methods=["POST"])
def api_regex_find():
    try:
        data = request.get_json(force=True) or {}
        pattern = data.get("pattern", "")
        text = data.get("text", "")
        rx = reg.compile(pattern)
        matches = [{"match": m.group(0), "start": m.start(), "end": m.end()} for m in rx.finditer(text)]
        return jsonify({"matches": matches})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/edit-distance/calculate", methods=["POST"])
def api_edit_distance():
    try:
        data = request.get_json()
        word1, word2 = data['word1'], data['word2']
        distance = nltk.edit_distance(word1, word2)
        return jsonify({'distance': distance})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/ngram/probability", methods=["POST"])
def api_ngram_probability():
    try:
        data = request.get_json()
        corpus = data.get("corpus", "").lower()
        n = data.get("n", 2)
        phrase = data.get("phrase", "").lower()

        tokens = word_tokenize(corpus)
        vocab_size = len(set(tokens))
        
        n_grams = list(ngrams(tokens, n))
        n_minus_1_grams = list(ngrams(tokens, n - 1))
        
        n_gram_counts = Counter(n_grams)
        n_minus_1_gram_counts = Counter(n_minus_1_grams)

        test_tokens = word_tokenize(phrase)
        if len(test_tokens) < n:
             return jsonify({"error": f"Phrase must be at least {n} words long."}), 400

        test_n_gram = tuple(test_tokens[-(n):])
        prefix = test_n_gram[:-1]

        numerator = n_gram_counts[test_n_gram] + 1
        denominator = n_minus_1_gram_counts[prefix] + vocab_size
        
        prob = numerator / denominator
        
        return jsonify({ "probability": prob })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/pos-tag", methods=["POST"])
def api_pos_tag():
    try:
        data = request.get_json()
        sentence = data.get("sentence", "")
        tokens = word_tokenize(sentence)
        tags = nltk.pos_tag(tokens)
        
        detailed_tags = []
        for word, tag in tags:
            description = POS_TAG_DESCRIPTIONS.get(tag, "N/A")
            detailed_tags.append({"word": word, "tag": tag, "description": description})

        return jsonify({"tags": detailed_tags})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/ner", methods=["POST"])
def api_ner():
    try:
        data = request.get_json()
        text = data.get("text", "")
        doc = nlp_spacy(text)
        
        entities = []
        for ent in doc.ents:
            entities.append({"text": ent.text, "label": ent.label_})
            
        return jsonify({"entities": entities})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/wsd/disambiguate", methods=["POST"])
def api_wsd_disambiguate():
    try:
        data = request.get_json()
        sentence = data.get("sentence", "")
        
        tokens = word_tokenize(sentence)
        tagged_sent = pos_tag(tokens)
        lemmatizer = WordNetLemmatizer()
        
        results = []
        for word, tag in tagged_sent:
            wn_pos = None
            if tag.startswith('J'): wn_pos = wn.ADJ
            elif tag.startswith('V'): wn_pos = wn.VERB
            elif tag.startswith('N'): wn_pos = wn.NOUN
            elif tag.startswith('R'): wn_pos = wn.ADV
            if not wn_pos: continue

            lemma = lemmatizer.lemmatize(word, pos=wn_pos)
            synsets = wn.synsets(lemma, pos=wn_pos)

            if not synsets: continue

            senses = [{"name": s.name(), "definition": s.definition()} for s in synsets]
            results.append({"word": word, "senses": senses})
            
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/sparse-vectors", methods=["POST"])
def api_sparse_vectors():
    try:
        data = request.get_json()
        corpus = data.get("corpus", "").split('\n')
        
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)
        vocab = vectorizer.get_feature_names_out().tolist()
        
        return jsonify({
            "vocabulary": vocab,
            "tfidf": tfidf_matrix.toarray().tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/dense-vectors/lab", methods=["POST"])
def api_dense_vectors_lab():
    try:
        data = request.get_json()
        corpus_text = data.get("corpus", "")
        target_word = data.get("target_word", "").lower()
        
        sentences = [word_tokenize(sent.lower()) for sent in corpus_text.split('\n')]
        model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
        
        if target_word in model.wv:
            similar_words = model.wv.most_similar(target_word)
        else:
            return jsonify({"error": f"Word '{target_word}' not in vocabulary."}), 400

        return jsonify({"similar_words": similar_words})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/sparse-vectors/classify", methods=["POST"])
def api_sparse_vectors_classify():
    try:
        data = request.get_json()
        training_data_raw = data.get("training_data", "")
        query = data.get("query", "")

        lines = training_data_raw.strip().split('\n')
        labels = [line.split(',', 1)[0].strip() for line in lines]
        texts = [line.split(',', 1)[1].strip() for line in lines]

        model = make_pipeline(TfidfVectorizer(), MultinomialNB())
        model.fit(texts, labels)
        prediction = model.predict([query])[0]

        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/dense-vectors/search", methods=["POST"])
def api_dense_vectors_search():
    try:
        data = request.get_json()
        corpus_text = data.get("corpus", "")
        query = data.get("query", "")

        documents = [doc for doc in corpus_text.strip().split('\n') if doc]
        
        tokenized_docs = [word_tokenize(doc.lower()) for doc in documents]
        tokenized_query = word_tokenize(query.lower())
        
        all_sentences = tokenized_docs + [tokenized_query]
        model = Word2Vec(all_sentences, vector_size=100, window=5, min_count=1, workers=4)
        
        def get_sentence_vector(tokens):
            vectors = [model.wv[word] for word in tokens if word in model.wv]
            return np.mean(vectors, axis=0) if len(vectors) > 0 else np.zeros(model.vector_size)

        doc_vectors = np.array([get_sentence_vector(doc) for doc in tokenized_docs])
        query_vector = get_sentence_vector(tokenized_query).reshape(1, -1)
        
        similarities = cosine_similarity(query_vector, doc_vectors)[0]
        
        results = sorted(zip(documents, similarities), key=lambda item: item[1], reverse=True)
        
        return jsonify({
            "results": [{"doc": doc, "score": float(score)} for doc, score in results]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)

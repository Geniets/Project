# Learn NLP: An Interactive Learning Platform

Welcome to **Learn Natural Language Processing**, a web-based, interactive platform for learning Natural Language Processing. This application features a 9-module curriculum with hands-on labs to guide users from fundamental concepts to advanced applications like text classification and semantic search.

## Features

-   **Comprehensive Curriculum:** 9 modules covering the entire NLP pipeline.
-   **Interactive Labs:** Real-time, in-browser exercises for every concept.
-   **Zero Setup:** All NLP processing is handled on the backend.
-   **Self-Assessment:** Quizzes in each module to reinforce learning.
-   **Pdf Summarizer:** Summarizes uploaded PDFs, highlights key information, and allows users to ask questions about the content.

## Technology Stack

-   **Backend:** Python, Flask
-   **Frontend:** HTML, CSS, JavaScript
-   **NLP Libraries:** NLTK, spaCy, Scikit-learn, Gensim , Hugging Face Transformers (including DistilBART)

## Local Setup

1.  **Clone the repository and create a virtual environment:**
    ```bash
    git clone [https://github.com/your-username/learn-nlp.git](https://github.com/your-username/learn-nlp.git)
    cd learn-nlp
    python3 -m venv venv && source venv/bin/activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt 
    ```
    *(Note: `requirements.txt` should contain Flask, NLTK, spaCy, scikit-learn, gensim, pandas, and numpy)*

3.  **Download NLP models:**
    ```python
    import nltk, spacy.cli
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    spacy.cli.download('en_core_web_sm')
    ```

4.  **Run the app:**
    ```bash
    python run.py
    ```
    The application will be available at `http://127.0.0.1:5000`.

</markdown>

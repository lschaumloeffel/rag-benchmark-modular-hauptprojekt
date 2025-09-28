## Setup & Installation

### Voraussetzungen
- Python 3.13+
- Optional: Docker, OpenAI API Key

### Installation
```bash
git clone <repository-url>
cd rag-benchmark-modular

# Dependencies
pip install -r requirements.txt

# NLP-Modelle installieren
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

 **Environment Variables konfigurieren:**

Erstelle `.env` Datei im Root-Verzeichnis:
```
OPENAI_API_KEY=sk-your-openai-api-key-here
NEO4J_PASSWORD=your-neo4j-password
```

## Daten vorbereiten

### FAQ-Korpus erstellen

Erstelle `data/faq_korpus.json`:
```json
[
  {
    "id": "doc_001",
    "question": "Was ist Retrieval-Augmented Generation?",
    "answer": "RAG kombiniert Language Models mit externem Wissen für genauere Antworten.",
    "category": "RAG Basics",
    "keywords": ["RAG", "retrieval", "language model"]
  }
]
```

### Test-Fragen erstellen

Erstelle `data/fragenliste.csv`:
```csv
id,question,expected_topics,difficulty,category
q001,Erkläre mir RAG,RAG basics,easy,basic
q002,Wie funktioniert Vector-Retrieval?,vector search,medium,technical
```

### Referenz-Antworten konfigurieren

Bearbeite `config/base_config.yaml` und ergänze unter `evaluation.reference_answers`:
```yaml
evaluation:
  reference_answers:
    q001: "RAG ist eine Technik die Language Models mit externen Datenquellen verbindet."
    q002: "Vector-Retrieval nutzt Embeddings für semantische Ähnlichkeitssuche."
```

## Verwendung

### 1. Konfiguration validieren

```bash
python run_experiment.py --validate-only
```

### 2. Quick Test durchführen

```bash
python run_experiment.py --quick-test --verbose
```

### 3. Vollständiges Experiment

```bash
# Alle konfigurierten Methoden
python run_experiment.py --verbose

# Spezifische Methoden
python run_experiment.py --methods baseline,vector --verbose

# Mit custom Config
python run_experiment.py --config config/base_config.yaml --methods baseline,vector
```

### 4. Ergebnisse analysieren

```bash
# Jupyter Notebook für Visualisierung
jupyter notebook notebooks/results_analysis.ipynb
```

## Experiment-Ausgaben

Nach einem Experiment befinden sich in `results/`:

- `rag_pipeline_results_*.csv` - Detaillierte Pipeline-Ergebnisse
- `evaluation_scores_*.csv` - BLEU/ROUGE Scores pro Frage
- `final_report_*.json` - Aggregierte Statistiken
- `experiment_summary_*.json` - Kurze Zusammenfassung


### Autor
Lukas Schaumlöffel - Master Informatik (HAW Hamburg)
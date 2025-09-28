"""
Data Loader für RAG-Benchmark Experimente

Dieses Modul stellt Funktionen zum Laden und Verarbeiten von FAQ-Korpus
und Testfragen bereit.

Author: Lukas Schaumlöffel
"""

import json
import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Hauptklasse für das Laden und Verarbeiten von Experimentdaten.

    Unterstützt:
    - FAQ-Korpus aus JSON
    - Testfragen aus CSV
    - Konfigurationsdateien (YAML/JSON)
    - Validation und Error Handling
    """

    def __init__(self, base_path: str = "../data"):
        """
        Initialize DataLoader

        Args:
            base_path: Pfad zum data/ Verzeichnis
        """
        self.base_path = Path(base_path)
        self.faq_documents = None
        self.test_questions = None
        self.config = None

        # Stelle sicher, dass data/ Verzeichnis existiert
        self.base_path.mkdir(exist_ok=True)

        logger.info(f"DataLoader initialisiert mit base_path: {self.base_path}")

    def load_faq_corpus(self, filename: str = "faq_korpus.json") -> List[Dict[str, Any]]:
        """
        Lädt FAQ-Korpus aus JSON-Datei

        Args:
            filename: Name der JSON-Datei

        Returns:
            Liste von FAQ-Dokumenten mit Schema:
            {
                "id": str,
                "question": str,
                "answer": str,
                "category": str,
                "keywords": List[str]
            }
        """
        filepath = self.base_path / filename

        if not filepath.exists():
            raise FileNotFoundError(f"FAQ-Korpus nicht gefunden: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            self.faq_documents = json.load(f)

        # Validation
        self._validate_faq_documents()

        logger.info(f"FAQ-Korpus geladen: {len(self.faq_documents)} Dokumente")
        return self.faq_documents

    def load_test_questions(self, filename: str = "fragenliste.csv") -> pd.DataFrame:
        """
        Lädt Testfragen aus CSV-Datei

        Args:
            filename: Name der CSV-Datei

        Returns:
            DataFrame mit Spalten: id, question, expected_topics, difficulty, category
        """
        filepath = self.base_path / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Testfragen nicht gefunden: {filepath}")

        self.test_questions = pd.read_csv(filepath, encoding='utf-8')

        # Validation
        self._validate_test_questions()

        logger.info(f"Testfragen geladen: {len(self.test_questions)} Fragen")
        return self.test_questions

    def load_config(self, filename: str = "base_config.yaml") -> Dict[str, Any]:
        """
        Lädt Konfiguration aus YAML oder JSON

        Args:
            filename: Name der Konfigurationsdatei

        Returns:
            Konfiguration als Dictionary
        """
        config_path = Path("../../config") / filename

        if not config_path.exists():
            logger.warning(f"Konfiguration nicht gefunden: {config_path}")
            return self._get_default_config()

        if filename.endswith('.yaml') or filename.endswith('.yml'):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        else:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)

        logger.info(f"Konfiguration geladen: {config_path}")
        return self.config

    def get_documents_for_embedding(self) -> List[str]:
        """
        Bereitet Dokumente für Embedding vor (kombiniert Frage + Antwort)

        Returns:
            Liste von kombinierten Texten für Embedding
        """
        if self.faq_documents is None:
            raise ValueError("FAQ-Dokumente müssen zuerst geladen werden")

        documents = []
        for doc in self.faq_documents:
            combined_text = f"Frage: {doc['question']}\n\nAntwort: {doc['answer']}"
            documents.append(combined_text)

        return documents

    def get_document_metadata(self) -> List[Dict[str, Any]]:
        """
        Extrahiert Metadata für die Zuordnung nach Retrieval

        Returns:
            Liste von Metadata-Dictionaries
        """
        if self.faq_documents is None:
            raise ValueError("FAQ-Dokumente müssen zuerst geladen werden")

        metadata = []
        for doc in self.faq_documents:
            metadata.append({
                'id': doc['id'],
                'question': doc['question'],
                'answer': doc['answer'],
                'category': doc['category'],
                'keywords': doc['keywords']
            })

        return metadata

    def get_questions_by_difficulty(self, difficulty: str) -> pd.DataFrame:
        """
        Filtert Testfragen nach Schwierigkeitsgrad

        Args:
            difficulty: "easy", "medium", oder "hard"

        Returns:
            Gefilterte Testfragen
        """
        if self.test_questions is None:
            raise ValueError("Testfragen müssen zuerst geladen werden")

        return self.test_questions[self.test_questions['difficulty'] == difficulty]

    def export_corpus_summary(self, output_path: str = "../results/corpus_summary.json"):
        """
        Exportiert Zusammenfassung des Korpus für Dokumentation

        Args:
            output_path: Pfad für Export-Datei
        """
        if self.faq_documents is None:
            raise ValueError("FAQ-Dokumente müssen zuerst geladen werden")

        # Kategorien analysieren
        categories = {}
        for doc in self.faq_documents:
            cat = doc['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(doc['id'])

        # Schwierigkeitsgrade analysieren
        difficulty_counts = {}
        if self.test_questions is not None:
            difficulty_counts = self.test_questions['difficulty'].value_counts().to_dict()

        summary = {
            "corpus_info": {
                "total_documents": len(self.faq_documents),
                "categories": {cat: len(docs) for cat, docs in categories.items()},
                "avg_answer_length": sum(len(doc['answer'].split()) for doc in self.faq_documents) / len(
                    self.faq_documents)
            },
            "test_questions": {
                "total_questions": len(self.test_questions) if self.test_questions is not None else 0,
                "difficulty_distribution": difficulty_counts
            },
            "detailed_categories": categories
        }

        # Export
        output_dir = Path(output_path).parent
        output_dir.mkdir(exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"Korpus-Zusammenfassung exportiert: {output_path}")
        return summary

    def _validate_faq_documents(self):
        """Validiert FAQ-Dokumente Schema"""
        required_fields = ['id', 'question', 'answer', 'category', 'keywords']

        for i, doc in enumerate(self.faq_documents):
            for field in required_fields:
                if field not in doc:
                    raise ValueError(f"Dokument {i} fehlt Feld: {field}")

            # Keywords sollten Liste sein
            if not isinstance(doc['keywords'], list):
                raise ValueError(f"Dokument {doc['id']}: keywords muss Liste sein")

    def _validate_test_questions(self):
        """Validiert Testfragen Schema"""
        required_columns = ['id', 'question', 'difficulty']

        for col in required_columns:
            if col not in self.test_questions.columns:
                raise ValueError(f"Testfragen fehlt Spalte: {col}")

        # Schwierigkeitsgrade prüfen
        valid_difficulties = ['easy', 'medium', 'hard']
        invalid = self.test_questions[~self.test_questions['difficulty'].isin(valid_difficulties)]

        if len(invalid) > 0:
            logger.warning(f"Ungültige Schwierigkeitsgrade gefunden: {invalid['difficulty'].unique()}")

    def _get_default_config(self) -> Dict[str, Any]:
        """Standard-Konfiguration falls keine Datei vorhanden"""
        return {
            "data": {
                "corpus_file": "faq_korpus.json",
                "questions_file": "fragenliste.csv",
                "min_corpus_size": 10
            },
            "retrieval": {
                "vector": {
                    "model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    "top_k": 3,
                    "similarity_threshold": 0.1
                },
                "graph": {
                    "neo4j_uri": "bolt://localhost:7687",
                    "neo4j_user": "neo4j",
                    "neo4j_password": "password",
                    "traversal_depth": 2,
                    "min_entity_score": 0.1
                },
                "hybrid": {
                    "weight_vector": 0.6,
                    "weight_graph": 0.4,
                    "fusion_method": "weighted_sum"
                }
            },
            "evaluation": {
                "metrics": ["bleu", "rouge1", "rouge2", "rougeL"],
                "output_format": "csv",
                "save_detailed_results": True
            },
            "logging": {
                "level": "INFO",
                "log_retrieval_details": True,
                "timestamp_format": "%Y-%m-%d %H:%M:%S"
            }
        }


# Convenience functions für direkten Import
def load_faq_corpus(filepath: str = "../data/faq_korpus.json") -> List[Dict[str, Any]]:
    """Convenience function zum direkten Laden des FAQ-Korpus"""
    loader = DataLoader()
    return loader.load_faq_corpus(Path(filepath).name)


def load_test_questions(filepath: str = "../data/fragenliste.csv") -> pd.DataFrame:
    """Convenience function zum direkten Laden der Testfragen"""
    loader = DataLoader()
    return loader.load_test_questions(Path(filepath).name)


def load_experiment_config(config_file: str = "../config/base_config.yaml") -> Dict[str, Any]:
    """Convenience function zum direkten Laden der Konfiguration"""
    loader = DataLoader()
    return loader.load_config(Path(config_file).name)


# Example usage
if __name__ == "__main__":
    # Demo des DataLoaders
    print("=== RAG-Benchmark DataLoader Demo ===")

    loader = DataLoader()

    try:
        # FAQ-Korpus laden
        faq_docs = loader.load_faq_corpus()
        print(f"FAQ-Korpus: {len(faq_docs)} Dokumente")

        # Testfragen laden
        questions = loader.load_test_questions()
        print(f"Testfragen: {len(questions)} Fragen")

        # Konfiguration laden (mit Fallback)
        config = loader.load_config()
        print(f"Konfiguration geladen")

        # Embedding-Daten vorbereiten
        embed_texts = loader.get_documents_for_embedding()
        print(f"{len(embed_texts)} Texte für Embedding vorbereitet")

        # Korpus-Zusammenfassung
        summary = loader.export_corpus_summary()
        print(f"Korpus-Analyse exportiert")

        # Beispiel-Ausgabe
        print("\n=== BEISPIEL-DATEN ===")
        print(f"Erste Frage: {questions.iloc[0]['question']}")
        print(f"Kategorien: {list(summary['corpus_info']['categories'].keys())}")

    except Exception as e:
        print(f"Fehler: {e}")
        logger.error(f"Demo fehlgeschlagen: {e}")

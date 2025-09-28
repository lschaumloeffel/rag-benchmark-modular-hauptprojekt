"""
Text Preprocessor für RAG-Benchmark Experimente

Dieses Modul stellt Funktionen für Text-Preprocessing, Entity-Extraction und
Chunk-Strategien bereit.

Author: Lukas Schaumlöffel
"""

import re
import spacy
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class ProcessedDocument:
    """Container für preprocesste Dokumente"""
    id: str
    original_text: str
    cleaned_text: str
    entities: List[Dict[str, Any]]
    concepts: List[Dict[str, Any]]
    keywords: List[str]
    chunk_texts: List[str] = None


class TextPreprocessor:
    """
    Hauptklasse für Text-Preprocessing in RAG-Experimenten

    Features:
    - Text-Cleaning und Normalisierung
    - Entity und Concept Extraction (für Graph-Retrieval)
    - Chunking-Strategien für größere Dokumente
    - Keyword-Enhancement
    """

    def __init__(self, spacy_model: str = "de_core_news_sm"):
        """
        Initialize TextPreprocessor

        Args:
            spacy_model: Name des Spacy-Modells für NLP
        """
        self.nlp = self._load_spacy_model(spacy_model)
        self.cleaning_patterns = self._get_cleaning_patterns()

        logger.info(f"TextPreprocessor initialisiert mit Modell: {spacy_model}")

    def process_document(self,
                         document: Dict[str, Any],
                         combine_qa: bool = True,
                         extract_entities: bool = True,
                         chunk_size: Optional[int] = None) -> ProcessedDocument:
        """
        Vollständige Preprocessing eines FAQ-Dokuments

        Args:
            document: FAQ-Dokument mit 'id', 'question', 'answer', etc.
            combine_qa: Ob Frage und Antwort kombiniert werden sollen
            extract_entities: Ob Entities/Concepts extrahiert werden sollen
            chunk_size: Optional chunking (für größere Dokumente)

        Returns:
            ProcessedDocument mit allen Preprocessing-Ergebnissen
        """
        # Text vorbereiten
        if combine_qa:
            original_text = f"Frage: {document['question']}\n\nAntwort: {document['answer']}"
        else:
            original_text = document['answer']

        # Text cleaning
        cleaned_text = self.clean_text(original_text)

        # Entity/Concept extraction
        entities, concepts = [], []
        if extract_entities:
            entities, concepts = self.extract_entities_and_concepts(cleaned_text)

        # Chunking (falls gewünscht)
        chunks = None
        if chunk_size:
            chunks = self.chunk_text(cleaned_text, chunk_size)

        return ProcessedDocument(
            id=document['id'],
            original_text=original_text,
            cleaned_text=cleaned_text,
            entities=entities,
            concepts=concepts,
            keywords=document.get('keywords', []),
            chunk_texts=chunks
        )

    def clean_text(self, text: str) -> str:
        """
        Text-Cleaning und Normalisierung

        Args:
            text: Roher Input-Text

        Returns:
            Bereinigter Text
        """
        # Grundlegende Bereinigung
        text = text.strip()

        # Mehrfache Whitespaces reduzieren
        text = re.sub(r'\s+', ' ', text)

        # Spezielle Zeichen normalisieren (aus deinen Notebooks)
        text = text.replace('Ã¤', 'ä').replace('Ã¶', 'ö').replace('Ã¼', 'ü')
        text = text.replace('Ã„', 'Ä').replace('Ã–', 'Ö').replace('Ãœ', 'Ü')
        text = text.replace('ÃŸ', 'ß')

        # Anführungszeichen normalisieren
        text = re.sub(r'[""„"]', '"', text)

        # Spezielle Patterns bereinigen
        for pattern, replacement in self.cleaning_patterns.items():
            text = re.sub(pattern, replacement, text)

        return text

    def extract_entities_and_concepts(self, text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extrahiert Entities und Konzepte für Graph-Retrieval

        Basiert auf deinem Code aus 03_graph_retrieval.ipynb

        Args:
            text: Input-Text

        Returns:
            Tuple von (entities, concepts)
        """
        if self.nlp is None:
            logger.warning("Kein Spacy-Modell verfügbar - überspringe Entity-Extraction")
            return [], []

        doc = self.nlp(text)

        # Named Entities extrahieren
        entities = []
        for ent in doc.ents:
            if len(ent.text.strip()) > 2:  # Mindestlänge
                entities.append({
                    'text': ent.text.strip(),
                    'label': ent.label_,
                    'type': 'ENTITY',
                    'start': ent.start_char,
                    'end': ent.end_char
                })

        # Wichtige Noun Phrases (Konzepte) extrahieren
        concepts = []
        for chunk in doc.noun_chunks:
            text_clean = chunk.text.strip().lower()

            # Filter für relevante Konzepte (aus deinem Code adaptiert)
            if (len(text_clean) > 3 and
                    not text_clean.startswith(('der', 'die', 'das', 'ein', 'eine')) and
                    text_clean not in ['frage', 'antwort', 'system', 'methode']):
                concepts.append({
                    'text': text_clean,
                    'label': 'CONCEPT',
                    'type': 'CONCEPT',
                    'start': chunk.start_char,
                    'end': chunk.end_char
                })

        # Deduplizierung
        entities = self._deduplicate_extractions(entities)
        concepts = self._deduplicate_extractions(concepts)

        logger.debug(f"Extrahiert: {len(entities)} Entities, {len(concepts)} Concepts")

        return entities, concepts

    def chunk_text(self, text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
        """
        Chunking für größere Dokumente

        Args:
            text: Input-Text
            chunk_size: Maximale Anzahl Wörter pro Chunk
            overlap: Überlappung zwischen Chunks in Wörtern

        Returns:
            Liste von Text-Chunks
        """
        words = text.split()

        if len(words) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            chunks.append(chunk_text)

            # Nächster Start mit Überlappung
            start = end - overlap
            if start >= len(words) - overlap:
                break

        logger.debug(f"Text in {len(chunks)} Chunks aufgeteilt")
        return chunks

    def enhance_keywords(self, document: Dict[str, Any], extracted_concepts: List[Dict[str, Any]]) -> List[str]:
        """
        Erweitert vorhandene Keywords um extrahierte Konzepte

        Args:
            document: Original FAQ-Dokument
            extracted_concepts: Extrahierte Konzepte

        Returns:
            Erweiterte Keyword-Liste
        """
        original_keywords = document.get('keywords', [])

        # Konzepte zu Keywords hinzufügen
        concept_keywords = [concept['text'] for concept in extracted_concepts]

        # Kombinieren und deduplizieren (case-insensitive)
        all_keywords = original_keywords + concept_keywords
        unique_keywords = []
        seen = set()

        for keyword in all_keywords:
            key_lower = keyword.lower()
            if key_lower not in seen:
                seen.add(key_lower)
                unique_keywords.append(keyword)

        return unique_keywords

    def batch_process_documents(self,
                                documents: List[Dict[str, Any]],
                                **kwargs) -> List[ProcessedDocument]:
        """
        Verarbeitet eine Liste von Dokumenten im Batch

        Args:
            documents: Liste von FAQ-Dokumenten
            **kwargs: Parameter für process_document()

        Returns:
            Liste von ProcessedDocument Objekten
        """
        processed_docs = []

        for i, doc in enumerate(documents):
            try:
                processed = self.process_document(doc, **kwargs)
                processed_docs.append(processed)

                if i % 5 == 0:
                    logger.info(f"Verarbeitet: {i + 1}/{len(documents)} Dokumente")

            except Exception as e:
                logger.error(f"Fehler bei Dokument {doc.get('id', i)}: {e}")
                continue

        logger.info(f"Batch-Processing abgeschlossen: {len(processed_docs)} Dokumente")
        return processed_docs

    def prepare_for_vector_retrieval(self, documents: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Bereitet Dokumente für Vector-Retrieval vor

        Args:
            documents: FAQ-Dokumente

        Returns:
            Tuple von (embedding_texts, metadata)
        """
        embedding_texts = []
        metadata = []

        for doc in documents:
            # Kombinierter Text für Embedding (wie in deinen Notebooks)
            combined_text = f"Frage: {doc['question']}\n\nAntwort: {doc['answer']}"
            cleaned_text = self.clean_text(combined_text)

            embedding_texts.append(cleaned_text)
            metadata.append({
                'id': doc['id'],
                'question': doc['question'],
                'answer': doc['answer'],
                'category': doc['category'],
                'keywords': doc['keywords']
            })

        logger.info(f"Vector-Retrieval Vorbereitung: {len(embedding_texts)} Texte")
        return embedding_texts, metadata

    def prepare_for_graph_retrieval(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Bereitet Dokumente für Graph-Retrieval vor (mit Entity-Extraction)

        Args:
            documents: FAQ-Dokumente

        Returns:
            Erweiterte Dokumente mit Entities und Concepts
        """
        graph_documents = []

        for doc in documents:
            processed = self.process_document(doc, extract_entities=True)

            # Erweitere Original-Dokument um Preprocessing-Ergebnisse
            graph_doc = doc.copy()
            graph_doc.update({
                'cleaned_text': processed.cleaned_text,
                'entities': processed.entities,
                'concepts': processed.concepts,
                'enhanced_keywords': self.enhance_keywords(doc, processed.concepts)
            })

            graph_documents.append(graph_doc)

        logger.info(f"Graph-Retrieval Vorbereitung: {len(graph_documents)} Dokumente")
        return graph_documents

    def _load_spacy_model(self, model_name: str):
        """Lädt Spacy-Modell mit Fallback"""
        try:
            nlp = spacy.load(model_name)
            logger.info(f"Spacy-Modell geladen: {model_name}")
            return nlp
        except OSError:
            # Fallback auf englisches Modell
            try:
                nlp = spacy.load("en_core_web_sm")
                logger.warning(f"Fallback auf en_core_web_sm (Original: {model_name} nicht gefunden)")
                return nlp
            except OSError:
                logger.error("Kein Spacy-Modell gefunden. Installation erforderlich:")
                logger.error("python -m spacy download de_core_news_sm")
                logger.error("python -m spacy download en_core_web_sm")
                return None

    def _get_cleaning_patterns(self) -> Dict[str, str]:
        """Definiert Regex-Patterns für Text-Cleaning"""
        return {
            r'https?://[^\s]+': '',  # URLs entfernen
            r'\b\w{1,2}\b': '',  # Sehr kurze Wörter entfernen
            r'[^\w\s\-.,!?äöüÄÖÜß]': '',  # Sonderzeichen (außer deutschen)
            r'\.{2,}': '.',  # Mehrfache Punkte
            r'\s*\n\s*': ' ',  # Zeilenumbrüche durch Leerzeichen
        }

    def _deduplicate_extractions(self, extractions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Entfernt Duplikate aus Entity/Concept-Listen"""
        unique_items = []
        seen = set()

        for item in extractions:
            key = item['text'].lower()
            if key not in seen:
                seen.add(key)
                unique_items.append(item)

        return unique_items

    def get_statistics(self, processed_docs: List[ProcessedDocument]) -> Dict[str, Any]:
        """
        Generiert Statistiken über preprocesste Dokumente

        Args:
            processed_docs: Liste von ProcessedDocument Objekten

        Returns:
            Statistik-Dictionary für Dokumentation
        """
        if not processed_docs:
            return {}

        total_entities = sum(len(doc.entities) for doc in processed_docs)
        total_concepts = sum(len(doc.concepts) for doc in processed_docs)
        total_keywords = sum(len(doc.keywords) for doc in processed_docs)

        text_lengths = [len(doc.cleaned_text.split()) for doc in processed_docs]

        stats = {
            "preprocessing_stats": {
                "total_documents": len(processed_docs),
                "total_entities": total_entities,
                "total_concepts": total_concepts,
                "total_keywords": total_keywords,
                "avg_entities_per_doc": total_entities / len(processed_docs),
                "avg_concepts_per_doc": total_concepts / len(processed_docs),
                "avg_text_length": sum(text_lengths) / len(text_lengths),
                "min_text_length": min(text_lengths),
                "max_text_length": max(text_lengths)
            },
            "entity_types": self._get_entity_type_distribution(processed_docs),
            "concept_frequency": self._get_concept_frequency(processed_docs)
        }

        return stats

    def _get_entity_type_distribution(self, processed_docs: List[ProcessedDocument]) -> Dict[str, int]:
        """Analysiert Verteilung der Entity-Typen"""
        entity_types = {}

        for doc in processed_docs:
            for entity in doc.entities:
                entity_type = entity['label']
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1

        return entity_types

    def _get_concept_frequency(self, processed_docs: List[ProcessedDocument]) -> Dict[str, int]:
        """Analysiert Häufigkeit von Konzepten"""
        concept_freq = {}

        for doc in processed_docs:
            for concept in doc.concepts:
                concept_text = concept['text']
                concept_freq[concept_text] = concept_freq.get(concept_text, 0) + 1

        # Sortiere nach Häufigkeit
        return dict(sorted(concept_freq.items(), key=lambda x: x[1], reverse=True))


class ChunkingStrategy:
    """
    Verschiedene Chunking-Strategien für Dokumente
    """

    @staticmethod
    def word_based_chunking(text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
        """Word-basiertes Chunking mit Überlappung"""
        words = text.split()

        if len(words) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            chunks.append(chunk_text)

            start = end - overlap
            if start >= len(words) - overlap:
                break

        return chunks

    @staticmethod
    def sentence_based_chunking(text: str, max_sentences: int = 3) -> List[str]:
        """Satz-basiertes Chunking"""
        # Einfache Satz-Trennung
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        for i in range(0, len(sentences), max_sentences):
            chunk_sentences = sentences[i:i + max_sentences]
            chunk_text = '. '.join(chunk_sentences) + '.'
            chunks.append(chunk_text)

        return chunks


# Convenience functions für direkten Import
def preprocess_faq_corpus(documents: List[Dict[str, Any]],
                          spacy_model: str = "de_core_news_sm",
                          **kwargs) -> List[ProcessedDocument]:
    """Convenience function für Batch-Preprocessing"""
    preprocessor = TextPreprocessor(spacy_model)
    return preprocessor.batch_process_documents(documents, **kwargs)


def prepare_documents_for_retrieval(documents: List[Dict[str, Any]],
                                    method: str = "vector") -> Any:
    """
    Convenience function zur Vorbereitung für spezifische Retrieval-Methoden

    Args:
        documents: FAQ-Dokumente
        method: "vector" oder "graph"

    Returns:
        Vorbereitete Daten je nach Methode
    """
    preprocessor = TextPreprocessor()

    if method == "vector":
        return preprocessor.prepare_for_vector_retrieval(documents)
    elif method == "graph":
        return preprocessor.prepare_for_graph_retrieval(documents)
    else:
        raise ValueError(f"Unbekannte Methode: {method}")


# Example usage
if __name__ == "__main__":
    # Demo des Preprocessors
    print("=== RAG-Benchmark TextPreprocessor Demo ===")

    # Beispiel-Dokument
    sample_doc = {
        "id": "doc_test",
        "question": "Was ist Vector-Retrieval?",
        "answer": "Vector-Retrieval nutzt Embeddings und FAISS für schnelle Ähnlichkeitssuche.",
        "category": "Vector Retrieval",
        "keywords": ["vector", "embeddings", "FAISS"]
    }

    try:
        preprocessor = TextPreprocessor()

        # Einzelnes Dokument verarbeiten
        processed = preprocessor.process_document(sample_doc)
        print(f"Processed Document ID: {processed.id}")
        print(f"Entities: {len(processed.entities)}")
        print(f"Concepts: {len(processed.concepts)}")
        print(f"Cleaned Text: {processed.cleaned_text[:100]}...")

        # Für Vector-Retrieval vorbereiten
        embed_texts, metadata = preprocessor.prepare_for_vector_retrieval([sample_doc])
        print(f"Vector-Preparation: {len(embed_texts)} Texte")

    except Exception as e:
        print(f"Demo fehlgeschlagen: {e}")
        logger.error(f"Demo fehlgeschlagen: {e}")

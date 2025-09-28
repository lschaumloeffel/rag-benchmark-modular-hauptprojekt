"""
Vector Retrieval Module für RAG-Benchmark Experimente

Implementiert FAISS-basiertes Vector Retrieval mit Sentence Transformers.

Author: Lukas Schaumlöffel
"""

import numpy as np
import faiss
import pickle
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Container für Retrieval-Ergebnisse"""
    query: str
    retrieved_docs: List[Dict[str, Any]]
    scores: List[float]
    retrieval_time: float
    num_retrieved: int


class VectorRetrieval:
    """
    FAISS-basiertes Vector Retrieval System

    Features:
    - Sentence Transformer Embeddings
    - FAISS-Index für schnelle Suche
    - Cosinus-Ähnlichkeit mit Normalisierung
    - Konfigurierbare Parameter
    - Performance-Tracking
    """

    def __init__(self,
                 model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                 index_type: str = "flat",
                 similarity_metric: str = "cosine"):
        """
        Initialize VectorRetrieval

        Args:
            model_name: Sentence Transformer Modell
            index_type: "flat" oder "ivf" für FAISS
            similarity_metric: "cosine" oder "euclidean"
        """
        self.model_name = model_name
        self.index_type = index_type
        self.similarity_metric = similarity_metric

        # Embedding-Modell laden
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()

        # Index und Metadata
        self.index = None
        self.document_metadata = None
        self.document_embeddings = None

        logger.info(f"VectorRetrieval initialisiert: {model_name} (dim: {self.embedding_dimension})")

    def build_index(self, documents: List[str], metadata: List[Dict[str, Any]]) -> None:
        """
        Baut FAISS-Index aus Dokumenten auf

        Args:
            documents: Liste von Dokumenttexten für Embedding
            metadata: Entsprechende Metadaten für jedes Dokument
        """
        logger.info(f"Baue Index für {len(documents)} Dokumente...")

        start_time = time.time()

        # 1. Embeddings generieren
        self.document_embeddings = self._generate_embeddings(documents)

        # 2. FAISS-Index erstellen
        self.index = self._create_faiss_index()

        # 3. Embeddings normalisieren (für Cosinus-Ähnlichkeit)
        if self.similarity_metric == "cosine":
            normalized_embeddings = self._normalize_embeddings(self.document_embeddings)
        else:
            normalized_embeddings = self.document_embeddings

        # 4. Zum Index hinzufügen
        self.index.add(normalized_embeddings.astype('float32'))

        # 5. Metadata speichern
        self.document_metadata = metadata

        build_time = time.time() - start_time
        logger.info(f"Index aufgebaut in {build_time:.2f}s - {self.index.ntotal} Vektoren")

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.0) -> RetrievalResult:
        """
        Führt Vector Retrieval für eine Query durch

        Args:
            query: Suchanfrage
            top_k: Anzahl zurückzugebender Dokumente
            min_score: Minimaler Ähnlichkeitsscore

        Returns:
            RetrievalResult mit gefundenen Dokumenten
        """
        if self.index is None:
            raise ValueError("Index muss zuerst mit build_index() erstellt werden")

        start_time = time.time()

        # 1. Query Embedding generieren
        query_embedding = self._generate_embeddings([query])

        # 2. Normalisieren falls Cosinus-Ähnlichkeit
        if self.similarity_metric == "cosine":
            query_embedding = self._normalize_embeddings(query_embedding)

        # 3. Suche im Index
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)

        # 4. Ergebnisse zusammenstellen
        retrieved_docs = []
        valid_scores = []

        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and score >= min_score:  # Gültiger Index und Score
                doc_metadata = self.document_metadata[idx].copy()
                doc_metadata['retrieval_score'] = float(score)
                retrieved_docs.append(doc_metadata)
                valid_scores.append(float(score))

        retrieval_time = time.time() - start_time

        return RetrievalResult(
            query=query,
            retrieved_docs=retrieved_docs,
            scores=valid_scores,
            retrieval_time=retrieval_time,
            num_retrieved=len(retrieved_docs)
        )

    def batch_retrieve(self, queries: List[str], **kwargs) -> List[RetrievalResult]:
        """
        Batch Retrieval für mehrere Queries

        Args:
            queries: Liste von Suchanfragen
            **kwargs: Parameter für retrieve()

        Returns:
            Liste von RetrievalResult Objekten
        """
        results = []

        for i, query in enumerate(queries):
            try:
                result = self.retrieve(query, **kwargs)
                results.append(result)

                if i % 5 == 0:
                    logger.info(f"Retrieval: {i + 1}/{len(queries)} Queries")

            except Exception as e:
                logger.error(f"Fehler bei Query '{query}': {e}")
                # Leeres Ergebnis für fehlgeschlagene Queries
                results.append(RetrievalResult(
                    query=query,
                    retrieved_docs=[],
                    scores=[],
                    retrieval_time=0.0,
                    num_retrieved=0
                ))

        logger.info(f"Batch Retrieval abgeschlossen: {len(results)} Queries")
        return results

    def save_index(self,
                   index_path: str = "../data/faiss_index.bin",
                   metadata_path: str = "../data/vector_metadata.pkl") -> None:
        """
        Speichert Index und Metadata für spätere Verwendung

        Args:
            index_path: Pfad für FAISS-Index
            metadata_path: Pfad für Pickle-Metadata
        """
        if self.index is None:
            raise ValueError("Kein Index zum Speichern vorhanden")

        # Verzeichnisse erstellen
        Path(index_path).parent.mkdir(exist_ok=True)
        Path(metadata_path).parent.mkdir(exist_ok=True)

        # FAISS Index speichern
        faiss.write_index(self.index, index_path)

        # Metadata speichern
        metadata = {
            'document_metadata': self.document_metadata,
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dimension,
            'index_type': self.index_type,
            'similarity_metric': self.similarity_metric,
            'num_documents': len(self.document_metadata) if self.document_metadata else 0
        }

        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(f"Index gespeichert: {index_path}")
        logger.info(f"Metadata gespeichert: {metadata_path}")

    def load_index(self,
                   index_path: str = "../data/faiss_index.bin",
                   metadata_path: str = "../data/vector_metadata.pkl") -> None:
        """
        Lädt gespeicherten Index und Metadata

        Args:
            index_path: Pfad zum FAISS-Index
            metadata_path: Pfad zur Pickle-Metadata
        """
        if not Path(index_path).exists():
            raise FileNotFoundError(f"Index nicht gefunden: {index_path}")

        if not Path(metadata_path).exists():
            raise FileNotFoundError(f"Metadata nicht gefunden: {metadata_path}")

        # FAISS Index laden
        self.index = faiss.read_index(index_path)

        # Metadata laden
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        self.document_metadata = metadata['document_metadata']

        # Model laden falls anders als aktuelles
        if metadata['model_name'] != self.model_name:
            logger.warning(f"Model mismatch: erwartet {metadata['model_name']}, geladen {self.model_name}")

        logger.info(f"Index geladen: {self.index.ntotal} Dokumente")

    def get_index_statistics(self) -> Dict[str, Any]:
        """
        Gibt Statistiken über den aktuellen Index zurück

        Returns:
            Statistik-Dictionary
        """
        if self.index is None:
            return {"error": "Kein Index verfügbar"}

        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "num_documents": self.index.ntotal,
            "index_type": str(type(self.index)),
            "similarity_metric": self.similarity_metric,
            "memory_usage_mb": self.index.ntotal * self.embedding_dimension * 4 / (1024 * 1024)  # float32
        }

    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generiert Embeddings mit Sentence Transformer"""
        logger.debug(f"Generiere Embeddings für {len(texts)} Texte...")

        start_time = time.time()
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
        embedding_time = time.time() - start_time

        logger.debug(f"Embeddings generiert in {embedding_time:.2f}s")
        return embeddings

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """L2-Normalisierung für Cosinus-Ähnlichkeit"""
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    def _create_faiss_index(self) -> faiss.Index:
        """Erstellt FAISS-Index basierend auf Konfiguration"""
        if self.index_type == "flat":
            if self.similarity_metric == "cosine":
                # IndexFlatIP für Inner Product (nach Normalisierung = Cosinus)
                index = faiss.IndexFlatIP(self.embedding_dimension)
            else:
                # IndexFlatL2 für L2-Distanz
                index = faiss.IndexFlatL2(self.embedding_dimension)
        else:
            raise ValueError(f"Index-Typ nicht implementiert: {self.index_type}")

        logger.debug(f"FAISS-Index erstellt: {type(index)}")
        return index


class VectorRetrievalPipeline:
    """
    High-level Pipeline für Vector Retrieval mit automatischem Setup
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Pipeline mit Konfiguration

        Args:
            config: Konfiguration mit 'retrieval.vector' Sektion
        """
        vector_config = config.get('retrieval', {}).get('vector', {})

        self.retrieval = VectorRetrieval(
            model_name=vector_config.get('model', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'),
            similarity_metric=vector_config.get('similarity', 'cosine')
        )

        self.top_k = vector_config.get('top_k', 3)
        self.min_score = vector_config.get('similarity_threshold', 0.0)

        self.config = config

    def setup_from_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Setup Pipeline aus FAQ-Dokumenten

        Args:
            documents: FAQ-Dokumente aus DataLoader
        """
        # Preprocessing für Vector Retrieval
        from src.data.preprocessor import TextPreprocessor

        preprocessor = TextPreprocessor()
        embedding_texts, metadata = preprocessor.prepare_for_vector_retrieval(documents)

        # Index aufbauen
        self.retrieval.build_index(embedding_texts, metadata)

        logger.info("Vector Retrieval Pipeline setup abgeschlossen")

    def query(self, question: str) -> RetrievalResult:
        """
        Führt Retrieval für eine Frage durch

        Args:
            question: Suchanfrage

        Returns:
            RetrievalResult
        """
        return self.retrieval.retrieve(question, self.top_k, self.min_score)

    def evaluate_on_questions(self, test_questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluiert Retrieval auf Testfragen

        Args:
            test_questions: DataFrame oder Liste mit 'id', 'question' Spalten

        Returns:
            Liste von Evaluations-Ergebnissen
        """
        results = []

        for question_data in test_questions:
            if isinstance(question_data, dict):
                q_id = question_data['id']
                question = question_data['question']
                difficulty = question_data.get('difficulty', 'unknown')
            else:
                # Pandas Row
                q_id = question_data.id
                question = question_data.question
                difficulty = question_data.difficulty

            # Retrieval durchführen
            retrieval_result = self.query(question)

            # Ergebnis für CSV-Export formatieren
            for i, (doc, score) in enumerate(zip(retrieval_result.retrieved_docs, retrieval_result.scores)):
                results.append({
                    'question_id': q_id,
                    'query': question,
                    'difficulty': difficulty,
                    'retrieved_doc_id': doc['id'],
                    'retrieval_score': score,
                    'retrieved_question': doc['question'],
                    'retrieved_category': doc['category'],
                    'rank': i + 1,
                    'retrieval_time': retrieval_result.retrieval_time
                })

        logger.info(f"Vector Retrieval evaluiert: {len(test_questions)} Fragen")
        return results

    def get_performance_stats(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Berechnet Performance-Statistiken

        Args:
            evaluation_results: Ergebnisse von evaluate_on_questions()

        Returns:
            Performance-Statistiken
        """
        if not evaluation_results:
            return {}

        import pandas as pd
        df = pd.DataFrame(evaluation_results)

        # Gruppiere nach Frage (nur beste Treffer pro Frage)
        best_per_question = df.loc[df.groupby('question_id')['retrieval_score'].idxmax()]

        stats = {
            "method": "vector",
            "num_questions": len(best_per_question),
            "avg_retrieval_score": float(best_per_question['retrieval_score'].mean()),
            "min_retrieval_score": float(best_per_question['retrieval_score'].min()),
            "max_retrieval_score": float(best_per_question['retrieval_score'].max()),
            "avg_retrieval_time": float(best_per_question['retrieval_time'].mean()),
            "success_rate": float((best_per_question['retrieval_score'] > self.min_score).mean()),
            "total_retrievals": len(evaluation_results)
        }

        # Performance by difficulty
        if 'difficulty' in best_per_question.columns:
            difficulty_stats = best_per_question.groupby('difficulty')['retrieval_score'].agg(['mean', 'count'])
            stats['by_difficulty'] = difficulty_stats.to_dict()

        return stats


# Convenience functions für direkten Import
def create_vector_retrieval(documents: List[Dict[str, Any]],
                            config: Optional[Dict[str, Any]] = None) -> VectorRetrievalPipeline:
    """
    Convenience function zur schnellen Erstellung einer Vector Retrieval Pipeline

    Args:
        documents: FAQ-Dokumente
        config: Optional Konfiguration

    Returns:
        Konfigurierte VectorRetrievalPipeline
    """
    if config is None:
        # Default config
        config = {
            'retrieval': {
                'vector': {
                    'model': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                    'top_k': 3,
                    'similarity_threshold': 0.0
                }
            }
        }

    pipeline = VectorRetrievalPipeline(config)
    pipeline.setup_from_documents(documents)

    return pipeline


def load_existing_vector_retrieval(index_path: str = "../data/faiss_index.bin",
                                   metadata_path: str = "../data/vector_metadata.pkl") -> VectorRetrieval:
    """
    Lädt bereits trainierten Vector Retrieval Index

    Args:
        index_path: Pfad zum FAISS-Index
        metadata_path: Pfad zur Metadata

    Returns:
        Konfiguriertes VectorRetrieval Objekt
    """
    # Lade Metadata um Modell-Name zu bestimmen
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    retrieval = VectorRetrieval(model_name=metadata['model_name'])
    retrieval.load_index(index_path, metadata_path)

    return retrieval


# Example usage
if __name__ == "__main__":
    # Demo des Vector Retrievals
    print("=== RAG-Benchmark Vector Retrieval Demo ===")

    # Beispiel-Dokumente
    sample_docs = [
        {
            "id": "doc_001",
            "question": "Was ist RAG?",
            "answer": "RAG kombiniert Language Models mit externem Wissen.",
            "category": "Basics",
            "keywords": ["RAG", "retrieval"]
        },
        {
            "id": "doc_002",
            "question": "Wie funktioniert FAISS?",
            "answer": "FAISS ermöglicht schnelle Vektorsuche über Ähnlichkeitsmetriken.",
            "category": "Technical",
            "keywords": ["FAISS", "vector", "similarity"]
        }
    ]

    try:
        # Pipeline erstellen und testen
        pipeline = create_vector_retrieval(sample_docs)

        # Test-Query
        test_query = "Erkläre mir RAG"
        result = pipeline.query(test_query)

        print(f"Query: {test_query}")
        print(f"Gefundene Dokumente: {result.num_retrieved}")
        print(f"Beste Ähnlichkeit: {max(result.scores) if result.scores else 0}")
        print(f"Retrieval Zeit: {result.retrieval_time:.3f}s")

        # Index-Statistiken
        stats = pipeline.retrieval.get_index_statistics()
        print(f"Index: {stats['num_documents']} Dokumente, {stats['memory_usage_mb']:.1f}MB")

    except Exception as e:
        print(f"Demo fehlgeschlagen: {e}")
        logger.error(f"Demo fehlgeschlagen: {e}")

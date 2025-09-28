"""
Hybrid Retrieval Module für RAG-Benchmark Experimente

Kombiniert Vector- und Graph-Retrieval mit konfigurierbaren Fusion-Strategien.

Author: Lukas Schaumlöffel
"""

import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HybridRetrievalResult:
    """Container für Hybrid Retrieval-Ergebnisse"""
    query: str
    retrieved_docs: List[Dict[str, Any]]
    fusion_scores: List[float]
    vector_scores: List[float]
    graph_scores: List[int]
    fusion_method: str
    retrieval_time: float
    num_retrieved: int
    vector_contribution: float
    graph_contribution: float


class HybridRetrieval:
    """
    Hybrid Retrieval System - kombiniert Vector und Graph Retrieval

    Features:
    - Gewichtete Score-Fusion
    - Rang-basierte Fusion (Reciprocal Rank Fusion)
    - Adaptive Gewichtung basierend auf Query-Eigenschaften
    - Performance-Analyse beider Komponenten
    """

    def __init__(self,
                 vector_retrieval,
                 graph_retrieval,
                 fusion_method: str = "weighted_sum",
                 vector_weight: float = 0.6,
                 graph_weight: float = 0.4,
                 rrf_k: int = 60):
        """
        Initialize HybridRetrieval

        Args:
            vector_retrieval: VectorRetrieval Instanz
            graph_retrieval: GraphRetrieval Instanz
            fusion_method: "weighted_sum", "rrf", oder "adaptive"
            vector_weight: Gewichtung für Vector-Scores
            graph_weight: Gewichtung für Graph-Scores
            rrf_k: Parameter für Reciprocal Rank Fusion
        """
        self.vector_retrieval = vector_retrieval
        self.graph_retrieval = graph_retrieval
        self.fusion_method = fusion_method
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight
        self.rrf_k = rrf_k

        # Normalisierung für Score-Fusion
        self.vector_score_range = (0.0, 1.0)  # Cosinus-Ähnlichkeit normiert
        self.graph_score_range = (0, 15)  # Empirisch basierend auf Scoring-System

        logger.info(f"HybridRetrieval initialisiert: {fusion_method} (v:{vector_weight}, g:{graph_weight})")

    def retrieve(self, query: str, top_k: int = 3) -> HybridRetrievalResult:
        """
        Führt Hybrid Retrieval für eine Query durch

        Args:
            query: Suchanfrage
            top_k: Anzahl zurückzugebender Dokumente

        Returns:
            HybridRetrievalResult mit fusionierter Rangliste
        """
        start_time = time.time()

        # 1. Beide Retrieval-Methoden ausführen
        vector_result = self.vector_retrieval.retrieve(query, top_k=top_k * 2)  # Mehr für bessere Fusion
        graph_result = self.graph_retrieval.retrieve(query, top_k=top_k * 2)

        # 2. Score-Fusion durchführen
        if self.fusion_method == "weighted_sum":
            fused_docs, fusion_scores, v_scores, g_scores = self._weighted_sum_fusion(
                vector_result, graph_result, top_k
            )
        elif self.fusion_method == "rrf":
            fused_docs, fusion_scores, v_scores, g_scores = self._reciprocal_rank_fusion(
                vector_result, graph_result, top_k
            )
        elif self.fusion_method == "adaptive":
            fused_docs, fusion_scores, v_scores, g_scores = self._adaptive_fusion(
                query, vector_result, graph_result, top_k
            )
        else:
            raise ValueError(f"Unbekannte Fusion-Methode: {self.fusion_method}")

        # 3. Beiträge der beiden Methoden berechnen
        vector_contrib, graph_contrib = self._calculate_contributions(v_scores, g_scores)

        retrieval_time = time.time() - start_time

        return HybridRetrievalResult(
            query=query,
            retrieved_docs=fused_docs,
            fusion_scores=fusion_scores,
            vector_scores=v_scores,
            graph_scores=g_scores,
            fusion_method=self.fusion_method,
            retrieval_time=retrieval_time,
            num_retrieved=len(fused_docs),
            vector_contribution=vector_contrib,
            graph_contribution=graph_contrib
        )

    def _weighted_sum_fusion(self, vector_result, graph_result, top_k: int) -> Tuple[
        List[Dict], List[float], List[float], List[int]]:
        """
        Gewichtete Summen-Fusion der Scores

        Normalisiert Vector- und Graph-Scores auf [0,1] und kombiniert gewichtet.
        """
        # Sammle alle einzigartigen Dokumente
        doc_scores = defaultdict(lambda: {'vector': 0.0, 'graph': 0})

        # Vector-Ergebnisse verarbeiten
        for doc, score in zip(vector_result.retrieved_docs, vector_result.scores):
            normalized_score = self._normalize_vector_score(score)
            doc_scores[doc['id']]['vector'] = normalized_score
            doc_scores[doc['id']]['doc'] = doc

        # Graph-Ergebnisse verarbeiten
        for doc, score in zip(graph_result.retrieved_docs, graph_result.scores):
            normalized_score = self._normalize_graph_score(score)
            doc_scores[doc['id']]['graph'] = normalized_score
            doc_scores[doc['id']]['doc'] = doc

        # Fusion-Scores berechnen
        fusion_results = []
        for doc_id, scores in doc_scores.items():
            fusion_score = (self.vector_weight * scores['vector'] +
                            self.graph_weight * scores['graph'])

            fusion_results.append({
                'doc': scores['doc'],
                'fusion_score': fusion_score,
                'vector_score': scores['vector'],
                'graph_score': scores['graph']
            })

        # Nach Fusion-Score sortieren
        fusion_results.sort(key=lambda x: x['fusion_score'], reverse=True)
        fusion_results = fusion_results[:top_k]

        # Ergebnisse extrahieren
        docs = [r['doc'] for r in fusion_results]
        fusion_scores = [r['fusion_score'] for r in fusion_results]
        vector_scores = [r['vector_score'] for r in fusion_results]
        graph_scores = [r['graph_score'] for r in fusion_results]

        return docs, fusion_scores, vector_scores, graph_scores

    def _reciprocal_rank_fusion(self, vector_result, graph_result, top_k: int) -> Tuple[
        List[Dict], List[float], List[float], List[int]]:
        """
        Reciprocal Rank Fusion (RRF) - kombiniert basierend auf Rängen

        RRF Score = Σ(1 / (k + rank_i)) für alle Listen wo Dokument vorkommt
        """
        doc_scores = defaultdict(lambda: {'rrf_score': 0.0, 'vector': 0.0, 'graph': 0, 'doc': None})

        # Vector-Ranks verarbeiten
        for rank, (doc, score) in enumerate(zip(vector_result.retrieved_docs, vector_result.scores)):
            doc_scores[doc['id']]['rrf_score'] += 1.0 / (self.rrf_k + rank + 1)
            doc_scores[doc['id']]['vector'] = score
            doc_scores[doc['id']]['doc'] = doc

        # Graph-Ranks verarbeiten
        for rank, (doc, score) in enumerate(zip(graph_result.retrieved_docs, graph_result.scores)):
            doc_scores[doc['id']]['rrf_score'] += 1.0 / (self.rrf_k + rank + 1)
            doc_scores[doc['id']]['graph'] = score
            if doc_scores[doc['id']]['doc'] is None:
                doc_scores[doc['id']]['doc'] = doc

        # Nach RRF-Score sortieren
        rrf_results = sorted(doc_scores.items(), key=lambda x: x[1]['rrf_score'], reverse=True)
        rrf_results = rrf_results[:top_k]

        # Ergebnisse extrahieren
        docs = [result[1]['doc'] for result in rrf_results]
        fusion_scores = [result[1]['rrf_score'] for result in rrf_results]
        vector_scores = [result[1]['vector'] for result in rrf_results]
        graph_scores = [result[1]['graph'] for result in rrf_results]

        return docs, fusion_scores, vector_scores, graph_scores

    def _adaptive_fusion(self, query: str, vector_result, graph_result, top_k: int) -> Tuple[
        List[Dict], List[float], List[float], List[int]]:
        """
        Adaptive Fusion - passt Gewichtung basierend auf Query-Eigenschaften an

        Heuristik:
        - Kurze, einfache Queries: höhere Vector-Gewichtung
        - Komplexe Queries mit Entities: höhere Graph-Gewichtung
        """
        # Query-Eigenschaften analysieren
        query_length = len(query.split())

        # Einfache Heuristik für adaptive Gewichtung
        if query_length <= 5:
            # Kurze Query: bevorzuge Vector
            adaptive_vector_weight = 0.8
            adaptive_graph_weight = 0.2
        elif query_length >= 10:
            # Lange Query: bevorzuge Graph
            adaptive_vector_weight = 0.4
            adaptive_graph_weight = 0.6
        else:
            # Standard-Gewichtung
            adaptive_vector_weight = self.vector_weight
            adaptive_graph_weight = self.graph_weight

        logger.debug(f"Adaptive Gewichtung: Vector {adaptive_vector_weight}, Graph {adaptive_graph_weight}")

        # Temporär Gewichtung ändern und weighted_sum ausführen
        original_v_weight, original_g_weight = self.vector_weight, self.graph_weight
        self.vector_weight, self.graph_weight = adaptive_vector_weight, adaptive_graph_weight

        result = self._weighted_sum_fusion(vector_result, graph_result, top_k)

        # Original-Gewichtung wiederherstellen
        self.vector_weight, self.graph_weight = original_v_weight, original_g_weight

        return result

    def _normalize_vector_score(self, score: float) -> float:
        """Normalisiert Vector-Score auf [0,1]"""
        min_score, max_score = self.vector_score_range
        return max(0.0, min(1.0, (score - min_score) / (max_score - min_score)))

    def _normalize_graph_score(self, score: int) -> float:
        """Normalisiert Graph-Score auf [0,1]"""
        min_score, max_score = self.graph_score_range
        return max(0.0, min(1.0, (score - min_score) / (max_score - min_score)))

    def _calculate_contributions(self, vector_scores: List[float], graph_scores: List[Union[int, float]]) -> Tuple[
        float, float]:
        """Berechnet relative Beiträge der beiden Methoden"""
        if not vector_scores and not graph_scores:
            return 0.0, 0.0

        total_vector = sum(vector_scores)
        total_graph = sum(graph_scores)
        total = total_vector + total_graph

        if total == 0:
            return 0.5, 0.5

        return total_vector / total, total_graph / total

    def batch_retrieve(self, queries: List[str], **kwargs) -> List[HybridRetrievalResult]:
        """
        Batch Retrieval für mehrere Queries

        Args:
            queries: Liste von Suchanfragen
            **kwargs: Parameter für retrieve()

        Returns:
            Liste von HybridRetrievalResult Objekten
        """
        results = []

        for i, query in enumerate(queries):
            try:
                result = self.retrieve(query, **kwargs)
                results.append(result)

                if i % 5 == 0:
                    logger.info(f"Hybrid Retrieval: {i + 1}/{len(queries)} Queries")

            except Exception as e:
                logger.error(f"Fehler bei Query '{query}': {e}")
                # Leeres Ergebnis für fehlgeschlagene Queries
                results.append(HybridRetrievalResult(
                    query=query,
                    retrieved_docs=[],
                    fusion_scores=[],
                    vector_scores=[],
                    graph_scores=[],
                    fusion_method=self.fusion_method,
                    retrieval_time=0.0,
                    num_retrieved=0,
                    vector_contribution=0.0,
                    graph_contribution=0.0
                ))

        logger.info(f"Hybrid Batch Retrieval abgeschlossen: {len(results)} Queries")
        return results

    def evaluate_on_questions(self, test_questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluiert Hybrid Retrieval auf Testfragen

        Args:
            test_questions: Liste mit 'id', 'question', 'difficulty'

        Returns:
            Liste von Evaluations-Ergebnissen für CSV-Export
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

            # Hybrid Retrieval durchführen
            retrieval_result = self.retrieve(question)

            # Ergebnis für CSV-Export formatieren
            for i, (doc, fusion_score, v_score, g_score) in enumerate(
                    zip(retrieval_result.retrieved_docs,
                        retrieval_result.fusion_scores,
                        retrieval_result.vector_scores,
                        retrieval_result.graph_scores)):
                results.append({
                    'question_id': q_id,
                    'query': question,
                    'difficulty': difficulty,
                    'retrieved_doc_id': doc['id'],
                    'retrieval_score': fusion_score,
                    'vector_score': v_score,
                    'graph_score': g_score,
                    'retrieved_question': doc['question'],
                    'retrieved_category': doc['category'],
                    'fusion_method': retrieval_result.fusion_method,
                    'vector_contribution': retrieval_result.vector_contribution,
                    'graph_contribution': retrieval_result.graph_contribution,
                    'rank': i + 1,
                    'retrieval_time': retrieval_result.retrieval_time
                })

        logger.info(f"Hybrid Retrieval evaluiert: {len(test_questions)} Fragen")
        return results

    def get_performance_stats(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Berechnet Performance-Statistiken für Hybrid Retrieval

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
            "method": "hybrid",
            "fusion_method": self.fusion_method,
            "vector_weight": self.vector_weight,
            "graph_weight": self.graph_weight,
            "num_questions": len(best_per_question),
            "avg_retrieval_score": float(best_per_question['retrieval_score'].mean()),
            "min_retrieval_score": float(best_per_question['retrieval_score'].min()),
            "max_retrieval_score": float(best_per_question['retrieval_score'].max()),
            "avg_retrieval_time": float(best_per_question['retrieval_time'].mean()),
            "success_rate": float((best_per_question['retrieval_score'] > 0).mean()),
            "avg_vector_contribution": float(best_per_question['vector_contribution'].mean()),
            "avg_graph_contribution": float(best_per_question['graph_contribution'].mean()),
            "total_retrievals": len(evaluation_results)
        }

        # Performance by difficulty
        if 'difficulty' in best_per_question.columns:
            difficulty_stats = best_per_question.groupby('difficulty')['retrieval_score'].agg(['mean', 'count'])
            stats['by_difficulty'] = difficulty_stats.to_dict()

        return stats

    def compare_fusion_methods(self, queries: List[str], methods: List[str] = None) -> Dict[str, Any]:
        """
        Vergleicht verschiedene Fusion-Methoden auf denselben Queries

        Args:
            queries: Test-Queries
            methods: Liste von Fusion-Methoden zu testen

        Returns:
            Vergleichsstatistiken
        """
        if methods is None:
            methods = ["weighted_sum", "rrf", "adaptive"]

        results_by_method = {}
        original_method = self.fusion_method

        for method in methods:
            self.fusion_method = method
            logger.info(f"Teste Fusion-Methode: {method}")

            method_results = []
            for query in queries:
                result = self.retrieve(query)
                method_results.append({
                    'query': query,
                    'best_score': max(result.fusion_scores) if result.fusion_scores else 0,
                    'num_retrieved': result.num_retrieved,
                    'retrieval_time': result.retrieval_time
                })

            # Statistiken berechnen
            avg_score = np.mean([r['best_score'] for r in method_results])
            avg_time = np.mean([r['retrieval_time'] for r in method_results])
            success_rate = np.mean([1 if r['num_retrieved'] > 0 else 0 for r in method_results])

            results_by_method[method] = {
                'avg_score': avg_score,
                'avg_time': avg_time,
                'success_rate': success_rate,
                'detailed_results': method_results
            }

        # Original-Methode wiederherstellen
        self.fusion_method = original_method

        return results_by_method


class HybridRetrievalPipeline:
    """
    High-level Pipeline für Hybrid Retrieval mit automatischem Setup
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Pipeline mit Konfiguration

        Args:
            config: Konfiguration mit 'retrieval' Sektion für alle Methoden
        """
        self.config = config

        # Einzelne Retrieval-Systeme initialisieren
        from .vector_retrieval import VectorRetrievalPipeline
        from .graph_retrieval import GraphRetrievalPipeline

        self.vector_pipeline = VectorRetrievalPipeline(config)
        self.graph_pipeline = GraphRetrievalPipeline(config)

        # Hybrid-Konfiguration
        hybrid_config = config.get('retrieval', {}).get('hybrid', {})

        self.hybrid_retrieval = None  # Wird nach Setup initialisiert
        self.fusion_method = hybrid_config.get('fusion_method', 'weighted_sum')
        self.vector_weight = hybrid_config.get('weight_vector', 0.6)
        self.graph_weight = hybrid_config.get('weight_graph', 0.4)

    def setup_from_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Setup Pipeline aus FAQ-Dokumenten

        Args:
            documents: FAQ-Dokumente aus DataLoader
        """
        logger.info("Setup Hybrid Retrieval Pipeline...")

        # Beide Einzelsysteme setup
        self.vector_pipeline.setup_from_documents(documents)
        self.graph_pipeline.setup_from_documents(documents)

        # Hybrid-System initialisieren
        self.hybrid_retrieval = HybridRetrieval(
            vector_retrieval=self.vector_pipeline.retrieval,
            graph_retrieval=self.graph_pipeline.retrieval,
            fusion_method=self.fusion_method,
            vector_weight=self.vector_weight,
            graph_weight=self.graph_weight
        )

        logger.info("Hybrid Retrieval Pipeline setup abgeschlossen")

    def query(self, question: str) -> HybridRetrievalResult:
        """
        Führt Hybrid Retrieval für eine Frage durch

        Args:
            question: Suchanfrage

        Returns:
            HybridRetrievalResult
        """
        if self.hybrid_retrieval is None:
            raise ValueError("Pipeline muss zuerst mit setup_from_documents() initialisiert werden")

        return self.hybrid_retrieval.retrieve(question)

    def run_fusion_comparison(self, test_questions: List[str]) -> Dict[str, Any]:
        """
        Vergleicht alle verfügbaren Fusion-Methoden

        Args:
            test_questions: Liste von Test-Queries

        Returns:
            Detaillierter Vergleich der Fusion-Methoden
        """
        if self.hybrid_retrieval is None:
            raise ValueError("Pipeline muss zuerst initialisiert werden")

        return self.hybrid_retrieval.compare_fusion_methods(test_questions)


# Convenience functions für direkten Import
def create_hybrid_retrieval(documents: List[Dict[str, Any]],
                            config: Optional[Dict[str, Any]] = None) -> HybridRetrievalPipeline:
    """
    Convenience function zur schnellen Erstellung einer Hybrid Retrieval Pipeline

    Args:
        documents: FAQ-Dokumente
        config: Optional Konfiguration

    Returns:
        Konfigurierte HybridRetrievalPipeline
    """
    if config is None:
        # Default config für alle drei Methoden
        config = {
            'retrieval': {
                'vector': {
                    'model': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                    'top_k': 3,
                    'similarity_threshold': 0.0
                },
                'graph': {
                    'neo4j_uri': 'bolt://localhost:7687',
                    'neo4j_user': 'neo4j',
                    'neo4j_password': 'password',
                    'top_k': 3,
                    'min_entity_score': 1
                },
                'hybrid': {
                    'fusion_method': 'weighted_sum',
                    'weight_vector': 0.6,
                    'weight_graph': 0.4,
                    'rrf_k': 60
                }
            }
        }

    pipeline = HybridRetrievalPipeline(config)
    pipeline.setup_from_documents(documents)

    return pipeline


# Example usage
if __name__ == "__main__":
    # Demo des Hybrid Retrievals
    print("=== RAG-Benchmark Hybrid Retrieval Demo ===")

    # Beispiel-Dokumente
    sample_docs = [
        {
            "id": "doc_001",
            "question": "Was ist RAG?",
            "answer": "RAG kombiniert Language Models mit externem Wissen.",
            "category": "Basics",
            "keywords": ["RAG", "retrieval", "language model"]
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
        pipeline = create_hybrid_retrieval(sample_docs)

        # Test-Query
        test_query = "Erkläre mir RAG und FAISS"
        result = pipeline.query(test_query)

        print(f"Query: {test_query}")
        print(f"Fusion-Methode: {result.fusion_method}")
        print(f"Gefundene Dokumente: {result.num_retrieved}")
        print(f"Beste Fusion-Score: {max(result.fusion_scores) if result.fusion_scores else 0}")
        print(f"Vector-Beitrag: {result.vector_contribution:.2f}")
        print(f"Graph-Beitrag: {result.graph_contribution:.2f}")
        print(f"Retrieval Zeit: {result.retrieval_time:.3f}s")

        # Fusion-Vergleich
        fusion_comparison = pipeline.run_fusion_comparison([test_query])
        print(f"Beste Fusion-Methode: {max(fusion_comparison, key=lambda x: fusion_comparison[x]['avg_score'])}")

    except Exception as e:
        print(f"Demo fehlgeschlagen: {e}")
        logger.error(f"Demo fehlgeschlagen: {e}")

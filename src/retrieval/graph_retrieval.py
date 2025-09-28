"""
Graph Retrieval Module für RAG-Benchmark Experimente

Implementiert Neo4j-basiertes Graph Retrieval mit Entity-Extraction und Cypher-Queries.

Author: Lukas Schaumlöffel
"""

import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


@dataclass
class GraphRetrievalResult:
    """Container für Graph Retrieval-Ergebnisse"""
    query: str
    retrieved_docs: List[Dict[str, Any]]
    scores: List[int]  # Integer scores basierend auf Matches
    matched_concepts: List[str]
    query_terms: List[str]
    retrieval_time: float
    num_retrieved: int


class GraphRetrieval:
    """
    Neo4j-basiertes Graph Retrieval System

    Features:
    - Knowledge Graph mit Documents, Entities, Concepts, Categories
    - Entity-Extraction mit Spacy
    - Cypher-basierte Graph-Traversierung
    - Scoring basierend auf strukturellen Matches
    - Konfigurierbare Traversal-Tiefe
    """

    def __init__(self,
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "password",
                 spacy_model: str = "de_core_news_sm"):
        """
        Initialize GraphRetrieval

        Args:
            neo4j_uri: Neo4j Verbindungs-URI
            neo4j_user: Neo4j Username
            neo4j_password: Neo4j Passwort
            spacy_model: Spacy-Modell für Entity-Extraction
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password

        # Neo4j Driver
        self.driver = None
        self.is_connected = False

        # Spacy für NLP
        self.nlp = None
        self.spacy_available = False

        # Scoring-Gewichte (aus deinem Notebook adaptiert)
        self.scoring_weights = {
            'concept_match': 3,
            'entity_match': 2,
            'question_match': 2,
            'answer_match': 1
        }

        self._connect_to_neo4j()
        self._load_spacy_model(spacy_model)

    def _connect_to_neo4j(self):
        """Stellt Verbindung zu Neo4j her"""
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password)
            )

            # Test-Verbindung
            with self.driver.session() as session:
                result = session.run("RETURN 'Neo4j verbunden!' AS message")
                test_result = result.single()

            self.is_connected = True
            logger.info(f"Neo4j-Verbindung hergestellt: {self.neo4j_uri}")

        except Exception as e:
            logger.error(f"Neo4j-Verbindung fehlgeschlagen: {e}")
            self.is_connected = False

    def _load_spacy_model(self, model_name: str):
        """Lädt Spacy-Modell für Entity-Extraction"""
        try:
            import spacy
            self.nlp = spacy.load(model_name)
            self.spacy_available = True
            logger.info(f"Spacy-Modell geladen: {model_name}")

        except (ImportError, OSError) as e:
            try:
                # Fallback auf englisches Modell
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
                self.spacy_available = True
                logger.warning(f"Fallback auf en_core_web_sm (Original: {model_name} nicht verfügbar)")

            except (ImportError, OSError):
                logger.error("Kein Spacy-Modell verfügbar - Entity-Extraction deaktiviert")
                self.spacy_available = False

    def build_graph(self, documents: List[Dict[str, Any]]) -> None:
        """
        Baut Knowledge Graph aus FAQ-Dokumenten auf

        Args:
            documents: FAQ-Dokumente mit Entity/Concept-Information
        """
        if not self.is_connected:
            raise ConnectionError("Neo4j-Verbindung nicht verfügbar")

        logger.info(f"Baue Knowledge Graph für {len(documents)} Dokumente...")

        start_time = time.time()

        # 1. Datenbank leeren und Schema erstellen
        self._clear_database()
        self._create_schema()

        # 2. Dokumente verarbeiten und einfügen
        for doc in documents:
            self._insert_document(doc)

        build_time = time.time() - start_time
        logger.info(f"Knowledge Graph aufgebaut in {build_time:.2f}s")

    def retrieve(self, query: str, top_k: int = 3, min_score: int = 1) -> GraphRetrievalResult:
        """
        Führt Graph Retrieval für eine Query durch

        Args:
            query: Suchanfrage
            top_k: Anzahl zurückzugebender Dokumente
            min_score: Minimaler Relevanz-Score

        Returns:
            GraphRetrievalResult mit gefundenen Dokumenten
        """
        if not self.is_connected:
            logger.warning("Neo4j nicht verfügbar - leeres Ergebnis")
            return GraphRetrievalResult(
                query=query,
                retrieved_docs=[],
                scores=[],
                matched_concepts=[],
                query_terms=[],
                retrieval_time=0.0,
                num_retrieved=0
            )

        start_time = time.time()

        # 1. Query-Begriffe extrahieren
        query_terms = self._extract_query_terms(query)

        # 2. Graph-Retrieval mit Cypher
        retrieved_docs, scores, matched_concepts = self._graph_search(query_terms, top_k)

        # 3. Ergebnisse filtern nach min_score
        filtered_docs = []
        filtered_scores = []

        for doc, score in zip(retrieved_docs, scores):
            if score >= min_score:
                filtered_docs.append(doc)
                filtered_scores.append(score)

        retrieval_time = time.time() - start_time

        return GraphRetrievalResult(
            query=query,
            retrieved_docs=filtered_docs,
            scores=filtered_scores,
            matched_concepts=matched_concepts,
            query_terms=query_terms,
            retrieval_time=retrieval_time,
            num_retrieved=len(filtered_docs)
        )

    def batch_retrieve(self, queries: List[str], **kwargs) -> List[GraphRetrievalResult]:
        """
        Batch Retrieval für mehrere Queries

        Args:
            queries: Liste von Suchanfragen
            **kwargs: Parameter für retrieve()

        Returns:
            Liste von GraphRetrievalResult Objekten
        """
        results = []

        for i, query in enumerate(queries):
            try:
                result = self.retrieve(query, **kwargs)
                results.append(result)

                if i % 5 == 0:
                    logger.info(f"Graph Retrieval: {i + 1}/{len(queries)} Queries")

            except Exception as e:
                logger.error(f"Fehler bei Query '{query}': {e}")
                # Leeres Ergebnis für fehlgeschlagene Queries
                results.append(GraphRetrievalResult(
                    query=query,
                    retrieved_docs=[],
                    scores=[],
                    matched_concepts=[],
                    query_terms=[],
                    retrieval_time=0.0,
                    num_retrieved=0
                ))

        logger.info(f"Graph Batch Retrieval abgeschlossen: {len(results)} Queries")
        return results

    def _extract_query_terms(self, text: str) -> List[str]:
        """
        Extrahiert relevante Begriffe aus Query

        Args:
            text: Query-Text

        Returns:
            Liste von Suchbegriffen
        """
        if not self.spacy_available:
            # Fallback: einfache Wort-Trennung
            terms = [word.lower().strip() for word in text.split()
                     if len(word) > 3 and word.lower() not in ['sind', 'eine', 'wie', 'was', 'kann']]
            return list(set(terms))

        doc = self.nlp(text)
        terms = []

        # Named Entities
        for ent in doc.ents:
            if len(ent.text.strip()) > 2:
                terms.append(ent.text.strip().lower())

        # Wichtige Noun Phrases
        for chunk in doc.noun_chunks:
            term = chunk.text.strip().lower()
            if (len(term) > 3 and
                    not term.startswith(('der', 'die', 'das', 'ein', 'eine')) and
                    term not in ['frage', 'antwort', 'system', 'methode']):
                terms.append(term)

        # Fallback: wichtige Einzelwörter
        simple_terms = [word.lower().strip() for word in text.split()
                        if len(word) > 3 and word.lower() not in ['sind', 'eine', 'wie', 'was', 'kann']]

        all_terms = list(set(terms + simple_terms))
        logger.debug(f"Query-Begriffe extrahiert: {all_terms}")

        return all_terms

    def _graph_search(self, query_terms: List[str], top_k: int) -> Tuple[List[Dict], List[int], List[str]]:
        """
        Führt Cypher-basierte Graph-Suche durch

        Args:
            query_terms: Extrahierte Query-Begriffe
            top_k: Anzahl Ergebnisse

        Returns:
            Tuple von (documents, scores, matched_concepts)
        """
        if not query_terms:
            return [], [], []

        # Cypher-Query für strukturelle Suche (aus deinem Notebook adaptiert)
        cypher_query = """
        UNWIND $query_terms AS term
        MATCH (d:Document)
        OPTIONAL MATCH (d)-[:MENTIONS]->(c:Concept)
        WHERE toLower(c.name) CONTAINS toLower(term)

        OPTIONAL MATCH (d)-[:CONTAINS]->(e:Entity) 
        WHERE toLower(e.name) CONTAINS toLower(term)

        WITH d, term,
             CASE WHEN c IS NOT NULL THEN 3 ELSE 0 END AS concept_score,
             CASE WHEN e IS NOT NULL THEN 2 ELSE 0 END AS entity_score,
             CASE WHEN toLower(d.question) CONTAINS toLower(term) THEN 2 ELSE 0 END AS question_score,
             CASE WHEN toLower(d.answer) CONTAINS toLower(term) THEN 1 ELSE 0 END AS answer_score,
             c.name AS matched_concept

        WITH d, term, concept_score + entity_score + question_score + answer_score AS term_score,
             matched_concept

        WHERE term_score > 0

        WITH d, SUM(term_score) AS total_score, 
             COLLECT(DISTINCT matched_concept) AS matched_concepts

        WHERE total_score > 0

        RETURN d.id AS doc_id, d.question AS question, d.answer AS answer, 
               d.category AS category, total_score, matched_concepts
        ORDER BY total_score DESC
        LIMIT $top_k
        """

        with self.driver.session() as session:
            result = session.run(cypher_query, query_terms=query_terms, top_k=top_k)

            documents = []
            scores = []
            all_matched_concepts = []

            for record in result:
                doc = {
                    'id': record['doc_id'],
                    'question': record['question'],
                    'answer': record['answer'],
                    'category': record['category']
                }

                documents.append(doc)
                scores.append(int(record['total_score']))

                # Sammle alle matched concepts
                concepts = [c for c in record['matched_concepts'] if c is not None]
                all_matched_concepts.extend(concepts)

        return documents, scores, list(set(all_matched_concepts))

    def _clear_database(self):
        """Leert die Neo4j-Datenbank"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.debug("Neo4j-Datenbank geleert")

    def _create_schema(self):
        """Erstellt Schema-Constraints für den Knowledge Graph"""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (cat:Category) REQUIRE cat.name IS UNIQUE"
        ]

        with self.driver.session() as session:
            for constraint in constraints:
                session.run(constraint)

        logger.debug("Graph-Schema erstellt")

    def _insert_document(self, document: Dict[str, Any]):
        """
        Fügt ein Dokument mit allen Beziehungen in den Graph ein

        Args:
            document: FAQ-Dokument mit entities/concepts aus Preprocessor
        """
        # Document Node erstellen
        with self.driver.session() as session:
            # 1. Document Node
            session.run("""
                CREATE (d:Document {
                    id: $doc_id,
                    question: $question,
                    answer: $answer,
                    category: $category
                })
            """,
                        doc_id=document['id'],
                        question=document['question'],
                        answer=document['answer'],
                        category=document['category']
                        )

            # 2. Category Node und Beziehung
            session.run("""
                MERGE (cat:Category {name: $category})
                WITH cat
                MATCH (d:Document {id: $doc_id})
                MERGE (d)-[:BELONGS_TO]->(cat)
            """,
                        category=document['category'],
                        doc_id=document['id']
                        )

            # 3. Entities hinzufügen (falls vorhanden)
            if 'entities' in document:
                for entity in document['entities']:
                    session.run("""
                        MERGE (e:Entity {name: $entity_name, type: $entity_type})
                        WITH e
                        MATCH (d:Document {id: $doc_id})
                        MERGE (d)-[:CONTAINS]->(e)
                    """,
                                entity_name=entity['text'].lower(),
                                entity_type=entity['label'],
                                doc_id=document['id']
                                )

            # 4. Concepts hinzufügen (aus Keywords + extrahierten Concepts)
            concepts_to_add = document.get('keywords', [])
            if 'concepts' in document:
                concepts_to_add.extend([c['text'] for c in document['concepts']])
            if 'enhanced_keywords' in document:
                concepts_to_add.extend(document['enhanced_keywords'])

            # Deduplizieren
            unique_concepts = list(set([c.lower() for c in concepts_to_add if len(c) > 2]))

            for concept in unique_concepts:
                session.run("""
                    MERGE (c:Concept {name: $concept_name})
                    WITH c
                    MATCH (d:Document {id: $doc_id})
                    MERGE (d)-[:MENTIONS]->(c)
                """,
                            concept_name=concept,
                            doc_id=document['id']
                            )

    def evaluate_on_questions(self, test_questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluiert Graph Retrieval auf Testfragen

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

            # Graph Retrieval durchführen
            retrieval_result = self.retrieve(question)

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
                    'matched_concepts': len(retrieval_result.matched_concepts),
                    'query_terms': len(retrieval_result.query_terms),
                    'rank': i + 1,
                    'retrieval_time': retrieval_result.retrieval_time
                })

        logger.info(f"Graph Retrieval evaluiert: {len(test_questions)} Fragen")
        return results

    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Gibt Statistiken über den Knowledge Graph zurück

        Returns:
            Graph-Statistiken
        """
        if not self.is_connected:
            return {"error": "Neo4j nicht verbunden"}

        stats = {}

        with self.driver.session() as session:
            # Node-Statistiken
            node_stats = session.run("""
                MATCH (n)
                RETURN labels(n)[0] AS label, count(n) AS count
            """)

            node_counts = {record['label']: record['count'] for record in node_stats}
            stats['node_counts'] = node_counts

            # Beziehungs-Statistiken
            rel_stats = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) AS relationship, count(r) AS count
            """)

            rel_counts = {record['relationship']: record['count'] for record in rel_stats}
            stats['relationship_counts'] = rel_counts

            # Graph-Dichte
            total_nodes = sum(node_counts.values())
            total_relationships = sum(rel_counts.values())

            stats['graph_metrics'] = {
                'total_nodes': total_nodes,
                'total_relationships': total_relationships,
                'avg_degree': total_relationships / total_nodes if total_nodes > 0 else 0
            }

        return stats

    def close(self):
        """Schließt Neo4j-Verbindung"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j-Verbindung geschlossen")


class GraphRetrievalPipeline:
    """
    High-level Pipeline für Graph Retrieval mit automatischem Setup
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Pipeline mit Konfiguration

        Args:
            config: Konfiguration mit 'retrieval.graph' Sektion
        """
        graph_config = config.get('retrieval', {}).get('graph', {})

        self.retrieval = GraphRetrieval(
            neo4j_uri=graph_config.get('neo4j_uri', 'bolt://localhost:7687'),
            neo4j_user=graph_config.get('neo4j_user', 'neo4j'),
            neo4j_password=graph_config.get('neo4j_password', 'password')
        )

        self.top_k = graph_config.get('top_k', 3)
        self.min_score = graph_config.get('min_entity_score', 1)

        self.config = config

    def setup_from_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Setup Pipeline aus FAQ-Dokumenten (mit Entity-Extraction)

        Args:
            documents: FAQ-Dokumente aus DataLoader
        """
        # Preprocessing für Graph Retrieval (mit Entity-Extraction)
        from src.data.preprocessor import TextPreprocessor

        preprocessor = TextPreprocessor()
        graph_documents = preprocessor.prepare_for_graph_retrieval(documents)

        # Graph aufbauen
        self.retrieval.build_graph(graph_documents)

        logger.info("Graph Retrieval Pipeline setup abgeschlossen")

    def query(self, question: str) -> GraphRetrievalResult:
        """
        Führt Graph Retrieval für eine Frage durch

        Args:
            question: Suchanfrage

        Returns:
            GraphRetrievalResult
        """
        return self.retrieval.retrieve(question, self.top_k, self.min_score)

    def get_performance_stats(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Berechnet Performance-Statistiken für Graph Retrieval

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
            "method": "graph",
            "num_questions": len(best_per_question),
            "avg_retrieval_score": float(best_per_question['retrieval_score'].mean()),
            "min_retrieval_score": float(best_per_question['retrieval_score'].min()),
            "max_retrieval_score": float(best_per_question['retrieval_score'].max()),
            "avg_retrieval_time": float(best_per_question['retrieval_time'].mean()),
            "success_rate": float((best_per_question['retrieval_score'] > self.min_score).mean()),
            "avg_matched_concepts": float(best_per_question['matched_concepts'].mean()),
            "avg_query_terms": float(best_per_question['query_terms'].mean()),
            "total_retrievals": len(evaluation_results)
        }

        # Performance by difficulty
        if 'difficulty' in best_per_question.columns:
            difficulty_stats = best_per_question.groupby('difficulty')['retrieval_score'].agg(['mean', 'count'])
            stats['by_difficulty'] = difficulty_stats.to_dict()

        return stats


# Convenience functions für direkten Import
def create_graph_retrieval(documents: List[Dict[str, Any]],
                           config: Optional[Dict[str, Any]] = None) -> GraphRetrievalPipeline:
    """
    Convenience function zur schnellen Erstellung einer Graph Retrieval Pipeline

    Args:
        documents: FAQ-Dokumente
        config: Optional Konfiguration

    Returns:
        Konfigurierte GraphRetrievalPipeline
    """
    if config is None:
        # Default config
        config = {
            'retrieval': {
                'graph': {
                    'neo4j_uri': 'bolt://localhost:7687',
                    'neo4j_user': 'neo4j',
                    'neo4j_password': 'password',
                    'top_k': 3,
                    'min_entity_score': 1
                }
            }
        }

    pipeline = GraphRetrievalPipeline(config)
    pipeline.setup_from_documents(documents)

    return pipeline


# Example usage
if __name__ == "__main__":
    # Demo des Graph Retrievals
    print("=== RAG-Benchmark Graph Retrieval Demo ===")

    # Beispiel-Dokumente
    sample_docs = [
        {
            "id": "doc_001",
            "question": "Was ist RAG?",
            "answer": "RAG kombiniert Language Models mit externem Wissen.",
            "category": "Basics",
            "keywords": ["RAG", "retrieval", "language model"]
        }
    ]

    try:
        # Pipeline erstellen und testen
        pipeline = create_graph_retrieval(sample_docs)

        # Test-Query
        test_query = "Erkläre mir RAG"
        result = pipeline.query(test_query)

        print(f"Query: {test_query}")
        print(f"Gefundene Dokumente: {result.num_retrieved}")
        print(f"Query Terms: {result.query_terms}")
        print(f"Matched Concepts: {result.matched_concepts}")
        print(f"Retrieval Zeit: {result.retrieval_time:.3f}s")

        # Graph-Statistiken
        stats = pipeline.retrieval.get_graph_statistics()
        print(f"Graph: {stats.get('node_counts', {})} Nodes")

    except Exception as e:
        print(f"Demo fehlgeschlagen: {e}")
        logger.error(f"Demo fehlgeschlagen: {e}")

"""
Automated Evaluator für RAG-Benchmark Experimente

Orchestriert die komplette Evaluation verschiedener RAG-Methoden.

Author: Lukas Schaumlöffel
"""

import json
import time
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration für ein RAG-Experiment"""
    experiment_name: str
    methods: List[str]  # ["vector", "graph", "hybrid", "baseline"]
    test_questions: List[Dict[str, Any]]
    reference_answers: Dict[str, str]
    output_dir: str
    llm_config: Dict[str, Any]
    evaluation_config: Dict[str, Any]


@dataclass
class MethodResult:
    """Ergebnis einer einzelnen Retrieval-Methode"""
    method: str
    question_id: str
    question: str
    answer: str
    retrieval_time: float
    retrieval_details: Dict[str, Any]
    success: bool
    error_message: str = None


class RAGEvaluator:
    """
    Hauptklasse für die automatisierte Evaluation aller RAG-Methoden

    Orchestriert:
    - Setup aller Retrieval-Pipelines
    - LLM-Integration für Antwortgenerierung
    - Metriken-Berechnung
    - Report-Generierung
    - CSV-Export für weitere Analyse
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize RAGEvaluator

        Args:
            config: Vollständige Experiment-Konfiguration
        """
        self.config = config
        self.results_dir = Path(config.get('output_dir', '../results'))
        self.results_dir.mkdir(exist_ok=True, parents=True)

        # Components initialization (lazy loading)
        self.pipelines = {}
        self.llm_client = None
        self.evaluator_pipeline = None

        # Experiment tracking
        self.experiment_start_time = None
        self.all_results = []

        logger.info("RAGEvaluator initialisiert")

    def setup_experiment(self, experiment_config: ExperimentConfig) -> None:
        """
        Setup für ein komplettes Experiment

        Args:
            experiment_config: Experiment-Konfiguration
        """
        self.experiment_config = experiment_config
        self.experiment_start_time = time.time()

        logger.info(f"Setup Experiment: {experiment_config.experiment_name}")

        # 1. Retrieval-Pipelines initialisieren
        self._setup_retrieval_pipelines(experiment_config.methods)

        # 2. LLM-Client initialisieren
        self._setup_llm_client(experiment_config.llm_config)

        # 3. Evaluation-Pipeline initialisieren
        self._setup_evaluation_pipeline(experiment_config.evaluation_config)

        logger.info("Experiment setup abgeschlossen")

    def run_complete_evaluation(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Führt komplette Evaluation aller Methoden durch

        Args:
            documents: FAQ-Dokumente für Retrieval-Setup

        Returns:
            Vollständiger Evaluations-Report
        """
        if not hasattr(self, 'experiment_config'):
            raise ValueError("Experiment muss zuerst mit setup_experiment() konfiguriert werden")

        logger.info("Starte komplette Evaluation...")

        # 1. Alle Pipelines mit Dokumenten setup
        self._setup_pipelines_with_documents(documents)

        # 2. Für jede Methode alle Fragen durchlaufen
        all_method_results = []

        for method in self.experiment_config.methods:
            logger.info(f"Evaluiere Methode: {method}")
            method_results = self._evaluate_single_method(method)
            all_method_results.extend(method_results)

        # 3. Metriken berechnen
        evaluation_results = self._calculate_evaluation_metrics(all_method_results)

        # 4. Report generieren
        final_report = self._generate_final_report(evaluation_results)

        # 5. Ergebnisse speichern
        self._save_all_results(all_method_results, evaluation_results, final_report)

        total_time = time.time() - self.experiment_start_time
        logger.info(f"Evaluation abgeschlossen in {total_time:.2f}s")

        return final_report

    def _setup_retrieval_pipelines(self, methods: List[str]) -> None:
        """Setup aller benötigten Retrieval-Pipelines"""
        for method in methods:
            if method == "baseline":
                continue  # Baseline braucht kein Retrieval

            try:
                if method == "vector":
                    from ..retrieval.vector_retrieval import VectorRetrievalPipeline
                    self.pipelines[method] = VectorRetrievalPipeline(self.config)

                elif method == "graph":
                    from ..retrieval.graph_retrieval import GraphRetrievalPipeline
                    self.pipelines[method] = GraphRetrievalPipeline(self.config)

                elif method == "hybrid":
                    from ..retrieval.hybrid_retrieval import HybridRetrievalPipeline
                    self.pipelines[method] = HybridRetrievalPipeline(self.config)

                else:
                    logger.warning(f"Unbekannte Methode: {method}")

            except Exception as e:
                logger.error(f"Fehler beim Setup {method}: {e}")
                self.pipelines[method] = None

    def _setup_llm_client(self, llm_config: Dict[str, Any]) -> None:
        """Setup LLM-Client für Antwortgenerierung (LangChain-basiert)"""
        try:
            # API Key prüfen
            api_key = llm_config.get('api_key') or os.getenv('OPENAI_API_KEY')
            if not api_key:
                logger.warning("Kein OpenAI API Key - LLM-Funktionen deaktiviert")
                self.llm_client = None
                return

            # LangChain ChatOpenAI Client setup
            from langchain_openai import ChatOpenAI

            self.llm_client = ChatOpenAI(
                model=llm_config.get('model', 'gpt-4.1-nano'),
                temperature=llm_config.get('temperature', 0.1),
                max_tokens=llm_config.get('max_tokens', 500),
                api_key=api_key
            )

            # Test-Aufruf mit LangChain
            from langchain_core.messages import HumanMessage
            test_response = self.llm_client.invoke([HumanMessage(content="Test")])

            logger.info(f"LangChain LLM-Client erfolgreich initialisiert: {llm_config.get('model')}")

        except Exception as e:
            logger.error(f"LangChain LLM-Client Setup fehlgeschlagen: {e}")
            self.llm_client = None

    def _setup_evaluation_pipeline(self, eval_config: Dict[str, Any]) -> None:
        """Setup Evaluation-Pipeline"""
        try:
            from .metrics import RAGEvaluationPipeline
            config_with_eval = {**self.config, 'evaluation': eval_config}
            self.evaluator_pipeline = RAGEvaluationPipeline(config_with_eval)
            logger.info("Evaluation-Pipeline initialisiert")

        except Exception as e:
            logger.error(f"Evaluation-Pipeline Setup fehlgeschlagen: {e}")
            self.evaluator_pipeline = None

    def _setup_pipelines_with_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Setup alle Pipelines mit FAQ-Dokumenten"""
        for method, pipeline in self.pipelines.items():
            if pipeline is None:
                continue

            try:
                logger.info(f"Setup {method} pipeline mit {len(documents)} Dokumenten")
                pipeline.setup_from_documents(documents)

            except Exception as e:
                logger.error(f"Pipeline setup fehlgeschlagen für {method}: {e}")
                self.pipelines[method] = None

    def _evaluate_single_method(self, method: str) -> List[MethodResult]:
        """Evaluiert eine einzelne Retrieval-Methode auf allen Testfragen"""
        results = []

        for question_data in self.experiment_config.test_questions:
            question_id = question_data['id']
            question = question_data['question']
            difficulty = question_data.get('difficulty', 'unknown')

            try:
                # Antwort generieren
                start_time = time.time()
                answer, retrieval_details = self._generate_answer(method, question)
                generation_time = time.time() - start_time

                # Ergebnis speichern
                result = MethodResult(
                    method=method,
                    question_id=question_id,
                    question=question,
                    answer=answer,
                    retrieval_time=generation_time,
                    retrieval_details=retrieval_details,
                    success=True
                )

                results.append(result)
                logger.debug(f"{method} - {question_id}: Erfolg ({generation_time:.2f}s)")

            except Exception as e:
                logger.error(f"{method} - {question_id}: Fehler {e}")

                # Fehler-Ergebnis speichern
                result = MethodResult(
                    method=method,
                    question_id=question_id,
                    question=question,
                    answer="",
                    retrieval_time=0.0,
                    retrieval_details={},
                    success=False,
                    error_message=str(e)
                )
                results.append(result)

        logger.info(f"{method}: {sum(1 for r in results if r.success)}/{len(results)} erfolgreich")
        return results

    def _generate_answer(self, method: str, question: str) -> Tuple[str, Dict[str, Any]]:
        """
        Generiert Antwort für eine Frage mit der gegebenen Methode

        Args:
            method: Retrieval-Methode
            question: Frage

        Returns:
            Tuple von (answer, retrieval_details)
        """
        if method == "baseline":
            return self._generate_baseline_answer(question)

        # Retrieval durchführen
        pipeline = self.pipelines.get(method)
        if pipeline is None:
            raise ValueError(f"Pipeline für {method} nicht verfügbar")

        retrieval_result = pipeline.query(question)

        # Context aus Retrieval-Ergebnissen zusammenstellen
        if hasattr(retrieval_result, 'retrieved_docs'):
            context_docs = retrieval_result.retrieved_docs
        else:
            context_docs = []

        # LLM-Antwort generieren
        answer = self._generate_llm_answer(question, context_docs, method)

        # Retrieval-Details für Analyse
        retrieval_details = {
            'num_retrieved': getattr(retrieval_result, 'num_retrieved', len(context_docs)),
            'retrieval_time': getattr(retrieval_result, 'retrieval_time', 0.0),
            'scores': getattr(retrieval_result, 'scores', []) or getattr(retrieval_result, 'fusion_scores', []),
            'method_specifics': self._extract_method_specifics(retrieval_result, method)
        }

        return answer, retrieval_details

    def _generate_baseline_answer(self, question: str) -> Tuple[str, Dict[str, Any]]:
        """Generiert Baseline-Antwort ohne Retrieval"""
        if self.llm_client is None:
            return "LLM nicht verfügbar", {}

        baseline_prompt = f"""Du bist ein hilfreicher Assistent für Fragen zu Retrieval-Augmented Generation (RAG) und verwandten Technologien.

Frage: {question}

Beantworte die Frage basierend auf deinem allgemeinen Wissen über RAG, Machine Learning und AI.

Antwort:"""

        try:
            from langchain_core.messages import HumanMessage
            messages = [HumanMessage(content=baseline_prompt)]
            response = self.llm_client.invoke(messages)
            answer = response.content.strip()
            return answer, {'method': 'baseline', 'context_used': False}

        except Exception as e:
            logger.error(f"Baseline LLM-Generierung fehlgeschlagen: {e}")
            return f"Baseline-Fehler: {e}", {}

    def _generate_llm_answer(self, question: str, context_docs: List[Dict[str, Any]], method: str) -> str:
        """Generiert LLM-Antwort mit Retrieval-Context"""
        if self.llm_client is None:
            return "LLM nicht verfügbar"

        # Context zusammenstellen
        context_parts = []
        for i, doc in enumerate(context_docs[:3]):  # Top 3 Dokumente
            context_parts.append(f"Dokument {i + 1}: {doc.get('answer', doc.get('content', ''))}")

        context = "\n\n".join(context_parts) if context_parts else "Keine relevanten Dokumente gefunden."

        # Prompt Template
        prompt = f"""Du bist ein hilfreicher Assistent für Fragen zu Retrieval-Augmented Generation (RAG) und verwandten Technologien.

Kontext-Informationen: {context}

Frage: {question}

Anweisungen:
- Beantworte die Frage basierend auf den bereitgestellten Kontext-Informationen
- Wenn die Informationen nicht ausreichen, sage das ehrlich
- Halte deine Antwort präzise und hilfreich
- Verwende die Fachbegriffe aus dem Kontext korrekt

Antwort:"""

        try:
            from langchain_core.messages import HumanMessage
            messages = [HumanMessage(content=prompt)]
            response = self.llm_client.invoke(messages)
            return response.content.strip()

        except Exception as e:
            logger.error(f"LLM-Generierung fehlgeschlagen: {e}")
            return f"LLM-Fehler: {e}"

    def _extract_method_specifics(self, retrieval_result, method: str) -> Dict[str, Any]:
        """Extrahiert methoden-spezifische Details"""
        specifics = {}

        if method == "vector":
            # Vector-spezifische Metriken
            pass

        elif method == "graph":
            if hasattr(retrieval_result, 'matched_concepts'):
                specifics['matched_concepts'] = len(retrieval_result.matched_concepts)
            if hasattr(retrieval_result, 'query_terms'):
                specifics['query_terms'] = len(retrieval_result.query_terms)

        elif method == "hybrid":
            if hasattr(retrieval_result, 'vector_contribution'):
                specifics['vector_contribution'] = retrieval_result.vector_contribution
            if hasattr(retrieval_result, 'graph_contribution'):
                specifics['graph_contribution'] = retrieval_result.graph_contribution
            if hasattr(retrieval_result, 'fusion_method'):
                specifics['fusion_method'] = retrieval_result.fusion_method

        return specifics

    def _calculate_evaluation_metrics(self, method_results: List[MethodResult]) -> List[Any]:
        """Berechnet Evaluation-Metriken für alle Ergebnisse"""
        if self.evaluator_pipeline is None:
            logger.warning("Evaluator-Pipeline nicht verfügbar")
            return []

        # Difficulty-Mapping erstellen für schnellen Lookup
        difficulty_map = {q['id']: q.get('difficulty', 'unknown')
                          for q in self.experiment_config.test_questions}

        # Convert zu Format für Evaluation-Pipeline
        rag_results = []
        for result in method_results:
            if result.success:
                rag_results.append({
                    'question_id': result.question_id,
                    'method': result.method,
                    'answer': result.answer,
                    'difficulty': difficulty_map.get(result.question_id, 'unknown'),  # FIX: Korrekte difficulty
                    'retrieved_docs': getattr(result, 'retrieved_docs', None)
                })

        # Evaluation durchführen
        evaluation_results = self.evaluator_pipeline.evaluate_rag_responses(
            rag_results,
            self.experiment_config.reference_answers
        )

        return evaluation_results

    def _generate_final_report(self, evaluation_results: List[Any]) -> Dict[str, Any]:
        """Generiert finalen Evaluations-Report"""
        if self.evaluator_pipeline is None or not evaluation_results:
            logger.warning("Keine Evaluation-Ergebnisse für Report verfügbar")
            return {}

        # Vollständigen Report generieren
        report = self.evaluator_pipeline.generate_evaluation_report(evaluation_results)

        # Experiment-Metadaten hinzufügen
        report['experiment_metadata'] = {
            'name': self.experiment_config.experiment_name,
            'methods_tested': self.experiment_config.methods,
            'total_questions': len(self.experiment_config.test_questions),
            'total_runtime': time.time() - self.experiment_start_time,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        return report

    def _save_all_results(self, method_results: List[MethodResult],
                          evaluation_results: List[Any],
                          final_report: Dict[str, Any]) -> None:
        """Speichert alle Ergebnisse in verschiedenen Formaten"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')

        # 1. Method Results als CSV (für Pipeline-Analyse)
        method_data = []
        for result in method_results:
            row = {
                'question_id': result.question_id,
                'method': result.method,
                'question': result.question,
                'answer': result.answer,
                'success': result.success,
                'retrieval_time': result.retrieval_time,
                'error_message': result.error_message or ''
            }

            # Method-spezifische Details hinzufügen
            if result.retrieval_details:
                row.update({f"detail_{k}": v for k, v in result.retrieval_details.items()
                            if isinstance(v, (int, float, str, bool))})

            method_data.append(row)

        method_df = pd.DataFrame(method_data)
        method_df.to_csv(self.results_dir / f'rag_pipeline_results_{timestamp}.csv',
                         index=False, encoding='utf-8')

        # 2. Evaluation Results als CSV (kompatibel mit bestehenden Formaten)
        if evaluation_results and hasattr(self.evaluator_pipeline.metrics_calculator, 'export_results_to_csv'):
            self.evaluator_pipeline.metrics_calculator.export_results_to_csv(
                evaluation_results,
                self.results_dir / f'evaluation_scores_{timestamp}.csv'
            )

        # 3. Final Report als JSON
        with open(self.results_dir / f'final_report_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)

        # 4. Summary für schnelle Übersicht
        summary = self._create_experiment_summary(final_report)
        with open(self.results_dir / f'experiment_summary_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Alle Ergebnisse gespeichert in: {self.results_dir}")

    def _create_experiment_summary(self, final_report: Dict[str, Any]) -> Dict[str, Any]:
        """Erstellt kompakte Zusammenfassung für schnelle Übersicht"""
        summary = {}

        if 'experiment_metadata' in final_report:
            summary['experiment'] = final_report['experiment_metadata']

        if 'evaluation_summary' in final_report:
            summary['results'] = final_report['evaluation_summary']

        # Top-Level Metriken extrahieren
        if 'method_comparison' in final_report:
            bleu_comparison = final_report['method_comparison'].get('bleu_score', {})
            summary['method_ranking'] = sorted(
                [(method, scores['mean']) for method, scores in bleu_comparison.items()],
                key=lambda x: x[1], reverse=True
            )

        return summary


# Convenience functions für direkten Import
def run_complete_rag_evaluation(documents: List[Dict[str, Any]],
                                test_questions: List[Dict[str, Any]],
                                reference_answers: Dict[str, str],
                                config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function für komplette RAG-Evaluation

    Args:
        documents: FAQ-Dokumente
        test_questions: Test-Fragen mit id, question, difficulty
        reference_answers: Referenz-Antworten (question_id -> answer)
        config: Optional Konfiguration

    Returns:
        Vollständiger Evaluations-Report
    """
    if config is None:
        config = {
            'retrieval': {
                'vector': {'model': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', 'top_k': 3},
                'graph': {'neo4j_uri': 'bolt://localhost:7687', 'neo4j_user': 'neo4j', 'neo4j_password': 'password',
                          'top_k': 3},
                'hybrid': {'fusion_method': 'weighted_sum', 'weight_vector': 0.6, 'weight_graph': 0.4}
            },
            'llm': {
                'model': 'gpt-4.1-nano',
                'max_tokens': 300,
                'temperature': 0.1
            },
            'evaluation': {
                'rouge_metrics': ['rouge1', 'rouge2', 'rougeL'],
                'custom_metrics': True
            },
            'output_dir': '../results'
        }

    # Experiment Config erstellen
    experiment_config = ExperimentConfig(
        experiment_name=f"RAG_Evaluation_{time.strftime('%Y%m%d_%H%M%S')}",
        methods=['baseline', 'vector', 'graph', 'hybrid'],
        test_questions=test_questions,
        reference_answers=reference_answers,
        output_dir=config.get('output_dir', '../results'),
        llm_config=config['llm'],
        evaluation_config=config['evaluation']
    )

    # Evaluator starten
    evaluator = RAGEvaluator(config)
    evaluator.setup_experiment(experiment_config)

    return evaluator.run_complete_evaluation(documents)


# Example usage
if __name__ == "__main__":
    # Demo des RAG Evaluators
    print("=== RAG-Benchmark Evaluator Demo ===")

    # Beispiel-Setup (würde normalerweise aus DataLoader kommen)
    sample_docs = [
        {
            "id": "doc_001",
            "question": "Was ist RAG?",
            "answer": "RAG kombiniert Language Models mit externem Wissen.",
            "category": "Basics",
            "keywords": ["RAG", "retrieval"]
        }
    ]

    sample_questions = [
        {"id": "q001", "question": "Erkläre mir RAG", "difficulty": "easy"}
    ]

    sample_references = {
        "q001": "RAG ist eine Technik die Language Models mit externen Datenquellen verbindet."
    }

    try:
        # Vollständige Evaluation (ohne LLM-API für Demo)
        config = {
            'retrieval': {'vector': {'model': 'sentence-transformers/all-MiniLM-L6-v2'}},
            'llm': {'model': 'gpt-4.1-nano'},  # Würde API Key brauchen
            'evaluation': {'custom_metrics': False},
            'output_dir': './demo_results'
        }

        evaluator = RAGEvaluator(config)
        print("Evaluator initialisiert")

        # In echter Nutzung würde hier run_complete_rag_evaluation aufgerufen
        print("Demo erfolgreich - für vollständige Evaluation OpenAI API Key erforderlich")

    except Exception as e:
        print(f"Demo Info: {e}")
        logger.error(f"Demo Info: {e}")

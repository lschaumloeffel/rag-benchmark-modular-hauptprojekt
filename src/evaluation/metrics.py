"""
Evaluation Metrics Module für RAG-Benchmark Experimente

Implementiert BLEU, ROUGE und weitere Metriken für automatische Evaluation.

Author: Lukas Schaumlöffel
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import re

# NLTK imports mit error handling
try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    nltk_available = True

    # Download required data if not present
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

except ImportError:
    nltk_available = False

# ROUGE imports mit error handling
try:
    from rouge_score import rouge_scorer

    rouge_available = True
except ImportError:
    rouge_available = False

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container für Evaluation-Ergebnisse"""
    question_id: str
    method: str
    reference_text: str
    generated_text: str
    bleu_score: float
    rouge1_f: float
    rouge2_f: float
    rougeL_f: float
    difficulty: str
    additional_metrics: Dict[str, float] = None


class MetricsCalculator:
    """
    Hauptklasse für die Berechnung von Evaluation-Metriken

    Features:
    - BLEU-Scores mit verschiedenen Smoothing-Funktionen
    - ROUGE-1, ROUGE-2, ROUGE-L Scores
    - Custom RAG-spezifische Metriken
    - Batch-Processing für große Datensätze
    - Robuste Error-Handling
    """

    def __init__(self,
                 bleu_smoothing: str = "method1",
                 rouge_metrics: List[str] = None,
                 custom_metrics: bool = True):
        """
        Initialize MetricsCalculator

        Args:
            bleu_smoothing: NLTK Smoothing-Methode für BLEU
            rouge_metrics: Liste der zu berechnenden ROUGE-Metriken
            custom_metrics: Ob custom RAG-Metriken berechnet werden sollen
        """
        self.bleu_smoothing = bleu_smoothing
        self.rouge_metrics = rouge_metrics or ['rouge1', 'rouge2', 'rougeL']
        self.custom_metrics = custom_metrics

        # Initialize components
        self._setup_bleu_smoother()
        self._setup_rouge_scorer()

        # Availability checks
        self.bleu_available = nltk_available
        self.rouge_available = rouge_available

        if not self.bleu_available:
            logger.warning("NLTK nicht verfügbar - BLEU-Scores deaktiviert")
        if not self.rouge_available:
            logger.warning("rouge-score nicht verfügbar - ROUGE-Scores deaktiviert")

        logger.info("MetricsCalculator initialisiert")

    def calculate_bleu_score(self, reference: str, candidate: str) -> float:
        """
        Berechnet BLEU-Score zwischen Referenz und Kandidat

        Args:
            reference: Referenz-Text (Ground Truth)
            candidate: Generierter Text

        Returns:
            BLEU-Score (0.0 - 1.0)
        """
        if not self.bleu_available:
            logger.warning("BLEU nicht verfügbar")
            return 0.0

        try:
            # Text preprocessing
            reference_tokens = self._tokenize_text(reference)
            candidate_tokens = self._tokenize_text(candidate)

            # BLEU berechnen mit Smoothing
            score = sentence_bleu(
                [reference_tokens],
                candidate_tokens,
                smoothing_function=self.smoother.method1
            )

            return float(score)

        except Exception as e:
            logger.error(f"BLEU-Berechnung fehlgeschlagen: {e}")
            return 0.0

    def calculate_rouge_scores(self, reference: str, candidate: str) -> Dict[str, float]:
        """
        Berechnet ROUGE-Scores zwischen Referenz und Kandidat

        Args:
            reference: Referenz-Text
            candidate: Generierter Text

        Returns:
            Dictionary mit ROUGE-Scores
        """
        if not self.rouge_available:
            logger.warning("ROUGE nicht verfügbar")
            return {metric: 0.0 for metric in self.rouge_metrics}

        try:
            scores = self.rouge_scorer.score(reference, candidate)

            rouge_results = {}
            for metric in self.rouge_metrics:
                if metric in scores:
                    rouge_results[f"{metric}_f"] = scores[metric].fmeasure
                    rouge_results[f"{metric}_p"] = scores[metric].precision
                    rouge_results[f"{metric}_r"] = scores[metric].recall
                else:
                    rouge_results[f"{metric}_f"] = 0.0

            return rouge_results

        except Exception as e:
            logger.error(f"ROUGE-Berechnung fehlgeschlagen: {e}")
            return {f"{metric}_f": 0.0 for metric in self.rouge_metrics}

    def calculate_custom_rag_metrics(self,
                                     reference: str,
                                     candidate: str,
                                     retrieved_docs: List[str] = None) -> Dict[str, float]:
        """
        Berechnet RAG-spezifische Metriken

        Args:
            reference: Referenz-Text
            candidate: Generierter Text
            retrieved_docs: Liste der abgerufenen Dokumente

        Returns:
            Dictionary mit custom Metriken
        """
        metrics = {}

        # 1. Semantic Overlap (einfache Token-basierte Approximation)
        metrics['semantic_overlap'] = self._calculate_semantic_overlap(reference, candidate)

        # 2. Answer Completeness
        metrics['answer_completeness'] = self._calculate_completeness(reference, candidate)

        # 3. Factual Consistency (bei verfügbaren retrieved docs)
        if retrieved_docs:
            metrics['factual_consistency'] = self._calculate_factual_consistency(candidate, retrieved_docs)

        # 4. Response Coherence
        metrics['response_coherence'] = self._calculate_coherence(candidate)

        return metrics

    def evaluate_single_answer(self,
                               question_id: str,
                               method: str,
                               reference_text: str,
                               generated_text: str,
                               difficulty: str = "unknown",
                               retrieved_docs: List[str] = None) -> EvaluationResult:
        """
        Vollständige Evaluation einer einzelnen Antwort

        Args:
            question_id: Eindeutige Fragen-ID
            method: Retrieval-Methode (vector, graph, hybrid, baseline)
            reference_text: Referenz-Antwort
            generated_text: Generierte Antwort
            difficulty: Schwierigkeitsgrad
            retrieved_docs: Abgerufene Dokumente für custom Metriken

        Returns:
            EvaluationResult mit allen Metriken
        """
        # BLEU-Score
        bleu_score = self.calculate_bleu_score(reference_text, generated_text)

        # ROUGE-Scores
        rouge_scores = self.calculate_rouge_scores(reference_text, generated_text)

        # Custom Metriken
        additional_metrics = {}
        if self.custom_metrics:
            additional_metrics = self.calculate_custom_rag_metrics(
                reference_text, generated_text, retrieved_docs
            )

        return EvaluationResult(
            question_id=question_id,
            method=method,
            reference_text=reference_text,
            generated_text=generated_text,
            bleu_score=bleu_score,
            rouge1_f=rouge_scores.get('rouge1_f', 0.0),
            rouge2_f=rouge_scores.get('rouge2_f', 0.0),
            rougeL_f=rouge_scores.get('rougeL_f', 0.0),
            difficulty=difficulty,
            additional_metrics=additional_metrics
        )

    def batch_evaluate(self, evaluation_data: List[Dict[str, Any]]) -> List[EvaluationResult]:
        """
        Batch-Evaluation für mehrere Antworten

        Args:
            evaluation_data: Liste mit Dicts containing:
                - question_id, method, reference_text, generated_text, difficulty

        Returns:
            Liste von EvaluationResult Objekten
        """
        results = []

        for i, data in enumerate(evaluation_data):
            try:
                result = self.evaluate_single_answer(
                    question_id=data['question_id'],
                    method=data['method'],
                    reference_text=data['reference_text'],
                    generated_text=data['generated_text'],
                    difficulty=data.get('difficulty', 'unknown'),
                    retrieved_docs=data.get('retrieved_docs', None)
                )
                results.append(result)

                if i % 10 == 0:
                    logger.info(f"Evaluation: {i + 1}/{len(evaluation_data)} Antworten")

            except Exception as e:
                logger.error(f"Fehler bei Evaluation {data.get('question_id', i)}: {e}")
                continue

        logger.info(f"Batch-Evaluation abgeschlossen: {len(results)} Ergebnisse")
        return results

    def compute_aggregate_statistics(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """
        Berechnet aggregierte Statistiken über alle Evaluations-Ergebnisse

        Args:
            results: Liste von EvaluationResult Objekten

        Returns:
            Aggregierte Statistiken
        """
        if not results:
            return {}

        # Gruppiere nach Methode
        by_method = {}
        for result in results:
            if result.method not in by_method:
                by_method[result.method] = []
            by_method[result.method].append(result)

        # Statistiken pro Methode
        method_stats = {}
        for method, method_results in by_method.items():
            bleu_scores = [r.bleu_score for r in method_results]
            rouge1_scores = [r.rouge1_f for r in method_results]
            rouge2_scores = [r.rouge2_f for r in method_results]
            rougeL_scores = [r.rougeL_f for r in method_results]

            method_stats[method] = {
                'num_questions': len(method_results),
                'bleu_score': {
                    'mean': float(np.mean(bleu_scores)),
                    'std': float(np.std(bleu_scores)),
                    'min': float(np.min(bleu_scores)),
                    'max': float(np.max(bleu_scores))
                },
                'rouge1_f': {
                    'mean': float(np.mean(rouge1_scores)),
                    'std': float(np.std(rouge1_scores))
                },
                'rouge2_f': {
                    'mean': float(np.mean(rouge2_scores)),
                    'std': float(np.std(rouge2_scores))
                },
                'rougeL_f': {
                    'mean': float(np.mean(rougeL_scores)),
                    'std': float(np.std(rougeL_scores))
                }
            }

            # Performance by difficulty
            difficulty_breakdown = {}
            for result in method_results:
                diff = result.difficulty
                if diff not in difficulty_breakdown:
                    difficulty_breakdown[diff] = []
                difficulty_breakdown[diff].append(result.bleu_score)

            for diff, scores in difficulty_breakdown.items():
                difficulty_breakdown[diff] = {
                    'mean_bleu': float(np.mean(scores)),
                    'count': len(scores)
                }

            method_stats[method]['by_difficulty'] = difficulty_breakdown

        return method_stats

    def export_results_to_csv(self, results: List[EvaluationResult], output_path: str) -> None:
        """
        Exportiert Evaluation-Ergebnisse als CSV (kompatibel mit bestehenden Formaten)

        Args:
            results: Liste von EvaluationResult Objekten
            output_path: Pfad für CSV-Export
        """
        import pandas as pd

        # Convert to DataFrame
        data = []
        for result in results:
            row = {
                'question_id': result.question_id,
                'method': result.method,
                'bleu_score': result.bleu_score,
                'rouge1_f': result.rouge1_f,
                'rouge2_f': result.rouge2_f,
                'rougeL_f': result.rougeL_f,
                'difficulty': result.difficulty
            }

            # Custom metrics hinzufügen
            if result.additional_metrics:
                row.update(result.additional_metrics)

            data.append(row)

        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False, encoding='utf-8')

        logger.info(f"Evaluation-Ergebnisse exportiert: {output_path}")

    def _setup_bleu_smoother(self):
        """Initialisiert BLEU Smoothing-Funktion"""
        if nltk_available:
            self.smoother = SmoothingFunction()
        else:
            self.smoother = None

    def _setup_rouge_scorer(self):
        """Initialisiert ROUGE Scorer"""
        if rouge_available:
            self.rouge_scorer = rouge_scorer.RougeScorer(self.rouge_metrics, use_stemmer=True)
        else:
            self.rouge_scorer = None

    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenisiert Text für BLEU-Berechnung"""
        # Einfache Tokenisierung als Fallback
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', ' ', text)  # Satzzeichen entfernen
        tokens = text.split()
        return [token for token in tokens if len(token) > 1]

    def _calculate_semantic_overlap(self, reference: str, candidate: str) -> float:
        """
        Berechnet semantische Überlappung zwischen Texten

        Einfache Token-basierte Approximation für semantische Ähnlichkeit.
        """
        ref_tokens = set(self._tokenize_text(reference))
        cand_tokens = set(self._tokenize_text(candidate))

        if not ref_tokens and not cand_tokens:
            return 1.0
        if not ref_tokens or not cand_tokens:
            return 0.0

        intersection = len(ref_tokens.intersection(cand_tokens))
        union = len(ref_tokens.union(cand_tokens))

        return intersection / union if union > 0 else 0.0

    def _calculate_completeness(self, reference: str, candidate: str) -> float:
        """
        Berechnet Answer Completeness - wie vollständig ist die Antwort?

        Approximation über Längen-Verhältnis und Token-Coverage.
        """
        ref_tokens = self._tokenize_text(reference)
        cand_tokens = self._tokenize_text(candidate)

        if not ref_tokens:
            return 1.0 if not cand_tokens else 0.0

        # Length ratio (capped at 1.0)
        length_ratio = min(1.0, len(cand_tokens) / len(ref_tokens))

        # Token coverage
        ref_token_set = set(ref_tokens)
        cand_token_set = set(cand_tokens)
        coverage = len(ref_token_set.intersection(cand_token_set)) / len(ref_token_set)

        # Gewichtete Kombination
        completeness = 0.4 * length_ratio + 0.6 * coverage

        return completeness

    def _calculate_factual_consistency(self, candidate: str, retrieved_docs: List[str]) -> float:
        """
        Berechnet Factual Consistency - wie konsistent ist die Antwort mit abgerufenen Docs?

        Einfache Approximation über Token-Overlap mit retrieved docs.
        """
        if not retrieved_docs:
            return 0.0

        cand_tokens = set(self._tokenize_text(candidate))

        # Sammle alle Tokens aus retrieved docs
        doc_tokens = set()
        for doc in retrieved_docs:
            doc_tokens.update(self._tokenize_text(doc))

        if not doc_tokens:
            return 0.0

        # Anteil der Candidate-Tokens, die in retrieved docs vorkommen
        overlap = len(cand_tokens.intersection(doc_tokens))
        consistency = overlap / len(cand_tokens) if cand_tokens else 0.0

        return consistency

    def _calculate_coherence(self, text: str) -> float:
        """
        Berechnet Response Coherence - wie kohärent ist der Text?

        Einfache Heuristik basierend auf Satzlänge und Wiederholungen.
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 0.0

        # Durchschnittliche Satzlänge (optimal: 10-25 Wörter)
        avg_sentence_length = np.mean([len(s.split()) for s in sentences])
        length_score = 1.0 - abs(avg_sentence_length - 17.5) / 17.5
        length_score = max(0.0, min(1.0, length_score))

        # Repetition penalty
        all_words = text.lower().split()
        unique_words = set(all_words)
        repetition_ratio = len(unique_words) / len(all_words) if all_words else 1.0

        # Gewichtete Kombination
        coherence = 0.6 * length_score + 0.4 * repetition_ratio

        return coherence


class RAGEvaluationPipeline:
    """
    High-level Pipeline für RAG-System Evaluation
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Evaluation Pipeline

        Args:
            config: Konfiguration mit 'evaluation' Sektion
        """
        eval_config = config.get('evaluation', {})

        self.metrics_calculator = MetricsCalculator(
            bleu_smoothing=eval_config.get('bleu_smoothing', 'method1'),
            rouge_metrics=eval_config.get('rouge_metrics', ['rouge1', 'rouge2', 'rougeL']),
            custom_metrics=eval_config.get('custom_metrics', True)
        )

        self.output_format = eval_config.get('output_format', 'csv')
        self.save_detailed = eval_config.get('save_detailed_results', True)

        self.config = config

    def evaluate_rag_responses(self,
                               rag_results: List[Dict[str, Any]],
                               reference_answers: Dict[str, str]) -> List[EvaluationResult]:
        """
        Evaluiert RAG-System Antworten gegen Referenz-Antworten

        Args:
            rag_results: Liste von RAG-Pipeline Ergebnissen mit 'question_id', 'answer', 'method'
            reference_answers: Dict mapping question_id -> reference_answer

        Returns:
            Liste von EvaluationResult Objekten
        """
        evaluation_data = []

        for rag_result in rag_results:
            q_id = rag_result['question_id']

            if q_id not in reference_answers:
                logger.warning(f"Keine Referenz-Antwort für {q_id}")
                continue

            evaluation_data.append({
                'question_id': q_id,
                'method': rag_result['method'],
                'reference_text': reference_answers[q_id],
                'generated_text': rag_result['answer'],
                'difficulty': rag_result.get('difficulty', 'unknown'),
                'retrieved_docs': rag_result.get('retrieved_docs', None)
            })

        return self.metrics_calculator.batch_evaluate(evaluation_data)

    def generate_evaluation_report(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """
        Generiert umfassenden Evaluation-Report

        Args:
            results: Liste von EvaluationResult Objekten

        Returns:
            Detaillierter Report für Dokumentation
        """
        # Basis-Statistiken
        aggregate_stats = self.metrics_calculator.compute_aggregate_statistics(results)

        # Method comparison
        method_comparison = self._create_method_comparison(results)

        # Best/Worst performance analysis
        performance_analysis = self._analyze_performance_extremes(results)

        # Summary für schnellen Überblick
        summary = self._create_evaluation_summary(aggregate_stats)

        report = {
            'evaluation_summary': summary,
            'aggregate_statistics': aggregate_stats,
            'method_comparison': method_comparison,
            'performance_analysis': performance_analysis,
            'total_evaluations': len(results),
            'available_metrics': {
                'bleu': self.metrics_calculator.bleu_available,
                'rouge': self.metrics_calculator.rouge_available,
                'custom': self.metrics_calculator.custom_metrics
            }
        }

        return report

    def _create_method_comparison(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Erstellt direkten Vergleich zwischen Methoden"""
        methods = list(set(r.method for r in results))

        comparison = {}
        for metric in ['bleu_score', 'rouge1_f', 'rouge2_f', 'rougeL_f']:
            comparison[metric] = {}
            for method in methods:
                method_results = [r for r in results if r.method == method]
                scores = [getattr(r, metric) for r in method_results]
                comparison[metric][method] = {
                    'mean': float(np.mean(scores)) if scores else 0.0,
                    'count': len(scores)
                }

        return comparison

    def _analyze_performance_extremes(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Analysiert beste und schlechteste Performance"""
        if not results:
            return {}

        # Beste und schlechteste BLEU-Scores
        sorted_by_bleu = sorted(results, key=lambda x: x.bleu_score, reverse=True)

        best_results = sorted_by_bleu[:3]
        worst_results = sorted_by_bleu[-3:]

        return {
            'best_performance': [
                {
                    'question_id': r.question_id,
                    'method': r.method,
                    'bleu_score': r.bleu_score,
                    'difficulty': r.difficulty
                } for r in best_results
            ],
            'worst_performance': [
                {
                    'question_id': r.question_id,
                    'method': r.method,
                    'bleu_score': r.bleu_score,
                    'difficulty': r.difficulty
                } for r in worst_results
            ]
        }

    def _create_evaluation_summary(self, aggregate_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Erstellt Executive Summary der Evaluation"""
        if not aggregate_stats:
            return {}

        # Finde beste Methode
        best_method = None
        best_bleu = 0.0

        for method, stats in aggregate_stats.items():
            method_bleu = stats.get('bleu_score', {}).get('mean', 0.0)
            if method_bleu > best_bleu:
                best_bleu = method_bleu
                best_method = method

        summary = {
            'best_method': best_method,
            'best_bleu_score': best_bleu,
            'methods_compared': list(aggregate_stats.keys()),
            'total_questions': sum(stats.get('num_questions', 0) for stats in aggregate_stats.values())
        }

        return summary


# Convenience functions für direkten Import
def evaluate_rag_system(rag_results: List[Dict[str, Any]],
                        reference_answers: Dict[str, str],
                        config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function für komplette RAG-System Evaluation

    Args:
        rag_results: RAG-Pipeline Ergebnisse
        reference_answers: Referenz-Antworten
        config: Optional Konfiguration

    Returns:
        Vollständiger Evaluation-Report
    """
    if config is None:
        config = {
            'evaluation': {
                'rouge_metrics': ['rouge1', 'rouge2', 'rougeL'],
                'custom_metrics': True,
                'output_format': 'csv'
            }
        }

    pipeline = RAGEvaluationPipeline(config)
    results = pipeline.evaluate_rag_responses(rag_results, reference_answers)
    report = pipeline.generate_evaluation_report(results)

    return report


# Example usage
if __name__ == "__main__":
    # Demo der Evaluation Metrics
    print("=== RAG-Benchmark Evaluation Metrics Demo ===")

    # Beispiel-Daten
    reference_text = "RAG kombiniert Language Models mit externem Wissen für bessere Antworten."
    generated_text = "RAG verbindet Sprachmodelle mit externen Datenquellen um genauere Antworten zu erzeugen."

    try:
        calculator = MetricsCalculator()

        # Einzelne Metriken testen
        bleu = calculator.calculate_bleu_score(reference_text, generated_text)
        rouge_scores = calculator.calculate_rouge_scores(reference_text, generated_text)
        custom_scores = calculator.calculate_custom_rag_metrics(reference_text, generated_text)

        print(f"BLEU Score: {bleu:.3f}")
        print(f"ROUGE-1 F1: {rouge_scores.get('rouge1_f', 0):.3f}")
        print(f"ROUGE-2 F1: {rouge_scores.get('rouge2_f', 0):.3f}")
        print(f"ROUGE-L F1: {rouge_scores.get('rougeL_f', 0):.3f}")
        print(f"Custom Metrics: {custom_scores}")

        # Vollständige Evaluation
        eval_result = calculator.evaluate_single_answer(
            question_id="test_001",
            method="demo",
            reference_text=reference_text,
            generated_text=generated_text,
            difficulty="easy"
        )

        print(f"Vollständige Evaluation: {eval_result.bleu_score:.3f} BLEU")

    except Exception as e:
        print(f"Demo fehlgeschlagen: {e}")
        logger.error(f"Demo fehlgeschlagen: {e}")

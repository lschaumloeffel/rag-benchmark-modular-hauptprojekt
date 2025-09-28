"""
Evaluation and Metrics Module

Provides BLEU/ROUGE metrics calculation and automated
RAG system evaluation orchestration.
"""

from .metrics import MetricsCalculator, RAGEvaluationPipeline, evaluate_rag_system
from .evaluator import RAGEvaluator, ExperimentConfig, MethodResult, run_complete_rag_evaluation

__all__ = [
    # Metrics
    'MetricsCalculator',
    'RAGEvaluationPipeline',
    'evaluate_rag_system',

    # Evaluator
    'RAGEvaluator',
    'ExperimentConfig',
    'MethodResult',
    'run_complete_rag_evaluation'
]
#!/usr/bin/env python3
"""
RAG-Benchmark Main Experiment Runner

Orchestriert alle Module und führt komplette RAG-Benchmarks durch.

Usage:
    python run_experiment.py --config config/base_config.yaml
    python run_experiment.py --config config/base_config.yaml --methods vector,graph
    python run_experiment.py --quick-test

Author: Lukas Schaumlöffel
Master Informatik (HAW Hamburg)
"""

import argparse
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from src.validation.config_validator import validate_and_load_config
    from src.data.loader import DataLoader
    from src.data.preprocessor import TextPreprocessor
    from src.evaluation.evaluator import RAGEvaluator, ExperimentConfig
    from src.evaluation.metrics import MetricsCalculator
except ImportError as e:
    print(f"Error: Could not import required modules. {e}")
    print("Make sure all dependencies are installed and src/ directory exists.")
    sys.exit(1)


class ExperimentRunner:
    """
    Main orchestration class für RAG-Benchmark Experimente

    Koordiniert:
    - Konfiguration loading und validation
    - Data loading und preprocessing
    - Retrieval pipeline setup
    - Experiment execution
    - Results collection und reporting
    """

    def __init__(self, config_path: str, verbose: bool = False):
        """
        Initialize ExperimentRunner

        Args:
            config_path: Pfad zur Konfigurationsdatei
            verbose: Ausführliche Ausgaben aktivieren
        """
        self.config_path = config_path
        self.verbose = verbose
        self.config = None
        self.data_loader = None
        self.results = {}

        # Logging setup
        self._setup_logging()

        # Logo und Info ausgeben
        if verbose:
            self._print_banner()

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"ExperimentRunner initialisiert: {config_path}")

    def run_complete_experiment(self, override_methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Führt komplettes RAG-Benchmark Experiment durch

        Args:
            override_methods: Optional override für enabled methods

        Returns:
            Vollständiger Experiment-Report
        """
        try:
            start_time = time.time()
            self.logger.info("=== STARTE RAG-BENCHMARK EXPERIMENT ===")

            # 1. Konfiguration laden und validieren
            self._load_and_validate_config()

            # 2. Methods override falls gewünscht
            if override_methods:
                self.config['methods']['enabled'] = override_methods
                self.logger.info(f"Methods override: {override_methods}")

            # 3. Daten laden
            documents, test_questions = self._load_experiment_data()

            # 4. Experiment konfigurieren
            experiment_config = self._create_experiment_config(test_questions)

            # 5. Evaluator setup und Experiment durchführen
            evaluator = RAGEvaluator(self.config)
            evaluator.setup_experiment(experiment_config)

            # 6. Komplette Evaluation durchführen
            final_report = evaluator.run_complete_evaluation(documents)

            # 7. Zusätzliche Analyse und Zusammenfassung
            self._enhance_report(final_report)

            total_time = time.time() - start_time
            self.logger.info(f"=== EXPERIMENT ABGESCHLOSSEN ({total_time:.2f}s) ===")

            # 8. Finale Ausgabe
            if self.verbose:
                self._print_experiment_summary(final_report, total_time)

            return final_report

        except Exception as e:
            self.logger.error(f"Experiment fehlgeschlagen: {e}")
            raise

    def run_quick_test(self) -> Dict[str, Any]:
        """
        Führt schnellen Test mit reduziertem Dataset durch

        Returns:
            Test-Ergebnisse
        """
        self.logger.info("=== QUICK TEST MODUS ===")

        self._load_and_validate_config()

        # Quick test: nur baseline und vector, weniger Fragen
        self.config['methods']['enabled'] = ['baseline', 'vector']
        self.config['development']['quick_test_mode'] = True

        return self.run_complete_experiment(['baseline', 'vector'])

    def _load_and_validate_config(self) -> None:
        """Lädt und validiert die Konfiguration"""
        self.logger.info(f"Lade Konfiguration: {self.config_path}")

        try:
            self.config = validate_and_load_config(self.config_path)

            enabled_methods = self.config.get('methods', {}).get('enabled', [])
            self.logger.info(f"Aktivierte Methoden: {enabled_methods}")

            # Development mode check
            if self.config.get('development', {}).get('debug_mode', False):
                self.logger.warning("DEBUG MODUS aktiviert - nur für Entwicklung verwenden")

        except Exception as e:
            self.logger.error(f"Konfiguration invalid: {e}")
            raise

    def _load_experiment_data(self) -> tuple:
        """Lädt und preprocessed Experiment-Daten"""
        self.logger.info("Lade Experiment-Daten...")

        try:
            # DataLoader initialisieren
            data_base_path = self.config.get('data', {}).get('base_path', './data')
            self.data_loader = DataLoader(base_path=data_base_path)

            # FAQ-Korpus laden
            corpus_file = self.config['data']['corpus_file']
            documents = self.data_loader.load_faq_corpus(corpus_file)
            self.logger.info(f"FAQ-Korpus geladen: {len(documents)} Dokumente")

            # Test-Fragen laden
            questions_file = self.config['data']['questions_file']
            test_questions_df = self.data_loader.load_test_questions(questions_file)
            test_questions = test_questions_df.to_dict('records')
            self.logger.info(f"Test-Fragen geladen: {len(test_questions)} Fragen")

            # Quick test mode: reduziere Dataset
            if self.config.get('development', {}).get('quick_test_mode', False):
                test_sample_size = self.config.get('development', {}).get('test_sample_size', 3)
                test_questions = test_questions[:test_sample_size]
                self.logger.info(f"Quick test: reduziert auf {len(test_questions)} Fragen")

            # Validierung
            min_corpus_size = self.config['data'].get('min_corpus_size', 10)
            if len(documents) < min_corpus_size:
                raise ValueError(f"Korpus zu klein: {len(documents)} < {min_corpus_size}")

            return documents, test_questions

        except Exception as e:
            self.logger.error(f"Fehler beim Laden der Daten: {e}")
            raise

    def _create_experiment_config(self, test_questions: List[Dict[str, Any]]) -> ExperimentConfig:
        """Erstellt ExperimentConfig aus geladenen Daten"""

        # Experiment name mit timestamp
        experiment_name = f"{self.config['experiment']['name']}_{time.strftime('%Y%m%d_%H%M%S')}"

        # Reference answers aus Config extrahieren
        reference_answers = self.config['evaluation']['reference_answers']

        # Validierung: alle Test-Fragen müssen Reference-Answers haben
        missing_refs = []
        for question in test_questions:
            q_id = question['id']
            if q_id not in reference_answers:
                missing_refs.append(q_id)

        if missing_refs:
            self.logger.warning(f"Fehlende Reference-Answers für: {missing_refs}")

        return ExperimentConfig(
            experiment_name=experiment_name,
            methods=self.config['methods']['enabled'],
            test_questions=test_questions,
            reference_answers=reference_answers,
            output_dir=self.config['experiment']['output_dir'],
            llm_config=self.config['llm'],
            evaluation_config=self.config['evaluation']
        )

    def _enhance_report(self, report: Dict[str, Any]) -> None:
        """Erweitert den Report um zusätzliche Analysen"""

        # System-Informationen hinzufügen
        import platform
        import psutil

        system_info = {
            'python_version': platform.python_version(),
            'platform': platform.platform(),
            'cpu_count': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024 ** 3), 2)
        }

        report['system_info'] = system_info

        # Config-Zusammenfassung
        report['config_summary'] = {
            'config_file': self.config_path,
            'enabled_methods': self.config['methods']['enabled'],
            'llm_model': self.config['llm']['model'],
            'vector_model': self.config.get('retrieval', {}).get('vector', {}).get('model', 'N/A'),
            'evaluation_metrics': self.config['evaluation']['metrics']
        }

        # Performance-Zusammenfassung
        if 'experiment_metadata' in report:
            metadata = report['experiment_metadata']
            questions_per_second = metadata['total_questions'] / metadata['total_runtime']
            report['performance_summary'] = {
                'questions_per_second': round(questions_per_second, 2),
                'avg_time_per_question': round(metadata['total_runtime'] / metadata['total_questions'], 2),
                'total_methods_tested': len(metadata['methods_tested'])
            }

    def _setup_logging(self) -> None:
        """Konfiguriert Logging für das Experiment"""

        # Log level basierend auf verbose flag
        log_level = logging.DEBUG if self.verbose else logging.INFO

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)

        # Root logger konfigurieren
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.addHandler(console_handler)

        # File logging falls in config aktiviert
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)

        file_handler = logging.FileHandler(
            logs_dir / f"rag_experiment_{time.strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    def _print_banner(self) -> None:
        """Gibt Banner mit Projekt-Informationen aus"""
        banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                           RAG-Benchmark Hauptprojekt                         ║
║                                                                              ║
║                       Modulare Benchmarking-Architektur für                  ║
║                Retrievalmethoden in Retrieval-Augmented Generation           ║
║                                                                              ║
║    Author: Lukas Schaumlöffel                                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        print(banner)

    def _print_experiment_summary(self, report: Dict[str, Any], total_time: float) -> None:
        """Gibt Experiment-Zusammenfassung aus"""

        print("\n" + "=" * 80)
        print("                        EXPERIMENT SUMMARY")
        print("=" * 80)

        # Basis-Informationen
        if 'experiment_metadata' in report:
            metadata = report['experiment_metadata']
            print(f"Experiment: {metadata['name']}")
            print(f"Methoden:   {', '.join(metadata['methods_tested'])}")
            print(f"Fragen:     {metadata['total_questions']}")
            print(f"Laufzeit:   {total_time:.2f}s")
            print()

        # Beste Methode
        if 'evaluation_summary' in report:
            summary = report['evaluation_summary']
            best_method = summary.get('best_method', 'Unknown')
            best_score = summary.get('best_bleu_score', 0.0)
            print(f"Beste Methode: {best_method} (BLEU: {best_score:.4f})")
            print()

        # Method Comparison
        if 'method_comparison' in report and 'bleu_score' in report['method_comparison']:
            print("BLEU Score Vergleich:")
            bleu_comparison = report['method_comparison']['bleu_score']

            # Sortiere nach Score
            sorted_methods = sorted(bleu_comparison.items(),
                                    key=lambda x: x[1]['mean'], reverse=True)

            for method, stats in sorted_methods:
                score = stats['mean']
                count = stats['count']
                print(f"  {method:>12}: {score:.4f} (n={count})")
            print()

        # Performance
        if 'performance_summary' in report:
            perf = report['performance_summary']
            print(f"Performance: {perf['questions_per_second']:.2f} Fragen/Sekunde")
            print()

        # Output location
        output_dir = Path(self.config['experiment']['output_dir'])
        print(f"Ergebnisse gespeichert in: {output_dir.resolve()}")

        print("=" * 80)


def main():
    """Main entry point mit argument parsing"""

    parser = argparse.ArgumentParser(
        description="RAG-Benchmark Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py --config config/base_config.yaml
  python run_experiment.py --config config/base_config.yaml --methods vector,graph
  python run_experiment.py --quick-test
  python run_experiment.py --config config/base_config.yaml --verbose
        """
    )

    parser.add_argument(
        '--config', '-c',
        default='config/base_config.yaml',
        help='Pfad zur Konfigurationsdatei (default: config/base_config.yaml)'
    )

    parser.add_argument(
        '--methods', '-m',
        help='Comma-separated Liste der zu testenden Methoden (override config)'
    )

    parser.add_argument(
        '--quick-test', '-q',
        action='store_true',
        help='Führt schnellen Test mit reduziertem Dataset durch'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Ausführliche Ausgaben aktivieren'
    )

    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Nur Konfiguration validieren, kein Experiment ausführen'
    )

    args = parser.parse_args()

    # Config file existiert?
    if not Path(args.config).exists():
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)

    try:
        # Validate-only mode
        if args.validate_only:
            from src.validation.config_validator import validate_config_file
            result = validate_config_file(args.config)

            if result.is_valid:
                print("Konfiguration ist valid")
                return 0
            else:
                print("Konfiguration ist invalid:")
                for error in result.errors:
                    print(f"  - {error}")
                return 1

        # Experiment Runner initialisieren
        runner = ExperimentRunner(args.config, verbose=args.verbose)

        # Methods override parsen
        override_methods = None
        if args.methods:
            override_methods = [m.strip() for m in args.methods.split(',')]

        # Experiment ausführen
        if args.quick_test:
            report = runner.run_quick_test()
        else:
            report = runner.run_complete_experiment(override_methods)

        # Erfolg
        if report.get('evaluation_summary', {}).get('best_method'):
            return 0  # Success
        else:
            print("Warning: Experiment abgeschlossen aber keine klaren Ergebnisse")
            return 1

    except KeyboardInterrupt:
        print("\nExperiment durch Benutzer abgebrochen")
        return 130

    except Exception as e:
        print(f"Error: Experiment fehlgeschlagen: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""
Configuration Validator für RAG-Benchmark Experimente

Validiert und normalisiert die Konfigurationsdateien für alle Module.

Author: Lukas Schaumlöffel
"""

import yaml
import json
import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Container für Validation-Ergebnisse"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    normalized_config: Dict[str, Any] = None


class ConfigValidator:
    """
    Hauptklasse für Konfiguration-Validation und Normalisierung

    Features:
    - Schema-Validation für alle Config-Sektionen
    - Environment Variable Resolution
    - Default Value Assignment
    - Cross-reference Validation
    - Method-specific Validation
    - Development/Production Mode Checks
    """

    def __init__(self, strict_mode: bool = False):
        """
        Initialize ConfigValidator

        Args:
            strict_mode: Bei True werden Warnings als Errors behandelt
        """
        self.strict_mode = strict_mode
        self.schema = self._get_config_schema()
        self.required_env_vars = ['OPENAI_API_KEY']  # Optional aber empfohlen

        logger.info(f"ConfigValidator initialisiert (strict_mode: {strict_mode})")

    def validate_config_file(self, config_path: str) -> ValidationResult:
        """
        Validiert eine Konfigurationsdatei vollständig

        Args:
            config_path: Pfad zur YAML/JSON Konfigurationsdatei

        Returns:
            ValidationResult mit Validierungs-Ergebnis
        """
        try:
            # Config laden
            config = self._load_config_file(config_path)

            # Vollständige Validation
            return self.validate_config(config, config_path)

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Fehler beim Laden der Config: {e}"],
                warnings=[]
            )

    def validate_config(self, config: Dict[str, Any], source: str = "dict") -> ValidationResult:
        """
        Validiert eine Konfiguration gegen das Schema

        Args:
            config: Konfiguration als Dictionary
            source: Quelle der Konfiguration (für Error Messages)

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []

        # 1. Schema-Validation
        schema_errors, schema_warnings = self._validate_schema(config)
        errors.extend(schema_errors)
        warnings.extend(schema_warnings)

        # 2. Environment Variables checken
        env_warnings = self._check_environment_variables()
        warnings.extend(env_warnings)

        # 3. Method-spezifische Validation
        method_errors, method_warnings = self._validate_methods(config)
        errors.extend(method_errors)
        warnings.extend(method_warnings)

        # 4. Cross-Reference Validation
        ref_errors = self._validate_cross_references(config)
        errors.extend(ref_errors)

        # 5. Path Validation
        path_errors, path_warnings = self._validate_paths(config)
        errors.extend(path_errors)
        warnings.extend(path_warnings)

        # 6. Normalisierte Config erstellen
        normalized_config = self._normalize_config(config)

        # 7. Final validation
        is_valid = len(errors) == 0 and (not self.strict_mode or len(warnings) == 0)

        if not is_valid:
            logger.error(f"Konfiguration invalid: {len(errors)} Fehler, {len(warnings)} Warnungen")
        else:
            logger.info(f"Konfiguration valid: {len(warnings)} Warnungen")

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            normalized_config=normalized_config
        )

    def _load_config_file(self, config_path: str) -> Dict[str, Any]:
        """Lädt Konfiguration aus YAML oder JSON"""
        path = Path(config_path)

        if not path.exists():
            raise FileNotFoundError(f"Konfigurationsdatei nicht gefunden: {config_path}")

        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            elif path.suffix.lower() == '.json':
                return json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")

    def _validate_schema(self, config: Dict[str, Any]) -> tuple[List[str], List[str]]:
        """Validiert gegen das erwartete Schema"""
        errors = []
        warnings = []

        # Required sections checken
        required_sections = ['experiment', 'data', 'retrieval', 'llm', 'evaluation', 'methods']

        for section in required_sections:
            if section not in config:
                errors.append(f"Required section missing: {section}")
                continue

            # Section-spezifische Validation
            section_errors, section_warnings = self._validate_section(section, config[section])
            errors.extend([f"{section}.{err}" for err in section_errors])
            warnings.extend([f"{section}.{warn}" for warn in section_warnings])

        return errors, warnings

    def _validate_section(self, section_name: str, section_config: Dict[str, Any]) -> tuple[List[str], List[str]]:
        """Validiert eine einzelne Config-Sektion"""
        errors = []
        warnings = []

        expected_fields = self.schema.get(section_name, {})

        # Required fields checken
        for field, field_schema in expected_fields.items():
            if field_schema.get('required', False) and field not in section_config:
                errors.append(f"Required field missing: {field}")

            if field in section_config:
                # Type validation
                expected_type = field_schema.get('type')
                actual_value = section_config[field]

                if expected_type and not self._check_type(actual_value, expected_type):
                    errors.append(
                        f"Invalid type for {field}: expected {expected_type}, got {type(actual_value).__name__}")

                # Enum validation
                if 'enum' in field_schema and actual_value not in field_schema['enum']:
                    errors.append(f"Invalid value for {field}: {actual_value} not in {field_schema['enum']}")

                # Range validation
                if 'min' in field_schema and isinstance(actual_value, (int, float)) and actual_value < field_schema[
                    'min']:
                    errors.append(f"Value too small for {field}: {actual_value} < {field_schema['min']}")

                if 'max' in field_schema and isinstance(actual_value, (int, float)) and actual_value > field_schema[
                    'max']:
                    warnings.append(f"Value large for {field}: {actual_value} > {field_schema['max']}")

        return errors, warnings

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Überprüft ob Wert dem erwarteten Typ entspricht"""
        type_map = {
            'str': str,
            'int': int,
            'float': (int, float),  # int ist auch als float akzeptabel
            'bool': bool,
            'list': list,
            'dict': dict
        }

        expected = type_map.get(expected_type, str)
        return isinstance(value, expected)

    def _check_environment_variables(self) -> List[str]:
        """Überprüft verfügbare Environment Variables"""
        warnings = []

        # OpenAI API Key (optional aber empfohlen)
        if not os.getenv('OPENAI_API_KEY'):
            warnings.append("OPENAI_API_KEY environment variable not set - LLM features will be disabled")

        # Neo4j Password (falls Graph-Retrieval aktiviert)
        if not os.getenv('NEO4J_PASSWORD'):
            warnings.append("NEO4J_PASSWORD not set - using config value (less secure)")

        return warnings

    def _validate_methods(self, config: Dict[str, Any]) -> tuple[List[str], List[str]]:
        """Validiert method-spezifische Konfiguration"""
        errors = []
        warnings = []

        methods = config.get('methods', {})
        enabled_methods = methods.get('enabled', [])

        # Prüfen, ob aktivierte Methoden korrekte Config aufweisen
        retrieval_config = config.get('retrieval', {})

        for method in enabled_methods:
            if method == 'baseline':
                continue  # Baseline braucht keine spezielle Config

            if method not in retrieval_config:
                errors.append(f"Method '{method}' enabled but no configuration found in retrieval.{method}")

            # Method-spezifische Checks
            if method == 'vector':
                self._validate_vector_config(retrieval_config.get('vector', {}), errors, warnings)
            elif method == 'graph':
                self._validate_graph_config(retrieval_config.get('graph', {}), errors, warnings)
            elif method == 'hybrid':
                self._validate_hybrid_config(retrieval_config.get('hybrid', {}), errors, warnings)

        return errors, warnings

    def _validate_vector_config(self, vector_config: Dict[str, Any], errors: List[str], warnings: List[str]) -> None:
        """Validiert Vector-Retrieval Konfiguration"""
        # Model validation
        model = vector_config.get('model', '')
        if not model.startswith('sentence-transformers/'):
            warnings.append(f"Vector model '{model}' is not a sentence-transformers model")

        # Top-k reasonable?
        top_k = vector_config.get('top_k', 3)
        if top_k > 10:
            warnings.append(f"Vector top_k very high: {top_k}")

        # Similarity threshold
        threshold = vector_config.get('similarity_threshold', 0.0)
        if threshold > 0.8:
            warnings.append(f"Vector similarity threshold very high: {threshold}")

    def _validate_graph_config(self, graph_config: Dict[str, Any], errors: List[str], warnings: List[str]) -> None:
        """Validiert Graph-Retrieval Konfiguration"""
        # Neo4j connection
        neo4j_uri = graph_config.get('neo4j_uri', '')
        if not neo4j_uri.startswith('bolt://'):
            errors.append(f"Invalid Neo4j URI format: {neo4j_uri}")

        # Spacy model
        spacy_model = graph_config.get('spacy_model', '')
        if not spacy_model:
            errors.append("Graph retrieval requires spacy_model configuration")

    def _validate_hybrid_config(self, hybrid_config: Dict[str, Any], errors: List[str], warnings: List[str]) -> None:
        """Validiert Hybrid-Retrieval Konfiguration"""
        # Gewichtungen müssen sich zu 1.0 addieren
        vector_weight = hybrid_config.get('weight_vector', 0.6)
        graph_weight = hybrid_config.get('weight_graph', 0.4)

        total_weight = vector_weight + graph_weight
        if abs(total_weight - 1.0) > 0.01:  # Kleine Toleranz für Float-Arithmetik
            errors.append(f"Hybrid weights don't sum to 1.0: {vector_weight} + {graph_weight} = {total_weight}")

        # Fusion method validation
        fusion_method = hybrid_config.get('fusion_method', 'weighted_sum')
        valid_methods = ['weighted_sum', 'rrf', 'adaptive']
        if fusion_method not in valid_methods:
            errors.append(f"Invalid fusion_method: {fusion_method} not in {valid_methods}")

    def _validate_cross_references(self, config: Dict[str, Any]) -> List[str]:
        """Validiert Cross-References zwischen Config-Sektionen"""
        errors = []

        # Evaluation reference_answers sollten zu test questions passen
        evaluation = config.get('evaluation', {})
        reference_answers = evaluation.get('reference_answers', {})

        # Warne wenn sehr wenige Referenz-Antworten
        if len(reference_answers) < 5:
            errors.append(f"Very few reference answers: {len(reference_answers)} (recommend at least 5)")

        # Check reference answer format
        for q_id, answer in reference_answers.items():
            if not isinstance(answer, str) or len(answer.strip()) < 10:
                errors.append(f"Reference answer for {q_id} too short or invalid")

        return errors

    def _validate_paths(self, config: Dict[str, Any]) -> tuple[List[str], List[str]]:
        """Validiert Pfad-Konfigurationen"""
        errors = []
        warnings = []

        # Output directory
        output_dir = config.get('experiment', {}).get('output_dir', '../results')
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create output directory {output_dir}: {e}")

        # Data files
        data_config = config.get('data', {})
        corpus_file = data_config.get('corpus_file', 'faq_korpus.json')
        questions_file = data_config.get('questions_file', 'fragenliste.csv')

        # Relative paths basierend auf ../data/
        data_dir = Path('../data')
        if not (data_dir / corpus_file).exists():
            warnings.append(f"Corpus file not found: {corpus_file} (will be created if needed)")

        if not (data_dir / questions_file).exists():
            warnings.append(f"Questions file not found: {questions_file}")

        return errors, warnings

    def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalisiert Konfiguration mit Default-Werten"""
        normalized = config.copy()

        # Environment variable substitution
        normalized = self._resolve_environment_variables(normalized)

        # Default values
        normalized = self._apply_defaults(normalized)

        # Path normalization
        normalized = self._normalize_paths(normalized)

        return normalized

    def _resolve_environment_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Ersetzt Environment Variables in der Config"""

        def replace_env_vars(obj):
            if isinstance(obj, dict):
                return {k: replace_env_vars(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_env_vars(item) for item in obj]
            elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
                env_var = obj[2:-1]
                return os.getenv(env_var, obj)  # Fallback zu Original wen nicht gefunden
            else:
                return obj

        return replace_env_vars(config)

    def _apply_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Wendet Default-Werte auf Config an"""
        defaults = {
            'experiment': {
                'log_level': 'INFO',
                'save_intermediate_results': True
            },
            'llm': {
                'provider': 'langchain_openai',
                'max_tokens': 500,
                'temperature': 0.1
            },
            'retrieval': {
                'vector': {
                    'top_k': 3,
                    'similarity_threshold': 0.0
                },
                'graph': {
                    'top_k': 3,
                    'min_entity_score': 1
                },
                'hybrid': {
                    'weight_vector': 0.6,
                    'weight_graph': 0.4
                }
            },
            'evaluation': {
                'metrics': ['bleu', 'rouge1', 'rouge2', 'rougeL'],
                'output_format': 'csv'
            },
            'methods': {
                'enabled': ['baseline', 'vector']
            }
        }

        def merge_defaults(target, defaults):
            for key, value in defaults.items():
                if key not in target:
                    target[key] = value
                elif isinstance(value, dict) and isinstance(target[key], dict):
                    merge_defaults(target[key], value)
            return target

        return merge_defaults(config.copy(), defaults)

    def _normalize_paths(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalisiert alle Pfade in der Konfiguration"""
        # Convert relative paths to absolute
        if 'experiment' in config:
            output_dir = config['experiment'].get('output_dir', '../results')
            config['experiment']['output_dir'] = str(Path(output_dir).resolve())

        # Cache directory
        if 'performance' in config and 'cache_dir' in config['performance']:
            cache_dir = config['performance']['cache_dir']
            config['performance']['cache_dir'] = str(Path(cache_dir).resolve())

        return config

    def _get_config_schema(self) -> Dict[str, Dict[str, Any]]:
        """Definiert das erwartete Konfiguration-Schema"""
        return {
            'experiment': {
                'name': {'type': 'str', 'required': True},
                'version': {'type': 'str', 'required': False},
                'output_dir': {'type': 'str', 'required': True}
            },
            'data': {
                'corpus_file': {'type': 'str', 'required': True},
                'questions_file': {'type': 'str', 'required': True},
                'min_corpus_size': {'type': 'int', 'min': 1, 'max': 1000}
            },
            'retrieval': {
                'vector': {'type': 'dict', 'required': False},
                'graph': {'type': 'dict', 'required': False},
                'hybrid': {'type': 'dict', 'required': False}
            },
            'llm': {
                'model': {'type': 'str', 'required': True},
                'provider': {'type': 'str', 'enum': ['openai', 'langchain_openai']},
                'max_tokens': {'type': 'int', 'min': 50, 'max': 4000},
                'temperature': {'type': 'float', 'min': 0.0, 'max': 2.0}
            },
            'evaluation': {
                'metrics': {'type': 'list', 'required': True},
                'reference_answers': {'type': 'dict', 'required': True}
            },
            'methods': {
                'enabled': {'type': 'list', 'required': True}
            }
        }

    def generate_validation_report(self, result: ValidationResult) -> str:
        """Generiert einen detaillierten Validation-Report"""
        report = []
        report.append("=== RAG-Benchmark Configuration Validation Report ===")
        report.append("")

        if result.is_valid:
            report.append("Configuration is VALID")
        else:
            report.append("Configuration is INVALID")

        report.append(f"  Errors: {len(result.errors)}")
        report.append(f"  Warnings: {len(result.warnings)}")
        report.append("")

        if result.errors:
            report.append("ERRORS:")
            for i, error in enumerate(result.errors, 1):
                report.append(f"   {i}. {error}")
            report.append("")

        if result.warnings:
            report.append("WARNINGS:")
            for i, warning in enumerate(result.warnings, 1):
                report.append(f"   {i}. {warning}")
            report.append("")

        # Recommendations
        if result.warnings or result.errors:
            report.append("RECOMMENDATIONS:")

            if any("OPENAI_API_KEY" in w for w in result.warnings):
                report.append("   - Set OPENAI_API_KEY environment variable for LLM functionality")

            if any("reference answers" in e.lower() for e in result.errors):
                report.append("   - Add more comprehensive reference answers for evaluation")

            if any("neo4j" in e.lower() for e in result.errors):
                report.append("   - Ensure Neo4j is running if graph retrieval is enabled")

            if any("model" in e.lower() for e in result.errors + result.warnings):
                report.append("   - Verify model names and availability")

        return "\n".join(report)


# Convenience functions für direkten Import
def validate_config_file(config_path: str, strict: bool = False) -> ValidationResult:
    """
    Convenience function für Config-File Validation

    Args:
        config_path: Pfad zur Konfigurationsdatei
        strict: Strict mode (Warnings als Errors behandeln)

    Returns:
        ValidationResult
    """
    validator = ConfigValidator(strict_mode=strict)
    return validator.validate_config_file(config_path)


def validate_and_load_config(config_path: str) -> Dict[str, Any]:
    """
    Lädt und validiert Config, gibt normalisierte Version zurück

    Args:
        config_path: Pfad zur Konfigurationsdatei

    Returns:
        Normalisierte und validierte Konfiguration

    Raises:
        ValueError: Bei invalider Konfiguration
    """
    result = validate_config_file(config_path)

    if not result.is_valid:
        error_msg = f"Configuration invalid:\n" + "\n".join(result.errors)
        raise ValueError(error_msg)

    if result.warnings:
        logger.warning(f"Configuration warnings:\n" + "\n".join(result.warnings))

    return result.normalized_config


# Example usage
if __name__ == "__main__":
    # Demo des Config Validators
    print("=== RAG-Benchmark Config Validator Demo ===")

    # Beispiel-Validierung der base_config.yaml
    try:
        config_path = "../../config/base_config.yaml"

        validator = ConfigValidator(strict_mode=False)
        result = validator.validate_config_file(config_path)

        # Report generieren
        report = validator.generate_validation_report(result)
        print(report)

        if result.is_valid:
            print("\nKonfiguration erfolgreich validiert!")
            print(f"Normalisierte Config hat {len(result.normalized_config)} Hauptsektionen")
        else:
            print("\nKonfiguration invalid - siehe Fehler oben")

    except Exception as e:
        print(f"Validation fehlgeschlagen: {e}")
        logger.error(f"Demo fehlgeschlagen: {e}")
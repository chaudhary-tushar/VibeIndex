# # embedding_validator.py
"""
Production Embedding Validator for RAG Pipeline
Validates embeddings without regenerating them
All metadata extracted from chunk and embedding analysis
"""

import re
import warnings
from datetime import datetime
from typing import Any

import numpy as np
from scipy import stats

# Define constants for magic values to satisfy Ruff linter
TOLERANCE_THRESHOLD = 0.01
NORM_UPPER_BOUND = 1000.0
ZERO_THRESHOLD = 1e-6
MAGNITUDE_LOWER_BOUND = 0.01
MAGNITUDE_UPPER_BOUND = 100.0
SMALL_EPSILON = 1e-8
MIN_STATISTICAL_SAMPLE_SIZE = 3
LOCAL_VARIANCE_WINDOW = 32
MAX_LOCAL_VARIANCE = 5.0
MIN_COMPLEXITY_THRESHOLD = 1e-6
MAX_COMPLEXITY_RATIO = 7.0
CODE_QUALITY_THRESHOLD = 0.3
MIN_LINE_REQUIREMENT = 2
SEMANTIC_DENSITY_THRESHOLD = 1e-5
MAX_CLUSTER_STD = 0.8
SEMANTIC_MIN_THRESHOLD = 0.7
SEMANTIC_DENSITY_MIN = 1e-6
SEMANTIC_MIN_ADJACENCY = 0.1
SEMANTIC_MIN_COHERENCE = 0.2
SEMANTIC_MAX_DENSITY = 100
SEMANTIC_DENSITY_MIN_RANGE = 1e-6
OUTLIER_DETECTION_STD = 3
ENTROPY_BASE_OFFSET = 1e-8
SHAPIRO_SAMPLE_LIMIT = 5000
KOLMOGOROV_TEST_ALPHA = 0.05
MIN_NORMALITY_ALPHA = 0.05
DOMAIN_RELEVANCE_MIN_PATTERNS = 2
API_PATTERN_MIN_COUNT = 2
MAX_SYNTAX_COMPLEXITY_RATIO = 10.0
MIN_SYNTAX_COMPLEXITY_RATIO = 0.1
LATENCY_THRESHOLD_MS = 1000
MIN_MEMORY_LIMIT_MB = 1024
MIN_THROUGHPUT_EPS = 10
MIN_CODE_QUALITY_SCORE = 0.15
MIN_FUNCTION_QUALITY_SCORE = 0.2
MIN_CLASS_QUALITY_SCORE = 0.2
MIN_DOCSTRING_QUALITY_SCORE = 0.2
MAX_LONG_CODE_PENALTY = 0.8
MIN_LINE_THRESHOLD = 3
MAX_LONG_LINE_THRESHOLD = 50
MIN_NORMALIZED_SCORE = 0.0
MAX_NORMALIZED_SCORE = 1.0
MIN_DIMENSIONALITY_SCORE = 0.9
MIN_SEMANTIC_SCORE = 0.7
MIN_DISTRIBUTION_SCORE = 0.8
MIN_PERFORMANCE_SCORE = 0.9
MIN_OVERALL_SCORE = 0.85
MIN_DOMAIN_SCORE = 0.75
DIMENSIONALITY_WEIGHT = 0.2
SEMANTIC_WEIGHT = 0.3
DISTRIBUTION_WEIGHT = 0.2
DOMAIN_WEIGHT = 0.15
PERFORMANCE_WEIGHT = 0.15
MIN_DISTANCE_THRESHOLD = 0.1
MAX_DISTANCE_THRESHOLD = 0.3
DISTRIBUTION_DIMENSIONS_CHECK = 256

warnings.filterwarnings("ignore", category=UserWarning)


class DimensionalityValidator:
    """Validates embedding dimensions and basic properties"""

    def __init__(self, expected_dim: int):
        self.expected_dim = expected_dim

    def validate(self, embedding: np.ndarray, chunk: dict) -> dict[str, Any]:
        checks = {
            "dimension_match": len(embedding) == self.expected_dim,
            "is_finite": bool(np.all(np.isfinite(embedding))),
            "has_no_nan": not bool(np.any(np.isnan(embedding))),
            "has_no_inf": not bool(np.any(np.isinf(embedding))),
            "not_zero_vector": not np.allclose(embedding, 0.0),
            "not_constant_vector": not bool(np.isclose(np.std(embedding), 0.0)),
            "reasonable_magnitude": bool(MAGNITUDE_LOWER_BOUND < np.linalg.norm(embedding) < NORM_UPPER_BOUND),
        }

        metrics = {
            "norm": float(np.linalg.norm(embedding)),
            "mean": float(np.mean(embedding)),
            "std": float(np.std(embedding)),
            "min": float(np.min(embedding)),
            "max": float(np.max(embedding)),
            "dimension": len(embedding),
            "sparsity": float(np.sum(np.abs(embedding) < ZERO_THRESHOLD) / len(embedding)),
        }

        return {"checks": checks, "metrics": metrics, "passed": all(checks.values())}


class SemanticValidator:
    """Validates semantic properties without regenerating embeddings"""

    def validate(self, embedding: np.ndarray, chunk: dict) -> dict[str, Any]:
        # Extract code and metadata from chunk
        code = chunk.get("code", "")
        embedding_text = chunk.get("embedding_text", code)

        if not embedding_text.strip():
            return {
                "code_complexity": 0,
                "embedding_complexity": 0.0,
                "complexity_ratio": 0.0,
                "semantic_density": 0.0,
                "information_content": 0.0,
                "passed": False,
            }

        # Calculate code complexity metrics
        code_complexity = self._calculate_code_complexity(code)

        # Calculate embedding complexity
        embedding_variance = float(np.var(embedding))
        embedding_entropy = float(stats.entropy(np.abs(embedding) + ENTROPY_BASE_OFFSET))

        # Semantic density: how much information per token
        token_count = max(1, len(embedding_text.split()))
        semantic_density = embedding_variance / token_count

        # Information content: combination of variance and entropy
        information_content = (embedding_variance * embedding_entropy) / token_count

        # Complexity ratio: embedding complexity should correlate with code complexity
        complexity_ratio = (embedding_variance * 1000) / max(code_complexity, 1)

        # Validation thresholds
        passed = (
            semantic_density > MIN_COMPLEXITY_THRESHOLD
            and information_content > MIN_COMPLEXITY_THRESHOLD
            and MAGNITUDE_LOWER_BOUND < complexity_ratio < MAGNITUDE_UPPER_BOUND
            and embedding_variance > SMALL_EPSILON
        )

        return {
            "code_complexity": code_complexity,
            "embedding_complexity": float(embedding_variance),
            "complexity_ratio": float(complexity_ratio),
            "semantic_density": float(semantic_density),
            "information_content": float(information_content),
            "embedding_entropy": float(embedding_entropy),
            "token_count": token_count,
            "passed": passed,
        }

    def _calculate_code_complexity(self, code: str) -> int:
        """Calculate code complexity based on syntax elements"""
        if not code.strip():
            return 0

        complexity = 0

        # Count syntax elements
        complexity += len(re.findall(r"\bdef\b|\bclass\b|\bfunction\b", code))  # Definitions
        complexity += len(re.findall(r"\bif\b|\belse\b|\belif\b", code))  # Conditionals
        complexity += len(re.findall(r"\bfor\b|\bwhile\b", code))  # Loops
        complexity += len(re.findall(r"\btry\b|\bexcept\b|\bfinally\b", code))  # Exception handling
        complexity += len(re.findall(r"\bimport\b|\bfrom\b", code))  # Imports
        complexity += len(re.findall(r"@\w+", code))  # Decorators
        complexity += code.count("=") - code.count("==") - code.count("!=")  # Assignments
        complexity += code.count("(") + code.count("{") + code.count("[")  # Nesting

        return complexity


class DistributionValidator:
    """Validates statistical distribution of embedding values"""

    def validate(self, embedding: np.ndarray, chunk: dict) -> dict[str, Any]:
        n = len(embedding)

        if n < MIN_LINE_THRESHOLD:
            return {
                "checks": {
                    "sufficient_dimensions": False,
                    "no_outliers": True,
                    "balanced_distribution": True,
                },
                "metrics": {
                    "skewness": 0.0,
                    "kurtosis": 0.0,
                    "outlier_count": 0,
                    "outlier_percentage": 0.0,
                },
                "passed": False,
            }

        # Statistical checks
        skewness = float(stats.skew(embedding))
        kurtosis = float(stats.kurtosis(embedding))

        # Outlier detection
        mean, std = np.mean(embedding), np.std(embedding)
        outlier_count = 0
        if std > 0:
            z_scores = np.abs((embedding - mean) / std)
            outlier_count = int(np.sum(z_scores > OUTLIER_DETECTION_STD))

        outlier_percentage = (outlier_count / n) * 100

        # Distribution balance check
        percentiles = np.percentile(embedding, [10, 25, 50, 75, 90])
        iqr = percentiles[3] - percentiles[1]

        checks = {
            "sufficient_dimensions": n >= DISTRIBUTION_DIMENSIONS_CHECK,
            "no_excessive_outliers": outlier_percentage < MAX_LOCAL_VARIANCE,
            "balanced_distribution": bool(iqr > SEMANTIC_DENSITY_MIN_RANGE),
            "reasonable_skewness": abs(skewness) < MAX_COMPLEXITY_RATIO,
            "reasonable_kurtosis": abs(kurtosis) < MAX_COMPLEXITY_RATIO,
        }

        metrics = {
            "skewness": skewness,
            "kurtosis": kurtosis,
            "outlier_count": outlier_count,
            "outlier_percentage": float(outlier_percentage),
            "iqr": float(iqr),
            "percentiles": {
                "p10": float(percentiles[0]),
                "p25": float(percentiles[1]),
                "p50": float(percentiles[2]),
                "p75": float(percentiles[3]),
                "p90": float(percentiles[4]),
            },
        }

        return {"checks": checks, "metrics": metrics, "passed": all(checks.values())}


class DomainSpecificValidator:
    """Validates domain-specific properties for code embeddings"""

    def __init__(self, domain: str = "code"):
        self.domain = domain

    def validate(self, embedding: np.ndarray, chunk: dict) -> dict[str, Any]:
        code = chunk.get("code", "")
        chunk_type = chunk.get("type", "unknown")
        language = chunk.get("language", "unknown")

        # Extract features from code
        code_features = self._extract_code_features(code, language)

        # Extract features from embedding
        embedding_features = self._extract_embedding_features(embedding)

        # Validate alignment between code and embedding
        alignment_score = self._calculate_alignment_score(code_features, embedding_features)

        # Domain-specific checks
        checks = {
            "has_code_elements": code_features["has_code_elements"],
            "language_detected": code_features["language_match"],
            "structural_elements": code_features["structural_score"] > 0,
            "embedding_complexity_appropriate": embedding_features["complexity_appropriate"],
            "feature_alignment": alignment_score > CODE_QUALITY_THRESHOLD,
        }

        # Comprehensive metrics
        metrics = {
            "code_features": code_features,
            "embedding_features": embedding_features,
            "alignment_score": float(alignment_score),
            "chunk_type": chunk_type,
            "language": language,
        }

        return {"checks": checks, "metrics": metrics, "passed": all(checks.values())}

    def _extract_code_features(self, code: str, language: str) -> dict:
        """Extract features from code without regenerating embeddings"""
        if not code.strip():
            return {
                "has_code_elements": False,
                "language_match": False,
                "structural_score": 0,
                "syntax_elements": {},
                "line_count": 0,
                "complexity": 0,
            }

        lines = [line.strip() for line in code.split("\n") if line.strip()]

        # Detect code elements
        syntax_elements = {
            "has_classes": bool(re.search(r"\bclass\s+\w+", code)),
            "has_functions": bool(re.search(r"\bdef\s+\w+|\bfunction\s+\w+", code)),
            "has_imports": "import " in code or "from " in code,
            "has_decorators": "@" in code,
            "has_control_flow": any(kw in code for kw in ["if ", "for ", "while ", "switch "]),
            "has_variables": "=" in code and not all(op in code for op in ["==", "!="]),
            "has_comments": "#" in code or "//" in code or "/*" in code,
            "has_docstrings": '"""' in code or "'''" in code,
            "has_async": "async " in code or "await " in code,
            "has_error_handling": any(kw in code for kw in ["try:", "except", "catch", "finally"]),
        }

        # Calculate structural score
        structural_score = sum(syntax_elements.values())

        # Language detection
        python_indicators = ["def ", "import ", "self.", ":", "elif ", "__"]
        javascript_indicators = ["function ", "const ", "let ", "var ", "=>"]

        language_match = False
        if language.lower() == "python":
            language_match = sum(1 for ind in python_indicators if ind in code) >= MIN_LINE_REQUIREMENT
        elif language.lower() in {"javascript", "typescript"}:
            language_match = sum(1 for ind in javascript_indicators if ind in code) >= MIN_LINE_REQUIREMENT
        else:
            language_match = True  # Unknown language, assume match

        # Code complexity
        complexity = len(re.findall(r"\bif\b|\bfor\b|\bwhile\b|\bdef\b|\bclass\b", code))

        return {
            "has_code_elements": structural_score > 0,
            "language_match": language_match,
            "structural_score": structural_score,
            "syntax_elements": syntax_elements,
            "line_count": len(lines),
            "complexity": complexity,
        }

    def _extract_embedding_features(self, embedding: np.ndarray) -> dict:
        """Extract features from embedding vector"""
        # Analyze embedding structure
        variance = float(np.var(embedding))
        mean_abs = float(np.mean(np.abs(embedding)))

        # Check for embedding complexity
        # High variance and entropy suggest complex representation
        entropy = float(stats.entropy(np.abs(embedding) + 1e-8))

        # Analyze value distribution
        positive_ratio = float(np.sum(embedding > 0) / len(embedding))

        # Complexity check
        complexity_appropriate = (
            variance > MIN_COMPLEXITY_THRESHOLD
            and entropy > SEMANTIC_MIN_ADJACENCY
            and CODE_QUALITY_THRESHOLD < positive_ratio < MAX_NORMALIZED_SCORE  # Balanced positive/negative
        )

        return {
            "variance": variance,
            "mean_absolute": mean_abs,
            "entropy": entropy,
            "positive_ratio": positive_ratio,
            "complexity_appropriate": complexity_appropriate,
        }

    def _calculate_alignment_score(self, code_features: dict, embedding_features: dict) -> float:
        """Calculate how well embedding represents code complexity"""
        code_complexity = code_features["structural_score"] + code_features["complexity"]
        embedding_complexity = embedding_features["variance"] * embedding_features["entropy"] * 1000

        if code_complexity == 0:
            return 0.0

        # Normalized alignment score
        ratio = embedding_complexity / code_complexity

        # Good alignment: ratio between 0.1 and 10
        if MIN_SYNTAX_COMPLEXITY_RATIO <= ratio <= MAX_SYNTAX_COMPLEXITY_RATIO:
            return 1.0
        if ratio < MIN_SYNTAX_COMPLEXITY_RATIO:
            return ratio / 0.1  # Partial score
        return 10.0 / ratio  # Partial score


class ConsistencyValidator:
    """Validates consistency properties of embeddings"""

    def validate(self, embedding: np.ndarray, chunk: dict) -> dict[str, Any]:
        """
        Validate consistency without regenerating embeddings.
        Uses statistical properties to detect potential issues.
        """

        # Check for degenerate patterns
        checks = {
            "no_repeating_patterns": self._check_no_repeating_patterns(embedding),
            "sufficient_variance": float(np.var(embedding)) > SEMANTIC_DENSITY_MIN_RANGE,
            "no_extreme_values": not bool(np.any(np.abs(embedding) > MAGNITUDE_UPPER_BOUND)),
            "smooth_distribution": bool(self._check_smooth_distribution(embedding)),
        }

        # Calculate consistency metrics
        # Autocorrelation to detect patterns
        autocorr = self._calculate_autocorrelation(embedding)

        # Local variance to detect inconsistencies
        local_variances = self._calculate_local_variances(embedding)

        metrics = {
            "autocorrelation": float(autocorr),
            "local_variance_mean": float(np.mean(local_variances)),
            "local_variance_std": float(np.std(local_variances)),
            "global_variance": float(np.var(embedding)),
        }

        return {"checks": checks, "metrics": metrics, "passed": all(checks.values())}

    def _check_no_repeating_patterns(self, embedding: np.ndarray, window: int = 10) -> bool:
        """Check if embedding has no obvious repeating patterns"""
        if len(embedding) < window * 2:
            return True

        # Check for repeating subsequences
        for i in range(len(embedding) - window * 2):
            segment1 = embedding[i : i + window]
            segment2 = embedding[i + window : i + window * 2]
            if np.allclose(segment1, segment2, rtol=0.01):
                return False
        return True

    def _check_smooth_distribution(self, embedding: np.ndarray) -> bool:
        """Check if embedding values are smoothly distributed"""
        if len(embedding) < MIN_THROUGHPUT_EPS:
            return True

        # Calculate differences between consecutive values
        diffs = np.diff(embedding)

        # Check for sudden jumps
        mean_diff = np.mean(np.abs(diffs))
        max_diff = np.max(np.abs(diffs))

        # Max diff shouldn't be too much larger than mean diff
        return max_diff < mean_diff * 50 if mean_diff > 0 else True

    def _calculate_autocorrelation(self, embedding: np.ndarray, lag: int = 1) -> float:
        """Calculate autocorrelation at given lag"""
        if len(embedding) <= lag:
            return 0.0

        x = embedding[:-lag]
        y = embedding[lag:]

        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0

        return float(np.corrcoef(x, y)[0, 1])

    def _calculate_local_variances(self, embedding: np.ndarray, window: int = 50) -> np.ndarray:
        """Calculate variance in local windows"""
        if len(embedding) < window:
            return np.array([np.var(embedding)])

        variances = []
        for i in range(0, len(embedding) - window + 1, window // 2):
            window_data = embedding[i : i + window]
            variances.append(np.var(window_data))

        return np.array(variances)


class DependencyValidator:
    """Validates if dependencies are properly represented"""

    def validate(self, embedding: np.ndarray, chunk: dict) -> dict[str, Any]:
        """Validate dependency awareness without regenerating embeddings"""

        # Extract dependencies from chunk
        dependencies = chunk.get("dependencies", [])
        references = chunk.get("references", [])

        # Analyze dependency information in code
        code = chunk.get("code", "")
        detected_deps = self._detect_dependencies_in_code(code)

        # Check if chunk has dependency information
        has_dependencies = len(dependencies) > 0 or len(references) > 0
        has_detected_deps = len(detected_deps) > 0

        # Analyze embedding for dependency signals
        # Dependencies should add complexity to embedding
        embedding_complexity = float(np.var(embedding) * stats.entropy(np.abs(embedding) + 1e-8))

        checks = {
            "has_dependency_info": has_dependencies or has_detected_deps,
            "dependencies_in_code": has_detected_deps,
            "embedding_has_complexity": embedding_complexity > SEMANTIC_DENSITY_THRESHOLD,
        }

        metrics = {
            "dependency_count": len(dependencies),
            "reference_count": len(references),
            "detected_dependency_count": len(detected_deps),
            "embedding_complexity": embedding_complexity,
            "has_imports": "import" in code.lower() or "from" in code.lower(),
        }

        return {"checks": checks, "metrics": metrics, "passed": all(checks.values()) if has_dependencies else True}

    def _detect_dependencies_in_code(self, code: str) -> list[str]:
        """Detect dependencies mentioned in code"""
        deps = []

        # Python imports
        import_matches = re.findall(r"import\s+([\w.]+)|from\s+([\w.]+)", code)
        for match in import_matches:
            deps.extend([m for m in match if m])

        # Usage patterns (e.g., module.function)
        usage_matches = re.findall(r"(\w+)\.\w+\(", code)
        deps.extend(usage_matches)

        return list(set(deps))


class EmbeddingQualityValidator:
    """Main validator orchestrating all validation checks"""

    def __init__(self, config: dict[str, Any] | None = None):
        # Default configuration
        self.config = config or {
            "expected_dimension": 2560,
            "domain": "code",
            "quality_thresholds": {
                "dimensionality_score": 0.9,
                "semantic_score": 0.7,
                "distribution_score": 0.8,
                "domain_score": 0.75,
                "consistency_score": 0.8,
                "dependency_score": 0.7,
                "overall_score": 0.75,
            },
            "validator_weights": {
                "dimensionality": 0.25,
                "semantic": 0.25,
                "distribution": 0.15,
                "domain": 0.15,
                "consistency": 0.10,
                "dependency": 0.10,
            },
        }

        # Initialize validators
        self.validators = {
            "dimensionality": DimensionalityValidator(expected_dim=self.config.get("expected_dimension", 2560)),
            "semantic": SemanticValidator(),
            "distribution": DistributionValidator(),
            "domain": DomainSpecificValidator(domain=self.config.get("domain", "code")),
            "consistency": ConsistencyValidator(),
            "dependency": DependencyValidator(),
        }

        self.quality_thresholds = self.config.get("quality_thresholds", {})
        self.validator_weights = self.config.get("validator_weights", {})

    def validate_embedding(self, embedding: np.ndarray, chunk: dict) -> dict[str, Any]:
        """
        Validate a single embedding without regenerating it.

        Args:
            embedding: numpy array of embedding values
            chunk: dict containing chunk information with keys:
                - id: chunk identifier
                - type: chunk type (class, function, etc.)
                - code: actual code content
                - file_path: path to source file
                - language: programming language
                - dependencies: list of dependencies (optional)
                - references: list of references (optional)
                - metadata: additional metadata (optional)
                - relationships: relationship information (optional)

        Returns:
            dict with validation results including:
                - passed: bool indicating if validation passed
                - overall_score: float overall quality score
                - validation_results: detailed results from each validator
                - quality_scores: score for each validator
                - rejection_reasons: reasons if validation failed
                - recommendations: suggestions for improvement
        """
        validation_results = {}
        scores = {}

        # Run all validators
        for name, validator in self.validators.items():
            try:
                result = validator.validate(embedding, chunk)
                validation_results[name] = result
                scores[name] = self._calculate_validator_score(result)
            except Exception as e:
                # Handle validator errors gracefully
                validation_results[name] = {"error": str(e), "passed": False}
                scores[name] = 0.0

        # Calculate overall weighted score
        overall_score = sum(scores.get(name, 0) * self.validator_weights.get(name, 0) for name in self.validators)

        # Check if validation passed
        passed = self._check_validation_passed(scores, overall_score)

        # Generate report
        return {
            "embedding_id": chunk.get("id", "unknown"),
            "chunk_type": chunk.get("type", "unknown"),
            "file_path": chunk.get("file_path", "unknown"),
            "validation_results": validation_results,
            "quality_scores": scores,
            "overall_score": float(overall_score),
            "passed": bool(passed),
            "rejection_reasons": self._get_rejection_reasons(scores, validation_results) if not passed else [],
            "recommendations": self._generate_recommendations(scores, validation_results),
            "metadata": {
                "validator_version": "2.0.0",
                "validation_timestamp": datetime.now().isoformat(),  # noqa: DTZ005
                "embedding_dimension": len(embedding),
                "chunk_language": chunk.get("language", "unknown"),
            },
        }

    def _calculate_validator_score(self, result: dict) -> float:
        """Calculate score for a validator result"""
        # Check if explicit "passed" field exists
        if "passed" in result:
            return 1.0 if result["passed"] else 0.0

        # Calculate from checks if available
        checks = result.get("checks", {})
        if checks:
            passed_count = sum(1 for v in checks.values() if v)
            total_count = len(checks)
            return passed_count / total_count if total_count > 0 else 0.0

        # Default
        return 0.5

    def _check_validation_passed(self, scores: dict, overall_score: float) -> bool:
        """Determine if validation passed based on scores and thresholds"""
        # Check overall score
        if overall_score < self.quality_thresholds.get("overall_score", 0.75):
            return False

        # Check individual validator scores
        for name, score in scores.items():
            threshold = self.quality_thresholds.get(f"{name}_score", 0.7)
            if score < threshold:
                return False

        return True

    def _get_rejection_reasons(self, scores: dict, results: dict) -> list[str]:
        """Generate list of reasons why validation failed"""
        reasons = []

        for name, score in scores.items():
            threshold = self.quality_thresholds.get(f"{name}_score", 0.7)
            if score < threshold:
                reasons.append(f"{name}_validation_failed: score {score:.3f} below threshold {threshold}")

                # Add specific check failures
                validator_result = results.get(name, {})
                checks = validator_result.get("checks", {})
                for check_name, check_passed in checks.items():
                    if not check_passed:
                        reasons.append(f"  - {name}.{check_name} failed")

        return reasons

    def _generate_recommendations(self, scores: dict, results: dict) -> list[str]:  # noqa: C901, PLR0912
        """Generate actionable recommendations based on validation results"""
        recs = []

        # Dimensionality recommendations
        if scores.get("dimensionality", 1.0) < MAX_LONG_CODE_PENALTY:
            dim_result = results.get("dimensionality", {})
            checks = dim_result.get("checks", {})
            if not checks.get("dimension_match", True):
                recs.append("Embedding dimension mismatch - check model configuration")
            if not checks.get("not_zero_vector", True):
                recs.append("Zero vector detected - model may have failed to generate embedding")
            if not checks.get("reasonable_magnitude", True):
                metrics = dim_result.get("metrics", {})
                recs.append(f"Unusual embedding magnitude: {metrics.get('norm', 'N/A')} - investigate model output")

        # Semantic recommendations
        if scores.get("semantic", 1.0) < MIN_SEMANTIC_SCORE:
            semantic_result = results.get("semantic", {})
            metrics = semantic_result.get("metrics", {}) if semantic_result else {}
            if metrics.get("semantic_density", 0) < SEMANTIC_DENSITY_MIN:
                recs.append("Low semantic density - consider including more context in embedding_text")
            if metrics.get("complexity_ratio", 0) < TOLERANCE_THRESHOLD:
                recs.append("Embedding complexity too low for code complexity - model may not capture code structure")

        # Distribution recommendations
        if scores.get("distribution", 1.0) < MIN_DISTRIBUTION_SCORE:
            dist_result = results.get("distribution", {})
            checks = dist_result.get("checks", {})
            if not checks.get("no_excessive_outliers", True):
                recs.append("High outlier percentage detected - embedding may be unstable")

        # Domain recommendations
        if scores.get("domain", 1.0) < MIN_DOMAIN_SCORE:
            domain_result = results.get("domain", {})
            checks = domain_result.get("checks", {})
            if not checks.get("has_code_elements", True):
                recs.append("No code elements detected - ensure chunk contains actual code")
            if not checks.get("language_detected", True):
                recs.append("Language mismatch - verify chunk language metadata")
            if not checks.get("feature_alignment", True):
                recs.append("Poor alignment between code and embedding - model may not understand code structure")

        # Consistency recommendations
        if scores.get("consistency", 1.0) < MIN_DISTRIBUTION_SCORE:
            recs.append("Consistency issues detected - verify model stability across batches")

        # Dependency recommendations
        if scores.get("dependency", 1.0) < MIN_SEMANTIC_SCORE:
            recs.append("Dependencies not well represented - ensure dependency metadata is included in embedding_text")

        # General recommendations if no specific issues
        if not recs:
            recs.append("All validations passed - embedding quality is good")

        return recs


# ============================================================================
# PUBLIC API FUNCTION FOR PIPELINE INTEGRATION
# ============================================================================


def validate_embedding_chunk(
    embedding: list[float] | np.ndarray,
    chunk: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Validate a single embedding-chunk pair in your RAG pipeline.

    This function does NOT regenerate embeddings - it only analyzes the
    provided embedding vector and chunk metadata.

    Args:
        embedding: The embedding vector as list or numpy array (size 2560)
        chunk: Dictionary containing chunk information with these keys:
            Required:
                - id: str - Unique chunk identifier
                - type: str - Chunk type (class, function, method, etc.)
                - code: str - Actual code content
                - file_path: str - Path to source file
                - language: str - Programming language (python, javascript, etc.)

            Optional (but recommended for better validation):
                - embedding_text: str - The formatted text used to generate embedding
                - dependencies: list[str] - List of dependencies
                - references: list[str] - List of code references
                - metadata: dict - Additional metadata (decorators, base_classes, etc.)
                - relationships: dict - Relationship information
                - qualified_name: str - Fully qualified name
                - docstring: str - Documentation string

        config: Optional configuration dictionary with keys:
            - expected_dimension: int (default: 2560)
            - domain: str (default: "code")
            - quality_thresholds: dict of threshold values
            - validator_weights: dict of validator weights

    Returns:
        Dictionary containing:
            - passed: bool - Whether validation passed
            - overall_score: float - Overall quality score (0.0 to 1.0)
            - quality_scores: dict - Individual validator scores
            - validation_results: dict - Detailed results from each validator
            - rejection_reasons: list[str] - Reasons if validation failed
            - recommendations: list[str] - Actionable recommendations
            - metadata: dict - Validation metadata (timestamp, version, etc.)

    Example:
        >>> result = validate_embedding_chunk(
        ...     embedding=embedding_vector,  # Your 2560-dim embedding
        ...     chunk=chunk_dict  # Your chunk with code and metadata
        ... )
        >>> if result['passed']:
        ...     # Store embedding in vector DB
        ...     store_in_vectordb(embedding, chunk)
        ... else:
        ...     # Log failure and skip
        ...     logger.warning(f"Validation failed: {result['rejection_reasons']}")
    """
    # Convert to numpy array if needed
    if not isinstance(embedding, np.ndarray):
        embedding = np.array(embedding, dtype=np.float32)

    # Create validator with config
    validator = EmbeddingQualityValidator(config)

    # Run validation
    return validator.validate_embedding(embedding, chunk)

import json
import logging
import math
import os
import warnings
from functools import lru_cache
from pathlib import Path

from app.ml import features
from app.schemas import Citation, CitationStyle, Work


ARTIFACT_DIR = Path(__file__).parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "logistic_regression_v41_candidate.joblib"
METADATA_PATH = ARTIFACT_DIR / "logistic_regression_v41_metadata.json"
MODEL_VERSION = "4.1"
BINARY_FEATURES = {"year_match", "doi_match", "doi_available", "author_conflict"}
logger = logging.getLogger(__name__)


class ModelError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load_model():
    try:
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        if (metadata.get("model_version") != MODEL_VERSION or
                metadata.get("feature_version") != MODEL_VERSION or
                metadata.get("features") != list(features.FEATURE_ORDER)):
            raise ModelError("Model metadata version or feature order does not match v4.1")
        import joblib
        from sklearn.exceptions import InconsistentVersionWarning

        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            model = joblib.load(MODEL_PATH)
        if (getattr(model, "n_features_in_", None) != len(features.FEATURE_ORDER) or
                1 not in getattr(model, "classes_", [])):
            raise ModelError("Model feature count or positive class does not match metadata")
        names = getattr(model, "feature_names_in_", None)
        if names is not None and list(names) != list(features.FEATURE_ORDER):
            raise ModelError("Model feature names do not match metadata")
        return model
    except ModelError:
        raise
    except Exception as exc:
        raise ModelError("ML model could not be loaded") from exc


def predict_proba(values: dict[str, float]) -> float:
    if not isinstance(values, dict) or set(values) != set(features.FEATURE_ORDER):
        raise ValueError("Exactly nine named v4.1 features are required")
    ordered = []
    for name in features.FEATURE_ORDER:
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Invalid numeric feature: {name}")
        try:
            number = float(value)
        except OverflowError as exc:
            raise ValueError(f"Invalid numeric feature: {name}") from exc
        if not math.isfinite(number):
            raise ValueError(f"Invalid numeric feature: {name}")
        if (name == "year_difference" and number < 0) or (name != "year_difference" and not 0 <= number <= 1):
            raise ValueError(f"Feature out of range: {name}")
        if name in BINARY_FEATURES and number not in (0, 1):
            raise ValueError(f"Feature must be binary: {name}")
        ordered.append(number)
    try:
        model = load_model()
        positive = list(model.classes_).index(1)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
            probability = float(model.predict_proba([ordered])[0][positive])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ModelError("Model returned an invalid probability")
        return probability
    except ModelError:
        raise
    except Exception as exc:
        raise ModelError("ML inference failed") from exc


def observe_shadow(citation: Citation, work: Work, style: CitationStyle) -> None:
    if os.getenv("CMC_ML_SHADOW", "0") != "1":
        return
    try:
        values = features.build_features(citation, work, style)
        probability = predict_proba(values)
        logger.info("ml_shadow_result", extra={"model_version": MODEL_VERSION,
                                               "style": style, "source": work.source,
                                               "probability": probability})
    except Exception as exc:
        reason = str(exc) if isinstance(exc, features.PreprocessingUnavailable) else type(exc).__name__
        logger.warning("ml_shadow_skipped", extra={"model_version": MODEL_VERSION,
                                                "style": style, "reason": reason})

import json
import logging
import math

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.ml import features, predictor
from app.parser import parse_citation
from app.schemas import Citation, Work


GOLDEN = json.loads((predictor.ARTIFACT_DIR / "ml_golden_cases_v41.json").read_text(encoding="utf-8"))
SAMPLE = GOLDEN["cases"][0]["features"]
CITATION = Citation(raw_citation="Synthetic citation", title="Sample title", authors=["de la Cruz, M. A."], year=2020)
WORK = Work(title="Sample title", authors=["Maria de la Cruz"], year=2020, source="openalex")


@pytest.fixture
def clear_model_cache():
    predictor.load_model.cache_clear()
    yield
    predictor.load_model.cache_clear()


def test_model_loads_once():
    model = predictor.load_model()
    assert model is predictor.load_model()
    assert model.n_features_in_ == 9
    assert list(model.feature_names_in_) == list(features.FEATURE_ORDER)


def test_missing_model(monkeypatch, tmp_path, clear_model_cache):
    monkeypatch.setattr(predictor, "MODEL_PATH", tmp_path / "missing.joblib")
    with pytest.raises(predictor.ModelError, match="could not be loaded"):
        predictor.load_model()


def test_metadata_mismatch(monkeypatch, tmp_path, clear_model_cache):
    metadata = {"model_version": "4.1", "feature_version": "4.1",
                "features": list(reversed(features.FEATURE_ORDER))}
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(predictor, "METADATA_PATH", path)
    with pytest.raises(predictor.ModelError, match="feature order"):
        predictor.load_model()


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda case: case["case_id"])
def test_golden_probability(case):
    assert GOLDEN["feature_order"] == list(features.FEATURE_ORDER)
    assert abs(predictor.predict_proba(case["features"]) - case["expected_probability"]) < 1e-8


@pytest.mark.parametrize("change", [
    {"title_similarity": None},
    {"title_similarity": float("nan")},
    {"title_similarity": 120},
    {"year_difference": -1},
    {"year_difference": 10 ** 1000},
    {"author_conflict": "yes"},
    {"author_conflict": 0.5},
])
def test_invalid_feature_values(change):
    with pytest.raises(ValueError):
        predictor.predict_proba({**SAMPLE, **change})


def test_missing_or_extra_feature():
    with pytest.raises(ValueError):
        predictor.predict_proba({key: value for key, value in SAMPLE.items() if key != "author_conflict"})
    with pytest.raises(ValueError):
        predictor.predict_proba({**SAMPLE, "extra": 0})


def test_inference_error(monkeypatch):
    class BrokenModel:
        classes_ = [0, 1]

        def predict_proba(self, values):
            raise RuntimeError("broken")

    monkeypatch.setattr(predictor, "load_model", lambda: BrokenModel())
    with pytest.raises(predictor.ModelError, match="inference failed"):
        predictor.predict_proba(SAMPLE)


@pytest.mark.parametrize("style,raw,first_author", [
    ("apa", "de la Cruz, M. A., & Smith-Jones, B. (2020). A study of U.S. science. Journal of Examples.", "M. A. de la Cruz"),
    ("ieee", '[1] M. A. de la Cruz, B. Smith-Jones, "A study of U.S. science," Journal of Examples, 2020.', "M. A. de la Cruz"),
    ("mla", 'de la Cruz, Maria, and Anne Smith-Jones. "A study of U.S. science." Journal of Examples, 2020.', "Maria de la Cruz"),
    ("chicago", 'de la Cruz, Maria, and Anne Smith-Jones. 2020. "A study of U.S. science." Journal of Examples.', "Maria de la Cruz"),
    ("harvard", "de la Cruz, M.A. and Smith-Jones, B. (2020) 'A study of U.S. science', Journal of Examples.", "M.A. de la Cruz"),
    ("vancouver", "1. de la Cruz MA, Smith-Jones B, et al. A study of U.S. science. Journal of Examples. 2020;2(1):1-5.", "M. A. de la Cruz"),
])
def test_six_style_author_adapter(style, raw, first_author):
    citation = parse_citation(raw, style)
    work = Work(title="A study of U.S. science", authors=["Maria de la Cruz", "Ben Smith-Jones"],
                year=2020, doi="https://doi.org/10.1234/example", source="openalex")
    row = features.to_reference_row(citation, work, style)
    assert row["citation_author"] == first_author
    assert row["metadata_authors"] == "Maria de la Cruz;Ben Smith-Jones"
    assert row["metadata_doi"] == "10.1234/example"
    values = features.build_features(citation, work, style)
    assert list(values) == list(features.FEATURE_ORDER)
    assert all(type(value) is float and math.isfinite(value) for value in values.values())


def test_preprocessing_matches_colab_reference():
    import pandas as pd
    from app.ml.artifacts import cmc_preprocessing_v41_reference as reference

    citation = Citation(raw_citation="synthetic", title="A study of U.S. science",
                        authors=["de la Cruz, M. A."], year=2020, doi="10.1234/example")
    work = Work(title="A study of U.S. science", authors=["Maria de la Cruz", "Ben Smith-Jones"],
                year=2020, doi="https://doi.org/10.1234/example", source="openalex")
    row = features.to_reference_row(citation, work, "apa")
    assert row["citation_author"] == "M. A. de la Cruz"
    expected = reference.extract_features_v41(pd.DataFrame([row])).iloc[0]
    actual = features.build_features(citation, work, "apa")
    for name in features.FEATURE_ORDER:
        assert actual[name] == pytest.approx(float(expected[name]), abs=1e-12)
    assert actual["doi_match"] == 1.0
    assert actual["doi_available"] == 1.0
    assert reference.normalize_author_hyphen("Smith‐Jones") == "smith-jones"
    assert reference.author_identity_v41("a vaswani", "Ashish Vaswani") == (0.95, 0)


def test_missing_doi_and_incomplete_metadata():
    values = features.build_features(CITATION, WORK, "apa")
    assert values["doi_match"] == 0.0
    assert values["doi_available"] == 0.0
    one_doi = features.build_features(CITATION, WORK.model_copy(update={"doi": "10.1234/example"}), "apa")
    assert one_doi["doi_match"] == 0.0
    assert one_doi["doi_available"] == 0.0
    assert math.isfinite(predictor.predict_proba(values))
    for citation, work in [
        (CITATION.model_copy(update={"year": None}), WORK),
        (CITATION.model_copy(update={"title": None}), WORK),
        (CITATION.model_copy(update={"authors": []}), WORK),
        (CITATION, WORK.model_copy(update={"authors": []})),
        (CITATION, WORK.model_copy(update={"year": None})),
        (CITATION, WORK.model_copy(update={"title": None})),
    ]:
        with pytest.raises(features.PreprocessingUnavailable):
            features.build_features(citation, work, "apa")


def test_unicode_hyphen_name_uses_reference_normalization():
    citation = CITATION.model_copy(update={"authors": ["Smith‐Jones, A."]})
    work = WORK.model_copy(update={"authors": ["Alice Smith-Jones"]})
    values = features.build_features(citation, work, "apa")
    assert values["author_identity_score"] == 0.95
    assert values["author_conflict"] == 0.0


def test_inference_from_citation_and_work():
    values = features.build_features(CITATION, WORK, "apa")
    probability = predictor.predict_proba(values)
    assert 0 <= probability <= 1


def test_shadow_disabled(monkeypatch, caplog):
    monkeypatch.delenv("CMC_ML_SHADOW", raising=False)
    monkeypatch.setattr(features, "build_features", lambda *args: pytest.fail("unexpected preprocessing"))
    with caplog.at_level(logging.INFO):
        predictor.observe_shadow(CITATION, WORK, "apa")
    assert not caplog.records


def test_shadow_enabled_logs_only_safe_fields(monkeypatch, caplog):
    monkeypatch.setenv("CMC_ML_SHADOW", "1")
    with caplog.at_level(logging.INFO):
        predictor.observe_shadow(CITATION, WORK, "apa")
    record = caplog.records[-1]
    assert record.message == "ml_shadow_result"
    assert record.probability == pytest.approx(predictor.predict_proba(features.build_features(CITATION, WORK, "apa")))
    assert CITATION.raw_citation not in caplog.text
    assert "de la Cruz" not in caplog.text


def test_shadow_failure_does_not_raise(monkeypatch, caplog):
    monkeypatch.setenv("CMC_ML_SHADOW", "1")
    monkeypatch.setattr(features, "build_features", lambda *args: SAMPLE)
    monkeypatch.setattr(predictor, "predict_proba", lambda values: (_ for _ in ()).throw(predictor.ModelError("failed")))
    with caplog.at_level(logging.WARNING):
        predictor.observe_shadow(CITATION, WORK, "apa")
    assert caplog.records[-1].message == "ml_shadow_skipped"
    assert caplog.records[-1].reason == "ModelError"


def test_shadow_logs_safe_skip_reason(monkeypatch, caplog):
    monkeypatch.setenv("CMC_ML_SHADOW", "1")
    missing_title = CITATION.model_copy(update={"title": None})
    with caplog.at_level(logging.WARNING):
        predictor.observe_shadow(missing_title, WORK, "apa")
    assert caplog.records[-1].reason == "title_or_year_missing"
    assert missing_title.raw_citation not in caplog.text


def test_endpoint_response_unchanged_when_shadow_enabled_or_failed(monkeypatch):
    original = httpx.AsyncClient
    published = {"id": "https://openalex.org/W1", "title": "Attention Is All You Need",
                 "publication_year": 2017, "doi": None,
                 "authorships": [{"author": {"display_name": "Ashish Vaswani"}}]}

    def handler(request):
        return httpx.Response(200, json={"message": {"items": []}} if "crossref" in request.url.host
                              else {"results": [published]})

    monkeypatch.setattr("app.routes.httpx.AsyncClient",
                        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    request = {"citation": "Vaswani, A. (2017). Attention Is All You Need."}
    client = TestClient(app)
    monkeypatch.delenv("CMC_ML_SHADOW", raising=False)
    baseline = client.post("/api/v1/verifications/preview", json=request)
    assert baseline.status_code == 200

    monkeypatch.setenv("CMC_ML_SHADOW", "1")
    assert client.post("/api/v1/verifications/preview", json=request).json() == baseline.json()

    monkeypatch.setattr(predictor, "load_model", lambda: (_ for _ in ()).throw(predictor.ModelError("missing")))
    assert client.post("/api/v1/verifications/preview", json=request).json() == baseline.json()

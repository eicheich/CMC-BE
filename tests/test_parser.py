import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.parser import parse_citation


# Synthetic references exercise recognizable syntax; they are not publication claims.
STYLES = [
    ("apa", "de la Cruz, M. A., & Smith-Jones, B. (2020). A study of U.S. science. Journal of Examples, 2(1), 1-5. https://doi.org/10.1234/ABC",
     ["de la Cruz, M. A.", "Smith-Jones, B."]),
    ("ieee", '[1] M. A. de la Cruz, B. Smith-Jones, "A study of U.S. science," Journal of Examples, vol. 2, no. 1, pp. 1-5, 2020, doi:10.1234/ABC.',
     ["M. A. de la Cruz", "B. Smith-Jones"]),
    ("mla", 'de la Cruz, Maria, and Anne Smith-Jones. "A study of U.S. science." Journal of Examples, vol. 2, no. 1, 2020, pp. 1-5. https://doi.org/10.1234/ABC',
     ["de la Cruz, Maria", "Anne Smith-Jones"]),
    ("chicago", 'de la Cruz, Maria, and Anne Smith-Jones. 2020. "A study of U.S. science." Journal of Examples 2 (1): 1-5. https://doi.org/10.1234/ABC',
     ["de la Cruz, Maria", "Anne Smith-Jones"]),
    ("harvard", "de la Cruz, M.A. and Smith-Jones, B. (2020) 'A study of U.S. science', Journal of Examples, 2(1), pp. 1-5. doi:10.1234/ABC",
     ["de la Cruz, M.A.", "Smith-Jones, B."]),
    ("vancouver", '1. de la Cruz MA, Smith-Jones B, et al. A study of U.S. science. Journal of Examples. 2020;2(1):1-5. doi:10.1234/ABC',
     ["de la Cruz, M. A.", "Smith-Jones, B."]),
]


@pytest.mark.parametrize("style,raw,authors", STYLES)
def test_style_parser(style, raw, authors):
    parsed = parse_citation(raw, style)
    assert parsed.raw_citation == raw
    assert parsed.title == "A study of U.S. science"
    assert parsed.authors == authors
    assert parsed.year == 2020
    assert parsed.doi == "10.1234/abc"


@pytest.mark.parametrize("style,raw,_", STYLES)
def test_style_without_doi(style, raw, _):
    raw = raw.split(" https://doi.org/")[0].split(" doi:10.")[0]
    assert parse_citation(raw, style).doi is None


@pytest.mark.parametrize("style,_,__", STYLES)
def test_unparseable_and_incomplete(style, _, __):
    unknown = parse_citation("Unstructured note", style)
    assert (unknown.title, unknown.authors, unknown.year, unknown.doi) == (None, [], None, None)
    incomplete = parse_citation("Reference fragment 2020", style)
    assert incomplete.title is None
    assert incomplete.authors == []
    assert incomplete.year == 2020


def test_legacy_apa_parser_call():
    parsed = parse_citation("Vaswani, A., et al. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems.")
    assert (parsed.title, parsed.authors, parsed.year) == ("Attention Is All You Need", ["Vaswani, A."], 2017)


def test_vancouver_does_not_invent_authors_from_prose():
    parsed = parse_citation("An ordinary sentence. Another sentence. 2020.", "vancouver")
    assert parsed.authors == []
    assert parsed.title is None


@pytest.mark.parametrize("style,raw,_", STYLES)
def test_endpoint_passes_style_and_preserves_response(monkeypatch, style, raw, _):
    original = httpx.AsyncClient

    def handler(request):
        return httpx.Response(200, json={"message": {"items": []}} if "crossref" in request.url.host
                              else {"results": []})

    monkeypatch.setattr("app.routes.httpx.AsyncClient",
                        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))
    response = TestClient(app).post("/api/v1/verifications/preview", json={"citation": raw, "style": style})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"status", "match_score", "input", "best_match", "comparison", "reasons"}
    assert set(body["input"]) == {"raw_citation", "title", "authors", "year", "doi"}
    assert body["input"]["title"] == "A study of U.S. science"


def test_default_style_and_unsupported_style():
    schema = app.openapi()["components"]["schemas"]["PreviewRequest"]
    assert schema["properties"]["style"]["default"] == "apa"
    response = TestClient(app).post("/api/v1/verifications/preview",
                                    json={"citation": "Example", "style": "unknown"})
    assert response.status_code == 422

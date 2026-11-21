import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.metadata import normalize_crossref, normalize_openalex
from app.parser import parse_citation
from app.schemas import Citation, Work
from app.verification import compare, verify


REFERENCE = "Vaswani, A., et al. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems."
WORK = {"id": "https://openalex.org/W1", "title": "Attention Is All You Need",
        "publication_year": 2017, "doi": "https://doi.org/10.1234/example",
        "authorships": [{"author": {"display_name": "Ashish Vaswani"}}]}


def test_parse_doi_and_year():
    parsed = parse_citation("Smith, J. (2021). Paper Title. https://doi.org/10.1234/ABC.1")
    assert parsed.doi == "10.1234/abc.1"
    assert parsed.year == 2021
    assert parsed.title == "Paper Title"
    assert parsed.authors == ["Smith, J."]


def test_incomplete_citation():
    parsed = parse_citation("Some obscure reference")
    assert parsed.title is None
    assert parsed.authors == []
    assert parsed.year is None
    assert parsed.doi is None
    assert parsed.raw_citation == "Some obscure reference"
    assert parse_citation("doi:10.1234/2017").year is None


def test_metadata_normalization():
    crossref = normalize_crossref({"title": ["A Paper"], "author": [{"given": "Jane", "family": "Doe"}],
                                   "published": {"date-parts": [[2020, 1]]}, "DOI": "10.1234/a", "URL": "https://doi.org/10.1234/a"})
    openalex = normalize_openalex(WORK)
    assert (crossref.title, crossref.authors, crossref.year, crossref.doi, crossref.source) == (
        "A Paper", ["Jane Doe"], 2020, "10.1234/a", "crossref")
    assert (openalex.title, openalex.authors, openalex.year, openalex.source_url) == (
        "Attention Is All You Need", ["Ashish Vaswani"], 2017, "https://openalex.org/W1")


def test_title_similarity_and_doi_mismatch():
    citation = Citation(raw_citation="", title="Attention Is All You Need", doi="10.1234/wrong")
    work = Work(title="Attention Is All You Need", doi="10.1234/example", source="openalex")
    score, comparison, _ = compare(citation, work)
    assert comparison.title_similarity == 100
    assert comparison.doi_match is False
    assert score < 100


def run_with_mock(citation: str, handler):
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await verify(citation, client)
    return asyncio.run(run())


def test_no_candidates_after_successful_search():
    def handler(request):
        return httpx.Response(200, json={"message": {"items": []}} if "crossref" in request.url.host
                              else {"results": []})
    result = run_with_mock(REFERENCE, handler)
    assert result.status == "not_found"
    assert result.best_match is None


def test_all_providers_failed():
    result = run_with_mock(REFERENCE, lambda request: httpx.Response(503))
    assert result.status == "verification_unavailable"
    assert result.match_score is None


def test_doi_points_to_different_title():
    citation = "Vaswani, A. (2017). Attention Is All You Need. doi:10.1234/example"

    def handler(request):
        if "crossref" in request.url.host:
            return httpx.Response(404)
        return httpx.Response(200, json={**WORK, "title": "A Completely Different Publication"})

    result = run_with_mock(citation, handler)
    assert result.status == "metadata_mismatch"
    assert result.comparison.doi_match is True
    assert any("DOI resolves to a different publication" in reason for reason in result.reasons)


def test_matching_title_with_wrong_doi():
    citation = "Vaswani, A. (2017). Attention Is All You Need. doi:10.1234/wrong"

    def handler(request):
        if "/works/" in request.url.path:
            return httpx.Response(404)
        return httpx.Response(200, json={"message": {"items": []}} if "crossref" in request.url.host
                              else {"results": [WORK]})

    result = run_with_mock(citation, handler)
    assert result.status == "metadata_mismatch"
    assert result.comparison.doi_match is False


def test_invalid_provider_responses():
    result = run_with_mock(REFERENCE, lambda request: httpx.Response(200, text="not json"))
    assert result.status == "verification_unavailable"


@pytest.mark.parametrize("citation,work,status,reason", [
    (REFERENCE, WORK, "verified_match", None),
    (REFERENCE.replace("2017", "2022"), WORK, "metadata_mismatch", "citation states 2022"),
    (REFERENCE.replace("Vaswani, A.", "Goodfellow, I."), WORK, "metadata_mismatch", "Author mismatch: Goodfellow, I."),
    ("Smith, J. (2024). Quantum Banana Networks for Intergalactic Coffee Optimization. Journal of Impossible Computing.", WORK, "not_found", None),
    ("Vaswani, A. (2017). Attention Is All You Need.", WORK, "verified_match", None),
    (REFERENCE, {**WORK, "authorships": []}, "possible_match", None),
    (REFERENCE, {**WORK, "authorships": [{"author": {"display_name": "Vaswani, Ashish"}}]}, "verified_match", None),
    (REFERENCE, {**WORK, "publication_year": None}, "possible_match", None),
])
def test_matching_regressions(citation, work, status, reason):
    def handler(request):
        return httpx.Response(200, json={"message": {"items": []}} if "crossref" in request.url.host
                              else {"results": [work]})

    result = run_with_mock(citation, handler)
    assert result.status == status
    if reason:
        assert any(reason in item for item in result.reasons)
    if work.get("authorships") == []:
        assert result.comparison.author_match is None


@pytest.mark.parametrize("citation", ["", "   ", "x" * 2001])
def test_invalid_input(citation):
    response = TestClient(app).post("/api/v1/verifications/preview", json={"citation": citation})
    assert response.status_code == 422


def test_preview_endpoint_with_mocked_api(monkeypatch):
    original = httpx.AsyncClient
    calls = []

    def handler(request):
        calls.append(str(request.url))
        if "crossref" in request.url.host:
            return httpx.Response(200, json={"message": {"items": []}})
        return httpx.Response(200, json={"results": [WORK]})

    def client_factory(**kwargs):
        return original(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr("app.routes.httpx.AsyncClient", client_factory)
    response = TestClient(app).post("/api/v1/verifications/preview", json={"citation": REFERENCE})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "verified_match"
    assert body["best_match"]["source"] == "openalex"
    assert body["comparison"]["year_match"] is True
    assert len(calls) == 2

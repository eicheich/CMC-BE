import re

import httpx
from rapidfuzz import fuzz

from app.metadata import fetch_candidates
from app.parser import parse_citation
from app.schemas import Citation, Comparison, PreviewResponse, Work


def clean_doi(value: str | None) -> str | None:
    return re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value, flags=re.I).lower() if value else None


def compare(citation: Citation, work: Work) -> tuple[float, Comparison, list[str]]:
    result = Comparison()
    reasons: list[str] = []
    weighted = 0.0
    total = 0.0
    if citation.title:
        result.title_similarity = round(fuzz.ratio(citation.title.casefold(), (work.title or "").casefold()), 1)
        weighted += result.title_similarity * 0.6
        total += 0.6
        reasons.append(f"Title similarity: {result.title_similarity:.1f}%.")
    if citation.authors:
        names = [name.split(",")[0].strip().casefold() for name in citation.authors]
        candidates = [name.split()[-1].casefold() for name in work.authors if name]
        result.author_match = round(max((fuzz.ratio(name, candidate) for name in names for candidate in candidates), default=0), 1)
        weighted += result.author_match * 0.1
        total += 0.1
        reasons.append(f"Author similarity: {result.author_match:.1f}%.")
    if citation.year is not None:
        result.year_match = citation.year == work.year
        weighted += 10 if result.year_match else 0
        total += 0.1
        reasons.append("Publication year matches." if result.year_match else "Publication year differs or is absent.")
    if citation.doi:
        result.doi_match = clean_doi(citation.doi) == clean_doi(work.doi)
        weighted += 20 if result.doi_match else 0
        total += 0.2
        reasons.append("DOI matches." if result.doi_match else "DOI differs or is absent.")
    return round(weighted / total, 1) if total else 0.0, result, reasons


async def verify(raw: str, client: httpx.AsyncClient) -> PreviewResponse:
    citation = parse_citation(raw)
    works, any_success = await fetch_candidates(client, citation)
    if not works:
        return PreviewResponse(status="not_found" if any_success else "verification_unavailable",
                               input=citation, reasons=["No candidate found in available metadata sources." if any_success
                                                        else "Metadata sources are unavailable."])

    score, comparison, reasons, best = max(
        ((*compare(citation, work), work) for work in works), key=lambda entry: entry[0]
    )
    title = comparison.title_similarity
    meaningful_title = title is not None and title >= 65
    mismatch = (citation.doi and comparison.doi_match is False and meaningful_title) or (
        comparison.doi_match is True and title is not None and title < 70
    ) or (meaningful_title and comparison.year_match is False)
    if mismatch:
        status = "metadata_mismatch"
    elif score >= 85 and title is not None and title >= 85 and (
        comparison.doi_match is True or comparison.year_match is True or
        (comparison.author_match is not None and comparison.author_match >= 80)
    ):
        status = "verified_match"
    elif score >= 65 and (meaningful_title or comparison.doi_match is True):
        status = "possible_match"
    else:
        status = "not_found"
        best = None
        comparison = Comparison()
        score = None
        reasons = ["No sufficiently relevant candidate found."]
    return PreviewResponse(status=status, match_score=score, input=citation,
                           best_match=best, comparison=comparison, reasons=reasons)

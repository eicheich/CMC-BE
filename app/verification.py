import re
import unicodedata

import httpx
from rapidfuzz import fuzz

from app.metadata import fetch_candidates
from app.ml.predictor import observe_shadow
from app.parser import parse_citation
from app.schemas import Citation, CitationStyle, Comparison, PreviewResponse, Work


TITLE_RELEVANT = 65
TITLE_STRONG = 85
TITLE_DOI_CONFLICT = 70
AUTHOR_CONFLICT = 65
AUTHOR_STRONG = 85
SURNAME_STRONG = 90
AUTHOR_INITIAL_DIFFERENT = 70
AUTHOR_SURNAME_ONLY = 80
SCORE_VERIFIED = 85
TITLE_WEIGHT = 0.6
AUTHOR_WEIGHT = 0.1
YEAR_WEIGHT = 0.1
DOI_WEIGHT = 0.2


def clean_doi(value: str | None) -> str | None:
    return re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value, flags=re.I).lower() if value else None


def author_identity(name: str) -> tuple[str, str | None]:
    normalized = "".join(char for char in unicodedata.normalize("NFKD", name.casefold())
                         if not unicodedata.combining(char))
    if "," in normalized:
        family, given = normalized.split(",", 1)
    else:
        parts = normalized.split()
        family, given = (parts[-1], " ".join(parts[:-1])) if parts else ("", "")
    family_words = re.findall(r"\w+", family)
    initial = next((char for char in given if char.isalpha()), None)
    return (family_words[-1] if family_words else "", initial)


def author_similarity(cited: list[str], published: list[str]) -> float | None:
    pairs = [(author_identity(left), author_identity(right))
             for left in cited for right in published if right.strip()]
    scores = []
    for (left_family, left_initial), (right_family, right_initial) in pairs:
        if not left_family or not right_family:
            continue
        surname_score = fuzz.ratio(left_family, right_family)
        if surname_score >= SURNAME_STRONG and left_initial and right_initial:
            scores.append(surname_score if left_initial == right_initial else AUTHOR_INITIAL_DIFFERENT)
        elif surname_score >= SURNAME_STRONG:
            scores.append(min(surname_score, AUTHOR_SURNAME_ONLY))
        else:
            scores.append(surname_score)
    return round(max(scores), 1) if scores else None


def compare(citation: Citation, work: Work) -> tuple[float, Comparison, list[str]]:
    result = Comparison()
    reasons: list[str] = []
    weighted = 0.0
    total = 0.0
    if citation.title and work.title:
        result.title_similarity = round(fuzz.ratio(citation.title.casefold(), work.title.casefold()), 1)
        weighted += result.title_similarity * TITLE_WEIGHT
        total += TITLE_WEIGHT
        reasons.append(f"Title similarity: {result.title_similarity:.1f}%.")
    if citation.authors:
        result.author_match = author_similarity(citation.authors, work.authors)
        if result.author_match is not None:
            weighted += result.author_match * AUTHOR_WEIGHT
            total += AUTHOR_WEIGHT
            reasons.append(f"Author similarity: {result.author_match:.1f}%.")
    if citation.year is not None and work.year is not None:
        result.year_match = citation.year == work.year
        weighted += 100 * YEAR_WEIGHT if result.year_match else 0
        total += YEAR_WEIGHT
        if result.year_match:
            reasons.append("Publication year matches.")
    if citation.doi and work.doi:
        result.doi_match = clean_doi(citation.doi) == clean_doi(work.doi)
        weighted += 100 * DOI_WEIGHT if result.doi_match else 0
        total += DOI_WEIGHT
        if result.doi_match:
            reasons.append("DOI matches.")
    return round(weighted / total, 1) if total else 0.0, result, reasons


def consistency_issues(citation: Citation, work: Work, comparison: Comparison) -> list[str]:
    title = comparison.title_similarity
    relevant = (title is not None and title >= TITLE_RELEVANT) or comparison.doi_match is True
    if not relevant:
        return []
    issues = []
    if comparison.doi_match is True and title is not None and title < TITLE_DOI_CONFLICT:
        year = str(work.year) if work.year is not None else "year unknown"
        issues.append(f"DOI resolves to a different publication: {work.title} ({year}).")
    elif comparison.doi_match is False and title is not None and title >= TITLE_RELEVANT:
        issues.append(f"DOI mismatch: citation states {citation.doi}, but metadata reports {work.doi}.")
    if comparison.year_match is False:
        issues.append(f"Publication year mismatch: citation states {citation.year}, but metadata reports {work.year}.")
    if (len(citation.authors) == 1 and comparison.author_match is not None
            and comparison.author_match < AUTHOR_CONFLICT):
        issues.append(f"Author mismatch: {citation.authors[0]} was not found among the publication authors.")
    return issues


def classify(citation: Citation, comparison: Comparison, score: float, issues: list[str]) -> str:
    title = comparison.title_similarity
    relevant = (title is not None and title >= TITLE_RELEVANT) or comparison.doi_match is True
    if not relevant:
        return "not_found"
    if issues:
        return "metadata_mismatch"
    complete = (not citation.authors or comparison.author_match is not None) and (
        citation.year is None or comparison.year_match is not None) and (
        citation.doi is None or comparison.doi_match is not None)
    strong_identity = comparison.doi_match is True or (
        comparison.author_match is not None and comparison.author_match >= AUTHOR_STRONG)
    if complete and strong_identity and score >= SCORE_VERIFIED and title is not None and title >= TITLE_STRONG:
        return "verified_match"
    return "possible_match"


async def verify(raw: str, client: httpx.AsyncClient, style: CitationStyle = "apa") -> PreviewResponse:
    citation = parse_citation(raw, style)
    works, any_success = await fetch_candidates(client, citation)
    if not works:
        return PreviewResponse(status="not_found" if any_success else "verification_unavailable",
                               input=citation, reasons=["No candidate found in available metadata sources." if any_success
                                                        else "Metadata sources are unavailable."])

    ranked = []
    for work in works:
        score, comparison, reasons = compare(citation, work)
        issues = consistency_issues(citation, work, comparison)
        relevant = (comparison.title_similarity is not None and
                    comparison.title_similarity >= TITLE_RELEVANT) or comparison.doi_match is True
        ranked.append((relevant, score, not issues, work, comparison, reasons, issues))
    relevant, score, _, best, comparison, reasons, issues = max(ranked, key=lambda item: item[:3])
    status = classify(citation, comparison, score, issues)
    if not relevant or status == "not_found":
        return PreviewResponse(status="not_found", input=citation,
                               reasons=["No sufficiently relevant candidate found."])
    response = PreviewResponse(status=status, match_score=score, input=citation,
                               best_match=best, comparison=comparison, reasons=reasons + issues)
    observe_shadow(citation, best, style)
    return response

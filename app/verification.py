import re
import unicodedata
from typing import Literal, NamedTuple

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

FieldAssessment = Literal["match", "mismatch", "unavailable"]
IdentityAssessment = Literal["same", "different", "uncertain"]


class MetadataAssessment(NamedTuple):
    title: FieldAssessment
    author: FieldAssessment
    year: FieldAssessment
    doi: FieldAssessment
    complete: bool


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


def assess_identity(comparison: Comparison) -> IdentityAssessment:
    title = comparison.title_similarity
    author = comparison.author_match
    if title is None:
        return "uncertain"
    if title < TITLE_DOI_CONFLICT and comparison.doi_match is True:
        return "different"
    if title < TITLE_RELEVANT:
        return "different"
    if title >= TITLE_STRONG:
        if author is not None and author >= AUTHOR_STRONG:
            return "uncertain" if comparison.year_match is False and comparison.doi_match is False else "same"
        if comparison.doi_match is True and comparison.year_match is True:
            return "same"
        if title == 100 and comparison.year_match is True and comparison.doi_match is not False:
            return "same"
        return "uncertain"
    if (author is not None and author < AUTHOR_CONFLICT and comparison.year_match is False
            and comparison.doi_match is False):
        return "different"
    return "uncertain"


def assess_metadata(citation: Citation, comparison: Comparison) -> MetadataAssessment:
    def similarity_state(value: float | None, minimum: float) -> FieldAssessment:
        if value is None:
            return "unavailable"
        return "match" if value >= minimum else "mismatch"

    def boolean_state(value: bool | None) -> FieldAssessment:
        if value is None:
            return "unavailable"
        return "match" if value else "mismatch"

    complete = ((not citation.authors or comparison.author_match is not None)
                and (citation.year is None or comparison.year_match is not None)
                and (citation.doi is None or comparison.doi_match is not None))
    return MetadataAssessment(
        title=similarity_state(comparison.title_similarity, TITLE_RELEVANT),
        author=similarity_state(comparison.author_match, AUTHOR_CONFLICT),
        year=boolean_state(comparison.year_match), doi=boolean_state(comparison.doi_match),
        complete=complete)


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


def classify(citation: Citation, comparison: Comparison, score: float, issues: list[str], *,
             identity: IdentityAssessment | None = None,
             metadata: MetadataAssessment | None = None) -> str:
    title = comparison.title_similarity
    relevant = (title is not None and title >= TITLE_RELEVANT) or comparison.doi_match is True
    if not relevant:
        return "not_found"
    identity = identity or assess_identity(comparison)
    metadata = metadata or assess_metadata(citation, comparison)
    if identity == "different":
        return "metadata_mismatch" if comparison.doi_match is True and title < TITLE_DOI_CONFLICT else "not_found"
    if identity == "uncertain":
        return "possible_match"
    if "mismatch" in metadata[:4] or issues:
        return "metadata_mismatch"
    strong_identity = comparison.doi_match is True or (
        comparison.author_match is not None and comparison.author_match >= AUTHOR_STRONG)
    if metadata.complete and strong_identity and score >= SCORE_VERIFIED and title is not None and title >= TITLE_STRONG:
        return "verified_match"
    return "possible_match"


async def verify(raw: str, client: httpx.AsyncClient, style: CitationStyle = "apa") -> PreviewResponse:
    citation = parse_citation(raw, style)

    def title_author_support(comparison: Comparison) -> bool:
        return (comparison.title_similarity is not None and comparison.title_similarity >= TITLE_STRONG
                and (not citation.authors or comparison.author_match is not None
                     and comparison.author_match >= AUTHOR_STRONG))

    works, any_success = await fetch_candidates(
        client, citation, lambda work: title_author_support(compare(citation, work)[1]))
    if not works:
        return PreviewResponse(status="not_found" if any_success else "verification_unavailable",
                               input=citation, reasons=["No candidate found in available metadata sources." if any_success
                                                        else "Metadata sources are unavailable."])

    ranked = []
    for work in works:
        score, comparison, reasons = compare(citation, work)
        identity = assess_identity(comparison)
        metadata = assess_metadata(citation, comparison)
        issues = consistency_issues(citation, work, comparison)
        relevant = (comparison.title_similarity is not None and
                    comparison.title_similarity >= TITLE_RELEVANT) or comparison.doi_match is True
        rank = {"same": 2, "uncertain": 1, "different": 0}[identity]
        ranked.append((relevant, rank, title_author_support(comparison), score, not issues,
                       work, comparison, reasons, issues, identity, metadata))
    relevant, _, _, score, _, best, comparison, reasons, issues, identity, metadata = max(
        ranked, key=lambda item: item[:5])
    status = classify(citation, comparison, score, issues, identity=identity, metadata=metadata)
    if not relevant or status == "not_found":
        return PreviewResponse(status="not_found", input=citation,
                               reasons=["No sufficiently relevant candidate found."])
    response = PreviewResponse(status=status, match_score=score, input=citation,
                               best_match=best, comparison=comparison, reasons=reasons + issues)
    observe_shadow(citation, best, style)
    return response

import re

from app.schemas import Citation, CitationStyle, Work


FEATURE_ORDER = (
    "title_similarity",
    "author_similarity",
    "year_match",
    "year_difference",
    "doi_match",
    "doi_available",
    "author_token_match",
    "author_identity_score",
    "author_conflict",
)

SURNAME_FIRST_STYLES = {"apa", "harvard", "chicago", "mla", "vancouver"}
STYLES = SURNAME_FIRST_STYLES | {"ieee"}
DOI_RE = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/)?(10\.\d{4,9}/\S+)$", re.I)


class PreprocessingUnavailable(RuntimeError):
    pass


def _given_first(name: str, *, surname_first: bool) -> str:
    name = " ".join(name.split())
    if surname_first:
        if name.count(",") != 1:
            raise PreprocessingUnavailable("author_name_format")
        family, given = (part.strip() for part in name.split(",", 1))
        if not family or not given:
            raise PreprocessingUnavailable("author_name_format")
        return f"{given} {family}"
    if "," in name or len(name.split()) < 2:
        raise PreprocessingUnavailable("author_name_format")
    return name


def _doi(value: str | None) -> str | None:
    if not value:
        return None
    match = DOI_RE.fullmatch(value.strip().rstrip(".,;)]}"))
    if not match:
        raise PreprocessingUnavailable("doi_format")
    return match.group(1).lower()


def to_reference_row(citation: Citation, work: Work, style: CitationStyle) -> dict:
    if style not in STYLES:
        raise ValueError(f"Unsupported citation style: {style}")
    if not citation.title or not work.title or citation.year is None or work.year is None:
        raise PreprocessingUnavailable("title_or_year_missing")
    if not citation.authors or not work.authors:
        raise PreprocessingUnavailable("authors_missing")
    cited_author = _given_first(citation.authors[0], surname_first=style in SURNAME_FIRST_STYLES)
    metadata_authors = ";".join(_given_first(name, surname_first="," in name) for name in work.authors)
    return {
        "citation_title": citation.title,
        "metadata_title": work.title,
        "citation_author": cited_author,
        "metadata_authors": metadata_authors,
        "citation_year": citation.year,
        "metadata_year": work.year,
        "citation_doi": _doi(citation.doi),
        "metadata_doi": _doi(work.doi),
    }


def build_features(citation: Citation, work: Work, style: CitationStyle) -> dict[str, float]:
    row = to_reference_row(citation, work, style)
    # Keep the eight Colab functions as the single source for feature calculations.
    import pandas as pd
    from app.ml.artifacts import cmc_preprocessing_v41_reference as reference

    output = reference.extract_features_v41(pd.DataFrame([row])).iloc[0]
    return {name: float(output[name]) for name in FEATURE_ORDER}

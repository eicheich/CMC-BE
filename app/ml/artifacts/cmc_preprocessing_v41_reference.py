"""
Check My Citation — ML v4.1 preprocessing reference.

Exported from the original Google Colab notebook.

IMPORTANT:
- Preserve feature calculations for model compatibility.
- This file is a reference, not a production-ready module.
- Author formatting and missing metadata require separate tests.
"""

import re
import unicodedata

import pandas as pd
from rapidfuzz import fuzz

def clean_text(value):
    if pd.isna(value):
        return ""
    return " ".join(
        re.sub(r"[^\w\s]", " ", str(value).lower()).split()
    )


def first_author(value):
    if pd.isna(value):
        return ""
    return str(value).split(";")[0].strip()


def extract_features(df):
    rows = []

    for _, row in df.iterrows():
        citation_title = clean_text(row["citation_title"])
        metadata_title = clean_text(row["metadata_title"])

        citation_author = clean_text(row["citation_author"])
        metadata_author = clean_text(
            first_author(row["metadata_authors"])
        )

        citation_doi = row["citation_doi"]
        metadata_doi = row["metadata_doi"]

        has_citation_doi = pd.notna(citation_doi)
        has_metadata_doi = pd.notna(metadata_doi)

        rows.append({
            "title_similarity": fuzz.ratio(
                citation_title,
                metadata_title
            ) / 100,

            "author_similarity": fuzz.ratio(
                citation_author,
                metadata_author
            ) / 100,

            "year_match": int(
                row["citation_year"] == row["metadata_year"]
            ),

            "year_difference": abs(
                int(row["citation_year"])
                - int(row["metadata_year"])
            ),

            "doi_match": int(
                has_citation_doi
                and has_metadata_doi
                and str(citation_doi).lower()
                == str(metadata_doi).lower()
            ),

            "doi_available": int(
                has_citation_doi and has_metadata_doi
            )
        })

    return pd.DataFrame(rows)


def author_token_match(citation_author, metadata_authors):
    citation = clean_text(citation_author)

    if not citation:
        return 0.0

    authors = str(metadata_authors).split(";")

    best_score = 0.0

    for author in authors:
        full_name = clean_text(author)

        if not full_name:
            continue

        tokens = full_name.split()

        # Exact full-name match
        if citation == full_name:
            return 1.0

        # Match salah satu token nama
        if citation in tokens:
            best_score = max(best_score, 1.0)

        # Similarity dengan nama lengkap
        best_score = max(
            best_score,
            fuzz.ratio(citation, full_name) / 100
        )

    return best_score


def extract_features_v2(df):
    features = extract_features(df)

    features["author_token_match"] = [
        author_token_match(
            row["citation_author"],
            row["metadata_authors"]
        )
        for _, row in df.iterrows()
    ]

    return features


def normalize_author_hyphen(value):
    if pd.isna(value):
        return ""

    text = unicodedata.normalize(
        "NFKC", str(value)
    ).casefold()

    # Samakan variasi tanda hubung Unicode
    text = re.sub(
        r"[\u2010-\u2015\u2212]",
        "-",
        text
    )

    # Pertahankan tanda hubung di dalam nama
    text = re.sub(
        r"[^\w\s-]",
        " ",
        text
    )

    return " ".join(text.split())


def author_identity_v41(citation_author, metadata_authors):
    citation = normalize_author_hyphen(citation_author)
    citation_tokens = citation.split()

    if not citation_tokens:
        return 0.0, 0

    best_score = 0.0
    has_conflict = False

    for author in str(metadata_authors).split(";"):
        metadata = normalize_author_hyphen(author)
        tokens = metadata.split()

        if not tokens:
            continue

        score = fuzz.ratio(citation, metadata) / 100

        if citation == metadata:
            return 1.0, 0

        # Surname-only match
        if (
            len(citation_tokens) == 1
            and citation_tokens[0] == tokens[-1]
        ):
            score = max(score, 0.90)

        # Initial + surname match
        elif (
            len(citation_tokens) == 2
            and len(tokens) >= 2
            and citation_tokens[-1] == tokens[-1]
            and len(citation_tokens[0]) == 1
            and citation_tokens[0][0] == tokens[0][0]
        ):
            score = max(score, 0.95)

        # Full name with conflicting first initial
        elif (
            len(citation_tokens) >= 2
            and len(tokens) >= 2
            and citation_tokens[-1] == tokens[-1]
            and citation_tokens[0][0] != tokens[0][0]
        ):
            has_conflict = True
            score = min(score, 0.50)

        best_score = max(best_score, score)

    conflict = int(
        has_conflict and best_score < 0.85
    )

    return best_score, conflict


def extract_features_v41(df):
    features = extract_features_v2(df)

    identity_results = [
        author_identity_v41(
            row["citation_author"],
            row["metadata_authors"]
        )
        for _, row in df.iterrows()
    ]

    features["author_identity_score"] = [
        result[0] for result in identity_results
    ]

    features["author_conflict"] = [
        result[1] for result in identity_results
    ]

    return features


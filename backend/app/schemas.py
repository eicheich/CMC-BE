from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PreviewRequest(BaseModel):
    citation: str = Field(min_length=1, max_length=2000)

    @field_validator("citation")
    @classmethod
    def non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Citation must not be blank")
        return value


class Citation(BaseModel):
    raw_citation: str
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None


class Work(BaseModel):
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    source: Literal["crossref", "openalex"]
    source_url: str | None = None


class Comparison(BaseModel):
    title_similarity: float | None = None
    author_match: float | None = None
    year_match: bool | None = None
    doi_match: bool | None = None


class PreviewResponse(BaseModel):
    status: Literal["verified_match", "possible_match", "metadata_mismatch", "not_found", "verification_unavailable"]
    match_score: float | None = None
    input: Citation
    best_match: Work | None = None
    comparison: Comparison = Field(default_factory=Comparison)
    reasons: list[str] = Field(default_factory=list)

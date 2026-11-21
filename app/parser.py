import re

from app.schemas import Citation


DOI_RE = re.compile(r"10\.\d{4,9}/[^\s<>\"']+", re.I)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
APA_RE = re.compile(r"^(?P<authors>.+?)\s*\((?P<year>(?:19|20)\d{2})\)\.?(?:\s+)(?P<rest>.+)$")
AUTHOR_RE = re.compile(r"[\w'-]+,\s*(?:[A-Z]\.)+(?:\s*[A-Z]\.)?", re.UNICODE)


def parse_citation(raw: str) -> Citation:
    doi_match = DOI_RE.search(raw)
    doi = doi_match.group().rstrip(".,;)").lower() if doi_match else None
    apa = APA_RE.match(raw.strip())
    text_without_doi = raw[:doi_match.start()] + raw[doi_match.end():] if doi_match else raw
    year_match = YEAR_RE.search(text_without_doi)
    title = None
    authors: list[str] = []
    if apa:
        authors = AUTHOR_RE.findall(apa.group("authors"))
        # APA-like titles end at the next sentence boundary after the year.
        title = re.split(r"\.\s+", apa.group("rest"), maxsplit=1)[0].strip(" .") or None
        if title and doi and doi in title.lower():
            title = None
    return Citation(raw_citation=raw, title=title, authors=authors,
                    year=int(year_match.group()) if year_match else None, doi=doi)

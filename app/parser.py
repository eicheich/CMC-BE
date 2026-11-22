import re

from app.schemas import Citation, CitationStyle


DOI_RE = re.compile(r"10\.\d{4,9}/[^\s<>\"']+", re.I)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
PAREN_YEAR_RE = re.compile(r"^(?P<authors>.+?)\s*\((?P<year>(?:19|20)\d{2})\)\.?\s+(?P<rest>.+)$")
CHICAGO_RE = re.compile(r"^(?P<authors>.+?)\.\s+(?P<year>(?:19|20)\d{2})\.\s+(?P<rest>.+)$")
QUOTE_RE = re.compile(r"^[\"“‘'](?P<title>.+?)[\"”’'](?=[,.\s]|$)")
NUMBER_RE = re.compile(r"^(?:\[\d+\]|\d+\.)\s*")
ET_AL_RE = re.compile(r"(?:,?\s*(?:&|and)?\s*et\s+al\.?)$", re.I)
JOURNAL_RE = re.compile(
    r"[.!?]\s+(?=(?:Journal|Proceedings|Advances|Transactions|IEEE|ACM)\b)", re.I
)


def normalize_space(value: str) -> str:
    return " ".join(value.split())


def clean_punctuation(value: str) -> str:
    return value.strip(" \t\r\n,.;: ")


def extract_doi(value: str) -> str | None:
    match = DOI_RE.search(value)
    return match.group().rstrip(".,;:)]}”’").lower() if match else None


def extract_year(value: str) -> int | None:
    without_doi = DOI_RE.sub("", value)
    without_urls = re.sub(r"https?://\S+", "", without_doi, flags=re.I)
    match = YEAR_RE.search(without_urls)
    return int(match.group()) if match else None


def normalize_author(value: str) -> str:
    author = normalize_space(re.sub(r"^(?:and|&)\s+", "", value.strip(), flags=re.I)).strip(" ,;")
    return author if re.search(r"\b[A-Z]\.$", author) else author.rstrip(".")


def surname_first_authors(value: str) -> list[str]:
    value = ET_AL_RE.sub("", value).strip(" ,;")
    # Split at the comma before the next surname, leaving each surname/given-name comma intact.
    parts = re.split(r",\s*(?=(?:and\s+|&\s*)?[\w'’\-]+(?:\s+[\w'’\-]+)*,\s*[A-Z])", value)
    authors = []
    for part in parts:
        for author in re.split(r",?\s+(?:and|&)\s+|,\s*(?:and|&)\s+", part, flags=re.I):
            author = normalize_author(author)
            if author:
                authors.append(author)
    return authors


def given_first_authors(value: str) -> list[str]:
    value = ET_AL_RE.sub("", value).strip(" ,;")
    return [author for part in re.split(r",\s*|\s+and\s+|\s+&\s+", value, flags=re.I)
            if (author := normalize_author(part)) and not re.fullmatch(r"et\s+al", author, re.I)]


def vancouver_authors(value: str) -> list[str]:
    authors = []
    for part in ET_AL_RE.sub("", value).strip(" ,.;").split(","):
        author = normalize_author(part)
        if not author:
            continue
        match = re.fullmatch(r"(?P<family>[\w'’\-]+(?:\s+[\w'’\-]+)*)\s+(?P<initials>(?:[A-Z]\.?){1,4})", author)
        if match:
            initials = " ".join(f"{letter}." for letter in re.findall(r"[A-Z]", match.group("initials")))
            authors.append(f"{match.group('family')}, {initials}")
    return authors


def unquoted_title(value: str) -> str | None:
    value = re.split(r"\s+(?:https?://|doi\s*:\s*10\.)", value, maxsplit=1, flags=re.I)[0]
    journal = JOURNAL_RE.search(value)
    if journal:
        return clean_punctuation(value[:journal.start()]) or None
    for boundary in re.finditer(r"[.!?]\s+", value):
        prefix = value[:boundary.start()]
        if re.search(r"(?:\b[A-Za-z]\.)+[A-Za-z]$", prefix):
            continue
        return clean_punctuation(prefix) or None
    return clean_punctuation(value) or None


def title_from_rest(value: str) -> str | None:
    value = value.strip()
    quoted = QUOTE_RE.match(value)
    return clean_punctuation(quoted.group("title")) or None if quoted else unquoted_title(value)


def parse_author_date(value: str) -> tuple[str | None, list[str], int | None]:
    match = PAREN_YEAR_RE.match(value)
    if not match:
        return None, [], extract_year(value)
    return title_from_rest(match.group("rest")), surname_first_authors(match.group("authors")), int(match.group("year"))


def parse_chicago(value: str) -> tuple[str | None, list[str], int | None]:
    match = CHICAGO_RE.match(value)
    if not match:
        return None, [], extract_year(value)
    return title_from_rest(match.group("rest")), surname_first_authors(match.group("authors")), int(match.group("year"))


def parse_mla(value: str) -> tuple[str | None, list[str], int | None]:
    quoted = re.search(r"[\"“‘'](.+?)[\"”’']", value)
    if not quoted:
        return None, [], extract_year(value)
    authors = surname_first_authors(value[:quoted.start()].rstrip(" .,"))
    title = clean_punctuation(quoted.group(1)) or None
    return title, authors, extract_year(value[quoted.end():])


def parse_ieee(value: str) -> tuple[str | None, list[str], int | None]:
    quoted = re.search(r"[\"“](.+?)[\"”]", value)
    if not quoted:
        return None, [], extract_year(value)
    authors = given_first_authors(value[:quoted.start()].rstrip(" ,"))
    title = clean_punctuation(quoted.group(1)) or None
    return title, authors, extract_year(value[quoted.end():])


def parse_vancouver(value: str) -> tuple[str | None, list[str], int | None]:
    match = re.match(r"^(?P<authors>.+?\b(?:et\s+al\.)?)\.\s+(?P<rest>.+)$", value, re.I)
    if not match:
        return None, [], extract_year(value)
    authors = vancouver_authors(match.group("authors"))
    if not authors:
        return None, [], extract_year(value)
    rest = match.group("rest")
    title = unquoted_title(rest)
    year = extract_year(rest[len(title):]) if title else extract_year(rest)
    return title, authors, year


PARSERS = {
    "apa": parse_author_date,
    "harvard": parse_author_date,
    "chicago": parse_chicago,
    "mla": parse_mla,
    "ieee": parse_ieee,
    "vancouver": parse_vancouver,
}


def parse_citation(raw: str, style: CitationStyle = "apa") -> Citation:
    if style not in PARSERS:
        raise ValueError(f"Unsupported citation style: {style}")
    value = NUMBER_RE.sub("", normalize_space(raw))
    title, authors, year = PARSERS[style](value)
    return Citation(raw_citation=raw, title=title, authors=authors, year=year, doi=extract_doi(value))

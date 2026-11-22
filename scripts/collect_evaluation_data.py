"""Collect real publication metadata and prepare unreviewed evaluation pairs."""

import argparse
import csv
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from rapidfuzz import fuzz

from scripts.evaluate_matching import REQUIRED_COLUMNS


TOPICS = ("machine learning", "climate change", "microplastics")
TARGETS = {"same_consistent": 15, "same_modified": 10, "different_similar_title": 10,
           "different_similar_author": 10, "hard_incomplete": 5}
CANDIDATE_COLUMNS = (
    "case_id", "category", "proposed_label", "reviewed_label", "review_status", "overlap_status",
    "modification_type", "modification_source_doi", "modification_source_url",
    "citation_title", "citation_author", "citation_year", "citation_doi",
    "metadata_title", "metadata_authors", "metadata_year", "metadata_doi",
    "intended_doi", "intended_openalex_id", "intended_source_url",
    "candidate_doi", "candidate_openalex_id", "candidate_source_url",
    "original_title", "original_authors", "original_first_author", "original_year", "original_doi",
)
REVIEW_COLUMNS = (
    "case_id", "category", "proposed_label", "reviewed_label", "review_status", "overlap_status",
    "modification_type", "citation_title", "citation_author", "citation_year", "citation_doi",
    "metadata_title", "metadata_authors", "metadata_year", "metadata_doi",
    "intended_doi", "intended_openalex_id", "intended_source_url",
    "candidate_doi", "candidate_openalex_id", "candidate_source_url",
)


def normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    value = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", value.strip(), flags=re.I)
    value = value.rstrip(".,;)]}").casefold()
    return value if re.fullmatch(r"10\.\d{4,9}/\S+", value) else None


def normalize_openalex_id(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(r"(?:^|/)(W\d+)$", value.strip(), re.I)
    return f"https://openalex.org/{match.group(1).upper()}" if match else None


@dataclass(frozen=True)
class Publication:
    title: str
    authors: tuple[str, ...]
    year: int
    doi: str
    openalex_id: str | None
    source_url: str
    first_author_id: str | None = None


def parse_openalex(item: dict) -> Publication | None:
    try:
        title, year, doi = item.get("title"), item.get("publication_year"), normalize_doi(item.get("doi"))
        authorships = item.get("authorships") or []
        authors = tuple(entry["author"]["display_name"].strip() for entry in authorships
                        if entry.get("author", {}).get("display_name"))
        work_id = normalize_openalex_id(item.get("id"))
        author_id = authorships[0].get("author", {}).get("id") if authorships else None
        if (not title or not isinstance(year, int) or not 2020 <= year <= 2026 or not doi or not work_id or
                not authors or any("," in name or len(name.split()) < 2 for name in authors)):
            return None
        return Publication(title.strip(), authors, year, doi, work_id,
                           f"https://api.openalex.org/works/{work_id.rsplit('/', 1)[-1]}", author_id)
    except (AttributeError, KeyError, TypeError):
        return None


def parse_crossref(item: dict) -> Publication | None:
    try:
        dates = item.get("published", {}).get("date-parts", [])
        year = dates[0][0] if dates and dates[0] else None
        title = (item.get("title") or [None])[0]
        doi = normalize_doi(item.get("DOI"))
        authors = tuple(f"{entry['given'].strip()} {entry['family'].strip()}" for entry in item.get("author", [])
                        if entry.get("given") and entry.get("family"))
        if (not title or not isinstance(year, int) or not 2020 <= year <= 2026 or not doi or
                not authors or any("," in name for name in authors)):
            return None
        return Publication(title.strip(), authors, year, doi, None,
                           f"https://api.crossref.org/works/{doi}")
    except (AttributeError, KeyError, TypeError, IndexError):
        return None


class ApiClient:
    def __init__(self, client: httpx.Client, cache_dir: Path | None = None, interval: float = 1.0,
                 sleep=time.sleep, clock=time.monotonic):
        self.client, self.cache_dir, self.interval = client, cache_dir, interval
        self.sleep, self.clock = sleep, clock
        self.last_request: float | None = None
        self.cache_hits = 0
        self.network_requests = 0

    def get_json(self, url: str, params: dict) -> dict:
        key = hashlib.sha256(json.dumps([url, params], sort_keys=True).encode()).hexdigest()
        cache_path = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if cache_path and cache_path.exists():
            self.cache_hits += 1
            return json.loads(cache_path.read_text(encoding="utf-8"))
        for attempt in range(3):
            if self.last_request is not None:
                self.sleep(max(0.0, self.interval - (self.clock() - self.last_request)))
            self.last_request = self.clock()
            try:
                self.network_requests += 1
                response = self.client.get(url, params=params)
                if response.status_code in {429, 500, 502, 503, 504}:
                    if attempt < 2:
                        self.sleep(2 ** attempt)
                        continue
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise ValueError("Invalid API response")
                if cache_path:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(json.dumps(data), encoding="utf-8")
                return data
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt == 2:
                    raise
                self.sleep(2 ** attempt)
        raise RuntimeError("API retries exhausted")


def load_exclusions(path: Path | None) -> tuple[set[str], set[str]]:
    dois, ids = set(), set()
    if path:
        for line in path.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            doi, work_id = normalize_doi(value), normalize_openalex_id(value)
            if doi:
                dois.add(doi)
            elif work_id:
                ids.add(work_id)
            else:
                raise ValueError("Exclusion file contains an invalid DOI or OpenAlex ID")
    return dois, ids


def collect(api: ApiClient, exclusions: tuple[set[str], set[str]]) -> tuple[list[Publication], dict]:
    publications: list[Publication] = []
    seen_doi, seen_id = set(), set()
    stats = {"duplicates_removed": 0, "overlaps_removed": 0, "incomplete_removed": 0, "api_errors": []}

    def add(item: dict, parser) -> None:
        pub = parser(item)
        if not pub:
            stats["incomplete_removed"] += 1
        elif pub.doi in exclusions[0] or (pub.openalex_id and pub.openalex_id in exclusions[1]):
            stats["overlaps_removed"] += 1
        elif pub.doi in seen_doi or (pub.openalex_id and pub.openalex_id in seen_id):
            stats["duplicates_removed"] += 1
        else:
            publications.append(pub)
            seen_doi.add(pub.doi)
            if pub.openalex_id:
                seen_id.add(pub.openalex_id)

    key = os.getenv("OPENALEX_API_KEY")
    common = {"per_page": 100, "select": "id,title,doi,publication_year,authorships,type"}
    if key:
        common["api_key"] = key
    year_filter = "publication_year:2020-2026,type:article,has_doi:true"
    for topic in TOPICS:
        try:
            data = api.get_json("https://api.openalex.org/works",
                                {**common, "search": topic, "filter": year_filter})
            for item in data.get("results", []):
                add(item, parse_openalex)
        except (httpx.HTTPError, ValueError) as exc:
            stats["api_errors"].append(f"openalex:{topic}:{type(exc).__name__}")

    author_ids = list(dict.fromkeys(pub.first_author_id for pub in publications if pub.first_author_id))[:5]
    for author_id in author_ids:
        try:
            data = api.get_json("https://api.openalex.org/works",
                                {**common, "filter": f"{year_filter},authorships.author.id:{author_id}"})
            for item in data.get("results", []):
                add(item, parse_openalex)
        except (httpx.HTTPError, ValueError) as exc:
            stats["api_errors"].append(f"openalex:author:{type(exc).__name__}")

    if len(publications) < 50:
        try:
            data = api.get_json("https://api.crossref.org/works",
                                {"query.bibliographic": "machine learning", "rows": 100,
                                 "filter": "from-pub-date:2020-01-01,until-pub-date:2026-12-31"})
            for item in data.get("message", {}).get("items", []):
                add(item, parse_crossref)
        except (httpx.HTTPError, ValueError) as exc:
            stats["api_errors"].append(f"crossref:{type(exc).__name__}")
    return publications, stats


def make_pairs(publications: list[Publication], max_pairs: int, overlap_status: str) -> list[dict]:
    pairs: list[dict] = []
    used = set()

    def add(category: str, intended: Publication, candidate: Publication,
            modification: str = "none", modification_source: Publication | None = None) -> bool:
        if len(pairs) >= max_pairs or sum(row["category"] == category for row in pairs) >= TARGETS[category]:
            return False
        identity = (category, intended.doi, candidate.doi, modification)
        if identity in used:
            return False
        used.add(identity)
        citation_year, citation_doi, citation_author = intended.year, intended.doi, intended.authors[0]
        metadata_doi = candidate.doi
        if modification == "year_plus_one":
            citation_year += 1
        elif modification == "doi_swapped" and modification_source:
            citation_doi = modification_source.doi
        elif modification == "author_swapped" and modification_source:
            citation_author = modification_source.authors[0]
        elif modification == "metadata_doi_omitted":
            metadata_doi = ""
        pairs.append({
            "case_id": f"eval_{len(pairs) + 1:03d}", "category": category,
            "proposed_label": 1 if intended.doi == candidate.doi else 0,
            "reviewed_label": "", "review_status": "pending", "overlap_status": overlap_status,
            "modification_type": modification,
            "modification_source_doi": modification_source.doi if modification_source else "",
            "modification_source_url": modification_source.source_url if modification_source else "",
            "citation_title": intended.title, "citation_author": citation_author,
            "citation_year": citation_year, "citation_doi": citation_doi,
            "metadata_title": candidate.title, "metadata_authors": ";".join(candidate.authors),
            "metadata_year": candidate.year, "metadata_doi": metadata_doi,
            "intended_doi": intended.doi, "intended_openalex_id": intended.openalex_id or "",
            "intended_source_url": intended.source_url, "candidate_doi": candidate.doi,
            "candidate_openalex_id": candidate.openalex_id or "", "candidate_source_url": candidate.source_url,
            "original_title": intended.title, "original_authors": ";".join(intended.authors),
            "original_first_author": intended.authors[0],
            "original_year": intended.year, "original_doi": intended.doi,
        })
        return True

    for pub in publications[:TARGETS["same_consistent"]]:
        add("same_consistent", pub, pub)
    for index, pub in enumerate(publications[TARGETS["same_consistent"]:]):
        if sum(row["category"] == "same_modified" for row in pairs) >= TARGETS["same_modified"]:
            break
        variant = ("year_plus_one", "doi_swapped", "author_swapped")[index % 3]
        other = next((item for item in publications if item.doi != pub.doi), None)
        add("same_modified", pub, pub, variant, other)

    distinct = [(left, right) for i, left in enumerate(publications) for right in publications[i + 1:]
                if left.doi != right.doi and left.openalex_id != right.openalex_id]
    titles = sorted(((fuzz.ratio(left.title.casefold(), right.title.casefold()), left, right)
                     for left, right in distinct), key=lambda item: item[0], reverse=True)
    title_used = set()
    for similarity, left, right in titles:
        if similarity < 65:
            break
        if left.doi in title_used or right.doi in title_used:
            continue
        if left.title.casefold() == right.title.casefold() and left.first_author_id == right.first_author_id:
            continue
        if add("different_similar_title", left, right):
            title_used.update((left.doi, right.doi))
    authors = sorted(((fuzz.ratio(left.authors[0].casefold(), right.authors[0].casefold()), left, right)
                      for left, right in distinct
                      if (left.first_author_id and left.first_author_id == right.first_author_id) or
                      fuzz.ratio(left.authors[0].casefold(), right.authors[0].casefold()) >= 85),
                     key=lambda item: item[0], reverse=True)
    author_used = set()
    for _, left, right in authors:
        if left.doi in author_used or right.doi in author_used:
            continue
        if add("different_similar_author", left, right):
            author_used.update((left.doi, right.doi))
    for pub in publications[TARGETS["same_consistent"] + TARGETS["same_modified"]:]:
        add("hard_incomplete", pub, pub, "metadata_doi_omitted")
    return pairs


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_approved(candidates_path: Path, output_path: Path) -> int:
    with candidates_path.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    approved = []
    for row in rows:
        if row.get("review_status") != "approved" or row.get("reviewed_label") not in {"0", "1"}:
            continue
        approved.append({**{name: row[name] for name in REQUIRED_COLUMNS if name != "label"},
                         "label": row["reviewed_label"]})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_path, REQUIRED_COLUMNS, approved)
    return len(approved)


def run(output_dir: Path, exclude_file: Path | None = None, max_pairs: int = 50,
        cache_dir: Path | None = None, client: httpx.Client | None = None) -> dict:
    if not 1 <= max_pairs <= 50:
        raise ValueError("max_pairs must be between 1 and 50")
    if (output_dir / "evaluation_candidates_v1.csv").exists():
        raise FileExistsError("Candidate file already exists; use a new output directory to preserve reviews")
    exclusions = load_exclusions(exclude_file)
    overlap_status = "checked_supplied_exclusions" if any(exclusions) else "unchecked"
    contact = os.getenv("CROSSREF_CONTACT")
    user_agent = "CheckMyCitation-EvaluationCollector/0.1"
    if contact:
        user_agent += f" (mailto:{contact})"
    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=10.0, headers={"User-Agent": user_agent})
    try:
        api = ApiClient(client, cache_dir=cache_dir)
        publications, stats = collect(api, exclusions)
    finally:
        if own_client:
            client.close()
    pairs = make_pairs(publications, max_pairs, overlap_status)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "evaluation_candidates_v1.csv", CANDIDATE_COLUMNS, pairs)
    write_csv(output_dir / "evaluation_review_template_v1.csv", REVIEW_COLUMNS, pairs)
    summary = {
        "real_publications_collected": len(publications),
        "candidate_pairs": len(pairs),
        "categories": {category: sum(row["category"] == category for row in pairs) for category in TARGETS},
        "unique_dois": len({pub.doi for pub in publications}),
        "unique_openalex_ids": len({pub.openalex_id for pub in publications if pub.openalex_id}),
        "duplicates_removed": stats["duplicates_removed"],
        "overlaps_removed": stats["overlaps_removed"],
        "incomplete_removed": stats["incomplete_removed"],
        "pending_review": len(pairs),
        "overlap_status": overlap_status,
        "exclusion_dois": len(exclusions[0]), "exclusion_openalex_ids": len(exclusions[1]),
        "network_requests": api.network_requests, "cache_hits": api.cache_hits,
        "api_errors": stats["api_errors"],
        "note": "Proposed labels need manual review. No final evaluation was run.",
    }
    (output_dir / "evaluation_collection_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--exclude-file", type=Path, help="One training DOI or OpenAlex ID per line")
    parser.add_argument("--cache-dir", type=Path, help="Optional JSON response cache")
    parser.add_argument("--max-pairs", type=int, default=50)
    parser.add_argument("--export-approved", action="store_true", help="Export reviewed approved rows only; no API calls")
    args = parser.parse_args()
    if args.export_approved:
        count = export_approved(args.output_dir / "evaluation_candidates_v1.csv",
                                args.output_dir / "evaluation_pairs_v1.csv")
        print(json.dumps({"approved_exported": count}))
    else:
        print(json.dumps(run(args.output_dir, args.exclude_file, args.max_pairs, args.cache_dir), indent=2))


if __name__ == "__main__":
    main()

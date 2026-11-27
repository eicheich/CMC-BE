"""Audit DOI/OpenAlex ID overlap without modifying evaluation candidates."""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


DOI_RE = re.compile(r"10\.\d{4,9}/[^\s;,<>\"']+", re.I)
OPENALEX_RE = re.compile(r"\bW\d+\b", re.I)
PUBLICATION_ROLES = {
    "intended": ("intended_doi", "intended_openalex_id", "intended_source_url", "original_doi", "doi", "openalex_id"),
    "candidate": ("candidate_doi", "candidate_openalex_id", "candidate_source_url", "metadata_doi"),
    "modification_source": ("modification_source_doi", "modification_source_url"),
}


def identifiers(row: dict) -> dict[tuple[str, str], set[str]]:
    found = defaultdict(set)
    for column, value in row.items():
        if not isinstance(value, str):
            continue
        for match in DOI_RE.finditer(value):
            doi = match.group().rstrip(".,;:)]}").casefold()
            found[("doi", doi)].add(column)
        for match in OPENALEX_RE.finditer(value):
            found[("openalex_id", f"W{match.group()[1:]}")].add(column)
    return dict(found)


def source_kind(path: Path) -> str:
    name = path.stem.casefold()
    if "train" in name:
        return "training_named"
    if "benchmark" in name or "develop" in name or name.startswith("dev_"):
        return "development_named"
    if "smoke" in name:
        return "synthetic_smoke"
    if "independent" in name:
        return "independent_named"
    return "unclassified_reference"


def source_paths(data_dir: Path, candidates: Path) -> list[Path]:
    paths = [path for path in (data_dir / "reference").glob("*.csv")
             if path.name not in {"cmc_exclusion_identifiers.csv", "cmc_exclusion_provenance.csv"}]
    paths.extend(path for path in data_dir.glob("*.csv")
                 if path.resolve() != candidates.resolve() and
                 not path.name.startswith(("evaluation_review_", "evaluation_pairs_")))
    return sorted(set(paths))


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def exclusion_key(row: dict) -> tuple[str, str] | None:
    kind = row.get("identifier_type", "").strip()
    if kind not in {"doi", "openalex_id"}:
        return None
    return next((key for key in identifiers({"identifier": row.get("identifier", "")}) if key[0] == kind), None)


def audit_exclusions(data_dir: Path, candidates: list[dict]) -> dict | None:
    directory = data_dir / "reference"
    identifiers_path = directory / "cmc_exclusion_identifiers.csv"
    provenance_path = directory / "cmc_exclusion_provenance.csv"
    if not identifiers_path.exists() or not provenance_path.exists():
        return None
    excluded_rows, provenance_rows = read_csv(identifiers_path), read_csv(provenance_path)
    excluded = {key for row in excluded_rows if (key := exclusion_key(row))}
    provenance = defaultdict(list)
    for row in provenance_rows:
        if key := exclusion_key(row):
            provenance[key].append({"source_file": row.get("source_file", ""),
                                    "source_column": row.get("source_column", "")})
    without_provenance = excluded - provenance.keys()
    provenance_not_excluded = provenance.keys() - excluded
    publication_columns = {"doi", "metadata_doi", "openalex_id"}
    affected = []
    publication_pairs, citation_pairs = set(), set()
    matched = set()
    matched_publications = set()
    for row in candidates:
        case_id = row.get("case_id", "")
        matches = []
        for role, columns in PUBLICATION_ROLES.items():
            role_ids = identifiers({column: row.get(column, "") for column in columns})
            canonical = next((key for key in role_ids if key[0] == "doi"), next(iter(role_ids), None))
            for key, matched_columns in role_ids.items():
                if key not in excluded:
                    continue
                matched.add(key)
                for source in provenance.get(key, []):
                    source_role = "publication" if source["source_column"] in publication_columns else (
                        "synthetic_citation" if source["source_column"] == "citation_doi" else "other")
                    overlap_type = f"publication_to_{source_role}"
                    matches.append({"overlap_type": overlap_type, "candidate_role": role,
                                    "candidate_columns": sorted(matched_columns),
                                    "identifier_type": key[0], "identifier": key[1], "provenance": source})
                    if source_role == "publication":
                        publication_pairs.add(case_id)
                        if canonical:
                            matched_publications.add(canonical)
        for key, matched_columns in identifiers({"citation_doi": row.get("citation_doi", "")}).items():
            if key not in excluded:
                continue
            matched.add(key)
            for source in provenance.get(key, []):
                source_role = "publication" if source["source_column"] in publication_columns else (
                    "synthetic_citation" if source["source_column"] == "citation_doi" else "other")
                matches.append({"overlap_type": f"synthetic_citation_to_{source_role}",
                                "candidate_role": "citation_doi", "candidate_columns": sorted(matched_columns),
                                "identifier_type": key[0], "identifier": key[1], "provenance": source})
                citation_pairs.add(case_id)
        if matches:
            affected.append({"case_id": case_id, "matches": matches})
    declared_sources = sorted({row.get("source_file", "") for row in provenance_rows if row.get("source_file")})
    coverage_complete = not without_provenance and not provenance_not_excluded
    status = "overlap_found" if publication_pairs else (
        "passed_for_supplied_sources" if coverage_complete else "incomplete")
    return {
        "status": status,
        "reason": "No original-publication identifier overlap in the supplied exclusion set. Full v4.1 training coverage is unverified."
        if status == "passed_for_supplied_sources" else
        "Original-publication overlap found." if status == "overlap_found" else
        "Exclusion identifiers and provenance do not cover the same set.",
        "exclusion_identifier_rows": len(excluded_rows),
        "exclusion_unique_identifiers": len(excluded),
        "exclusion_provenance_rows": len(provenance_rows),
        "exclusion_identifiers_without_provenance": len(without_provenance),
        "provenance_identifiers_not_excluded": len(provenance_not_excluded),
        "provenance_source_files": declared_sources,
        "training_coverage_status": "unverified_completeness",
        "publication_overlap_pairs": len(publication_pairs),
        "synthetic_citation_overlap_pairs": len(citation_pairs),
        "unique_overlapping_candidate_publications": len(matched_publications),
        "overlapping_pairs": len(affected),
        "unique_matched_dois": sum(kind == "doi" for kind, _ in matched),
        "unique_matched_openalex_ids": sum(kind == "openalex_id" for kind, _ in matched),
        "affected_pairs": affected,
    }


def audit(data_dir: Path) -> dict:
    candidates_path = data_dir / "evaluation_candidates_v1.csv"
    candidates = read_csv(candidates_path)
    index = defaultdict(list)
    sources = []
    for path in source_paths(data_dir, candidates_path):
        rows = read_csv(path)
        kind = source_kind(path)
        source_ids = set()
        for row_number, row in enumerate(rows, start=2):
            for identifier, columns in identifiers(row).items():
                index[identifier].append({"file": path.relative_to(data_dir).as_posix(), "kind": kind,
                                          "row": row_number, "columns": sorted(columns)})
                source_ids.add(identifier)
        sources.append({"file": path.relative_to(data_dir).as_posix(), "kind": kind,
                        "rows": len(rows), "rows_with_identifiers": sum(bool(identifiers(row)) for row in rows),
                        "unique_dois": sum(kind == "doi" for kind, _ in source_ids),
                        "unique_openalex_ids": sum(kind == "openalex_id" for kind, _ in source_ids)})

    affected = []
    matched_ids = set()
    matched_publications = set()
    candidate_publications = set()
    for row in candidates:
        for role, columns in PUBLICATION_ROLES.items():
            role_ids = identifiers({column: row.get(column, "") for column in columns})
            canonical = next((identifier for identifier in role_ids if identifier[0] == "doi"),
                             next(iter(role_ids), None))
            if canonical:
                candidate_publications.add(canonical)
                if any(identifier in index for identifier in role_ids):
                    matched_publications.add(canonical)
        matches = []
        for identifier, columns in identifiers(row).items():
            for source in index.get(identifier, []):
                matches.append({"identifier_type": identifier[0], "identifier": identifier[1],
                                "candidate_columns": sorted(columns), "source": source})
                matched_ids.add(identifier)
        if matches:
            affected.append({"case_id": row.get("case_id"), "matches": matches})

    training_present = any(source["kind"] == "training_named" for source in sources)
    benchmark_present = any(source["kind"] == "development_named" for source in sources)
    status = "incomplete" if not (training_present and benchmark_present) else ("overlap_found" if affected else "passed")
    report = {
        "status": status,
        "training_dataset_available": training_present,
        "development_benchmark_available": benchmark_present,
        "reason": "Training and/or development benchmark publication CSV is missing; independence cannot be established."
        if status == "incomplete" else "Compared against named training and development CSV files; verify their coverage before claiming independence.",
        "comparison_method": "Normalized DOI and OpenAlex ID exact match across all CSV columns; no title-based inference.",
        "candidate_pairs_compared": len(candidates),
        "candidate_unique_publications_referenced": len(candidate_publications),
        "source_rows_compared": sum(source["rows"] for source in sources),
        "sources": sources,
        "overlapping_pairs": len(affected),
        "unique_matched_dois": sum(kind == "doi" for kind, _ in matched_ids),
        "unique_matched_openalex_ids": sum(kind == "openalex_id" for kind, _ in matched_ids),
        "unique_overlapping_candidate_publications": len(matched_publications),
        "affected_pairs": affected,
        "limitations": [
            "A filename does not prove that a source contains the complete v4.1 training set.",
            "Identifier-only comparison cannot detect the same publication when both datasets omit or use different identifiers.",
            "The 359-work collection pool was not saved as a publication CSV; this audit covers publications referenced by the 50 candidate pairs.",
        ],
    }
    if exclusion := audit_exclusions(data_dir, candidates):
        report.update(exclusion)
        report["comparison_method"] = (
            "Exact normalized DOI/OpenAlex ID comparison against supplied exclusion identifiers; "
            "candidate publication identities and citation_doi are classified separately using provenance."
        )
        report["limitations"].append("The supplied exclusion provenance names older source files; completeness for all v4.1 training/development data was not independently verified.")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/evaluation_overlap_report.json"))
    args = parser.parse_args()
    report = audit(args.data_dir)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "candidate_pairs_compared", "source_rows_compared", "overlapping_pairs")}))


if __name__ == "__main__":
    main()

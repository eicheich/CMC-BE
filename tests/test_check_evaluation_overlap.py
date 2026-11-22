"""Identifier audit tests use synthetic IDs, not actual publication claims."""

import csv

from scripts.check_evaluation_overlap import audit, identifiers


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def candidate(case_id, doi, work_id):
    return {"case_id": case_id, "intended_doi": doi, "intended_openalex_id": work_id,
            "candidate_doi": doi, "candidate_openalex_id": work_id,
            "modification_source_doi": "", "citation_doi": doi, "metadata_doi": doi}


def test_doi_url_and_openalex_id_extracted_from_any_column():
    found = identifiers({"metadata_pair": "https://doi.org/10.5555/ABC.1;10.5555/other",
                         "source_link": "https://api.openalex.org/works/W123"})
    assert ("doi", "10.5555/abc.1") in found
    assert ("doi", "10.5555/other") in found
    assert ("openalex_id", "W123") in found


def test_overlap_report_lists_source_and_affected_pairs(tmp_path):
    reference = tmp_path / "reference"
    reference.mkdir()
    write_csv(reference / "training_pairs.csv", [
        {"pair_metadata": "https://doi.org/10.5555/ABC.1", "other_id": "https://openalex.org/W123"},
        {"pair_metadata": "10.5555/unrelated", "other_id": "W999"},
    ])
    write_csv(reference / "development_benchmark.csv", [{"doi": "10.5555/dev", "openalex_id": "W888"}])
    write_csv(tmp_path / "evaluation_candidates_v1.csv", [
        candidate("hit", "10.5555/abc.1", "https://openalex.org/W123"),
        candidate("clear", "10.5555/clear", "https://openalex.org/W555"),
    ])
    report = audit(tmp_path)
    assert report["status"] == "overlap_found"
    assert report["candidate_pairs_compared"] == 2
    assert report["source_rows_compared"] == 3
    assert report["overlapping_pairs"] == 1
    assert report["unique_matched_dois"] == 1
    assert report["unique_matched_openalex_ids"] == 1
    assert report["unique_overlapping_candidate_publications"] == 1
    assert report["affected_pairs"][0]["case_id"] == "hit"
    assert report["affected_pairs"][0]["matches"][0]["source"]["file"] == "reference/training_pairs.csv"


def test_missing_training_file_keeps_status_incomplete(tmp_path):
    reference = tmp_path / "reference"
    reference.mkdir()
    write_csv(reference / "cmc_independent_publications.csv", [{"doi": "10.5555/same", "openalex_id": "W7"}])
    write_csv(tmp_path / "evaluation_candidates_v1.csv", [candidate("hit", "10.5555/same", "W7")])
    report = audit(tmp_path)
    assert report["status"] == "incomplete"
    assert report["training_dataset_available"] is False
    assert report["development_benchmark_available"] is False
    assert report["overlapping_pairs"] == 1


def test_exclusion_provenance_distinguishes_both_publications_and_citation(tmp_path):
    reference = tmp_path / "reference"
    reference.mkdir()
    write_csv(reference / "cmc_exclusion_identifiers.csv", [
        {"identifier_type": "doi", "identifier": "https://doi.org/10.5555/INTENDED"},
        {"identifier_type": "openalex_id", "identifier": "https://openalex.org/W456"},
        {"identifier_type": "doi", "identifier": "10.5555/synthetic"},
    ])
    write_csv(reference / "cmc_exclusion_provenance.csv", [
        {"identifier_type": "doi", "identifier": "10.5555/intended",
         "source_file": "cmc_train_v3.csv", "source_column": "doi"},
        {"identifier_type": "openalex_id", "identifier": "W456",
         "source_file": "cmc_labeled_pairs_v2.csv", "source_column": "openalex_id"},
        {"identifier_type": "doi", "identifier": "10.5555/synthetic",
         "source_file": "cmc_train_v3.csv", "source_column": "citation_doi"},
    ])
    row = candidate("both-sides", "10.5555/intended", "W123")
    row.update(candidate_doi="10.5555/candidate", candidate_openalex_id="W456",
               metadata_doi="10.5555/candidate", citation_doi="10.5555/synthetic")
    write_csv(tmp_path / "evaluation_candidates_v1.csv", [row])

    report = audit(tmp_path)
    assert report["status"] == "overlap_found"
    assert report["candidate_pairs_compared"] == 1
    assert report["publication_overlap_pairs"] == 1
    assert report["synthetic_citation_overlap_pairs"] == 1
    matches = report["affected_pairs"][0]["matches"]
    assert {(match["candidate_role"], match["overlap_type"], match["provenance"]["source_file"])
            for match in matches} == {
                ("intended", "publication_to_publication", "cmc_train_v3.csv"),
                ("candidate", "publication_to_publication", "cmc_labeled_pairs_v2.csv"),
                ("citation_doi", "synthetic_citation_to_synthetic_citation", "cmc_train_v3.csv"),
            }


def test_synthetic_overlap_does_not_fail_publication_audit(tmp_path):
    reference = tmp_path / "reference"
    reference.mkdir()
    write_csv(reference / "cmc_exclusion_identifiers.csv", [
        {"identifier_type": "doi", "identifier": "10.5555/only-citation"},
    ])
    write_csv(reference / "cmc_exclusion_provenance.csv", [
        {"identifier_type": "doi", "identifier": "10.5555/only-citation",
         "source_file": "cmc_train_v3.csv", "source_column": "citation_doi"},
    ])
    row = candidate("citation-only", "10.5555/clear", "W123")
    row["citation_doi"] = "https://doi.org/10.5555/ONLY-CITATION"
    write_csv(tmp_path / "evaluation_candidates_v1.csv", [row])

    report = audit(tmp_path)
    assert report["status"] == "passed_for_supplied_sources"
    assert report["publication_overlap_pairs"] == 0
    assert report["synthetic_citation_overlap_pairs"] == 1
    assert report["affected_pairs"][0]["matches"][0]["overlap_type"] == (
        "synthetic_citation_to_synthetic_citation")


def test_publication_id_seen_only_as_old_synthetic_citation_is_separate(tmp_path):
    reference = tmp_path / "reference"
    reference.mkdir()
    write_csv(reference / "cmc_exclusion_identifiers.csv", [
        {"identifier_type": "doi", "identifier": "10.5555/old-citation"},
        {"identifier_type": "doi", "identifier": "10.5555/no-provenance"},
    ])
    write_csv(reference / "cmc_exclusion_provenance.csv", [
        {"identifier_type": "doi", "identifier": "10.5555/old-citation",
         "source_file": "cmc_train_v3.csv", "source_column": "citation_doi"},
    ])
    row = candidate("publication-versus-citation", "10.5555/old-citation", "W123")
    row["citation_doi"] = ""
    write_csv(tmp_path / "evaluation_candidates_v1.csv", [row])

    report = audit(tmp_path)
    assert report["status"] == "incomplete"
    assert report["exclusion_identifiers_without_provenance"] == 1
    assert report["publication_overlap_pairs"] == 0
    assert report["affected_pairs"][0]["matches"][0]["overlap_type"] == (
        "publication_to_synthetic_citation")

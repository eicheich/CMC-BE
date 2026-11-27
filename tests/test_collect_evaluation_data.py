"""Synthetic API fixtures test collection logic; no fixture is a claimed real publication."""

import csv
import json

import httpx
import pytest

from scripts import collect_evaluation_data as collector
from scripts.evaluate_matching import REQUIRED_COLUMNS


def openalex_item(doi="10.5555/test.1", work_id="W1", title="Graph Methods for Libraries",
                  author="Alice Rivera", author_id="A1"):
    return {"id": f"https://openalex.org/{work_id}", "title": title, "publication_year": 2022,
            "doi": f"https://doi.org/{doi}",
            "authorships": [{"author": {"display_name": author, "id": f"https://openalex.org/{author_id}"}}]}


def publication(number, title=None, author="Alice Rivera", author_id="A1"):
    return collector.Publication(title or f"Graph Methods for Library Search {number}", (author,), 2022,
                                 f"10.5555/test.{number}", f"https://openalex.org/W{number}",
                                 f"https://api.openalex.org/works/W{number}", author_id)


def test_parse_crossref_and_openalex_and_normalize_doi():
    openalex = collector.parse_openalex(openalex_item())
    crossref = collector.parse_crossref({"title": ["Graph Methods for Libraries"],
                                        "author": [{"given": "Alice", "family": "Rivera"}],
                                        "published": {"date-parts": [[2022, 3, 1]]},
                                        "DOI": "10.5555/TEST.1", "URL": "https://doi.org/10.5555/test.1"})
    assert openalex.doi == crossref.doi == "10.5555/test.1"
    assert openalex.authors == crossref.authors == ("Alice Rivera",)
    assert openalex.openalex_id == "https://openalex.org/W1"
    assert crossref.source_url == "https://api.crossref.org/works/10.5555/test.1"
    assert collector.normalize_doi("https://doi.org/10.5555/TEST.1") == "10.5555/test.1"
    assert collector.normalize_openalex_id("W123") == "https://openalex.org/W123"


def test_deduplication_and_exclusions(tmp_path):
    excluded = tmp_path / "exclude.txt"
    excluded.write_text("# training works\nhttps://doi.org/10.5555/test.2\nW3\n", encoding="utf-8")

    class Api:
        def get_json(self, url, params):
            if "openalex" in url:
                return {"results": [openalex_item(), openalex_item(),
                                    openalex_item("10.5555/test.2", "W2"),
                                    openalex_item("10.5555/test.3", "W3")]}
            return {"message": {"items": []}}

    publications, stats = collector.collect(Api(), collector.load_exclusions(excluded))
    assert len(publications) == 1
    assert publications[0].doi == "10.5555/test.1"
    assert stats["duplicates_removed"] > 0
    assert stats["overlaps_removed"] > 0


def test_rate_limit_retry_timeout_and_cache(tmp_path):
    now = [0.0]
    waits = []
    responses = iter([429, 200])

    def handler(request):
        return httpx.Response(next(responses), json={"results": []})

    def sleep(seconds):
        waits.append(seconds)
        now[0] += seconds

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        api = collector.ApiClient(client, tmp_path, interval=1.0, sleep=sleep, clock=lambda: now[0])
        assert api.get_json("https://api.openalex.org/works", {"search": "test"}) == {"results": []}
        assert api.network_requests == 2
        assert waits and sum(waits) >= 1
        assert api.get_json("https://api.openalex.org/works", {"search": "test"}) == {"results": []}
        assert api.cache_hits == 1
        assert api.network_requests == 2

    calls = [0]

    def timeout_then_success(request):
        calls[0] += 1
        if calls[0] == 1:
            raise httpx.ReadTimeout("synthetic timeout")
        return httpx.Response(200, json={"results": []})

    with httpx.Client(transport=httpx.MockTransport(timeout_then_success)) as client:
        api = collector.ApiClient(client, interval=0, sleep=lambda _: None)
        assert api.get_json("https://api.openalex.org/works", {}) == {"results": []}
        assert calls[0] == 2


def test_pair_categories_and_review_defaults():
    works = [publication(number) for number in range(1, 41)]
    pairs = collector.make_pairs(works, 50, "unchecked")
    categories = {name: sum(row["category"] == name for row in pairs) for name in collector.TARGETS}
    assert len(pairs) <= 50
    assert categories["same_consistent"] == 15
    assert categories["same_modified"] == 10
    assert categories["different_similar_title"] > 0
    assert categories["different_similar_author"] > 0
    assert categories["hard_incomplete"] > 0
    assert all(row["review_status"] == "pending" and row["reviewed_label"] == "" for row in pairs)
    assert all(row["overlap_status"] == "unchecked" for row in pairs)
    assert all(row["proposed_label"] == (1 if row["intended_doi"] == row["candidate_doi"] else 0)
               for row in pairs)
    assert all(row["intended_source_url"] and row["candidate_source_url"] for row in pairs)
    for category in ("different_similar_title", "different_similar_author"):
        group = [row for row in pairs if row["category"] == category]
        assert len({row["intended_doi"] for row in group}) == len(group)
        assert len({row["candidate_doi"] for row in group}) == len(group)


def test_export_only_approved_rows_and_runner_format(tmp_path):
    source, target = tmp_path / "candidates.csv", tmp_path / "evaluation_pairs_v1.csv"
    rows = collector.make_pairs([publication(1), publication(2)], 2, "unchecked")
    rows[0]["review_status"], rows[0]["reviewed_label"] = "approved", "1"
    rows[1]["review_status"], rows[1]["reviewed_label"] = "uncertain", "0"
    collector.write_csv(source, collector.CANDIDATE_COLUMNS, rows)
    assert collector.export_approved(source, target) == 1
    with target.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        exported = list(reader)
    assert reader.fieldnames == list(REQUIRED_COLUMNS)
    assert len(exported) == 1 and exported[0]["label"] == "1"
    assert exported[0]["citation_author"] == "Alice Rivera"


def test_run_outputs_summary_without_reviewed_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(collector, "collect", lambda api, exclusions: (
        [publication(number) for number in range(1, 4)],
        {"duplicates_removed": 0, "overlaps_removed": 0, "incomplete_removed": 0, "api_errors": []}))
    with httpx.Client(transport=httpx.MockTransport(lambda request: pytest.fail("network called"))) as client:
        summary = collector.run(tmp_path / "out", client=client)
    assert summary["real_publications_collected"] == 3
    assert summary["overlap_status"] == "unchecked"
    assert summary["pending_review"] == summary["candidate_pairs"]
    with (tmp_path / "out" / "evaluation_candidates_v1.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert all(row["review_status"] == "pending" and not row["reviewed_label"] for row in rows)
    assert (tmp_path / "out" / "evaluation_review_template_v1.csv").exists()
    assert json.loads((tmp_path / "out" / "evaluation_collection_summary.json").read_text())["candidate_pairs"] == len(rows)
    with pytest.raises(FileExistsError):
        collector.run(tmp_path / "out", client=client)


def test_failed_apis_do_not_create_fake_publications():
    class FailedApi:
        def get_json(self, url, params):
            raise httpx.ReadTimeout("synthetic timeout")

    publications, stats = collector.collect(FailedApi(), (set(), set()))
    assert publications == []
    assert collector.make_pairs(publications, 50, "unchecked") == []
    assert stats["api_errors"]

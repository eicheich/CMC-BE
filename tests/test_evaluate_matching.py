"""Synthetic fixtures test the offline runner; they are not a real-world benchmark."""

import csv
import json
import sys

import pytest

from app.schemas import Citation, Work
from app.verification import classify, compare, consistency_issues
from scripts import evaluate_matching as runner


BASE = {
    "case_id": "same", "citation_title": "Attention Is All You Need", "citation_author": "A. Vaswani",
    "citation_year": "2017", "citation_doi": "", "metadata_title": "Attention Is All You Need",
    "metadata_authors": "Ashish Vaswani;Noam Shazeer", "metadata_year": "2017",
    "metadata_doi": "", "label": "1",
}


def write_dataset(path, rows):
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=runner.REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def read_results(path):
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def test_valid_dataset_outputs_and_production_rule_unchanged(tmp_path):
    dataset = tmp_path / "pairs.csv"
    output = tmp_path / "out"
    negative = {**BASE, "case_id": "different", "citation_title": "Quantum Banana Networks", "label": "0"}
    ambiguous = {**BASE, "case_id": "year_error", "citation_year": "2022"}
    write_dataset(dataset, [BASE, negative, ambiguous])

    summary = runner.evaluate(dataset, output)
    results = read_results(output / "evaluation_results.csv")
    saved = json.loads((output / "evaluation_summary.json").read_text(encoding="utf-8"))
    assert summary == saved
    assert summary["counts"]["rows"] == 3
    assert summary["counts"]["rule_evaluated"] == 2
    assert summary["counts"]["ml_evaluated"] == 3
    assert summary["counts"]["common_evaluated"] == 2
    assert (results[0]["rule_status"], results[0]["rule_prediction"]) == ("verified_match", "1")
    assert (results[1]["rule_status"], results[1]["rule_prediction"]) == ("not_found", "0")
    assert results[2]["rule_status"] == "metadata_mismatch"
    assert results[2]["rule_prediction"] == ""
    assert all(row["ml_probability"] and row["ml_prediction"] in {"0", "1"} for row in results)

    citation = Citation(raw_citation="", title=BASE["citation_title"], authors=[BASE["citation_author"]], year=2017)
    work = Work(title=BASE["metadata_title"], authors=BASE["metadata_authors"].split(";"),
                year=2017, source="openalex")
    score, comparison, _ = compare(citation, work)
    assert float(results[0]["rule_score"]) == score
    assert results[0]["rule_status"] == classify(citation, comparison, score,
                                                  consistency_issues(citation, work, comparison))


def test_invalid_label_and_missing_metadata_are_skipped(tmp_path):
    dataset = tmp_path / "pairs.csv"
    output = tmp_path / "out"
    write_dataset(dataset, [{**BASE, "case_id": "bad", "label": "maybe"},
                            {**BASE, "case_id": "missing", "metadata_title": ""}])
    summary = runner.evaluate(dataset, output)
    rows = read_results(output / "evaluation_results.csv")
    assert rows[0]["error"] == "invalid_label"
    assert rows[0]["ml_prediction"] == ""
    assert rows[0]["ml_skip_reason"] == "invalid_label"
    assert rows[1]["rule_prediction"] == ""
    assert rows[1]["ml_skip_reason"] == "title_or_year_missing"
    assert summary["counts"]["valid_labels"] == 1
    assert summary["counts"]["common_evaluated"] == 0
    assert summary["metrics"]["common_subset"]["ml"]["accuracy"] is None


def test_empty_doi_and_ml_failure(tmp_path, monkeypatch):
    dataset = tmp_path / "pairs.csv"
    output = tmp_path / "out"
    write_dataset(dataset, [BASE])
    monkeypatch.setattr(runner, "predict_proba", lambda values: (_ for _ in ()).throw(RuntimeError("offline failure")))
    summary = runner.evaluate(dataset, output)
    row = read_results(output / "evaluation_results.csv")[0]
    assert row["rule_prediction"] == "1"
    assert row["ml_prediction"] == ""
    assert row["ml_skip_reason"] == "RuntimeError"
    assert summary["counts"]["ml_skipped"] == 1


def test_metrics():
    result = runner.metrics([(1, 1), (1, 0), (0, 1), (0, 0)])
    assert result["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}
    assert all(result[name] == 0.5 for name in ("accuracy", "precision", "recall", "f1", "false_positive_rate", "false_negative_rate"))
    assert runner.metrics([])["accuracy"] is None


def test_runner_is_offline_and_cli_works(tmp_path, monkeypatch, capsys):
    dataset = tmp_path / "pairs.csv"
    output = tmp_path / "out"
    write_dataset(dataset, [BASE])
    monkeypatch.setattr("app.verification.fetch_candidates", lambda *args: pytest.fail("network retrieval called"))
    monkeypatch.setattr(sys, "argv", ["evaluate_matching", "--input", str(dataset),
                                  "--output-dir", str(output), "--citation-author-format", "given-first"])
    runner.main()
    assert (output / "evaluation_results.csv").exists()
    assert "common_evaluated" in capsys.readouterr().out


def test_missing_columns_and_input_overwrite_rejected(tmp_path):
    dataset = tmp_path / "pairs.csv"
    dataset.write_text("case_id,label\na,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="required columns"):
        runner.evaluate(dataset, tmp_path / "out")
    output = tmp_path / "out"
    output.mkdir()
    dataset = output / "evaluation_results.csv"
    write_dataset(dataset, [BASE])
    with pytest.raises(ValueError, match="Input path must differ"):
        runner.evaluate(dataset, output)

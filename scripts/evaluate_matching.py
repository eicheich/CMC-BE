"""Offline pair matching evaluation. Labels mean same publication, not a fully correct citation."""

import argparse
import csv
import json
import re
from pathlib import Path

from app.ml.features import FEATURE_ORDER, PreprocessingUnavailable, build_features
from app.ml.predictor import METADATA_PATH, MODEL_VERSION, predict_proba
from app.schemas import Citation, Work
from app.verification import TITLE_RELEVANT, classify, compare, consistency_issues


REQUIRED_COLUMNS = (
    "case_id", "citation_title", "citation_author", "citation_year", "citation_doi",
    "metadata_title", "metadata_authors", "metadata_year", "metadata_doi", "label",
)
RESULT_COLUMNS = (
    "case_id", "label", "rule_score", "rule_status", "rule_prediction", "rule_skip_reason",
    "ml_probability", "ml_prediction", "ml_skip_reason", "consistency_issues", "error",
)


def year(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    if not re.fullmatch(r"(?:19|20)\d{2}(?:\.0)?", value):
        raise ValueError("invalid_year")
    return int(float(value))


def binary_rule(status: str, title_similarity: float | None) -> tuple[int | None, str]:
    if status == "verified_match":
        return 1, ""
    if status == "not_found" and title_similarity is not None and title_similarity < TITLE_RELEVANT:
        return 0, ""
    return None, "ambiguous_status" if status != "not_found" else "title_unavailable"


def metrics(pairs: list[tuple[int, int]]) -> dict:
    tp = sum(truth == 1 and predicted == 1 for truth, predicted in pairs)
    tn = sum(truth == 0 and predicted == 0 for truth, predicted in pairs)
    fp = sum(truth == 0 and predicted == 1 for truth, predicted in pairs)
    fn = sum(truth == 1 and predicted == 0 for truth, predicted in pairs)
    n = len(pairs)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        "samples": n,
        "accuracy": (tp + tn) / n if n else None,
        "precision": precision,
        "recall": recall,
        "f1": (2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        if precision is not None and recall is not None else None,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "false_negative_rate": fn / (fn + tp) if fn + tp else None,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def evaluate_row(row: dict[str, str], author_format: str, threshold: float) -> dict:
    def cell(name: str) -> str:
        return row[name] or ""

    result = {name: "" for name in RESULT_COLUMNS}
    result["case_id"] = cell("case_id").strip()
    result["label"] = cell("label").strip()
    result["consistency_issues"] = "[]"
    if not result["case_id"]:
        result["error"] = "missing_case_id"
        result["rule_skip_reason"] = result["ml_skip_reason"] = result["error"]
        return result
    if result["label"] not in {"0", "1"}:
        result["error"] = "invalid_label"
        result["rule_skip_reason"] = result["ml_skip_reason"] = result["error"]
        return result

    try:
        citation = Citation(raw_citation="", title=cell("citation_title").strip() or None,
                            authors=[cell("citation_author").strip()] if cell("citation_author").strip() else [],
                            year=year(cell("citation_year")), doi=cell("citation_doi").strip() or None)
        work = Work(title=cell("metadata_title").strip() or None,
                    authors=[name.strip() for name in cell("metadata_authors").split(";") if name.strip()],
                    year=year(cell("metadata_year")), doi=cell("metadata_doi").strip() or None,
                    source="openalex")  # Source is required by Work but unused in offline scoring.
        score, comparison, _ = compare(citation, work)
        issues = consistency_issues(citation, work, comparison)
        status = classify(citation, comparison, score, issues)
        prediction, skip_reason = binary_rule(status, comparison.title_similarity)
        result.update(rule_score=score, rule_status=status, rule_prediction=prediction,
                      rule_skip_reason=skip_reason, consistency_issues=json.dumps(issues))
    except (TypeError, ValueError) as exc:
        result["error"] = str(exc) if str(exc) == "invalid_year" else "invalid_row"
        result["rule_skip_reason"] = result["ml_skip_reason"] = result["error"]
        return result

    try:
        style = "ieee" if author_format == "given-first" else "apa"
        probability = predict_proba(build_features(citation, work, style))
        result["ml_probability"] = probability
        result["ml_prediction"] = int(probability >= threshold)
    except Exception as exc:
        result["ml_skip_reason"] = str(exc) if isinstance(exc, PreprocessingUnavailable) else type(exc).__name__
    return result


def evaluate(input_path: Path, output_dir: Path, author_format: str = "given-first") -> dict:
    if author_format not in {"given-first", "surname-first"}:
        raise ValueError("Unsupported citation author format")
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    threshold = metadata.get("threshold")
    if (threshold != 0.5 or metadata.get("model_version") != MODEL_VERSION or
            metadata.get("features") != list(FEATURE_ORDER)):
        raise ValueError("Expected v4.1 model metadata, nine ordered features, and threshold 0.5")
    result_path = output_dir / "evaluation_results.csv"
    summary_path = output_dir / "evaluation_summary.json"
    if input_path.resolve() in {result_path.resolve(), summary_path.resolve()}:
        raise ValueError("Input path must differ from output paths")

    with input_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or set(REQUIRED_COLUMNS) - set(reader.fieldnames):
            raise ValueError("Dataset is missing required columns")
        rows = [evaluate_row(row, author_format, threshold) for row in reader]

    valid_label_rows = sum(row["label"] in {"0", "1"} for row in rows)
    valid = [row for row in rows if row["label"] in {"0", "1"} and not row["error"]]
    rule = [row for row in valid if row["rule_prediction"] in (0, 1)]
    ml = [row for row in valid if row["ml_prediction"] in (0, 1)]
    common = [row for row in valid if row["rule_prediction"] in (0, 1) and row["ml_prediction"] in (0, 1)]

    def pairs(subset: list[dict], method: str) -> list[tuple[int, int]]:
        return [(int(row["label"]), row[f"{method}_prediction"]) for row in subset]

    summary = {
        "label_definition": "1 = same publication; 0 = different publication, not full citation correctness",
        "rule_binary_definition": "verified_match = 1; not_found = 0 only when both titles compare below TITLE_RELEVANT; other statuses abstain",
        "citation_author_format": author_format,
        "model_version": MODEL_VERSION,
        "ml_threshold": threshold,
        "rule_title_relevance_threshold": TITLE_RELEVANT,
        "counts": {"rows": len(rows), "valid_labels": valid_label_rows,
                   "usable_rows": len(valid), "invalid_or_unusable_rows": len(rows) - len(valid),
                   "rule_evaluated": len(rule), "rule_skipped": len(valid) - len(rule),
                   "ml_evaluated": len(ml), "ml_skipped": len(valid) - len(ml),
                   "common_evaluated": len(common)},
        "metrics": {"rule_based": metrics(pairs(rule, "rule")), "ml": metrics(pairs(ml, "ml")),
                    "common_subset": {"rule_based": metrics(pairs(common, "rule")),
                                      "ml": metrics(pairs(common, "ml"))}},
        "limitations": [
            "Only labeled pairs are scored; this does not evaluate retrieval or full citation correctness.",
            "Ambiguous rule-based statuses abstain, so coverage differs between methods.",
            "The common subset is required for direct comparison and may not represent all citations.",
            "Author format is a dataset-wide assumption; mixed formats may skip ML inference.",
            "Do not treat development data used to select v4.1 as an independent final test.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Labeled pair CSV")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for CSV and JSON results")
    parser.add_argument("--citation-author-format", choices=("given-first", "surname-first"),
                        default="given-first", help="Format of citation_author for the whole CSV")
    args = parser.parse_args()
    summary = evaluate(args.input, args.output_dir, args.citation_author_format)
    print(json.dumps(summary["counts"], indent=2))


if __name__ == "__main__":
    main()

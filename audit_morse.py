"""Validate final benchmark provenance/results and summarize repeated medians."""
import hashlib
import json
from pathlib import Path, PureWindowsPath
import statistics

HERE = Path(__file__).resolve().parent
DATA = HERE / "performance_morse_20261019"


def artifact_path(relative):
    """Historical logs use Windows separators; archives work on either OS.

    Laplacian's identical gen5 core is represented by the included gen5
    snapshot, NOT by a claimed Laplacian implementation in this release.
    """
    path = PureWindowsPath(relative)
    if path.anchor:
        raise ValueError("absolute archive path is not permitted")
    parts = path.parts
    if parts == ("laplacian", "core.py"):
        parts = ("_baseline_gen5_20261019", "core.py")
    if any(part in ("..", ".") or ":" in part for part in parts):
        raise ValueError("archive source must stay within this repository")
    return HERE.joinpath(*parts)


def main():
    names = ("repeated.jsonl", "ablation.jsonl", "extended.jsonl", "sparse1000.jsonl", "field3.jsonl")
    records, signatures = [], {}
    for name in names:
        lines = [json.loads(line) for line in (DATA / name).read_text(encoding="utf-8").splitlines()]
        metadata = lines[0]["metadata"]
        for relative, expected in metadata["sha256"].items():
            assert hashlib.sha256(artifact_path(relative).read_bytes()).hexdigest() == expected, relative
        for record in lines[1:]:
            record["source_log"] = name
            records.append(record)
            if record["status"] == "ok":
                key = (record["graph_sha256"], str(record["field"]), record["h"])
                value = {k: record[k] for k in ("betti", "omega", "cycles", "boundaries")}
                assert key not in signatures or signatures[key] == value
                signatures[key] = value
                assert record["representatives_closed"]
                assert record["h_counts"] == record["betti"]
    verification = json.loads((DATA / "verification.json").read_text(encoding="utf-8"))
    assert all(suite["exit_code"] == 0 for suite in verification["suites"])
    for relative, expected in verification["source_sha256"].items():
        assert hashlib.sha256(artifact_path(relative).read_bytes()).hexdigest() == expected
    oracle = json.loads((DATA / "rank_oracle.json").read_text(encoding="utf-8"))
    assert len(oracle["checks"]) == len(signatures)
    assert oracle["source_sha256"] == hashlib.sha256(
        (HERE / "_baseline_gen5_20261019" / "core.py").read_bytes()).hexdigest()
    assert oracle["driver_sha256"] == hashlib.sha256(
        (HERE / "reference" / "verify_morse_dimensions_20261019.py").read_bytes()).hexdigest()
    groups = {}
    for r in records:
        if r["source_log"] not in ("repeated.jsonl", "sparse1000.jsonl"):
            continue
        groups.setdefault((r["index"], r["engine"]), []).append(r)
    summaries = []
    for (index, engine), group in sorted(groups.items()):
        assert len(group) == 3 and all(r["status"] == "ok" for r in group)
        summaries.append(dict(index=index, engine=engine, median_seconds=statistics.median(
            r["seconds"] for r in group), time_range=[min(r["seconds"] for r in group),
                                                      max(r["seconds"] for r in group)],
            median_private_mib=statistics.median(r["peak_private_mib"] for r in group)))
    report = dict(final_runs=len(records), successful_runs=sum(r["status"] == "ok" for r in records),
                  graph_field_cases=len(signatures), regression_suites=len(verification["suites"]),
                  independent_rank_checks=len(oracle["checks"]),
                  repeated_medians=summaries,
                  other_runs=[r for r in records if r["source_log"] in
                              ("ablation.jsonl", "extended.jsonl", "field3.jsonl")])
    output = HERE / ".local-results" / "historical_audit.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Independent gen5 rank-only cross-check of EVERY final benchmark graph."""
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from benchmark_morse import CASES, graph

HERE = Path(__file__).resolve().parent
DATA = HERE / "performance_morse_20261019"


def main():
    source = HERE / "_baseline_gen5_20261019" / "core.py"
    spec = importlib.util.spec_from_file_location("rank_oracle", source)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    measured = {}
    for name in ("repeated.jsonl", "ablation.jsonl", "extended.jsonl", "sparse1000.jsonl", "field3.jsonl"):
        for line in (DATA / name).read_text(encoding="utf-8").splitlines()[1:]:
            record = json.loads(line)
            if record["status"] == "ok":
                measured.setdefault(record["index"], []).append(record)
    checks = []
    for index in sorted(measured):
        case = CASES[index]
        vertices, edges = graph(case)
        graph_hash = hashlib.sha256(json.dumps(edges, separators=(",", ":")).encode()).hexdigest()
        start = time.perf_counter()
        oracle = old.path_homology(vertices, edges, max_dim=case["h"]+1, field=case["field"],
                                   loops=False, regular=False, generators=False, store_paths=False)
        for record in measured[index]:
            assert record["graph_sha256"] == graph_hash
            for recorded, attr in (("betti", "betti"), ("omega", "dim_omega"),
                                   ("cycles", "dim_cycles"), ("boundaries", "dim_boundaries")):
                expected = {int(p): value for p, value in record[recorded].items()}
                actual = getattr(oracle, attr)
                assert all(expected.get(p, 0) == actual.get(p, 0)
                           for p in set(expected) | set(actual)), (index, recorded)
        checks.append(dict(index=index, graph_sha256=graph_hash,
                           oracle_seconds=time.perf_counter()-start, field=case["field"],
                           betti=oracle.betti, completed_runs_checked=len(measured[index])))
        print(f"PASS gen5 rank oracle: case {index}, {len(measured[index])} runs", flush=True)
    (DATA / "rank_oracle.json").write_text(json.dumps(dict(
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), checks=checks),
        ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

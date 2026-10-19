"""Isolated, sequential gen5/full vs Morse/full vs Morse/no-dense-d bench.

Resource thresholds are sampled every 25ms, NOT OS-enforced hard limits.
All three modes return explicit Omega/Z/B/H chains; only morse_sparse omits
the additional dense allowed-path boundary matrices. Bases need not agree.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from benchmark_large import graph, formal_counts
from benchmark_limits import machine_info, process_memory

HERE = Path(__file__).resolve().parent
CASES = [
    dict(family="sparse", n=150, h=2, field="Q"),
    dict(family="sparse", n=300, h=2, field="Q"),
    dict(family="sparse", n=300, h=2, field=2),
    dict(family="sparse", n=500, h=2, field="Q"),
    dict(family="torus", n=125, h=3, field="Q", side=5, dimension=3),
    dict(family="torus", n=216, h=3, field=3, side=6, dimension=3),
    dict(family="dag", n=14, h=3, field=2),
    dict(family="dense", n=16, h=2, field="Q"),
    dict(family="sparse", n=1000, h=2, field=2),
]
ENGINES = ("linear", "linear_guard", "morse", "morse_sparse")


def worker(case, engine):
    source = (HERE / "_baseline_gen5_20261019" / "core.py"
              if engine in ("linear", "linear_guard") else HERE / "core.py")
    spec = importlib.util.spec_from_file_location("measured_core", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    vertices, edges = graph(case)
    graph_hash = hashlib.sha256(json.dumps(edges, separators=(",", ":")).encode()).hexdigest()
    print(json.dumps(dict(event="start", graph_sha256=graph_hash, edges=len(edges),
                          formal_paths=formal_counts(case, edges))), flush=True)
    options = dict(max_dim=case["h"]+1, field=case["field"], loops=False,
                   regular=False, generators=True)
    if engine == "linear_guard":
        original_matmul = module._field_matmul
        def guarded(left, right):
            if right.shape[1] == 0:
                kind, prime = module.resolve_field(case["field"])
                return module._field_zeros((left.shape[0], 0), kind, prime)
            return original_matmul(left, right)
        module._field_matmul = guarded
    if engine not in ("linear", "linear_guard"):
        options.update(homology_method="morse", store_matrices=engine == "morse")
    start = time.perf_counter()
    result = module.path_homology(vertices, edges, **options)
    seconds = time.perf_counter() - start
    # A matrix-free check of ORIGINAL-chain representatives, outside timing.
    from morse import Arithmetic
    arithmetic = Arithmetic(None if case["field"] == "Q" else case["field"])
    for p, chains in result.homology_basis.items():
        if p == 0:
            continue
        for chain in chains:
            bd = {}
            for path, value in chain.items():
                for face, sign in module.boundary_of_path(path):
                    arithmetic.axpy(bd, {face: sign}, arithmetic.scalar(value))
            assert not bd, (p, chain)
    print(json.dumps(dict(event="done", seconds=seconds, betti=result.betti,
                          omega=result.dim_omega, cycles=result.dim_cycles,
                          boundaries=result.dim_boundaries,
                          h_counts={p: len(v) for p, v in result.homology_basis.items()},
                          morse_stats=getattr(result, "morse_stats", {}),
                          dense_matrices=result.matrices_materialized,
                          representatives_closed=True)), flush=True)


def measure(case, engine, timeout, memory_mib):
    command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()),
               "--worker", json.dumps(case), "--engine", engine]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, encoding="utf-8", env=env,
                               creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    start, peak, rss, status = time.perf_counter(), 0, 0, "ok"
    while process.poll() is None:
        try:
            current_rss, current_peak = process_memory(int(process._handle))
            peak, rss = max(peak, current_peak), max(rss, current_rss)
        except OSError:
            pass
        if peak > memory_mib * 2**20 or time.perf_counter()-start > timeout:
            status = "memory_limit" if peak > memory_mib * 2**20 else "timeout"
            process.kill()
            break
        time.sleep(.025)
    out, error = process.communicate()
    record = dict(case, engine=engine, status=status, wall_seconds=time.perf_counter()-start,
                  peak_private_mib=peak/2**20, peak_rss_mib=rss/2**20, seconds=None)
    for line in out.splitlines():
        record.update({k: v for k, v in json.loads(line).items() if k != "event"})
    if status == "ok" and (process.returncode != 0 or record["seconds"] is None):
        record.update(status="error", error=error[-4000:])
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker")
    parser.add_argument("--engine", choices=ENGINES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--indices", default="0,1,2,4,5,6,7")
    parser.add_argument("--engines", default="linear,morse,morse_sparse")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=35)
    parser.add_argument("--memory-mib", type=int, default=1200)
    args = parser.parse_args()
    if args.worker:
        worker(json.loads(args.worker), args.engine)
        return
    if args.output is None:
        parser.error("--output required")
    indices = [int(i) for i in args.indices.split(",")]
    engines = args.engines.split(",")
    if (args.repeat < 1 or args.timeout <= 0 or args.memory_mib <= 0
            or any(i < 0 or i >= len(CASES) for i in indices)
            or any(e not in ENGINES for e in engines)):
        parser.error("invalid arguments")
    paths = [HERE / "core.py", HERE / "morse.py", Path(__file__).resolve(),
             HERE / "_baseline_gen5_20261019" / "core.py"]
    metadata = dict(date="2026-10-19", machine=machine_info(), timeout=args.timeout,
                    memory_mib=args.memory_mib, sampling_ms=25,
                    sha256={str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in paths})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = {}
    with args.output.open("w", encoding="utf-8") as output:
        output.write(json.dumps(dict(metadata=metadata))+"\n")
        for index in indices:
            for repetition in range(args.repeat):
                order = engines if repetition % 2 == 0 else list(reversed(engines))
                for engine in order:
                    record = measure(CASES[index], engine, args.timeout, args.memory_mib)
                    record.update(index=index, repetition=repetition)
                    if record["status"] == "ok":
                        key = (record["graph_sha256"], str(record["field"]), record["h"])
                        signature = {k: record[k] for k in ("betti", "omega", "cycles", "boundaries")}
                        assert key not in completed or signature == completed[key]
                        completed[key] = signature
                    output.write(json.dumps(record)+"\n")
                    output.flush()
                    print(json.dumps({k: record.get(k) for k in
                                      ("index", "engine", "repetition", "field", "n", "status",
                                       "seconds", "peak_private_mib", "error")}), flush=True)


if __name__ == "__main__":
    main()

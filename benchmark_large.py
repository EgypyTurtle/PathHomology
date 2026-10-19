"""Larger gen4/gen5 digraphs: isolated sequential workers with hard resource caps.

The parent never materializes the graph. Input generation, input hashing and
output hashing count toward wall time but not the path_homology call time.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
import time

from benchmark_limits import machine_info, process_memory

HERE = Path(__file__).resolve().parent
CASES = []


def add(family, n, h, field="Q", mode="stream", **extra):
    CASES.append(dict(family=family, n=n, h=h, field=field, mode=mode, **extra))


for n in (200000, 300000, 500000):
    add("sparse", n, 2)
add("sparse", 200000, 4, 2)
add("sparse", 300000, 2, 2)
add("cycle", 500000, 6)
add("cycle", 1000000, 6, 2)
for side, dimension, h, field in ((10, 3, 3, "Q"), (16, 3, 3, "Q"),
                                  (20, 3, 3, 2), (30, 2, 2, "Q"), (80, 2, 2, "Q")):
    add("torus", side ** dimension, h, field, side=side, dimension=dimension)
for n in (40, 48, 64):
    add("dense", n, 2)
add("dense", 40, 2, 2)
add("dense", 36, 3, 2)
add("dense", 40, 3)
add("dag", 36, 3, 2)
add("dag", 44, 3, 2)
add("dag", 64, 2)
add("dag", 80, 2)
add("sparse", 150, 2, mode="full")
add("sparse", 300, 2, mode="full")
add("sparse", 300, 2, 2, mode="full")
add("sparse", 500, 2, mode="full")
add("dense", 32, 3, 2)  # Extended time window for the prior unresolved case.


def graph(case):
    n, family = case["n"], case["family"]
    rng = random.Random(20261018 + n)
    if family == "sparse":
        edges = []
        for u in range(n):
            targets = set()
            while len(targets) < min(3, n - 1):
                v = rng.randrange(n)
                if u != v:
                    targets.add(v)
            edges.extend((u, v) for v in sorted(targets))
    elif family == "cycle":
        edges = [(u, (u + 1) % n) for u in range(n)]
    elif family == "dense":
        edges = [(u, v) for u in range(n) for v in range(n)
                 if u != v and rng.random() < .3]
    elif family == "dag":
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    elif family == "torus":
        side, dimension = case["side"], case["dimension"]
        edges = []
        for u in range(n):
            for axis in range(dimension):
                stride = side ** axis
                coordinate = u // stride % side
                v = u + stride if coordinate + 1 < side else u - (side - 1) * stride
                edges.append((u, v))
    else:
        raise ValueError(family)
    return list(range(n)), edges


def formal_counts(case, edges):
    n, degree, family = case["n"], case["h"] + 1, case["family"]
    if family == "dag":
        return [math.comb(n, p + 1) if p < n else 0 for p in range(degree + 1)]
    if family in ("sparse", "cycle", "torus"):
        d = (min(3, n - 1) if family == "sparse" else
             1 if family == "cycle" else case["dimension"])
        return [n * d ** p for p in range(degree + 1)]
    counts = [1] * n
    totals = [n]
    for _ in range(degree):
        following = [0] * n
        for u, v in edges:
            following[v] += counts[u]
        counts = following
        totals.append(sum(counts))
    return totals


def worker(core_path, case):
    spec = importlib.util.spec_from_file_location("large_measured_core", core_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    vertices, edges = graph(case)
    digest = hashlib.sha256()
    for u, v in edges:
        digest.update(struct.pack("<II", u, v))
    counts = formal_counts(case, edges)
    print(json.dumps(dict(event="start", edges=len(edges), paths=counts,
                          graph_sha256=digest.hexdigest())), flush=True)
    start = time.perf_counter()
    result = module.path_homology(vertices, edges, max_dim=case["h"] + 1,
                                 loops=False, regular=False, field=case["field"],
                                 generators=case["mode"] == "full",
                                 store_paths=case["mode"] == "full")
    seconds = time.perf_counter() - start
    basis_hash = None
    if case["mode"] == "full":
        digest = hashlib.sha256()
        for name in ("omega_paths", "cycles", "homology_basis"):
            for p, chains in sorted(getattr(result, name).items()):
                digest.update(f"{name}:{p}:{len(chains)};".encode())
                for chain in chains:
                    entries = []
                    for path, value in sorted(chain.items()):
                        scalar = (str(module.Fraction(value)) if case["field"] == "Q"
                                  else int(value) % case["field"])
                        entries.append((path, scalar))
                    digest.update(json.dumps(entries, separators=(",", ":")).encode())
        basis_hash = digest.hexdigest()
    print(json.dumps(dict(event="done", seconds=seconds,
                          betti={p: result.betti.get(p, 0) for p in range(case["h"] + 1)},
                          omega=result.dim_omega, cycles=result.dim_cycles,
                          boundaries=result.dim_boundaries, dim_allowed=result.dim_allowed,
                          candidate_counts=getattr(result, "candidate_counts", {}),
                          support_pruning=getattr(result, "support_pruning", {}),
                          basis_sha256=basis_hash)), flush=True)


def measure(core_path, case, timeout, memory_mib):
    command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()),
               "--worker", json.dumps(case), "--core", str(core_path)]
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
        if peak > memory_mib * 2**20 or time.perf_counter() - start > timeout:
            status = "memory_limit" if peak > memory_mib * 2**20 else "timeout"
            process.kill()
            break
        time.sleep(.025)
    out, error = process.communicate()
    record = dict(case, status=status, wall_seconds=time.perf_counter() - start,
                  peak_private_mib=peak / 2**20, peak_rss_mib=rss / 2**20)
    for line in out.splitlines():
        message = json.loads(line)
        record.update({k: v for k, v in message.items() if k != "event"})
    if "seconds" not in record:
        record["seconds"] = None
        if status == "ok":
            record.update(status="error", error=error[-3000:])
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--indices", default=",".join(map(str, range(26))))
    parser.add_argument("--versions", default="gen4,gen5")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=25)
    parser.add_argument("--memory-mib", type=int, default=1200)
    parser.add_argument("--worker")
    parser.add_argument("--core", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.core, json.loads(args.worker))
        return
    if args.output is None:
        parser.error("--output is required")
    sources = dict(gen4=HERE / "_baseline_gen4_20261019" / "core.py", gen5=HERE / "core.py")
    indices = [int(i) for i in args.indices.split(",")]
    versions = args.versions.split(",")
    if (args.repeat < 1 or args.timeout <= 0 or args.memory_mib <= 0
            or any(i < 0 or i >= len(CASES) for i in indices)
            or any(v not in sources for v in versions)):
        parser.error("invalid case, version, repetition, or resource limit")
    metadata = dict(date="2026-10-19", machine=machine_info(), timeout=args.timeout,
                    memory_mib=args.memory_mib, repeat=args.repeat,
                    source_sha256={v: hashlib.sha256(sources[v].read_bytes()).hexdigest() for v in versions},
                    driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(exist_ok=True, parents=True)
    with args.output.open("w", encoding="utf-8") as output:
        output.write(json.dumps(dict(metadata=metadata)) + "\n")
        for i in indices:
            for repetition in range(args.repeat):
                order = versions if repetition % 2 == 0 else list(reversed(versions))
                for version in order:
                    record = measure(sources[version], CASES[i], args.timeout, args.memory_mib)
                    record.update(index=i, version=version, repetition=repetition)
                    output.write(json.dumps(record) + "\n")
                    output.flush()
                    print(json.dumps({k: record.get(k) for k in
                                      ("index", "version", "repetition", "family", "n", "h", "field",
                                       "mode", "status", "seconds", "peak_private_mib")}), flush=True)


if __name__ == "__main__":
    main()

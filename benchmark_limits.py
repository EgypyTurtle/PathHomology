"""Reproducible, resource-bounded path-homology capacity measurements.

Every case runs in its own process, sequentially. H_h is measured by enumerating
through degree h+1; the artificial top Betti number of the truncation is omitted.
Windows peak private commit is measured by the OS, including NumPy allocations.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time


class MemoryCounters(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in (
            "peak_rss", "rss", "peak_paged", "paged", "peak_nonpaged",
            "nonpaged", "private", "peak_private", "private_usage",
        )
    ]


def process_memory(handle):
    counters = MemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(
        wintypes.HANDLE(handle), ctypes.byref(counters), counters.cb
    )
    if not ok:
        raise ctypes.WinError()
    return counters.peak_rss, counters.peak_private


def machine_info():
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
            (name, ctypes.c_ulonglong) for name in (
                "total", "available", "page_total", "page_available",
                "virtual_total", "virtual_available", "extended",
            )
        ]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    info = {
        "python": sys.version, "platform": platform.platform(),
        "processor": platform.processor(), "logical_cpus": os.cpu_count(),
        "ram_gib": status.total / 2**30, "available_gib": status.available / 2**30,
    }
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                           r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            info["processor_name"] = winreg.QueryValueEx(key, "ProcessorNameString")[0]
    except OSError:
        pass
    return info


def graph(case):
    n = case["n"]
    family = case["family"]
    if family == "dag":
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    elif family == "cycle":
        edges = [(i, (i + 1) % n) for i in range(n)]
    elif family == "sparse":
        rng = random.Random(20261018 + n)
        edges = []
        for i in range(n):
            targets = set()
            while len(targets) < min(3, n - 1):
                j = rng.randrange(n)
                if i != j:
                    targets.add(j)
            edges.extend((i, j) for j in sorted(targets))
    elif family == "dense":
        rng = random.Random(20261018 + n)
        edges = [(i, j) for i in range(n) for j in range(n)
                 if i != j and rng.random() < .3]
    else:
        raise ValueError(family)
    return list(range(n)), edges


def path_counts(vertices, edges, degree):
    incoming = {v: [] for v in vertices}
    for u, v in edges:
        incoming[v].append(u)
    counts = dict.fromkeys(vertices, 1)
    totals = [len(vertices)]
    for _ in range(degree):
        counts = {v: sum(counts[u] for u in incoming[v]) for v in vertices}
        totals.append(sum(counts.values()))
    return totals


def cases():
    result = []
    def add(family, n, h, field="Z2", mode="dims"):
        result.append(dict(family=family, n=n, h=h, field=field, mode=mode))
    for n in (12, 16, 20, 24, 28):
        for h in (2, 4):
            add("dag", n, h)
    for n, h in ((16, 4), (20, 4), (24, 3), (28, 3)):
        add("dag", n, h, "Q")
    for n, h in ((100, 2), (1000, 2), (3000, 2), (10000, 2),
                 (100, 4), (500, 4), (1000, 4), (100, 6)):
        add("sparse", n, h)
    for n, h in ((100, 2), (1000, 2), (3000, 2), (100, 4), (500, 4)):
        add("sparse", n, h, "Q")
    for n in (8, 12, 16, 20, 24):
        for h in (1, 2, 3):
            add("dense", n, h)
    for n, h in ((12, 2), (16, 2), (16, 3), (20, 2)):
        add("dense", n, h, "Q")
    for n in (1000, 10000):
        add("cycle", n, 6)
    for field in ("Z2", "Q"):
        for family, n, h in (("dag", 10, 3), ("dag", 12, 3), ("dag", 14, 3),
                             ("sparse", 20, 2), ("sparse", 50, 2),
                             ("dense", 8, 2), ("dense", 12, 2)):
            add(family, n, h, field, "full")
    # Extended frontier cases, appended so earlier recorded indices stay fixed.
    for field in ("Z2", "Q"):
        for n, h in ((30, 4), (32, 4), (24, 5), (20, 6)):
            add("dag", n, h, field)
        for n, h in ((28, 3), (32, 3), (24, 4), (32, 2), (40, 2), (60, 1)):
            add("dense", n, h, field)
        for n, h in ((10000, 6), (30000, 2), (100000, 2), (1000, 12)):
            add("sparse", n, h, field)
    for field in ("Z2", "Q"):
        for family, n, h in (("dag", 16, 3), ("dag", 18, 3),
                             ("sparse", 100, 2), ("dense", 16, 2)):
            add(family, n, h, field, "full")
    return result


def worker(core_path, case):
    spec = importlib.util.spec_from_file_location("measured_core", core_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    vertices, edges = graph(case)
    totals = path_counts(vertices, edges, case["h"] + 1)
    print(json.dumps({"event": "start", "edges": len(edges), "paths": totals}),
          flush=True)
    start = time.perf_counter()
    options = {"store_paths": False} if case["mode"] == "stream" else {}
    result = module.path_homology(
        vertices, edges, max_dim=case["h"] + 1, field=case["field"],
        generators=case["mode"] == "full",
        **options,
    )
    elapsed = time.perf_counter() - start
    print(json.dumps({
        "event": "done", "seconds": elapsed,
        "betti": {p: result.betti.get(p, 0) for p in range(case["h"] + 1)},
        "omega": result.dim_omega, "cycles": result.dim_cycles,
        "boundaries": result.dim_boundaries,
        "pruned_from_dimension": getattr(result, "pruned_from_dimension", None),
        "candidate_counts": getattr(result, "candidate_counts", {}),
        "support_pruning": getattr(result, "support_pruning", {}),
    }), flush=True)


def measure(core_path, case, timeout, memory_mib, path_cap):
    vertices, edges = graph(case)
    totals = path_counts(vertices, edges, case["h"] + 1)
    record = dict(case, edges=len(edges), paths=totals)
    if sum(totals) > path_cap and case["mode"] != "stream":
        return dict(record, status="path_cap", seconds=None)
    command = [sys.executable, "-u", str(Path(__file__).resolve()),
               "--worker", json.dumps(case), "--core", str(core_path)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, env=env,
                             creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    start = time.perf_counter()
    status = "ok"
    peak_rss = peak_private = 0
    while child.poll() is None:
        try:
            rss, private = process_memory(int(child._handle))
            peak_rss = max(peak_rss, rss)
            peak_private = max(peak_private, private)
        except OSError:
            pass
        if peak_private > memory_mib * 2**20:
            status = "memory_limit"
            child.kill()
            break
        if time.perf_counter() - start > timeout:
            status = "timeout"
            child.kill()
            break
        time.sleep(.025)
    stdout, stderr = child.communicate()
    record.update(status=status, wall_seconds=time.perf_counter() - start,
                  peak_rss_mib=peak_rss / 2**20,
                  peak_private_mib=peak_private / 2**20)
    done = None
    for line in stdout.splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if item.get("event") == "done":
            done = item
    if done is not None:
        record.update({k: v for k, v in done.items() if k != "event"})
    elif status == "ok":
        record.update(status="error", seconds=None, error=stderr[-3000:])
    else:
        record["seconds"] = None
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--core", type=Path, default=Path(__file__).with_name("core.py"))
    ap.add_argument("--output", type=Path)
    ap.add_argument("--timeout", type=float, default=20)
    ap.add_argument("--memory-mib", type=int, default=1200)
    ap.add_argument("--path-cap", type=int, default=700000)
    ap.add_argument("--worker")
    ap.add_argument("--indices", help="comma-separated case indices")
    ap.add_argument("--stream", action="store_true", help="use streaming mode for dimensions cases")
    ap.add_argument("--repeat", type=int, default=1, help="independent worker runs per selected case")
    ap.add_argument("--date", default="2026-10-19", help="client-local measurement date")
    args = ap.parse_args()
    if args.worker:
        worker(args.core, json.loads(args.worker))
        return
    if args.output is None:
        ap.error("--output is required")
    all_cases = cases()
    if args.stream:
        for case in all_cases:
            if case["mode"] == "dims":
                case["mode"] = "stream"
    selected = (list(range(len(all_cases))) if not args.indices
                else [int(i) for i in args.indices.split(",")])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(machine=machine_info(), core=str(args.core.resolve()),
                    core_sha256=hashlib.sha256(args.core.read_bytes()).hexdigest(),
                    timeout=args.timeout, memory_mib=args.memory_mib,
                    path_cap=args.path_cap, date=args.date)
    print(json.dumps(metadata), flush=True)
    with args.output.open("w", encoding="utf-8") as output:
        output.write(json.dumps({"metadata": metadata}) + "\n")
        for index in selected:
            for repetition in range(args.repeat):
                item = measure(args.core, all_cases[index], args.timeout,
                               args.memory_mib, args.path_cap)
                item["index"] = index
                item["repetition"] = repetition
                output.write(json.dumps(item) + "\n")
                output.flush()
                print(json.dumps(item), flush=True)


if __name__ == "__main__":
    main()

"""Portable release checks. Does not require pytest or external working trees."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def main():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    commands = [[name] for name in ("test_morse.py", "test_round5.py", "test_optimization.py",
                                    "test_path_homology.py", "test_fields.py", "examples.py",
                                    "verify_morse_dimensions.py", "audit_morse.py")]
    commands += [["cli.py", "run", "--edges", "0-1,1-2,2-3,3-0", "--max-dim", "2",
                  "--homology-method", "morse", "--no-boundary-matrices", "--json"],
                 ["cli.py", "run", "--edges", "0-1,1-2,2-3,3-0", "--max-dim", "2", "--stream", "--json"]]
    records = []
    for command in commands:
        start = time.perf_counter()
        proc = subprocess.run([sys.executable, "-B", *command], cwd=HERE, env=env,
                              capture_output=True, text=True, encoding="utf-8", timeout=180)
        passed = proc.returncode == 0
        if command[0] == "cli.py" and passed:
            result = json.loads(proc.stdout)
            assert result["betti"]["0"] == result["betti"]["1"] == 1
            assert result["matrices_materialized"] is False
            assert result["generators_materialized"] == ("--stream" not in command)
        records.append(dict(command=command, exit_code=proc.returncode,
                            seconds=time.perf_counter()-start, stdout=proc.stdout, stderr=proc.stderr))
        print(f"{'PASS' if passed else 'FAIL'} {' '.join(command)}", flush=True)
        if not passed:
            print(proc.stderr or proc.stdout, flush=True)
    tracked_hashes = {}
    for line in (HERE / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        expected, relative = line.split("  ", 1)
        actual = hashlib.sha256((HERE / relative).read_bytes()).hexdigest()
        assert actual == expected, f"release file hash mismatch: {relative}"
        tracked_hashes[relative] = actual
    output = HERE / ".local-results" / "release_verification.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(dict(python=sys.version, files_checked=len(tracked_hashes),
                                     source_sha256=tracked_hashes, suites=records),
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    if any(record["exit_code"] for record in records):
        raise SystemExit("Release verification failed; inspect .local-results/release_verification.json")
    print(f"All {len(records)} checks passed; {len(tracked_hashes)} release hashes verified.")


if __name__ == "__main__":
    main()

"""Check the existing repository's four historical validation entry points.

Some original scripts only print a verdict, without a failing exit status.
Require their success markers as well; never modify archived results/.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def main():
    suites = {
        "check_paper_examples.py": ["ALL PASS"],
        "check_circulant.py": ["Theorem 1.1 holds (dim Omega_n = 10 for n = 1..7): True",
                                "Theorem 1.2 holds in every case above: True"],
        "check_torsion.py": ["torsion subgroup is exactly Z/nZ for every n tested: True"],
        "verify_gf2_paths.py": ["ALL GF(2) FAST PATHS AGREE: True"],
    }
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    records = []
    for filename, markers in suites.items():
        start = time.perf_counter()
        try:
            proc = subprocess.run([sys.executable, "-B", filename], cwd=HERE, env=env,
                                  text=True, encoding="utf-8", capture_output=True, timeout=120)
            passed = proc.returncode == 0 and all(marker in proc.stdout for marker in markers)
            record = dict(file=filename, passed=passed, exit_code=proc.returncode,
                          seconds=time.perf_counter()-start, stdout=proc.stdout, stderr=proc.stderr)
        except subprocess.TimeoutExpired:
            record = dict(file=filename, passed=False, seconds=time.perf_counter()-start,
                          error="120-second timeout; no completion claimed")
        records.append(record)
        print(f"{'PASS' if record['passed'] else 'FAIL'} {filename} ({record['seconds']:.3f}s)", flush=True)
    output = HERE / ".local-results" / "repository_verification.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(dict(suites=records), ensure_ascii=False, indent=2), encoding="utf-8")
    if not all(record["passed"] for record in records):
        raise SystemExit("Historical validation failed; inspect .local-results/repository_verification.json")
    print("All four repository compatibility checks passed.")


if __name__ == "__main__":
    main()

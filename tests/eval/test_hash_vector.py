"""The eval fallback hash vector is stable across processes (PYTHONHASHSEED)."""

import os
import subprocess
import sys
from pathlib import Path


def test_hash_vector_is_stable_across_hash_seeds():
    outputs = []
    for seed in ("0", "7"):
        result = subprocess.run(  # noqa: S603 - fixed argv, no untrusted input
            [
                sys.executable,
                "-c",
                "from tests.eval.conftest import _hash_vector; "
                "print(_hash_vector('q5x-embedding-model', 8))",
            ],
            env={**os.environ, "PYTHONHASHSEED": seed},
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]

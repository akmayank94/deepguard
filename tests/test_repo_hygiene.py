import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_repo_hygiene():
    r = subprocess.run([sys.executable, "tools/check_repo.py"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_notebooks_are_valid():
    import nbformat
    for nb in (ROOT / "notebooks").rglob("*.ipynb"):
        nbformat.validate(nbformat.read(nb, as_version=4))

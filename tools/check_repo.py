"""Repository hygiene checks (run in CI and locally: python tools/check_repo.py).

Fails if the repo contains anything that must never be committed (D-013, D-018):
large files, videos or model weights, secrets, or notebooks saved with outputs.
"""
import json
import re
import subprocess
import sys

MAX_BYTES = 5_000_000
BLOCKED_EXT = (".mp4", ".avi", ".mov", ".mkv", ".zip", ".pt", ".pth", ".onnx", ".npy", ".npz", ".parquet", ".h5")
SECRET_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{30,}"),                      # Hugging Face token
    re.compile(r"ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"),
    re.compile(r"KAGGLE_(API_TOKEN|KEY)\s*=\s*['\"][A-Za-z0-9_\-]{16,}['\"]"),
    re.compile(r'"key"\s*:\s*"[0-9a-f]{32}"'),               # legacy kaggle.json
    re.compile(r"kaldir\.vc\.in\.tum\.de/faceforensics_download"),   # FF++ script link (private)
    re.compile(r"[?&]Signature=[A-Za-z0-9%]{20,}"),          # signed download URLs
]


def tracked_files():
    out = subprocess.run(["git", "ls-files", "-z"], capture_output=True, text=True, check=True).stdout
    return [f for f in out.split("\0") if f]


def main():
    problems = []
    for path in tracked_files():
        try:
            data = open(path, "rb").read()
        except FileNotFoundError:
            continue
        if len(data) > MAX_BYTES:
            problems.append(f"{path}: {len(data)/1e6:.1f} MB (limit {MAX_BYTES/1e6:.0f} MB)")
        if path.lower().endswith(BLOCKED_EXT):
            problems.append(f"{path}: data/model files must not be committed")
        if path.endswith(".env") or path.endswith("kaggle.json"):
            problems.append(f"{path}: secrets file")
        text = data.decode("utf-8", errors="ignore")
        for pat in SECRET_PATTERNS:
            if pat.search(text):
                problems.append(f"{path}: looks like a secret or private link ({pat.pattern[:40]}…)")
        if path.endswith(".ipynb") and not path.startswith("legacy/"):
            nb = json.loads(text)
            if any(c.get("outputs") for c in nb.get("cells", []) if c.get("cell_type") == "code"):
                problems.append(f"{path}: notebook saved with outputs (clear them; results go to docs/OUTPUTS.md)")
    for p in problems:
        print("FAIL", p)
    print(f"checked {len(tracked_files())} files: {'OK' if not problems else f'{len(problems)} problem(s)'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

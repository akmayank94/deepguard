"""
scripts/verify_dataset.py

Verifies the preprocessed FaceForensics++ dataset before any training begins.

Checks:
    1. Root folder exists and is readable
    2. real/ subfolder — count clips, frames per clip, image integrity
    3. fake/ subfolder — count clips per manipulation type, frames, integrity
    4. Pilot sampling — confirms exactly 100 real + 20×5 fake can be drawn
    5. Frame dimension check — all JPGs are 224×224
    6. Corrupt file detection — any JPG that fails to open is flagged
    7. Summary report — counts, warnings, go/no-go verdict

Expected structure:
    DATA_ROOT/
        real/
            real_000/  000.jpg ... 019.jpg
            real_001/  ...
            ...
            real_999/
        fake/
            Deepfakes_xxx_xxx/       000.jpg ... 019.jpg
            Face2Face_xxx_xxx/       ...
            FaceShifter_xxx_xxx/     ...
            FaceSwap_xxx_xxx/        ...
            NeuralTextures_xxx_xxx/  ...

Usage:
    python scripts/verify_dataset.py
    python scripts/verify_dataset.py --data_root "D:/path/to/deepfakes_preprocessed"
    python scripts/verify_dataset.py --full      # check every frame (slow)
"""

import os
import sys
import argparse
import random
from pathlib import Path
from collections import defaultdict

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(override=True)

# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def _pass(msg):  print(f"  {GREEN}PASS{RESET}  {msg}")
def _fail(msg):  print(f"  {RED}FAIL{RESET}  {msg}")
def _warn(msg):  print(f"  {YELLOW}WARN{RESET}  {msg}")
def _info(msg):  print(f"  {CYAN}INFO{RESET}  {msg}")


# ---------------------------------------------------------------------------
# Manipulation type prefixes — must match your folder naming exactly
# ---------------------------------------------------------------------------
MANIPULATION_TYPES = [
    "Deepfakes",
    "Face2Face",
    "FaceShifter",
    "FaceSwap",
    "NeuralTextures",
]

FRAMES_PER_CLIP  = 20
EXPECTED_SIZE    = (224, 224)
PILOT_REAL       = 100
PILOT_FAKE_EACH  = 20   # per manipulation type


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_open_image(path: Path) -> tuple:
    """
    Try to open a JPG with PIL.
    Returns (ok: bool, width: int, height: int, error: str)
    """
    try:
        from PIL import Image
        with Image.open(path) as img:
            img.verify()        # catches truncated files
        # Re-open after verify (verify closes the file)
        with Image.open(path) as img:
            w, h = img.size
        return True, w, h, ""
    except Exception as e:
        return False, 0, 0, str(e)


def _scan_clip_folder(clip_path: Path,
                       check_images: bool = True,
                       sample_only: bool = True) -> dict:
    """
    Scan one clip folder and return statistics.

    Args:
        clip_path:    Path to clip folder
        check_images: whether to open and verify images
        sample_only:  if True, only check first + last frame (fast mode)

    Returns dict with keys:
        n_frames, missing_frames, corrupt_files, wrong_size, frame_list
    """
    jpgs = sorted(clip_path.glob("*.jpg"))
    n_frames = len(jpgs)
    missing  = []
    corrupt  = []
    wrong_sz = []

    # Check expected frame names 000.jpg … 019.jpg
    expected_names = {f"{i:03d}.jpg" for i in range(FRAMES_PER_CLIP)}
    actual_names   = {f.name for f in jpgs}
    missing = sorted(expected_names - actual_names)

    if check_images:
        if sample_only:
            # Only check first and last frame for speed
            frames_to_check = [jpgs[0], jpgs[-1]] if len(jpgs) >= 2 else jpgs
        else:
            frames_to_check = jpgs

        for f in frames_to_check:
            ok, w, h, err = _try_open_image(f)
            if not ok:
                corrupt.append((f.name, err))
            elif (w, h) != EXPECTED_SIZE:
                wrong_sz.append((f.name, w, h))

    return {
        "n_frames":      n_frames,
        "missing_frames": missing,
        "corrupt_files": corrupt,
        "wrong_size":    wrong_sz,
        "frame_list":    jpgs,
    }


def _get_manip_type(folder_name: str) -> str:
    """Return the manipulation type prefix for a fake clip folder name."""
    for mt in MANIPULATION_TYPES:
        if folder_name.startswith(mt):
            return mt
    return "Unknown"


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------

def check_root(data_root: Path) -> bool:
    print(f"\n{BOLD}Check 1: Root folder{RESET}")
    if not data_root.exists():
        _fail(f"DATA_ROOT not found: {data_root}")
        _info("Set DATA_ROOT in your .env file or pass --data_root")
        return False
    if not data_root.is_dir():
        _fail(f"DATA_ROOT is not a directory: {data_root}")
        return False
    _pass(f"Root exists: {data_root}")
    return True


def check_real(data_root: Path, full: bool = False) -> dict:
    """Scan real/ subfolder. Returns summary dict."""
    print(f"\n{BOLD}Check 2: real/ subfolder{RESET}")
    real_root = data_root / "real"

    if not real_root.exists():
        _fail(f"real/ subfolder not found at {real_root}")
        return {}

    clip_dirs = sorted([d for d in real_root.iterdir() if d.is_dir()])
    n_clips = len(clip_dirs)
    _info(f"Found {n_clips} clip folders in real/")

    if n_clips == 0:
        _fail("No clip folders found in real/")
        return {}

    # Scan all clips (fast: sample first+last frame only unless --full)
    total_frames  = 0
    incomplete    = []   # clips with wrong frame count
    corrupt_clips = []
    wrong_sz_clips= []
    all_corrupt   = []
    all_wrong_sz  = []

    check_every = 1 if full else max(1, n_clips // 20)  # sample ~20 clips in fast mode

    for i, clip_dir in enumerate(clip_dirs):
        stats = _scan_clip_folder(
            clip_dir,
            check_images=(i % check_every == 0),
            sample_only=not full,
        )
        total_frames += stats["n_frames"]

        if stats["missing_frames"]:
            incomplete.append((clip_dir.name, stats["missing_frames"]))
        if stats["corrupt_files"]:
            corrupt_clips.append(clip_dir.name)
            all_corrupt.extend([(clip_dir.name, f, e) for f, e in stats["corrupt_files"]])
        if stats["wrong_size"]:
            wrong_sz_clips.append(clip_dir.name)
            all_wrong_sz.extend([(clip_dir.name, f, w, h) for f, w, h in stats["wrong_size"]])

    # Report
    avg_frames = total_frames / n_clips if n_clips > 0 else 0
    _info(f"Total frames      : {total_frames}")
    _info(f"Avg frames/clip   : {avg_frames:.1f}  (expected {FRAMES_PER_CLIP})")

    ok = True
    if not incomplete:
        _pass(f"All {n_clips} clips have correct frame counts")
    else:
        _warn(f"{len(incomplete)} clips have missing frames:")
        for name, missing in incomplete[:5]:
            print(f"      {name}: missing {missing}")
        if len(incomplete) > 5:
            print(f"      ... and {len(incomplete) - 5} more")
        ok = False

    if not all_corrupt:
        _pass("No corrupt JPG files detected (sampled)")
    else:
        _fail(f"{len(all_corrupt)} corrupt files found:")
        for clip, fname, err in all_corrupt[:3]:
            print(f"      {clip}/{fname}: {err}")
        ok = False

    if not all_wrong_sz:
        _pass(f"All checked frames are {EXPECTED_SIZE[0]}×{EXPECTED_SIZE[1]}")
    else:
        _warn(f"{len(all_wrong_sz)} frames have unexpected dimensions:")
        for clip, fname, w, h in all_wrong_sz[:3]:
            print(f"      {clip}/{fname}: {w}×{h}")

    return {
        "n_clips":      n_clips,
        "total_frames": total_frames,
        "clip_dirs":    clip_dirs,
        "incomplete":   incomplete,
        "corrupt":      all_corrupt,
        "wrong_size":   all_wrong_sz,
        "ok":           ok,
    }


def check_fake(data_root: Path, full: bool = False) -> dict:
    """Scan fake/ subfolder. Returns summary dict."""
    print(f"\n{BOLD}Check 3: fake/ subfolder{RESET}")
    fake_root = data_root / "fake"

    if not fake_root.exists():
        _fail(f"fake/ subfolder not found at {fake_root}")
        return {}

    clip_dirs = sorted([d for d in fake_root.iterdir() if d.is_dir()])
    n_clips_total = len(clip_dirs)
    _info(f"Found {n_clips_total} clip folders in fake/")

    if n_clips_total == 0:
        _fail("No clip folders found in fake/")
        return {}

    # Group by manipulation type
    by_type = defaultdict(list)
    unknown = []
    for clip_dir in clip_dirs:
        mt = _get_manip_type(clip_dir.name)
        if mt == "Unknown":
            unknown.append(clip_dir.name)
        else:
            by_type[mt].append(clip_dir)

    # Report per-type counts
    all_types_present = True
    for mt in MANIPULATION_TYPES:
        count = len(by_type[mt])
        if count > 0:
            _pass(f"{mt:<22}: {count} clips")
        else:
            _fail(f"{mt:<22}: 0 clips — NOT FOUND")
            all_types_present = False

    if unknown:
        _warn(f"{len(unknown)} folders with unrecognized prefix: {unknown[:3]}")

    # Scan a sample of clips per type
    total_frames  = 0
    incomplete    = []
    all_corrupt   = []
    all_wrong_sz  = []

    for mt, clips in by_type.items():
        check_every = max(1, len(clips) // 10)
        for i, clip_dir in enumerate(clips):
            stats = _scan_clip_folder(
                clip_dir,
                check_images=(i % check_every == 0),
                sample_only=not full,
            )
            total_frames += stats["n_frames"]
            if stats["missing_frames"]:
                incomplete.append((clip_dir.name, stats["missing_frames"]))
            if stats["corrupt_files"]:
                all_corrupt.extend(
                    [(clip_dir.name, f, e) for f, e in stats["corrupt_files"]])
            if stats["wrong_size"]:
                all_wrong_sz.extend(
                    [(clip_dir.name, f, w, h) for f, w, h in stats["wrong_size"]])

    _info(f"Total fake frames : {total_frames}")

    ok = all_types_present
    if not incomplete:
        _pass("All sampled fake clips have correct frame counts")
    else:
        _warn(f"{len(incomplete)} fake clips have missing frames (first 3):")
        for name, missing in incomplete[:3]:
            print(f"      {name}: missing {missing}")
        ok = False

    if not all_corrupt:
        _pass("No corrupt fake JPGs detected (sampled)")
    else:
        _fail(f"{len(all_corrupt)} corrupt fake files found")
        ok = False

    return {
        "n_clips_total": n_clips_total,
        "by_type":       dict(by_type),
        "total_frames":  total_frames,
        "incomplete":    incomplete,
        "corrupt":       all_corrupt,
        "wrong_size":    all_wrong_sz,
        "ok":            ok,
    }


def check_pilot_sampling(real_info: dict, fake_info: dict) -> bool:
    """
    Check 4: Confirm pilot sampling is possible.
    Need >= 100 real clips and >= 20 clips per manipulation type.
    """
    print(f"\n{BOLD}Check 4: Pilot sampling feasibility{RESET}")
    ok = True

    # Real
    n_real = real_info.get("n_clips", 0)
    if n_real >= PILOT_REAL:
        _pass(f"Real clips: {n_real} >= {PILOT_REAL} needed for pilot")
    else:
        _fail(f"Real clips: {n_real} < {PILOT_REAL} needed — pilot not possible")
        ok = False

    # Fake per type
    by_type = fake_info.get("by_type", {})
    for mt in MANIPULATION_TYPES:
        count = len(by_type.get(mt, []))
        if count >= PILOT_FAKE_EACH:
            _pass(f"{mt:<22}: {count} clips >= {PILOT_FAKE_EACH} needed")
        else:
            _fail(f"{mt:<22}: {count} clips < {PILOT_FAKE_EACH} needed")
            ok = False

    if ok:
        # Simulate the actual sampling
        random.seed(42)
        real_clips    = real_info["clip_dirs"]
        sampled_real  = random.sample(real_clips, PILOT_REAL)
        sampled_fake  = {}
        for mt in MANIPULATION_TYPES:
            clips = by_type.get(mt, [])
            sampled_fake[mt] = random.sample(clips, PILOT_FAKE_EACH)

        total_pilot_clips  = PILOT_REAL + PILOT_FAKE_EACH * len(MANIPULATION_TYPES)
        total_pilot_frames = total_pilot_clips * FRAMES_PER_CLIP
        _pass(f"Pilot sample drawn: {PILOT_REAL} real + "
              f"{PILOT_FAKE_EACH}×{len(MANIPULATION_TYPES)} fake "
              f"= {total_pilot_clips} clips / {total_pilot_frames} frames")
        _info(f"First 3 sampled real  : {[d.name for d in sampled_real[:3]]}")
        for mt in MANIPULATION_TYPES:
            _info(f"Sample [{mt}]: {[d.name for d in sampled_fake[mt][:2]]}")

    return ok


def check_loadable(real_info: dict, fake_info: dict) -> bool:
    """
    Check 5: Load one real and one fake frame through PIL and confirm shape.
    This proves the full load path works end-to-end.
    """
    print(f"\n{BOLD}Check 5: End-to-end frame load{RESET}")
    ok = True

    try:
        import numpy as np
        from PIL import Image

        # Load one real frame
        real_clip = real_info["clip_dirs"][0]
        real_frame = sorted(real_clip.glob("*.jpg"))[0]
        with Image.open(real_frame) as img:
            arr = np.array(img)
        if arr.shape == (224, 224, 3):
            _pass(f"Real frame loaded: {real_frame.name} → shape {arr.shape} dtype {arr.dtype}")
        else:
            _warn(f"Real frame shape: {arr.shape}  (expected (224,224,3))")

        # Load one fake frame
        by_type = fake_info["by_type"]
        first_type = MANIPULATION_TYPES[0]
        fake_clip  = by_type[first_type][0]
        fake_frame = sorted(fake_clip.glob("*.jpg"))[0]
        with Image.open(fake_frame) as img:
            arr = np.array(img)
        if arr.shape == (224, 224, 3):
            _pass(f"Fake frame loaded: {fake_frame.name} → shape {arr.shape} dtype {arr.dtype}")
        else:
            _warn(f"Fake frame shape: {arr.shape}  (expected (224,224,3))")

        # Run CT on one real frame as full integration check
        from contourlet import ContourletTransform
        ct = ContourletTransform(num_levels=2, directions_per_level=[4, 8])
        result = ct.decompose_image(arr)
        n_subbands_l1 = len(result['subbands']['R'][0])
        _pass(f"CT decomposition on real frame: level-1 subbands = {n_subbands_l1}  (expected 4)")

    except Exception as e:
        _fail(f"Frame load or CT failed: {e}")
        import traceback
        traceback.print_exc()
        ok = False

    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Verify preprocessed FaceForensics++ dataset structure")
    parser.add_argument(
        "--data_root", type=str, default=None,
        help="Path to deepfakes_preprocessed/ folder. Overrides .env DATA_ROOT.")
    parser.add_argument(
        "--full", action="store_true",
        help="Check every frame (slow but thorough). Default: fast sampling.")
    args = parser.parse_args()

    # Resolve data root
    data_root_str = args.data_root or os.environ.get("DATA_ROOT", "")
    if not data_root_str:
        print(f"{RED}ERROR:{RESET} DATA_ROOT not set. "
              f"Add it to .env or pass --data_root")
        sys.exit(1)

    data_root = Path(data_root_str)

    print(f"\n{'='*60}")
    print(f"{BOLD}  FaceForensics++ Dataset Verification{RESET}")
    print(f"{'='*60}")
    print(f"  Data root : {data_root}")
    print(f"  Mode      : {'full (every frame)' if args.full else 'fast (sampled)'}")

    # Run checks
    if not check_root(data_root):
        sys.exit(1)

    real_info = check_real(data_root, full=args.full)
    if not real_info:
        sys.exit(1)

    fake_info = check_fake(data_root, full=args.full)
    if not fake_info:
        sys.exit(1)

    pilot_ok = check_pilot_sampling(real_info, fake_info)
    load_ok  = check_loadable(real_info, fake_info)

    # Final verdict
    print(f"\n{'='*60}")
    print(f"{BOLD}  Summary{RESET}")
    print(f"{'='*60}")

    all_ok = (
        real_info.get("ok", False)
        and fake_info.get("ok", False)
        and pilot_ok
        and load_ok
    )

    checks = [
        ("Root folder",         True),
        ("real/ subfolder",     real_info.get("ok", False)),
        ("fake/ subfolder",     fake_info.get("ok", False)),
        ("Pilot feasibility",   pilot_ok),
        ("End-to-end load+CT",  load_ok),
    ]
    for name, result in checks:
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        print(f"  {status}  {name}")

    # Dataset size summary
    print(f"\n  Dataset summary:")
    print(f"    Real clips        : {real_info.get('n_clips', '?')}")
    print(f"    Fake clips total  : {fake_info.get('n_clips_total', '?')}")
    by_type = fake_info.get("by_type", {})
    for mt in MANIPULATION_TYPES:
        print(f"      {mt:<22}: {len(by_type.get(mt, []))}")
    total_clips = (real_info.get('n_clips', 0) +
                   fake_info.get('n_clips_total', 0))
    total_frames = total_clips * FRAMES_PER_CLIP
    print(f"    Total clips       : {total_clips}")
    print(f"    Total frames      : {total_frames}")

    if all_ok:
        print(f"\n  {GREEN}{BOLD}Dataset verified. Ready to proceed.{RESET}")
        print(f"  Next step: python scripts/precompute_features.py --pilot\n")
    else:
        print(f"\n  {RED}{BOLD}Issues found. Fix them before proceeding.{RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
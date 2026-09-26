"""Build the code zip that accompanies a portal upload.

    python notebooks/make_code_zip.py submissions/v4/v4_code.zip
"""
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(os.environ.get("AMLC_ROOT") or Path(__file__).resolve().parents[1])
out = Path(sys.argv[1])
files = [p for p in (ROOT / "code/business_entity_resolution").rglob("*")
         if p.is_file() and "__pycache__" not in p.parts]
PKG = "code/business_entity_resolution"
PIPELINE = ("run_v3.py", "run_v4.py", "run_v5.py", "run_v7.py", "build_fit3.py", "ce_export.py", "ce_blend.py",
            "ce_blend2.py", "make_submission.py", "write_candidates.py", "make_code_zip.py", "lbsim.py", "fr_fit.py")
extra = {ROOT / "notebooks" / n: f"{PKG}/pipeline/{n}" for n in PIPELINE}
for k in ("embed_knn", "ce_kernel"):
    extra |= {p: f"{PKG}/kaggle/{k}/{p.name}" for p in (ROOT / "kaggle" / k).glob("*") if p.is_file()}
out.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in files:
        z.write(p, p.relative_to(ROOT).as_posix())
    for p, arc in extra.items():
        if p.exists():
            z.write(p, arc)
print(f"{out}: {len(z.namelist()) if False else len(files) + sum(p.exists() for p in extra)} files")

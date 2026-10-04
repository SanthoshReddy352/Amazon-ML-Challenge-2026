"""Build the final package <team>_submission.zip (README "Final Submission Package" layout).

    python notebooks/make_final_zip.py --matching submissions/v13/matching_results.tsv \
        --candidates output/candidate_pairs.tsv --out submissions/Team_Bloom_submission.zip

Layout:
    output/matching_results.tsv, output/candidate_pairs.tsv
    code/business_entity_resolution/{src/, pipeline/, kaggle/, tests/, README.md, requirements.txt}
    Documentation_template.md   (the filled methodology document, docs/Documentation.md)
"""
import os
import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(os.environ.get("AMLC_ROOT") or Path(__file__).resolve().parents[1])
ap = argparse.ArgumentParser()
ap.add_argument("--matching", type=Path, required=True)
ap.add_argument("--candidates", type=Path, required=True)
ap.add_argument("--doc", type=Path, default=ROOT / "docs/Documentation.md")
ap.add_argument("--out", type=Path, default=ROOT / "submissions/Team_Bloom_submission.zip")
ap.add_argument("--no-candidate-check", action="store_true",
                help="validate matching only (the validator's soft candidate check loads every candidate ID into memory: "
                     "~192M pairs do not fit 16 GB); check containment separately (notebooks/check_candidates.py)")
a = ap.parse_args()

cand = [] if a.no_candidate_check else ["--candidate", str(a.candidates.resolve())]
r = subprocess.run([sys.executable, "utils/validate_submission.py", "--matching", str(a.matching.resolve())] + cand +
                   ["--test-dir", "dataset/test", "--check-ids"],
                   cwd=ROOT / "student_resource", capture_output=True, text=True)
print(r.stdout[-1500:], r.stderr[-500:])
if r.returncode != 0:
    sys.exit("validator failed: package not built")

code_zip = ROOT / "submissions/_final_code.zip"
subprocess.run([sys.executable, str(ROOT / "notebooks/make_code_zip.py"), str(code_zip)], check=True)
a.out.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    z.write(a.matching, "output/matching_results.tsv")
    z.write(a.candidates, "output/candidate_pairs.tsv")
    with zipfile.ZipFile(code_zip) as cz:
        for n in cz.namelist():
            z.writestr(n, cz.read(n))
    z.write(a.doc, "Documentation_template.md")
code_zip.unlink()
with zipfile.ZipFile(a.out) as z:
    names = z.namelist()
print(f"{a.out}: {len(names)} entries, {a.out.stat().st_size / 1e6:.0f} MB")
print("\n".join(n for n in names if not n.startswith("code/business_entity_resolution/src/")))

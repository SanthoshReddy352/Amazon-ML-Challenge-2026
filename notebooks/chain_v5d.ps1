$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
Set-Location "D:\Amazon-ML-Challenge-2026"
$py = ".\.venv\Scripts\python.exe"
& $py -W ignore notebooks/run_v4.py --tag v4d --folds 5 --test *> artifacts/eda/run_v4d.txt
Remove-Item artifacts/refine/ext_val_set.parquet, artifacts/refine/ext_test_set.parquet -ErrorAction SilentlyContinue
& $py -W ignore notebooks/run_v5.py --main-tag v4d --top 5 --folds 5 --test *> artifacts/eda/run_v5d.txt
"DONE" | Out-File artifacts/eda/chain_v5d.done

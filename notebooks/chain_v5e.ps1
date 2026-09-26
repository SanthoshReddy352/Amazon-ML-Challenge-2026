$env:PYTHONIOENCODING = "utf-8"
Set-Location "D:\Amazon-ML-Challenge-2026"
$py = ".\.venv\Scripts\python.exe"
& $py -W ignore notebooks/run_v4.py --tag v4e --folds 5 --extra refine/fit3_set.parquet --test 2>&1 | Out-File -Encoding utf8 artifacts/eda/run_v4e.txt
Remove-Item artifacts/refine/ext_val_set.parquet, artifacts/refine/ext_test_set.parquet -ErrorAction SilentlyContinue
& $py -W ignore notebooks/run_v5.py --main-tag v4e --top 5 --folds 5 --test 2>&1 | Out-File -Encoding utf8 artifacts/eda/run_v5e.txt
Copy-Item artifacts/test_scores_ext.parquet artifacts/refine/test_scores_ext_v5e.parquet
Copy-Item artifacts/refine/ext_val_oof.parquet artifacts/refine/ext_val_oof_v5e.parquet
"DONE" | Out-File artifacts/eda/chain_v5e.done

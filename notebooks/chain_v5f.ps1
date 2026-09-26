$env:PYTHONIOENCODING = "utf-8"
Set-Location "D:\Amazon-ML-Challenge-2026"
while (-not (Test-Path artifacts/eda/chain_fit4.done)) { Start-Sleep -Seconds 30 }
$py = ".\.venv\Scripts\python.exe"
& $py -W ignore notebooks/run_v4.py --tag v4f --folds 5 --extra refine/fit3_set.parquet refine/fit4_set.parquet --test 2>&1 | Out-File -Encoding utf8 artifacts/eda/run_v4f.txt
Remove-Item artifacts/refine/ext_val_set.parquet, artifacts/refine/ext_test_set.parquet -ErrorAction SilentlyContinue
& $py -W ignore notebooks/run_v5.py --main-tag v4f --top 5 --folds 5 --test 2>&1 | Out-File -Encoding utf8 artifacts/eda/run_v5f.txt
Copy-Item artifacts/test_scores_ext.parquet artifacts/refine/test_scores_ext_v5f.parquet
Copy-Item artifacts/refine/ext_val_oof.parquet artifacts/refine/ext_val_oof_v5f.parquet
"DONE" | Out-File artifacts/eda/chain_v5f.done

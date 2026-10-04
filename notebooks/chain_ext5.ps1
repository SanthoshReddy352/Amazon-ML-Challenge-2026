# ext5 chain (run detached): merge neighbour lists -> ext5 pairs/features/LightGBM on val -> test scores
$env:PYTHONUTF8 = "1"
Set-Location D:\Amazon-ML-Challenge-2026
$py = "D:\Amazon-ML-Challenge-2026\.venv\Scripts\python.exe"
& $py notebooks\bfwd_merge.py *> logs\ext5_merge.log
& $py notebooks\run_ext3.py --source bknn2 --max-rank 15 --tag g *> logs\ext5_val.log
& $py notebooks\run_ext3.py --source bknn2 --max-rank 15 --tag g --test --reuse-models *> logs\ext5_test.log
"done" | Out-File logs\ext5_done.txt

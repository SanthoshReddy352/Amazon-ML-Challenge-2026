# after the wide reverse blocking: ext2w val model (5-fold, negatives subsampled) then test scoring
$env:PYTHONUTF8 = '1'
Set-Location 'D:\Amazon-ML-Challenge-2026'
while (-not (Test-Path 'artifacts\eda\chain_ext2w.done')) { Start-Sleep -Seconds 30 }
& 'D:\Amazon-ML-Challenge-2026\.venv\Scripts\python.exe' -u notebooks\run_ext2.py --tag w --neg-frac 0.35 > artifacts\eda\run_ext2w_val.txt 2> artifacts\eda\run_ext2w_val.err
& 'D:\Amazon-ML-Challenge-2026\.venv\Scripts\python.exe' -u notebooks\run_ext2.py --tag w --neg-frac 0.35 --test --reuse-models > artifacts\eda\run_ext2w_test.txt 2> artifacts\eda\run_ext2w_test.err
'done' | Out-File artifacts\eda\chain_ext2w_model.done

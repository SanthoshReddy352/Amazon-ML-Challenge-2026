# wait for the train reverse-blocking run, then run the test one (detached chain; survives the session)
$env:PYTHONUTF8 = '1'
Set-Location 'D:\Amazon-ML-Challenge-2026'
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*run_ext2_block.py train*' }) { Start-Sleep -Seconds 20 }
& 'D:\Amazon-ML-Challenge-2026\.venv\Scripts\python.exe' -u notebooks\run_ext2_block.py test > artifacts\eda\ext2_block_test.txt 2> artifacts\eda\ext2_block_test.err
'done' | Out-File artifacts\eda\chain_ext2.done

# wider reverse blocking (top-10 total, top-3 name, top-3 address): train (val S1) then test
$env:PYTHONUTF8 = '1'
Set-Location 'D:\Amazon-ML-Challenge-2026'
& 'D:\Amazon-ML-Challenge-2026\.venv\Scripts\python.exe' -u notebooks\run_ext2_block.py train --tag w --k-total 10 --k-name 3 --k-addr 3 > artifacts\eda\ext2w_block_train.txt 2> artifacts\eda\ext2w_block_train.err
& 'D:\Amazon-ML-Challenge-2026\.venv\Scripts\python.exe' -u notebooks\run_ext2_block.py test --tag w --k-total 10 --k-name 3 --k-addr 3 > artifacts\eda\ext2w_block_test.txt 2> artifacts\eda\ext2w_block_test.err
'done' | Out-File artifacts\eda\chain_ext2w.done

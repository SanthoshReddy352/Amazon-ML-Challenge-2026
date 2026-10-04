$env:PYTHONUTF8 = "1"
Set-Location D:\Amazon-ML-Challenge-2026
$py = "D:\Amazon-ML-Challenge-2026\.venv\Scripts\python.exe"
& $py notebooks\bcos_local.py val *> logs\bcos_val.log
& $py notebooks\bcos_local.py test *> logs\bcos_test.log

$env:PYTHONIOENCODING = "utf-8"
Set-Location "D:\Amazon-ML-Challenge-2026\code\business_entity_resolution"
$py = "D:\Amazon-ML-Challenge-2026\.venv\Scripts\python.exe"
& $py -W ignore -m src.features --split train --cands ../../artifacts/blocking/union_train --subset fit_sample3 --fit-sample 440000 --knn ../../artifacts/kaggle_out/knn_train.parquet --out ../../artifacts/features/fit_sample3_u 2>&1 | Out-File -Encoding utf8 ../../artifacts/eda/fit3_features.txt
Set-Location "D:\Amazon-ML-Challenge-2026"
& $py -W ignore notebooks/build_fit3.py 2>&1 | Out-File -Encoding utf8 artifacts/eda/fit3_build.txt
"DONE" | Out-File artifacts/eda/chain_fit3.done

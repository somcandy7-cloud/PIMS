@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
"C:\Users\somca\AppData\Local\Programs\Python\Python313\python.exe" -m streamlit run app.py --server.port 8501
pause

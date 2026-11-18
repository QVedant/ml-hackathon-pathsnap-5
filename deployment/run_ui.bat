@echo off
REM Laptop 2 (Windows): serve the evaluator UI. Then open http://localhost:5500 and enter laptop 1's IP.
cd /d "%~dp0..\ui"
python -m http.server 5500

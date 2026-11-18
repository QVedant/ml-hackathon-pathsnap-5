@echo off
REM Laptop 1 (Windows): start the PathSnap API.  Usage: deployment\run_api.bat [C:\path\to\evaluator_trips]
cd /d "%~dp0.."
if not "%~1"=="" set PATHSNAP_EVAL_DIR=%~1
ipconfig | findstr IPv4
uv sync -q
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000

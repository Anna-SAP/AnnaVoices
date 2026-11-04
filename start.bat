@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [setup] creating virtual environment .venv ...
  py -3.12 -m venv .venv 2>nul || python -m venv .venv
  if not exist ".venv\Scripts\python.exe" goto :novenv
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
)

echo [setup] checking models ...
".venv\Scripts\python.exe" -m voiceextract.models
if errorlevel 1 goto :modelerr

echo [run] http://127.0.0.1:8765  (Ctrl+C to stop)
".venv\Scripts\python.exe" -m voiceextract.server --port 8765 --open
goto :eof

:novenv
echo Python 3.10+ not found. Please install Python 3.12 and run again.
pause
goto :eof

:modelerr
echo Model download failed. Check the network and run again.
pause

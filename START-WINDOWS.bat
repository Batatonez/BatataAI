@echo off
setlocal
cd /d "%~dp0"
title BatataAI

if exist "runtime\python-win\python.exe" (
  "runtime\python-win\python.exe" app.py
  goto :eof
)

where py >nul 2>nul
if %errorlevel%==0 (
  py -3 app.py
  goto :eof
)

where python >nul 2>nul
if %errorlevel%==0 (
  python app.py
  goto :eof
)

echo.
echo ERRO: Python 3 nao foi encontrado neste PC.
echo Instale Python 3 ou coloque uma versao portatil em:
echo runtime\python-win\python.exe
echo.
pause

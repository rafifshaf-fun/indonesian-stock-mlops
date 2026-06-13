@echo off
title Stock Prediction CLI
color 0E

if "%1"=="" (
    echo Usage: predict.bat BBCA.JK
    echo        predict.bat BBCA.JK BBRI.JK TLKM.JK
    echo        predict.bat --all
    exit /b 1
)

if exist "venv\Scripts\python.exe" (
    venv\Scripts\python.exe cli.py predict %*
) else (
    python cli.py predict %*
)

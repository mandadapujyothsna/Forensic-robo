@echo off
title RoboForensic Rover - Webcam Mode
echo ========================================================
echo  RoboForensic Autonomous Rover Vision Engine
echo  Camera Source : Webcam 0
echo  Serial Port   : COM21
echo ========================================================
echo.
cd /d "%~dp0forensic-robot\ai\scripts"
py -u main.py 0 COM21
pause

@echo off
title RoboForensic Rover - ESP32-CAM Stream
echo ========================================================
echo  RoboForensic Autonomous Rover Vision Engine
echo  Camera Source : http://192.168.4.1/stream (ESP32-CAM)
echo  Serial Port   : COM21
echo ========================================================
echo.
cd /d "%~dp0forensic-robot\ai\scripts"
py -u main.py http://192.168.4.1/stream COM21
pause

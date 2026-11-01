@echo off
title RoboForensic Mission Control Server
echo ========================================================
echo  Deploying RoboForensic Mission Control Web Server
echo  Local URL  : http://localhost:8000
echo  Network URL: http://192.168.4.2:8000
echo ========================================================
echo.
cd /d "%~dp0forensic-robot\ai"
py -u server.py
pause

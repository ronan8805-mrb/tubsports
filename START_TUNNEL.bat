@echo off
title Cloudflare Tunnel - tubsports
echo ========================================
echo   Cloudflare Tunnel - tubsports.com
echo ========================================
echo.
echo  Tunnel ID: c5a09d03-e57a-47b8-b837-bfcf369d7d29
echo  Routes:
echo    api.tubsports.com  ->  localhost:8002
echo.
echo  Press Ctrl+C to stop the tunnel.
echo ========================================
echo.

"C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel run tubsports

@echo off
echo Stopping Nginx...
:: Replace this path with exactly where your nginx.exe is located
cd K:\projects\nginx-1.31.3
nginx -s quit

echo Stopping Flask Servers...
:: This targets the exact window titles we created in the start script
:: The /T flag ensures it kills the child Python process, not just the CMD window
taskkill /F /FI "WINDOWTITLE eq Flask 8080" /T
taskkill /F /FI "WINDOWTITLE eq Flask 8081" /T
taskkill /F /FI "WINDOWTITLE eq Flask 8082" /T

echo All servers safely shut down.
pause
@echo off
echo Starting Flask Load Balanced Servers...

:: Start instance 1 on port 8080
start "Flask 8080" cmd /k "skinclassifier\Scripts\activate && waitress-serve --port=8080 app:app"

:: Start instance 2 on port 8081
start "Flask 8081" cmd /k "skinclassifier\Scripts\activate && waitress-serve --port=8081 app:app"

:: Start instance 3 on port 8082
start "Flask 8082" cmd /k "skinclassifier\Scripts\activate && waitress-serve --port=8082 app:app"
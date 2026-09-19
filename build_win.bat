@echo off
rem build_win.bat - double-click wrapper: runs build_win.sh in Git Bash.
rem Usage: build_win.bat [engine|test]   (same args as build_win.sh)
chcp 65001 >nul
setlocal
set "ROOT=%~dp0"
call :find_bash || exit /b 1
"%BASH%" "%ROOT%build_win.sh" %*
if errorlevel 1 (echo. & pause & exit /b 1)
goto :eof

:find_bash
if exist "%ProgramFiles%\Git\bin\bash.exe" set "BASH=%ProgramFiles%\Git\bin\bash.exe" & goto :eof
if exist "%LOCALAPPDATA%\Programs\Git\bin\bash.exe" set "BASH=%LOCALAPPDATA%\Programs\Git\bin\bash.exe" & goto :eof
rem PATH fallback: skip System32\bash.exe (that one is WSL, incompatible)
for /f "delims=" %%i in ('where bash 2^>nul') do (
  echo %%i | findstr /i /c:"System32" >nul || (set "BASH=%%i" & goto :eof)
)
echo Error: Git Bash not found. Install Git for Windows first: https://git-scm.com/download/win 1>&2
pause
exit /b 1

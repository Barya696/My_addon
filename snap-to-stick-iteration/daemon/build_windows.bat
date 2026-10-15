@echo off
REM Build wrapDaemon (one-folder build) on Windows.
REM Run this from the daemon\ folder on a Windows machine with Python 3.10+.

python -m venv .venv-build
call .venv-build\Scripts\activate.bat

pip install --upgrade pip
pip install -r requirements.txt pyinstaller

REM The spec makes a one-folder build and bundles Intel MKL (for PARDISO) in it
pyinstaller --noconfirm wrapDaemon.spec

echo.
echo Done. The build is in dist\wrapDaemon: wrapDaemon.exe and its _internal folder.
echo Copy both into your Blender add-on's bin folder: add-on-wrap-for-blender\bin\
echo Replace any old bin\_internal folder completely instead of merging into it.
pause

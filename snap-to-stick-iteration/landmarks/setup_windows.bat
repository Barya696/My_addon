@echo off
rem Face landmarks environment for the Wrap add-on (CPU only)
rem Creates .venv next to this file and installs PyTorch (CPU) and VTK.
cd /d "%~dp0"
py -3 -m venv .venv || goto :error
.venv\Scripts\python.exe -m pip install --upgrade pip || goto :error
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu || goto :error
.venv\Scripts\python.exe -m pip install vtk numpy imageio matplotlib scipy || goto :error
echo.
echo Done. In the add-on preferences, set "Face Landmarks" to this folder:
echo %~dp0
exit /b 0
:error
echo Setup failed.
exit /b 1

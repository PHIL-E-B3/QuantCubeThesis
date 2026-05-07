@echo off
REM === FOMC Sentiment Pipeline - Windows Setup ===
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo Installing project dependencies...
pip install -r requirements.txt

echo.
echo === Setup complete! ===
echo Activate with: venv\Scripts\activate.bat
pause

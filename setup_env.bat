@echo off
echo Creating virtual environment...
python -m venv venv
call venv\Scripts\activate
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt
echo Checking CUDA...
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
echo Done. Activate with: venv\Scripts\activate
pause
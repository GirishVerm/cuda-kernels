@echo off
REM Build script for CUDA Attention Kernels on Windows

echo ===================================
echo CUDA Attention Kernels - Build
echo ===================================
echo.

echo Checking Python...
python --version
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.10 or 3.11
    pause
    exit /b 1
)
echo.

echo Checking CUDA...
nvcc --version
if errorlevel 1 (
    echo ERROR: CUDA not found! Please install CUDA Toolkit
    pause
    exit /b 1
)
echo.

echo Checking PyTorch CUDA...
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
if errorlevel 1 (
    echo ERROR: PyTorch CUDA not available!
    echo Install with: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pause
    exit /b 1
)
echo.

echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo.

echo Building CUDA extensions...
python setup.py develop
if errorlevel 1 (
    echo ERROR: Build failed!
    pause
    exit /b 1
)
echo.

echo Running tests...
pytest tests/ -v
if errorlevel 1 (
    echo WARNING: Some tests failed
)
echo.

echo ===================================
echo Build complete!
echo ===================================
echo.
echo Quick start:
echo   python python/benchmarks/compare_implementations.py
echo   python python/benchmarks/benchmark_attention.py
echo   python python/benchmarks/visualize_results.py
echo.
pause


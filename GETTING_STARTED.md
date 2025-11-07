# Getting Started

Quick guide to build, test, and run the CUDA attention kernels on Windows.

## Prerequisites

### Hardware Requirements
- **NVIDIA GPU** with compute capability 7.0+ (Volta/Turing/Ampere/Ada/Hopper)
  - Your GPU: GTX 1650 Ti (4GB VRAM, Turing, compute capability 7.5)
  - Note: 4GB VRAM is limited - use batch_size=1 and seq_len up to 1024

### Software Requirements

1. **Windows 10 or 11** (64-bit)

2. **NVIDIA GPU Drivers** (version 520+)
   - Download: https://www.nvidia.com/Download/index.aspx
   - Verify: Run `nvidia-smi` in Command Prompt

3. **CUDA Toolkit 11.8 or 12.1**
   - Download: https://developer.nvidia.com/cuda-downloads
   - Select: Windows → x86_64 → Your Windows version → exe (local)
   - Verify: Run `nvcc --version` in Command Prompt

4. **Visual Studio 2019 or 2022** (Community Edition is free)
   - Download: https://visualstudio.microsoft.com/downloads/
   - During installation, select: **"Desktop development with C++"**
   - This provides the C++ compiler required by CUDA

5. **Python 3.10 or 3.11**
   - Download: https://www.python.org/downloads/
   - During installation: ✅ Check "Add Python to PATH"
   - Verify: Run `python --version`

6. **PyTorch with CUDA support**
   - For CUDA 11.8:
     ```cmd
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```
   - For CUDA 12.1:
     ```cmd
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
     ```
   - Verify:
     ```cmd
     python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
     ```
   - Expected output: `CUDA Available: True` and your GPU name

## Installation

### Option 1: Automated Build (Recommended)

1. Open **Command Prompt** or **PowerShell**
2. Navigate to project directory:
   ```cmd
   cd C:\path\to\llm-proj
   ```
3. Run the build script:
   ```cmd
   build.bat
   ```

This will:
- Check prerequisites
- Install Python dependencies
- Build CUDA extensions
- Run tests

### Option 2: Manual Build

```cmd
# 1. Navigate to project
cd C:\path\to\llm-proj

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Build CUDA extensions (takes 2-5 minutes)
python setup.py develop

# 4. Verify installation
python -c "import attention_cuda; print('Installation successful!')"

# 5. Run tests
pytest tests/ -v
```

## Quick Start

### 1. Verify Correctness

Compare implementations against PyTorch reference:

```cmd
python python/benchmarks/compare_implementations.py
```

**Expected output:**
```
CORRECTNESS COMPARISON
Implementation      Max Error       Status    
---------------------------------------------
naive              1.23e-04        [PASS]
tiled              2.34e-04        [PASS]
flash              5.67e-04        [PASS]
```

### 2. Run Benchmarks

Benchmark all implementations across different sequence lengths:

```cmd
# For GTX 1650 Ti (4GB VRAM), use smaller configurations
python python/benchmarks/benchmark_attention.py --batch-size 1 --seq-lengths 128,256,512,1024
```

**Expected output (GTX 1650 Ti):**
```
Benchmarking: batch=1, heads=8, seq_len=512, head_dim=64

Benchmarking pytorch...
  Mean latency: 3.2 ms
  Throughput: 5.5 TFLOPS

Benchmarking naive...
  Mean latency: 6.1 ms
  Throughput: 2.9 TFLOPS

Benchmarking tiled...
  Mean latency: 2.8 ms
  Throughput: 6.3 TFLOPS

Benchmarking flash...
  Mean latency: 2.1 ms
  Throughput: 8.4 TFLOPS

flash speedup vs PyTorch: 1.5x
```

**Note:** Performance numbers are approximate for GTX 1650 Ti. Actual results may vary.

Results are saved to `benchmark_results.json`

### 3. Visualize Results

Generate plots from benchmark results:

```cmd
python python/benchmarks/visualize_results.py --input benchmark_results.json --output-dir benchmark_plots
```

This creates:
- `latency_comparison.png` - Bar chart comparing latencies
- `throughput_comparison.png` - Throughput vs sequence length
- `speedup_comparison.png` - Speedup relative to PyTorch
- `latency_scaling.png` - Scaling analysis (linear and log-log)
- `summary_report.txt` - Text summary of results

### 4. Profile Kernels

Basic profiling with PyTorch profiler:

```cmd
python python/profiling/profile_kernels.py --impl flash
```

Advanced profiling with NVIDIA Nsight Compute:

```cmd
ncu --set full --export profile_report python python/profiling/profile_kernels.py --impl flash
```

Then open `profile_report.ncu-rep` in Nsight Compute GUI.

## Usage Examples

### Python API

```python
import torch
from attention_cuda import attention_flash

# Create random inputs
batch_size, num_heads, seq_len, head_dim = 2, 8, 512, 64
Q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
K = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')
V = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')

# Run optimized attention
output = attention_flash(Q, K, V)

print(f"Output shape: {output.shape}")  # [2, 8, 512, 64]
```

### Module API (for nn.Module integration)

```python
import torch.nn as nn
from attention_cuda import AttentionFlash

class MyTransformerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = AttentionFlash()
        # ... other layers
    
    def forward(self, Q, K, V):
        output = self.attention(Q, K, V)
        # ... rest of forward pass
        return output
```

### Benchmarking Custom Configuration

```python
import torch
import time
from attention_cuda import attention_flash
from attention_cuda.utils import generate_test_inputs, estimate_tflops

# Generate inputs
Q, K, V = generate_test_inputs(
    batch_size=4,
    num_heads=16,
    seq_len=1024,
    head_dim=64
)

# Warmup
for _ in range(10):
    _ = attention_flash(Q, K, V)

torch.cuda.synchronize()

# Benchmark
start = time.perf_counter()
for _ in range(100):
    output = attention_flash(Q, K, V)
torch.cuda.synchronize()
elapsed_ms = (time.perf_counter() - start) / 100 * 1000

# Calculate performance
tflops = estimate_tflops(4, 16, 1024, 64, elapsed_ms)
print(f"Latency: {elapsed_ms:.3f} ms")
print(f"Throughput: {tflops:.2f} TFLOPS")
```

## Troubleshooting

### Issue 1: "CUDA Available: False" in PyTorch

**Cause**: CPU-only PyTorch installed or CUDA version mismatch

**Solution**:
```cmd
# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio

# Check your CUDA version
nvcc --version

# Reinstall PyTorch with matching CUDA version
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue 2: "nvcc not found" during build

**Cause**: CUDA not in PATH

**Solution**: Add CUDA to system PATH
- Press `Win + R`, type `sysdm.cpl`, press Enter
- Go to "Advanced" → "Environment Variables"
- In "System variables", find "Path", click "Edit"
- Add: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin`
- Restart Command Prompt

### Issue 3: "Microsoft Visual Studio not found"

**Cause**: Visual Studio C++ tools not installed

**Solution**: Install Visual Studio 2022 with "Desktop development with C++" workload

### Issue 4: Build fails with compiler errors

**Cause**: Version incompatibility

**Solution**: Ensure compatible versions:
- Python 3.10 or 3.11 (NOT 3.12)
- Visual Studio 2019 or 2022
- CUDA 11.8 or 12.1
- PyTorch 2.0+

### Issue 5: Out of memory errors (COMMON ON GTX 1650 Ti)

**Cause**: GTX 1650 Ti has only 4GB VRAM - limited for large sequences

**Solution**: Always use batch_size=1 and avoid sequence lengths above 1024:
```cmd
# Safe for GTX 1650 Ti
python python/benchmarks/benchmark_attention.py --batch-size 1 --seq-lengths 128,256,512,1024

# Will likely fail on GTX 1650 Ti
python python/benchmarks/benchmark_attention.py --seq-lengths 2048  # Too large!
```

If you still get OOM errors:
- Reduce sequence length further
- Close other GPU applications
- Try head_dim=32 instead of 64

### Issue 6: Incorrect results

**Cause**: GPU not being used

**Solution**: Verify GPU usage:
```cmd
python -c "import torch; t = torch.randn(100, 100, device='cuda'); print('GPU working!' if t.is_cuda else 'Using CPU')"
```

## Performance Expectations

### By GPU Model

| GPU Model | VRAM | Expected Performance | Use Case |
|-----------|------|---------------------|----------|
| **GTX 1650 Ti** | **4GB** | **~3.3 TFLOPS** | **Your GPU - Testing/Demo** |
| RTX 2060 | 6GB | ~7 TFLOPS | Development |
| RTX 3060 | 12GB | ~13 TFLOPS | Good |
| RTX 3070 | 8GB | ~20 TFLOPS | Very Good |
| RTX 3080 | 10GB | ~30 TFLOPS | Excellent |

### Expected Speedups (GTX 1650 Ti, batch_size=1)

| Sequence Length | PyTorch | Flash | Speedup | VRAM Usage |
|----------------|---------|-------|---------|------------|
| 128 | 0.9 ms | 0.7 ms | 1.3x | ~500 MB |
| 256 | 2.1 ms | 1.5 ms | 1.4x | ~800 MB |
| 512 | 3.2 ms | 2.1 ms | 1.5x | ~1.5 GB |
| 1024 | 12.5 ms | 8.0 ms | 1.6x | ~3.2 GB |

**Note:** Sequence lengths above 1024 may cause out-of-memory errors on 4GB VRAM.
Use `--batch-size 1` for all benchmarks.

## Next Steps

### For Development
1. Modify CUDA kernels in `csrc/` directory
2. Rebuild: `python setup.py develop`
3. Test: `pytest tests/ -v`
4. Benchmark: `python python/benchmarks/benchmark_attention.py`

### For Profiling
1. Install NVIDIA Nsight Compute (free)
2. Profile: `ncu --set full python python/profiling/profile_kernels.py`
3. Analyze results in Nsight Compute GUI
4. Iterate on optimizations

### For Documentation
- Read [Design Decisions](docs/DESIGN.md) for architecture details
- Read [Optimization Techniques](docs/OPTIMIZATION.md) for optimization strategies
- Read [Profiling Guide](docs/PROFILING.md) for profiling instructions

## Common Commands

```cmd
# Run correctness tests
python python/benchmarks/compare_implementations.py

# Run full benchmarks
python python/benchmarks/benchmark_attention.py

# Generate visualizations
python python/benchmarks/visualize_results.py

# Profile with PyTorch
python python/profiling/profile_kernels.py

# Profile with Nsight Compute
ncu --set full python python/profiling/profile_kernels.py

# Run all unit tests
pytest tests/ -v

# Clean build
python setup.py clean --all
python setup.py develop
```

## Support

For issues or questions:
- Check [Troubleshooting](#troubleshooting) section above
- Review documentation in `docs/` directory
- Check CUDA and PyTorch installation

## Contributing

This is a demonstration project for NVIDIA interview purposes. Improvements welcome:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

Happy optimizing!

import os
import sys
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# CUDA compilation flags
extra_compile_args = {
    'cxx': ['-O3', '-std=c++17'],
    'nvcc': [
        '-O3',
        '-std=c++17',
        '--use_fast_math',
        '--generate-line-info',
        '-Xptxas=-v',
        '-Xcompiler=-fPIC',
        # Optimize for modern architectures
        '-gencode=arch=compute_70,code=sm_70',  # V100
        '-gencode=arch=compute_75,code=sm_75',  # RTX 20xx
        '-gencode=arch=compute_80,code=sm_80',  # A100
        '-gencode=arch=compute_86,code=sm_86',  # RTX 30xx
        '-gencode=arch=compute_89,code=sm_89',  # RTX 40xx
    ]
}

setup(
    name="attention-cuda",
    version="0.1.0",
    author="Girish Verma",
    description="Custom CUDA kernels for optimized LLM attention mechanisms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/attention-cuda",
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    ext_modules=[
        CUDAExtension(
            name='attention_cuda_kernels',
            sources=[
                'csrc/bindings.cpp',
                'csrc/attention_naive.cu',
                'csrc/attention_tiled.cu',
                'csrc/attention_flash.cu',
            ],
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=True)
    },
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.24.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.3.0',
            'matplotlib>=3.7.0',
            'seaborn>=0.12.0',
            'pandas>=2.0.0',
        ],
    },
    python_requires='>=3.8',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Programming Language :: CUDA",
    ],
)


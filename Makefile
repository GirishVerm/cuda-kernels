.PHONY: build install test benchmark profile clean help

help:
	@echo "CUDA Attention Kernels - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  build       - Build CUDA extensions"
	@echo "  install     - Install dependencies and build"
	@echo "  test        - Run all tests"
	@echo "  benchmark   - Run benchmarks"
	@echo "  profile     - Profile kernels"
	@echo "  visualize   - Generate visualization plots"
	@echo "  clean       - Clean build artifacts"
	@echo "  help        - Show this help message"

install:
	pip install -r requirements.txt
	python setup.py develop

build:
	python setup.py develop

test:
	pytest tests/ -v

test-correctness:
	pytest tests/test_correctness.py -v

test-gradients:
	pytest tests/test_gradients.py -v

benchmark:
	python python/benchmarks/benchmark_attention.py \
		--seq-lengths 128,256,512,1024,2048 \
		--output benchmark_results.json

compare:
	python python/benchmarks/compare_implementations.py

profile:
	python python/profiling/profile_kernels.py

profile-ncu:
	ncu --set full --export ncu_report \
		python python/profiling/profile_kernels.py

profile-nsys:
	nsys profile --trace=cuda,nvtx --output=nsys_report \
		python python/profiling/profile_kernels.py

visualize:
	python python/benchmarks/visualize_results.py \
		--input benchmark_results.json \
		--output-dir benchmark_plots

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf **/__pycache__
	rm -rf **/*.pyc
	rm -rf .pytest_cache
	rm -f benchmark_results.json
	rm -rf benchmark_plots/
	rm -f *.ncu-rep
	rm -f *.nsys-rep
	rm -f profiling_results/*.json

format:
	black python/ tests/

lint:
	flake8 python/ tests/ --max-line-length=100

check: lint test

all: clean install test benchmark visualize


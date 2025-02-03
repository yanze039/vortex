.PHONY: setup clean dev lint format submodules

PYTHON := python3.11
UV := uv
VENV_BIN := $(CURDIR)/.venv/bin

PYTHON_VERSION := $(shell $(PYTHON) -c 'import sys; print(sys.version_info[0])')
CUDA_PATH := /usr/local/cuda
CUDA_INCLUDE_PATH := $(CUDA_PATH)/include
CUDA_LIB_PATH := $(CUDA_PATH)/lib64
CPATH := $(CUDA_INCLUDE_PATH):/usr/local/cuda/include
CUDACXX := /usr/local/cuda/bin/nvcc
	
setup: submodules
	$(UV) venv
	. $(VENV_BIN)/activate
	$(UV) pip install ninja cmake pybind11 numpy psutil
	$(UV) pip install -e .
	$(UV) pip install transformer_engine[pytorch] --no-build-isolation 
	cd vortex/ops/attn && MAX_JOBS=32 $(UV) pip install -v -e  . --no-build-isolation

submodules:
	git submodule update --init --recursive

clean:
	rm -rf .venv
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:
	ruff check .
	black --check .

format:
	ruff check --fix .
	black .

rebuild: clean setup 
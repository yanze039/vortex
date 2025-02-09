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
	
define _SANITY_ENV_CHECK_MESSAGE
Test
endef
export _SANITY_ENV_CHECK_MESSAGE

_check_env_enabled:
ifeq ($(CONDA_DEFAULT_ENV), base)
	@echo "You are trying to install vortex in the base conda environment, skipping"
	exit 1
else ifeq ($(IS_CONDA_ENV), true)
	@echo "Using conda environment: $(CONDA_PREFIX) $(shell python3 --version)"
else ifeq ($(IS_VENV), true)
	@echo "Using venv: $(VIRTUAL_ENV) $(shell python3 --version)"
else
	@echo "$$_SANITY_ENV_CHECK_MESSAGE"
	exit 1
endif
ifeq ($(PYTHON_OK), no)
	@echo "Python version is less than $(PYTHON_MIN_VERSION)"
	@echo "$$_SANITY_ENV_CHECK_MESSAGE"
	exit 1
endif


setup-full: submodules
	$(UV) venv
	. $(VENV_BIN)/activate
	$(UV) pip install ninja cmake pybind11 numpy psutil
	$(UV) pip install -e .
	$(UV) pip install transformer_engine[pytorch] --no-build-isolation 
	cd vortex/ops/attn && MAX_JOBS=32 $(UV) pip install -v -e  . --no-build-isolation

setup-vortex-ops: submodules
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
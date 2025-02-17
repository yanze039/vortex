<div align="center">

# Vortex

</div>


This repository contains implementations of computational primitives for convolutional multi-hybrid models (Hyena-[SE, MR, LI], StripedHyena 2, Evo 2)

For training, please refer to the [savanna](https://github.com/Zymrael/savanna/) project.

## Interface

There are two main ways to interface with `vortex`:

1. Use `vortex` as the inference engine for pre-trained multi-hybrid (e.g., [Evo 2 40B](configs/shc-evo2-40b-1M.yml)). For this, we recommend installing `vortex` in a new environment (see below).
2. Import from `vortex` specific classes, kernels or utilities to work with custom convolutional multi-hybrid (e.g., sourcing from `hyena_ops.interface`).


## 1. Quick install for vortex ops

```bash
make setup-vortex-ops
```
Note that this is does not install all dependencies required to run autoregressive inference with larger pre-trained models.

## 2. Building a custom inference environment

### Using Docker

Refer to the [Dockerfile](Dockerfile) for a sample build.

### Using conda, venv or uv

To run e2e installation in a uv environment, use the following command:
```bash
make setup-full
```
Note that the `setup-full` step will compile various CUDA kernels, which usually takes at most several minutes. It may be necessary to customize CUDA header and library paths in `Makefile`.  

## Inference Quickstart

### In Docker

To run 40b generation sample: `./run`.

To run 7b generation sample: `sz=7 ./run`.

To run tests: `./run ./run_tests`.

To interactively execute commands in docker environment: `./run bash`.


### Outside Docker

```bash
python3 generate.py \
    --config_path <PATH_TO_CONFIG> \
    --checkpoint_path <PATH_TO_CHECKPOINT> \
    --input_file <PATH_TO_INPUT_FILE> \
    --cached_generation
```
`--cached_generation` activates KV-caching and custom caching for different variants of Hyena layers, reducing peak memory usage and latency.


## Acknowledgements

This project is built and maintained by [Zymrael](https://github.com/Zymrael) (Michael Poli), [garykbrixi](https://github.com/garykbrixi) (Garyk Brixi), [antonvnv](https://github.com/antonvnv) (Anton Vorontsov), [jeromeku](https://github.com/jeromeku) (Jerome Ku).

## Cite

If you find this project useful, consider citing the following [references](CITE.md).


<div align="center">

![alt text](assets/sh2.png)

</div>

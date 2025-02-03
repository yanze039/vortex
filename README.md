<div align="center">

# 🌀 Vortex 🌀

</div>

Standalone implementation of computational primitives for deep signal processing model architectures. For training, please refer to the [savanna](https://github.com/Zymrael/savanna/) project.

## Interface

While installation of the a `vortex`-specific environment is recommended for generation with `vortex`, we also provide an interface for using the primitives with a lightweight installation process:

## Inference

### In Docker environment

To run 40b generation sample, simply execute:

```bash
./run
```

To run 7b generation sample, simply execute:

```bash
sz=7 ./run
```

To run tests:

```bash
./run ./run_tests
```

To interactively execute commands in docker environment:

```bash
./run bash
```

### Without Docker

#### Environment setup (uv)

To run e2e installation in a uv environment, use the following command:
```bash
make setup
```
Note that the `setup` step will compile various CUDA kernels, which usually takes at most several minutes. It may be necessary to customize CUDA header and library paths in `Makefile`.  

After installation, use 
```bash
source .venv/bin/activate
```
to activate the environment.

## Environment setup (conda)

Manually follow the `make setup` steps in your preferred conda environment.

## Environment setup (docker)

TODO

## Quickstart

```bash
python3 generate.py \
    --config_path <PATH_TO_CONFIG> \
    --checkpoint_path <PATH_TO_CHECKPOINT> \
    --input_file <PATH_TO_INPUT_FILE> \
    --cached_generation
```
`--cached_generation` turns on KV-caching and custom caching for different variants of Hyena layers.


## Acknowledgements

This project is built and maintained by: 

## Cite

`vortex` provides implementation of deep signal processing primitives spanning many projects: StripedHyena-1 and 2, Evo-1 and 2. If useful, consider citing the following:


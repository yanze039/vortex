# To run 40b generation sample: `./run`.

# To run 7b generation sample: `sz=7 ./run`.

# To run tests: `./run ./run_tests`.

# To interactively execute commands in docker environment: `./run bash`.

from nvcr.io/nvidia/pytorch:24.10-py3 as base
arg REQUIREMENTS=requirements.txt.frozen
copy ${REQUIREMENTS} .
run --mount=type=cache,target=/root/.cache \
    pip install -r ${REQUIREMENTS}
workdir /workdir

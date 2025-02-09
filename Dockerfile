from nvcr.io/nvidia/pytorch:24.10-py3 as base
arg REQUIREMENTS=requirements.txt.frozen
copy ${REQUIREMENTS} .
run --mount=type=cache,target=/root/.cache \
    pip install -r ${REQUIREMENTS}
workdir /workdir

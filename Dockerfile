from nvcr.io/nvidia/pytorch:25.04-py3 as base
run apt-get update && apt-get install -y git python3-pip python3-tomli && rm -rf /var/lib/apt/lists/*

# Extract dependencies from pyproject: can't use pip-tools' pip-compile as it
# requires whole tree to be copied to Docker context, which makes it impossible
# to develop in docker envrionment.
copy pyproject.toml .
run python3 -c 'import tomli;\
                p = tomli.load(open("pyproject.toml", "rb"))["project"];\
                print("\n".join(p["dependencies"] + p.get("optional-dependencies", {"special": []})["special"]))'\
    > requirements.txt
run pip install -r requirements.txt

workdir /workdir
#!/bin/bash
set -euo pipefail

# mise (uv only — a full `mise install` would put the pinned Python ahead of
# the version under test)
eval "$(mise activate bash)"
mise install aqua:astral-sh/uv

# CI (compatibility.yml) runs this once per supported Python version — the
# requires-python floor upward — by setting UV_PYTHON.
# Locally: UV_PYTHON=3.12 ./compatibility.sh
export UV_PROJECT_ENVIRONMENT=".venv-compat/${UV_PYTHON:-default}"

uv sync
# Tests import all of src, but nothing imports the example agent, so
# byte-compile it to catch syntax incompatibilities there.
uv run --no-sync python -m compileall -q examples
uv run --no-sync pytest

# The same suite again, with every declared dependency dropped to the floor of
# its range, so the `>=` in pyproject.toml stays a tested claim. The floors are
# read from the manifest rather than repeated here, and read with a requirement
# parser rather than by hand, so extras and markers cannot turn into a package
# name that does not exist. `langchain-core` — which carries no floor of its
# own — stays at the version LangGraph pulls in.
mapfile -t floors < <(
  uv run --no-sync python - <<'PY'
import tomllib

from packaging.requirements import Requirement
from packaging.version import Version

with open("pyproject.toml", "rb") as f:
    for dep in tomllib.load(f)["project"]["dependencies"]:
        req = Requirement(dep)
        floors = [s.version for s in req.specifier if s.operator == ">="]
        if floors:
            print(f"{req.name}=={max(floors, key=Version)}")
PY
)
# --no-deps swaps the named packages alone. The dev group pins a current
# LangGraph for the example agent, so a resolver would honor that pin and leave
# the floor untested. The example is out of scope here for the same reason.
uv pip install --python "$UV_PROJECT_ENVIRONMENT" --no-deps "${floors[@]}"

# What is actually installed, not what was asked for: a range read wrong, or a
# `uv run` that syncs instead of honoring --no-sync, would put the pinned
# version back and leave the floor untested while the run stays green.
uv run --no-sync python - "${floors[@]}" <<'PY'
import sys
from importlib.metadata import version

for floor in sys.argv[1:]:
    name, _, want = floor.partition("==")
    got = version(name)
    if got != want:
        raise SystemExit(f"compatibility.sh: expected {floor}, found {got}")
    print(f"compatibility.sh: testing against {name}=={got}")
PY

uv run --no-sync pytest

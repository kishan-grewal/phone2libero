# phone2libero. `just` lists the recipes. Times are for one RTX 4090.

export MUJOCO_GL := "egl"

default:
    @just --list --unsorted

test:
    uv run pytest -q

lint:
    uv run ruff check . && uv run ruff format --check .

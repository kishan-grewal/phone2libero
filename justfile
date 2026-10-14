# phone2libero. `just` lists the recipes. Times are for one RTX 4090.

export MUJOCO_GL := "egl"

default:
    @just --list --unsorted

# Retarget one recorded take onto one task and init state; video + plot in outputs/replay/ (~20 s, no model)
replay-take take="G2_gap12/01" task="0" init="10":
    uv run python -m phone2libero.replay --take {{take}} --task {{task}} --init {{init}}

test:
    uv run pytest -q

lint:
    uv run ruff check . && uv run ruff format --check .

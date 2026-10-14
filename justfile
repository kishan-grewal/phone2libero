# phone2libero. `just` lists the recipes. Times are for one RTX 4090.

export MUJOCO_GL := "egl"

default:
    @just --list --unsorted

# Retarget one recorded take onto one task and init state; video + plot in outputs/replay/ (~20 s, no model)
replay-take take="G2_gap12/01" task="0" init="10":
    uv run python -m phone2libero.replay --take {{take}} --task {{task}} --init {{init}}

# The 5-shot LIBERO demos (from VGAS, arXiv 2602.07399) -> data/5shot
download-5shot:
    uv run hf download SemyonXu616/HF_LIBERO_5_SHOT --repo-type dataset --include "libero_spatial/*" --local-dir data/5shot

# Takes -> replays on the table-level tasks x init states 10-49 -> passing replays in data/phone_table
retarget: download-5shot
    uv run python -m phone2libero.grid run --tasks 0 1 2 6 8 --out outputs/grid/table
    uv run python -m phone2libero.dataset build --results outputs/grid/table/replays.jsonl --root data/phone_table --overwrite

test:
    uv run pytest -q

lint:
    uv run ruff check . && uv run ruff format --check .

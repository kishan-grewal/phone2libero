# phone2libero. `just` lists the recipes. Times are for one RTX 4090.

export MUJOCO_GL := "egl"

model := "kishan-grewal/phone2libero-smolvla"

default:
    @just --list --unsorted

# Watch the trained policy in LIBERO: one task, 10 episodes (~1 min); videos in outputs/evals/demo_task<task>/
demo task="3":
    uv run python -m phone2libero.evaluate run --policy {{model}} --name demo_task{{task}} --seeds 0 --task-ids {{task}}

# Retarget one recorded take onto one task and init state; video + plot in outputs/replay/ (~20 s, no model)
replay-take take="G2_gap12/01" task="0" init="10":
    uv run python -m phone2libero.replay --take {{take}} --task {{task}} --init {{init}}

# The 5-shot LIBERO demos (from VGAS, arXiv 2602.07399) -> data/5shot
download-5shot:
    uv run hf download SemyonXu616/HF_LIBERO_5_SHOT --repo-type dataset --include "libero_spatial/*" --local-dir data/5shot

# The phone recordings (2.6 GB) -> data/raw, and MediaPipe's hand model -> data/models
download-videos:
    uv run hf download kishan-grewal/phone2libero-recordings --repo-type dataset --local-dir data/raw
    mkdir -p data/models
    curl -sSL -o data/models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

# Every dataset this project made (phone demos, self-play rounds; 5.6 GB) -> data/demos
download-demos:
    uv run hf download kishan-grewal/phone2libero-demos --repo-type dataset --local-dir data/demos

# Recordings -> calibrated hand and bowl tracks, cut into takes -> data/takes (own env in capture/; ~40 min)
capture:
    uv run --project capture python -m phone_capture.session

# Takes -> replays on every task x init state 10-49 -> passing replays as LeRobot data in data/phone_* (~2.5 h)
retarget: download-5shot
    uv run python -m phone2libero.grid run --tasks 0 1 2 6 8 --out outputs/grid/table
    uv run python -m phone2libero.grid pick-side --tasks 3 5 7
    uv run python -m phone2libero.grid run --tasks 3 5 7 --out outputs/grid/raised
    uv run python -m phone2libero.grid run --tasks 4 --out outputs/grid/drawer_cabinet
    uv run python -m phone2libero.grid run --tasks 9 --sides 1 -1 --out outputs/grid/drawer_cabinet
    uv run python -m phone2libero.dataset build --results outputs/grid/table/replays.jsonl --root data/phone_table --overwrite
    uv run python -m phone2libero.dataset build --results outputs/grid/raised/replays.jsonl --root data/phone_raised --overwrite
    uv run python -m phone2libero.dataset build --results outputs/grid/drawer_cabinet/replays.jsonl --root data/phone_drawer_cabinet --spare 10 --overwrite

# Record a policy's own successes on init states 10-49, 20 per task (~30 min)
selfplay policy out tasks="0 1 2 3 4 5 6 7 8 9":
    uv run python -m phone2libero.selfplay --policy {{policy}} --out {{out}} --tasks {{tasks}} --overwrite

# Train SmolVLA on the 5-shot set plus the given datasets, 40k steps (~2 h 45 min) -> outputs/train/<name>
train name +datasets: download-5shot
    uv run python -m phone2libero.dataset merge --out data/train_{{name}} {{datasets}}
    uv run lerobot-train --config_path=configs/train_smolvla.json --dataset.repo_id=local/train_{{name}} \
        --dataset.root=data/train_{{name}} --output_dir=outputs/train/{{name}} --job_name={{name}}

# Evaluate a policy on all 10 tasks over 10 seeds, 1000 episodes (~35 min) -> outputs/evals/<name>
eval policy name:
    uv run python -m phone2libero.evaluate run --policy {{policy}} --name {{name}}

# Success rates with 95% intervals, and paired differences against the first name
compare ref +names:
    uv run python -m phone2libero.evaluate compare {{ref}} {{names}} --ref {{ref}}

test:
    uv run pytest -q

lint:
    uv run ruff check . && uv run ruff format --check .

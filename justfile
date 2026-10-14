# phone2libero. `just` lists the recipes. Times are for one RTX 4090.

export MUJOCO_GL := "egl"

default:
    @just --list --unsorted

# The phone recordings (3.3 GB) -> data/raw, and MediaPipe's hand model -> data/models
download-videos:
    uv run hf download kishan-grewal/phone2libero-recordings --repo-type dataset --local-dir data/raw
    mkdir -p data/models
    curl -sSL -o data/models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

# Recordings -> calibrated hand and bowl tracks, cut into takes -> data/takes (own env in capture/; ~40 min)
capture:
    uv run --project capture python -m phone_capture.session

test:
    uv run pytest -q

lint:
    uv run ruff check . && uv run ruff format --check .

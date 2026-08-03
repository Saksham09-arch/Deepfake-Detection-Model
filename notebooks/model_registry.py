"""
model_registry.py

Checkpoints do NOT live in git. As you retrain and models grow, this
is what keeps the repo (and any Docker image) small forever: weights
live in a Hugging Face Hub model repo, versioned independently, and
get pulled down into SAVED_MODELS_DIR at startup -- once per machine,
cached after that.

To publish a newly retrained checkpoint:
    huggingface-cli upload <your-hf-username>/deepfake-detection-models \
        saved_models/efficientnet_best.pth efficientnet_best.pth

To roll back to a previous version: change REVISION for that entry
below to an earlier commit SHA from your HF repo's commit history, or
tag good checkpoints (e.g. "v1", "v2") and pin the tag name instead.

Usage (call this ONCE at app startup, before building the pipeline):
    from model_registry import ensure_models
    ensure_models()  # downloads whatever's missing, skips what's already there
"""

import os
from pathlib import Path
from config import SAVED_MODELS_DIR


HF_MODEL_REPO = os.getenv("HF_MODEL_REPO", "vipulbhattt/deepfake-detection-models")


MODEL_MANIFEST = {
    "efficientnet_best.pth":            {"revision": "main"},
    "efficientnet_finetuned.pth":       {"revision": "main"},
    "efficientnet_faceswap_best.pth":   {"revision": "main"},
    "asvspoof_efficientnet_best.pth":   {"revision": "main"},
}


def ensure_models(needed: list = None, hf_token: str = None) -> dict:
    """
    Downloads any checkpoint in `needed` (defaults to everything in
    MODEL_MANIFEST) that isn't already sitting in SAVED_MODELS_DIR.
    Returns {filename: local_path_str} for everything that ended up
    available (downloaded or already present); a checkpoint that fails
    to download is simply omitted, so the caller can decide whether
    that branch (e.g. audio) should just be skipped for this run.
    """
    from huggingface_hub import hf_hub_download

    hf_token = hf_token or os.getenv("HF_TOKEN")
    targets = needed or list(MODEL_MANIFEST.keys())
    available = {}

    for filename in targets:
        local_path = SAVED_MODELS_DIR / filename
        if local_path.exists():
            print(f"[model_registry] {filename} already present, skipping download.")
            available[filename] = str(local_path)
            continue

        if filename not in MODEL_MANIFEST:
            print(f"[model_registry] WARNING: {filename} not in MODEL_MANIFEST, can't fetch.")
            continue

        revision = MODEL_MANIFEST[filename]["revision"]
        print(f"[model_registry] Downloading {filename} from {HF_MODEL_REPO}@{revision} ...")
        try:
            cached_path = hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename=filename,
                revision=revision,
                token=hf_token,
            )
            import shutil
            shutil.copy2(cached_path, local_path)
            available[filename] = str(local_path)
            print(f"[model_registry] {filename} ready at {local_path}")
        except Exception as e:
            print(f"[model_registry] FAILED to download {filename}: {e}")

    return available


if __name__ == "__main__":
    result = ensure_models()
    print("Available checkpoints:", list(result.keys()))

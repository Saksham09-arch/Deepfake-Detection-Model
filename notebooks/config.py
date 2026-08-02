"""
config.py
Every notebook imports from here instead of hardcoding an absolute path 

How it works: BASE_DIR is computed from THIS file's own location on
disk. As long as config.py sits at the repo root next to your
notebooks, BASE_DIR is always correct.

Usage in any notebook:
    from config import BASE_DIR, DATASETS_DIR, SAVED_MODELS_DIR, OUTPUTS_DIR, REPORTS_DIR
    checkpoint_path = str(SAVED_MODELS_DIR / "efficientnet_best.pth")

No .env edits, no per-machine setup required for paths. 
"""

import os
from pathlib import Path
 
try:
    # config.py -> notebooks/ -> repo root (two levels up, not one)
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    # Fallback for exec()'d contexts where __file__ isn't defined
    # (shouldn't normally happen for a notebook `from config import ...`)
    BASE_DIR = Path(os.getcwd()).parent
 
DATASETS_DIR = BASE_DIR / "datasets"
DATASETS_FACESWAP_DIR = BASE_DIR / "datasets_faceswap"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
OUTPUTS_DIR = BASE_DIR / "outputs"
REPORTS_DIR = BASE_DIR / "reports"
 
# ASVspoof audio-fake dataset lives nested under datasets/ (raw .flac
# extraction + rendered spectrogram PNGs + the source zip), not as a
# separate top-level sibling like datasets_faceswap/.
ASVSPOOF_ZIP_PATH = DATASETS_DIR / "asvpoof-2019-dataset.zip"
ASVSPOOF_EXTRACTED_DIR = DATASETS_DIR / "asvspoof_extracted"
ASVSPOOF_SPECTROGRAMS_DIR = DATASETS_DIR / "spectrograms_asvspoof"
 
# Auto-create the expected folder skeleton so a fresh clone doesn't
# immediately throw FileNotFoundError on the first os.makedirs-less cell.
for _d in (DATASETS_DIR, DATASETS_FACESWAP_DIR, SAVED_MODELS_DIR, OUTPUTS_DIR, REPORTS_DIR,
           ASVSPOOF_EXTRACTED_DIR, ASVSPOOF_SPECTROGRAMS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
 
# Sanity check: if BASE_DIR ends up being the notebooks/ folder itself
# (e.g. this file got moved to the repo root by mistake), the folders
# above just got created in the wrong place. Warn loudly instead of
# silently creating a second, empty, parallel datasets/ tree.
if BASE_DIR.name == "notebooks":
    print(
        f"WARNING: BASE_DIR resolved to {BASE_DIR}, which is a 'notebooks' "
        "folder. Expected BASE_DIR to be its PARENT (the repo root). "
        "Check config.py's location -- it should sit inside notebooks/, "
        "not at the repo root, given this project's folder layout."
    )
 
if __name__ == "__main__":
    print("BASE_DIR:", BASE_DIR)
    print("DATASETS_DIR:", DATASETS_DIR)
    print("SAVED_MODELS_DIR:", SAVED_MODELS_DIR)
    print("OUTPUTS_DIR:", OUTPUTS_DIR)
    print("REPORTS_DIR:", REPORTS_DIR)
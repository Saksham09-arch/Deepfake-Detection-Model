"""
audio_deepfake_module.py

Matches the model actually trained in 08_voice_detection.ipynb:
EfficientNet-B0 fine-tuned on RENDERED mel-spectrogram PNG images
(via librosa + matplotlib), not a raw-waveform CNN. Classes are
['bonafide', 'spoof'] (ImageFolder alphabetical order), where
bonafide = genuine speech, spoof = synthetic/TTS/voice-converted.

This replaces an earlier draft of this file that assumed a different,
smaller architecture and 'real'/'fake' class names -- that draft was
written before the actual ASVspoof training notebook existed and does
NOT match the real checkpoint. If you have an old audio_cnn_best.pth
from that draft lying around, it is NOT compatible with this module
and won't load -- only asvspoof_efficientnet_best.pth (from
08_voice_detection.ipynb) works here.
"""

import os
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")  # no GUI backend needed, just rendering to file
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

IMG_SIZE = 224
SAMPLE_RATE = 16000
N_MELS = 128

AUDIO_CLASS_NAMES = ["bonafide", "spoof"]  # matches ImageFolder alphabetical order from training

AUDIO_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------
# Step 1: pull audio out of a video (for the Video tab -- unrelated to
# the ASVspoof training data itself, this is just how we get a .wav to
# classify from an uploaded video file)
# ---------------------------------------------------------------------

def extract_audio_from_video(video_path: str, out_wav_path: str, sr: int = SAMPLE_RATE):
    """Requires moviepy (pip install moviepy)."""
    from moviepy import VideoFileClip
    clip = VideoFileClip(video_path)
    if clip.audio is None:
        raise ValueError(f"No audio track found in {video_path}")
    clip.audio.write_audiofile(out_wav_path, fps=sr, logger=None)
    clip.close()
    return out_wav_path


# ---------------------------------------------------------------------
# Step 2: audio -> spectrogram PNG (identical logic to
# audio_to_spectrogram_image_v3 in 08_voice_detection.ipynb -- kept
# consistent on purpose, since inference preprocessing must match
# training preprocessing)
# ---------------------------------------------------------------------

def audio_to_spectrogram_image(audio_path: str, output_path: str, sr: int = SAMPLE_RATE, n_mels: int = N_MELS) -> bool:
    try:
        y, _ = librosa.load(audio_path, sr=sr)
    except Exception as e:
        print(f"Failed to load {audio_path}: {e}")
        return False
    if len(y) == 0:
        return False

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
    ax.axis('off')
    librosa.display.specshow(mel_spec_db, sr=sr, ax=ax, cmap='magma')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------
# Step 3: model
# ---------------------------------------------------------------------

def build_asv_model(pretrained: bool = False) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(AUDIO_CLASS_NAMES))
    return model


def load_audio_model(checkpoint_path: str, device: str = None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_asv_model(pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()
    return model, device


# ---------------------------------------------------------------------
# Step 4: inference
# ---------------------------------------------------------------------

def predict_audio(model, device, audio_path: str, tmp_spectrogram_path: str = None) -> dict:
    """
    audio_path: a .wav or .flac file to classify.
    tmp_spectrogram_path: where to render the intermediate spectrogram
    PNG. Defaults to a sibling temp file if not given; caller is
    responsible for cleanup if a fixed path is passed in and reused.
    """
    cleanup_spectrogram = tmp_spectrogram_path is None
    if tmp_spectrogram_path is None:
        tmp_spectrogram_path = audio_path + "_spec_tmp.png"

    ok = audio_to_spectrogram_image(audio_path, tmp_spectrogram_path)
    if not ok:
        raise ValueError(f"Could not render spectrogram for {audio_path} (empty or unreadable audio)")

    try:
        img = Image.open(tmp_spectrogram_path).convert("RGB")
        tensor = AUDIO_TRANSFORM(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = int(probs.argmax())
        return {
            "prediction": AUDIO_CLASS_NAMES[pred_idx],  # "bonafide" or "spoof"
            "confidence": float(probs[pred_idx]) * 100,
            "bonafide_probability": float(probs[0]),
            "spoof_probability": float(probs[1]),
        }
    finally:
        if cleanup_spectrogram and os.path.exists(tmp_spectrogram_path):
            os.remove(tmp_spectrogram_path)


def predict_audio_from_video(model, device, video_path: str, tmp_wav: str = "temp_audio_pipeline.wav") -> dict:
    """Convenience wrapper for the Video tab: pulls audio out of the
    uploaded video, renders a spectrogram, classifies it, cleans up
    both temp files."""
    extract_audio_from_video(video_path, tmp_wav)
    try:
        return predict_audio(model, device, tmp_wav)
    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)

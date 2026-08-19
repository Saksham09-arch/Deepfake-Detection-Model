"""
pipeline_core.py

Shared module referenced in 05_video_inference.ipynb's own setup note:
"the clean way to share logic across notebooks is... converting shared
logic into a proper .py module in your models/ folder (which we'll
actually do when we build the master notebook)". This is that module.

Place this file in your repo root (same folder your notebooks import
from) alongside your notebooks. 09_master_pipeline.ipynb imports it
directly with `from pipeline_core import DeepfakeDetectionPipeline`.

Consolidates logic that used to be duplicated across 04_inference.ipynb,
05_video_inference.ipynb, and the old gradio-app notebook (since removed
in favor of this module + 09_master_pipeline.ipynb):
  - face detection + crop (crop_face_from_frame)
  - preprocessing transform
  - model loading
  - image / video / webcam prediction
  - explanation generation

Adds on top of what you already have:
  - 3-class face-swap-aware prediction (real / faceswap / other_fake)
  - optional audio-fake branch for video
  - Grad-CAM region localization
  - grounded explanations (concrete signals -> LLM, not just
    prediction+confidence -> LLM)
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

# ---------------------------------------------------------------------
# NOTE on paths: all paths now resolve through config.py (BASE_DIR,
# SAVED_MODELS_DIR, etc.) instead of being hardcoded per-notebook. This
# module takes paths as constructor arguments for the same reason --
# so a mismatched or stale checkpoint path can't silently slip in.
#
# This module takes paths as constructor arguments instead of hardcoding
# them, specifically to prevent that class of bug going forward.
# ---------------------------------------------------------------------

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Class orders — MUST match your ImageFolder alphabetical ordering.
BINARY_CLASS_NAMES = ["fake", "real"]  # matches your existing efficientnet_best.pth
# For the new 3-class face-swap-aware model, ImageFolder will alphabetize
# ["faceswap", "other_fake", "real"] regardless of the order you created
# the folders in -- verify this against your own train_data.classes
# printout before trusting it blindly.
FACESWAP_CLASS_NAMES = ["faceswap", "other_fake", "real"]


# ---------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------

def build_efficientnet(num_classes: int, pretrained: bool = False):
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    return model


# ---------------------------------------------------------------------
# Grad-CAM (see gradcam_module.py for the standalone version /
# explanation of why this is here -- same logic, folded into the
# pipeline class so the master notebook only needs one import)
# ---------------------------------------------------------------------

class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer or model.features[-1]
        self.activations = None
        self.gradients = None
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        input_tensor = input_tensor.clone().requires_grad_(True)
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        self.model.zero_grad()
        logits[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx, probs.detach().cpu().numpy()[0]


REGION_GRID_LABELS = [
    ["forehead / hairline", "forehead / hairline", "forehead / hairline"],
    ["left eye / temple", "eyes / nose bridge", "right eye / temple"],
    ["jaw / cheek (left)", "mouth / chin", "jaw / cheek (right)"],
]


def heatmap_to_regions(heatmap: np.ndarray, top_k=2):
    h, w = heatmap.shape
    ch, cw = h // 3, w // 3
    scores = {}
    for i in range(3):
        for j in range(3):
            cell = heatmap[i * ch:(i + 1) * ch, j * cw:(j + 1) * cw]
            label = REGION_GRID_LABELS[i][j]
            scores[label] = scores.get(label, 0.0) + float(cell.mean())
    total = sum(scores.values()) + 1e-8
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(label, s / total) for label, s in ranked[:top_k]]


# ---------------------------------------------------------------------
# Grounded explanation (matches your existing InferenceClient pattern
# from 04_inference.ipynb -- same client, same
# deepseek-ai model, just a prompt built from real signals instead of
# only prediction+confidence)
# ---------------------------------------------------------------------

def confidence_margin(probs_dict: dict) -> float:
    vals = sorted(probs_dict.values(), reverse=True)
    return vals[0] - vals[1] if len(vals) > 1 else vals[0]


def build_grounded_prompt(signals: dict) -> str:
    lines = [f"Detection result: {signals['prediction'].upper()}",
              f"Confidence: {signals['confidence']:.2f}%"]

    if "probabilities" in signals:
        for cls, p in signals["probabilities"].items():
            lines.append(f"{cls} probability: {p*100:.2f}%")
        margin = confidence_margin(signals["probabilities"])
        lines.append(
            f"Confidence margin between top two classes: {margin*100:.1f} points "
            f"({'a decisive call' if margin >= 0.15 else 'a borderline call'})."
        )

    if "top_regions" in signals:
        region_str = ", ".join(f"{name} ({w*100:.0f}% of activation)" for name, w in signals["top_regions"])
        lines.append(f"Grad-CAM attention concentrated on: {region_str}.")

    if "frame_agreement" in signals:
        fa = signals["frame_agreement"]
        lines.append(f"Per-frame agreement across sampled video frames: {fa*100:.0f}%.")

    if "audio_prediction" in signals:
        lines.append(f"Audio track separately classified as: {signals['audio_prediction']}.")
        if signals.get("audio_visual_agree") is False:
            lines.append("Audio and visual verdicts DISAGREE -- flag as lower overall confidence.")

    facts_block = "\n".join(lines)

    return f"""Detection result summary:
{facts_block}

Write a short, 3-4 sentence explanation for the user about this deepfake
detection result. Use ONLY the facts above -- do not invent additional
pixel-level details you weren't given. Mention the confidence margin and
Grad-CAM region if provided, since those explain WHY the model reached
this verdict, not just WHAT it concluded. Be honest this is a statistical
prediction, not certainty."""


class ExplanationClient:
    """Thin wrapper matching your existing HF InferenceClient usage."""

    def __init__(self, hf_token: str = None, model: str = "deepseek-ai/DeepSeek-V3-0324"):
        from huggingface_hub import InferenceClient
        self.client = InferenceClient(api_key=hf_token, provider="auto", timeout=15)
        self.model = model

    def generate(self, signals: dict) -> str:
        prompt = build_grounded_prompt(signals)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=180,
            )
            return response.choices[0].message.content
        except Exception as e:
            print("EXPLANATION ERROR:", type(e).__name__, str(e))
            return self._fallback(signals)

    @staticmethod
    def _fallback(signals: dict) -> str:
        parts = [f"The model predicted '{signals['prediction']}' with {signals['confidence']:.1f}% confidence."]
        if "probabilities" in signals:
            margin = confidence_margin(signals["probabilities"])
            parts.append("This was a decisive call." if margin >= 0.15 else "This was a borderline call.")
        if signals.get("top_regions"):
            parts.append(f"The strongest signal came from the {signals['top_regions'][0][0]} area.")
        if signals.get("audio_visual_agree") is False:
            parts.append("Audio and visual verdicts disagreed, which lowers overall confidence.")
        parts.append("This is a statistical estimate, not proof.")
        return " ".join(parts)


# ---------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------

class DeepfakeDetectionPipeline:
    def __init__(self, face_checkpoint: str, face_class_names: list,
                 audio_checkpoint: str = None, hf_token: str = None,
                 device: str = None):
        """
        face_checkpoint: path to either efficientnet_best.pth (binary) or
                          efficientnet_faceswap_best.pth (3-class) --
                          pass the matching face_class_names list.
        face_class_names: BINARY_CLASS_NAMES or FACESWAP_CLASS_NAMES.
        """
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.class_names = face_class_names

        self.face_model = build_efficientnet(num_classes=len(face_class_names), pretrained=False)
        self.face_model.load_state_dict(torch.load(face_checkpoint, map_location=self.device))
        self.face_model.to(self.device).eval()

        self.audio_model = None
        if audio_checkpoint and os.path.exists(audio_checkpoint):
            from audio_deepfake_module import load_audio_model
            self.audio_model, self.audio_device = load_audio_model(audio_checkpoint)

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        self.explainer = ExplanationClient(hf_token=hf_token) if hf_token else None

    # --- shared preprocessing (single source of truth, fixes the
    # duplication across 04/05/06) ---
    def crop_face_from_frame(self, bgr_frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) > 0:
            faces_sorted = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w, h = faces_sorted[0]
            return bgr_frame[y:y + h, x:x + w]
        return bgr_frame

    def _to_tensor(self, bgr_frame: np.ndarray) -> torch.Tensor:
        face_crop = self.crop_face_from_frame(bgr_frame)
        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        return self.transform(pil_img).unsqueeze(0)

    def _predict_tensor(self, tensor: torch.Tensor) -> dict:
        tensor = tensor.to(self.device)
        with torch.no_grad():
            logits = self.face_model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(probs.argmax())
        return {
            "prediction": self.class_names[pred_idx],
            "confidence": float(probs[pred_idx]) * 100,
            "probabilities": {c: float(p) for c, p in zip(self.class_names, probs)},
        }

    def _gradcam_regions(self, tensor: torch.Tensor) -> list:
        cam_engine = GradCAM(self.face_model)
        heatmap, _, _ = cam_engine.generate(tensor.to(self.device))
        return heatmap_to_regions(heatmap)

    # --- entry point: image ---
    def analyze_image(self, image_path: str, explain: bool = True) -> dict:
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Could not read image"}
        tensor = self._to_tensor(img)
        result = self._predict_tensor(tensor)
        result["top_regions"] = self._gradcam_regions(tensor)
        if explain and self.explainer:
            result["explanation"] = self.explainer.generate(result)
        return result

    # --- entry point: video ---
    def analyze_video(self, video_path: str, num_frames: int = 5, explain: bool = True) -> dict:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return {"error": "Could not read video"}

        frame_indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
        per_frame_preds, all_probs = [], []
        last_tensor = None

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = cap.read()
            if not success:
                continue
            tensor = self._to_tensor(frame)
            last_tensor = tensor
            frame_result = self._predict_tensor(tensor)
            per_frame_preds.append(frame_result["prediction"])
            all_probs.append(list(frame_result["probabilities"].values()))
        cap.release()

        if not all_probs:
            return {"error": "No frames could be processed"}

        avg_probs = np.mean(all_probs, axis=0)
        pred_idx = int(np.argmax(avg_probs))
        result = {
            "prediction": self.class_names[pred_idx],
            "confidence": float(avg_probs[pred_idx]) * 100,
            "probabilities": {c: float(p) for c, p in zip(self.class_names, avg_probs)},
            "majority_vote": max(set(per_frame_preds), key=per_frame_preds.count),
            "per_frame_predictions": per_frame_preds,
            "frame_agreement": per_frame_preds.count(max(set(per_frame_preds), key=per_frame_preds.count)) / len(per_frame_preds),
        }

        if last_tensor is not None:
            result["top_regions"] = self._gradcam_regions(last_tensor)

        if self.audio_model is not None:
            from audio_deepfake_module import predict_audio, extract_audio_from_video
            try:
                tmp_wav = "temp_audio_pipeline.wav"
                extract_audio_from_video(video_path, tmp_wav)
                audio_result = predict_audio(self.audio_model, self.audio_device, tmp_wav)
                os.remove(tmp_wav)
                result["audio_prediction"] = audio_result["prediction"]
                visual_is_fake = result["prediction"] != "real"
                audio_is_fake = audio_result["prediction"] == "spoof"
                result["audio_visual_agree"] = (visual_is_fake == audio_is_fake)
                result["overall_verdict"] = "LIKELY MANIPULATED" if (visual_is_fake or audio_is_fake) else "LIKELY AUTHENTIC"
            except ValueError:
                pass

        if explain and self.explainer:
            result["explanation"] = self.explainer.generate(result)
        return result

    # --- entry point: single webcam frame ---
    def analyze_frame(self, bgr_frame: np.ndarray, explain: bool = False) -> dict:
        tensor = self._to_tensor(bgr_frame)
        result = self._predict_tensor(tensor)
        if explain and self.explainer:
            result["top_regions"] = self._gradcam_regions(tensor)
            result["explanation"] = self.explainer.generate(result)
        return result

# --- entry point: standalone audio file ---
    def analyze_audio(self, audio_path: str, explain: bool = True) -> dict:
        if self.audio_model is None:
            return {"error": "Audio model not loaded. Check ASVSPOOF checkpoint path."}

        from audio_deepfake_module import predict_audio
        audio_result = predict_audio(self.audio_model, self.audio_device, audio_path)

        result = {
            "prediction": "real" if audio_result["prediction"] == "bonafide" else "fake",
            "confidence": audio_result.get("confidence", 0.0),
            "audio_prediction": audio_result["prediction"],
        }
        if "probabilities" in audio_result:
            result["probabilities"] = audio_result["probabilities"]

        if explain and self.explainer:
            result["explanation"] = self.explainer.generate(result)
        return result
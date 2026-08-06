"""
app.py

Render entrypoint. Start command: python app.py

What this does that the notebook version didn't:
  1. Pulls checkpoints from Hugging Face Hub via model_registry.py
     instead of assuming they're already on disk (they won't be --
     saved_models/ is gitignored on purpose, see model_registry.py).
  2. Binds Gradio to 0.0.0.0 and Render's dynamic $PORT, instead of
     Gradio's default localhost:7860 (Render's health check and proxy
     won't reach a server only listening on localhost).
  3. Degrades gracefully: if the faceswap or audio checkpoint fails to
     download, the app still starts with whatever's available instead
     of crashing on import.
"""

import os
import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()  # HF_TOKEN, HF_MODEL_REPO, etc. -- set these as Render environment variables in production, not a committed .env

from config import SAVED_MODELS_DIR, OUTPUTS_DIR
from model_registry import ensure_models
from pipeline_core import DeepfakeDetectionPipeline, BINARY_CLASS_NAMES, FACESWAP_CLASS_NAMES

HF_TOKEN = os.getenv("HF_TOKEN")

# --- Fetch whatever checkpoints are available; missing ones are
# skipped (logged), not fatal. ---
downloaded = ensure_models(hf_token=HF_TOKEN)

USE_FACESWAP_MODEL = "efficientnet_faceswap_best.pth" in downloaded

if USE_FACESWAP_MODEL:
    face_checkpoint = str(SAVED_MODELS_DIR / "efficientnet_faceswap_best.pth")
    class_names = FACESWAP_CLASS_NAMES
elif "efficientnet_best.pth" in downloaded:
    face_checkpoint = str(SAVED_MODELS_DIR / "efficientnet_best.pth")
    class_names = BINARY_CLASS_NAMES
else:
    raise RuntimeError(
        "No face model checkpoint available (neither efficientnet_faceswap_best.pth "
        "nor efficientnet_best.pth downloaded). Check HF_MODEL_REPO / HF_TOKEN and "
        "that the model repo actually contains these files."
    )

audio_checkpoint = str(SAVED_MODELS_DIR / "asvspoof_efficientnet_best.pth")
audio_available = "asvspoof_efficientnet_best.pth" in downloaded

pipeline = DeepfakeDetectionPipeline(
    face_checkpoint=face_checkpoint,
    face_class_names=class_names,
    audio_checkpoint=audio_checkpoint if audio_available else None,
    hf_token=HF_TOKEN,
)

print("Pipeline ready. Class names:", pipeline.class_names)
print("Audio branch loaded:", pipeline.audio_model is not None)


# --- Gradio app (same tabs as 10_master_pipeline.ipynb) ---
import gradio as gr


def image_tab_fn(pil_image):
    if pil_image is None:
        return "Please upload an image."
    tmp_path = str(OUTPUTS_DIR / "temp_master_upload.jpg")
    pil_image.save(tmp_path)
    result = pipeline.analyze_image(tmp_path)

    lines = [f"Prediction: {result['prediction'].upper()} ({result['confidence']:.1f}%)", ""]
    for cls, p in result["probabilities"].items():
        lines.append(f"  {cls}: {p*100:.1f}%")
    if result.get("top_regions"):
        lines.append("")
        lines.append("Model attention (Grad-CAM): " + ", ".join(f"{n} ({w*100:.0f}%)" for n, w in result["top_regions"]))
    if "explanation" in result:
        lines.append("")
        lines.append("Explanation:")
        lines.append(result["explanation"])
    return "\n".join(lines)


def video_tab_fn(video_path):
    if video_path is None:
        return "Please upload a video."
    result = pipeline.analyze_video(video_path)
    if "error" in result:
        return result["error"]

    lines = [f"Prediction: {result['prediction'].upper()} ({result['confidence']:.1f}%)",
              f"(Averaged across sampled frames; majority vote: {result['majority_vote']}, "
              f"frame agreement: {result['frame_agreement']*100:.0f}%)", ""]
    for cls, p in result["probabilities"].items():
        lines.append(f"  {cls}: {p*100:.1f}%")
    if result.get("top_regions"):
        lines.append("")
        lines.append("Model attention (Grad-CAM, last sampled frame): " +
                      ", ".join(f"{n} ({w*100:.0f}%)" for n, w in result["top_regions"]))
    if "audio_prediction" in result:
        lines.append("")
        lines.append(f"Audio verdict: {result['audio_prediction'].upper()}")
        lines.append(f"Overall: {result['overall_verdict']}")
    if "explanation" in result:
        lines.append("")
        lines.append("Explanation:")
        lines.append(result["explanation"])
    return "\n".join(lines)


def voice_tab_fn(audio_path):
    if audio_path is None:
        return "Please upload or record audio."
    if pipeline.audio_model is None:
        return "Audio model not available in this deployment."
    result = pipeline.analyze_audio(audio_path)
    if "error" in result:
        return result["error"]

    lines = [f"Prediction: {result['prediction'].upper()} ({result['confidence']:.1f}%)", ""]
    if "probabilities" in result:
        for cls, p in result["probabilities"].items():
            lines.append(f"  {cls}: {p*100:.1f}%")
    if "explanation" in result:
        lines.append("")
        lines.append("Explanation:")
        lines.append(result["explanation"])
    return "\n".join(lines)


def webcam_tab_fn(pil_image):
    if pil_image is None:
        return "No frame captured -- click the camera icon, then Analyze."
    img_array = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    result = pipeline.analyze_frame(img_array, explain=False)
    return f"Prediction: {result['prediction'].upper()} ({result['confidence']:.1f}%)"


with gr.Blocks(title="AI-Powered Deepfake Detection System") as demo:
    gr.Markdown("# AI-Powered Deepfake Detection System")
    gr.Markdown("Detect deepfakes in images, videos, audio, or live webcam feed.")

    with gr.Tab("Image"):
        img_input = gr.Image(type="pil", label="Upload an image")
        img_button = gr.Button("Analyze Image")
        img_output = gr.Textbox(label="Result", lines=12)
        img_button.click(fn=image_tab_fn, inputs=img_input, outputs=img_output)

    with gr.Tab("Video"):
        vid_input = gr.Video(label="Upload a video")
        vid_button = gr.Button("Analyze Video")
        vid_output = gr.Textbox(label="Result", lines=14)
        vid_button.click(fn=video_tab_fn, inputs=vid_input, outputs=vid_output)

    with gr.Tab("Webcam"):
        gr.Markdown("Click the camera icon, then Analyze Snapshot.")
        webcam_input = gr.Image(type="pil", label="Webcam", sources=["webcam"])
        webcam_button = gr.Button("Analyze Snapshot", variant="primary")
        webcam_output = gr.Textbox(label="Result")
        webcam_button.click(fn=webcam_tab_fn, inputs=webcam_input, outputs=webcam_output)

    with gr.Tab("Voice"):
        gr.Markdown("Detects AI-synthesized/cloned speech (ASVspoof 2019-trained).")
        voice_input = gr.Audio(type="filepath", label="Upload or record audio")
        voice_button = gr.Button("Analyze Voice")
        voice_output = gr.Textbox(label="Result", lines=10)
        voice_button.click(fn=voice_tab_fn, inputs=voice_input, outputs=voice_output)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=port)

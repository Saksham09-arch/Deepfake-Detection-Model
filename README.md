# 🎭 AI-Powered Deepfake Detection System

An end-to-end, multi-modal deepfake detection system — detects manipulated faces in images, videos, live webcam feeds, and AI-synthesized voice audio, with Grad-CAM visual explanations and GenAI-generated natural-language reasoning for each prediction.

**🔴 Live demo:** [deepfake-detection-model-t8be.onrender.com](https://deepfake-detection-model-t8be.onrender.com)

Built entirely using **Anaconda + Jupyter Notebook**, trained on **FaceForensics++**, **ASVspoof 2019**, and the **140k Real and Fake Faces** dataset. Powered by **EfficientNet-B0**, deployed on **Render** with models hosted on **Hugging Face Hub**.

---

## 🚀 Features

- **Image detection** — upload a face image, get a Real/Fake verdict with confidence, Grad-CAM heatmap region, and a plain-language explanation
- **Video detection** — samples frames across a video, averages predictions, cross-checks with majority vote, and optionally checks the audio track too
- **Live webcam detection** — real-time face detection and classification
- **Voice deepfake detection** — detects AI-cloned/synthesized speech, trained on ASVspoof 2019 (89% accuracy on synthesis systems never seen during training)
- **Face-swap detection** — a 3-class model distinguishing real / other-manipulation / face-swap specifically
- **GenAI explanations** — natural-language reasoning grounded in real signals (confidence margin, Grad-CAM region, frame agreement), not just a prediction label
- **Cloud deployment** — live on Render, with model checkpoints pulled from Hugging Face Hub at startup (keeping the repo and deploy image small)

---

## 🧠 Models & Accuracy

| Model | Task | Result |
|---|---|---|
| `efficientnet_best.pth` | Real vs Fake (image, FF++ only) | 98% test accuracy, 0.9987 ROC AUC |
| `efficientnet_finetuned.pth` | Real vs Fake (generalization-fixed) | 98% test accuracy; fixed a real-world webcam misclassification bug (67.95% wrong → 100% correct) |
| `efficientnet_faceswap_best.pth` | Real / other-fake / face-swap (3-class) | Trained via gradient accumulation + early stopping (see `07_faceswap_training.ipynb`) |
| `asvspoof_efficientnet_best.pth` | Voice: bonafide vs spoof | 89% accuracy on ASVspoof eval set — synthesis systems unseen during training |

**Note on the voice model:** an earlier attempt trained on the Fake-or-Real dataset achieved 100% validation accuracy but collapsed to ~50% (random chance) on genuinely unseen data — traced to two independent data-leakage artifacts in that dataset (inconsistent audio encoding and loudness normalization between real/fake source files). Switching to ASVspoof 2019, a rigorously-designed academic benchmark, fixed this. See the full manual for the complete diagnosis.

---

## 📁 Project Structure

```
Deepfake-Detection/
├── notebooks/
│   ├── 00_environment_setup.ipynb
│   ├── 01_dataset_download.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_inference.ipynb
│   ├── 05_video_inference.ipynb
│   ├── 06_generalization_improvement.ipynb
│   ├── 07_faceswap_training.ipynb
│   ├── 08_voice_detection.ipynb
│   ├── 09_faceswap_saksham.ipynb
│   ├── 09_gradio_app.ipynb
│   ├── 10_master_pipeline.ipynb
│   ├── app.py                  ← Render deployment entrypoint
│   ├── config.py               ← auto-resolves project paths
│   ├── pipeline_core.py        ← shared detection pipeline class
│   ├── model_registry.py       ← pulls checkpoints from Hugging Face Hub
│   └── requirements.txt        ← CPU-only deps for deployment
├── saved_models/                (git-ignored — pulled from HF Hub or regenerated)
├── datasets/                    (git-ignored — regenerate via notebooks)
├── .env                         (git-ignored — secrets)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🛠️ Local Setup

### 1. Install Anaconda, then:
```bash
conda create -n deepfake_env python=3.11
conda activate deepfake_env
conda install -n deepfake_env jupyter notebook -y
python -m ipykernel install --user --name deepfake_env --display-name "Python (deepfake_env)"
```

### 2. Install dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install numpy pandas opencv-python==4.10.0.84 matplotlib seaborn tqdm scikit-learn
pip install transformers huggingface_hub gradio Pillow moviepy python-dotenv kaggle librosa soundfile
conda install -c conda-forge ffmpeg -y
```

### 3. Set up secrets
Create `.env` in the project root:
```
HF_TOKEN=your_huggingface_token_here
```
Kaggle credentials go in `~/.kaggle/kaggle.json` (from kaggle.com → Settings → API).

### 4. Get the datasets and models
- **Datasets** aren't included in this repo (too large) — run notebooks `01`, `02`, `06`, `07`, `08` in order to regenerate them, or ask a maintainer for a shared link.
- **Trained models** are hosted on Hugging Face Hub and pulled automatically:
```bash
cd notebooks
python model_registry.py
```

### 5. Run the app locally
```bash
python app.py
```
Or open `10_master_pipeline.ipynb` / `09_gradio_app.ipynb` in Jupyter for the notebook version.

---

## ☁️ Deployment

The live app runs on [Render](https://render.com), using `notebooks/app.py` as the entrypoint. On startup it:
1. Pulls whatever model checkpoints are available from a Hugging Face Hub model repo (via `model_registry.py`)
2. Degrades gracefully if a checkpoint is missing (e.g. runs without the face-swap or audio branch rather than crashing)
3. Binds Gradio to `0.0.0.0` and Render's dynamic `$PORT`

To deploy your own instance: fork this repo, create a Render web service pointing at `notebooks/app.py` with `notebooks/requirements.txt`, and set an `HF_TOKEN` environment variable in Render's dashboard.

---

## 🤝 Contributing

1. Fork this repository
2. Create a feature branch (`git checkout -b your-feature`)
3. Commit and push to your fork
4. Open a Pull Request against `main` — it'll be reviewed before merging

---

## ⚠️ Known Limitations

- Trained primarily on FaceForensics++ and ASVspoof 2019 — broader cross-dataset generalization (Celeb-DF, DFDC, other voice benchmarks) is future work
- Face detection uses Haar Cascade (fast, but less robust than deep-learning detectors on extreme angles)
- GenAI explanations are grounded in real signals (confidence margin, Grad-CAM region) but are not themselves a certified forensic tool

## 🔮 Future Work

- Reconcile the two face-swap notebooks into one
- Broader cross-dataset generalization testing
- Expand voice detection beyond ASVspoof

---

*Built with PyTorch, EfficientNet-B0, OpenCV, librosa, Gradio, Hugging Face, and Render.*
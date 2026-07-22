\# 🎭 AI-Powered Deepfake Detection System



An end-to-end deepfake detection system built with Generative AI and Computer Vision — detects manipulated faces in images, videos, and live webcam feeds, with AI-generated natural-language explanations for each prediction.



Built entirely using \*\*Anaconda Prompt + Jupyter Notebook\*\*, trained on \*\*FaceForensics++\*\*, powered by \*\*EfficientNet-B0\*\* transfer learning and a free-tier LLM for explainability.



\---



\## 🚀 Features



\- \*\*Image detection\*\* — upload any face image, get a Real/Fake verdict with confidence score

\- \*\*Video detection\*\* — samples frames across a video and aggregates predictions for a final verdict

\- \*\*Live webcam detection\*\* — real-time face detection and classification via OpenCV, plus a Gradio-based webcam snapshot mode

\- \*\*GenAI explanations\*\* — natural-language reasoning for each prediction, generated via a free LLM (Hugging Face Inference Providers)

\- \*\*Interactive Gradio UI\*\* — tabbed interface for image, video, and webcam input

\- \*\*Generalization fine-tuning\*\* — improved real-world robustness by fine-tuning on diverse everyday-style face images, not just benchmark footage



\---



\## 🧠 Model \& Approach



| Component | Choice | Why |

|---|---|---|

| Architecture | EfficientNet-B0 (transfer learning) | Best accuracy-per-effort tradeoff for a single-GPU beginner-to-intermediate setup |

| Framework | PyTorch (CUDA 12.4) | Strong deepfake-detection research ecosystem, intuitive debugging |

| Face detection | OpenCV Haar Cascade | Fast, dependency-free, built into OpenCV |

| Explainability | Free LLM (DeepSeek-V3 via Hugging Face) | Natural-language reasoning layer on top of raw predictions |



\### Results



| Metric | Score |

|---|---|

| Test Accuracy | 98% |

| Precision / Recall / F1 (both classes) | 0.98 |

| ROC AUC | 0.9987 |



\*\*Generalization fine-tuning:\*\* the original model, trained only on FaceForensics++ footage, misclassified real-world webcam photos (67.95% "fake" on a genuinely real face) — a textbook domain-shift/generalization gap. Fine-tuning on a small sample of the \*\*140k Real and Fake Faces\*\* dataset (everyday-style photos) fixed this (now 100% correct on the same test) \*\*without\*\* losing accuracy on the original FaceForensics++ test set (still 98%).



\---



\## 📁 Project Structure



```

Deepfake-Detection/

├── datasets/

│   ├── train/real, train/fake

│   ├── validation/real, validation/fake

│   └── test/real, test/fake

│   (raw videos, extracted frames, and cropped-face intermediates are git-ignored — see Dataset Setup below)

├── notebooks/

│   ├── 00\_environment\_setup.ipynb

│   ├── 01\_dataset\_download.ipynb

│   ├── 02\_preprocessing.ipynb

│   ├── 03\_model\_training.ipynb

│   ├── 04\_inference.ipynb

│   ├── 05\_video\_inference.ipynb

│   ├── 06\_gradio\_app.ipynb

│   └── 07\_generalization\_improvement.ipynb

├── saved\_models/          (trained weights — git-ignored, see below)

├── outputs/

├── reports/                (evaluation plots, etc.)

├── .env                    (secrets — git-ignored, you create this)

├── .gitignore

└── README.md

```



\---



\## 🛠️ Environment Setup (from scratch)



\### 1. Install Anaconda

Download and install from \[anaconda.com](https://www.anaconda.com/). Open \*\*Anaconda Prompt\*\* afterward.



\### 2. Create the project folder (on your chosen drive)

```bash

D:

mkdir Deepfake-Detection

cd Deepfake-Detection

```



\### 3. Create and activate the conda environment

```bash

conda create -n deepfake\_env python=3.11

conda activate deepfake\_env

```



\### 4. Install Jupyter and register the kernel

```bash

conda install -n deepfake\_env jupyter notebook -y

pip install ipykernel

python -m ipykernel install --user --name deepfake\_env --display-name "Python (deepfake\_env)"

```



\### 5. Install core libraries

```bash

pip install numpy pandas opencv-python==4.10.0.84 matplotlib seaborn tqdm

```



\### 6. Install PyTorch (GPU-enabled, adjust CUDA version to your driver via `nvidia-smi`)

```bash

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

```



\### 7. Install remaining libraries

```bash

pip install transformers huggingface\_hub albumentations gradio streamlit Pillow moviepy scikit-learn python-dotenv kaggle

conda install -c conda-forge ffmpeg -y

```



\### 8. Verify GPU is detected

```python

import torch

print(torch.cuda.is\_available())  # should print True

```



\---



\## 🔑 Secrets Setup



Create a `.env` file in the project root (never committed — see `.gitignore`):

```

HF\_TOKEN=your\_huggingface\_token\_here

```

Get a free token at huggingface.co → Settings → Access Tokens.



Kaggle credentials go in `C:\\Users\\<you>\\.kaggle\\kaggle.json` (downloaded from kaggle.com → Settings → API → Create Legacy API Key) — this lives outside the project folder and is never part of the repo.



\---



\## 📦 Dataset Setup



Dataset used: \*\*FaceForensics++ (c23)\*\*, mirrored on Kaggle (`xdxd003/ff-c23`), plus a supplementary sample from \*\*140k Real and Fake Faces\*\* (`xhlulu/140k-real-and-fake-faces`) for generalization fine-tuning.



Run `notebooks/01\_dataset\_download.ipynb` to download a sample of real + Deepfakes videos via the Kaggle API. Then run `02\_preprocessing.ipynb` to:

1\. Extract 5 evenly-spaced frames per video

2\. Face-crop each frame (OpenCV Haar Cascade, falling back to full frame if no face detected — zero frames lost)

3\. Split into train/validation/test (70/15/15)



Raw videos, extracted frames, and intermediate crops are \*\*not\*\* included in this repo (too large, regenerable) — see `.gitignore`. Re-run notebooks 01–02 to reproduce them locally.



\---



\## 🏋️ Training



Run `notebooks/03\_model\_training.ipynb`:

\- Loads EfficientNet-B0 with ImageNet pretrained weights, replaces the final layer for binary classification

\- Trains with Adam optimizer, saves the best checkpoint by validation accuracy (not just the final epoch, to avoid keeping an overfit model)

\- Best checkpoint saved to `saved\_models/efficientnet\_best.pth` (git-ignored — retrain locally, or contact the repo owner for the weights file)



\### Generalization fine-tuning

Run `notebooks/07\_generalization\_improvement.ipynb` to fine-tune the trained model on a mixed dataset (original FF++ frames + everyday-style real/fake images), producing `saved\_models/efficientnet\_finetuned.pth` — the recommended model for real-world (non-benchmark) input like webcam photos.



\---



\## 📊 Evaluation



Also in `03\_model\_training.ipynb` / re-run against `07`'s fine-tuned checkpoint: classification report, confusion matrix, and ROC curve, saved to `reports/evaluation\_plots.png`.



\---



\## 🔍 Inference



\- \*\*Image:\*\* `04\_inference.ipynb` — single-image prediction with confidence + GenAI explanation

\- \*\*Video:\*\* `05\_video\_inference.ipynb` — samples frames, averages predictions, majority-vote cross-check

\- \*\*Live webcam (OpenCV window):\*\* also in `05\_video\_inference.ipynb` — real-time bounding box + label overlay



\---



\## 🖥️ Gradio Web App



Run `notebooks/06\_gradio\_app.ipynb` to launch the full interactive demo (Image / Video / Webcam tabs, GenAI explanations included). Opens at `http://127.0.0.1:7860`.



\---



\## 🤝 Contributing



This project is open to contributions via the standard fork-and-PR workflow:

1\. Fork this repository

2\. Create a feature branch (`git checkout -b your-feature`)

3\. Commit your changes and push to your fork

4\. Open a Pull Request against `main` — it'll be reviewed before merging



\---



\## ⚠️ Known Limitations



\- Trained primarily on FaceForensics++ (Deepfakes manipulation method) — cross-manipulation and cross-dataset generalization is an active area of improvement (see `07\_generalization\_improvement.ipynb`)

\- Face detection uses Haar Cascade (fast, but less accurate than deep-learning face detectors on extreme angles/occlusion)

\- GenAI explanations describe confidence levels in natural language but are not grounded in pixel-level model interpretability (that would require Grad-CAM or similar, planned as a future addition)



\## 🔮 Future Work

\- Grad-CAM heatmap visualization

\- Face-swap and voice deepfake detection

\- Cross-dataset generalization testing (Celeb-DF, DFDC)

\- Cloud deployment (Google Cloud Run)


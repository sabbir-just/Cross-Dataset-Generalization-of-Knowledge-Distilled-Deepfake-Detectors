# Cross-Dataset Generalization of Knowledge-Distilled Deepfake Detectors

Code for the paper **"Cross-Dataset Generalization of Knowledge-Distilled Deepfake Detectors"**.

We distil a dual-branch Vision Transformer teacher (**DP-ViT**: an RGB branch plus a high-frequency/Laplacian branch) into compact **EfficientNet** students, one per source dataset, and then evaluate every student on every dataset to build a **4 × 4 cross-domain AUC transfer matrix** across:

| Dataset | Type of manipulation |
|---|---|
| FaceForensics++ (FF++) | Face2Face, Deepfakes, NeuralTextures, FaceShifter, FaceSwap |
| Celeb-DF v2 | High-quality face swaps |
| DFDC | Multi-method face swaps (Facebook challenge) |
| Diffusion | DiffSwap and SDv15_DS0.3 images |

## Repository contents

Each script trains one student on one dataset, with the matching pre-trained DP-ViT teacher kept frozen.

| Script | Dataset | Teacher | Student | KD terms |
|---|---|---|---|---|
| `KD FF Dataset Teacher DPViT Std Effcnt.py` | FF++ frames | DP-ViT (ViT-B/16 x2) | EfficientNet-B0 | soft-label + feature + hard (BCE, label smoothing) |
| `KD Celeb DF Dataset Teacher DPViT Std Effcnt.py` | Celeb-DF frames | DP-ViT (variant inferred from checkpoint) | EfficientNet-B0 | soft-label + feature + hard; **Phase-2 fine-tune** of an existing student checkpoint with warm-restart cosine LR and SWA |
| `DFDC KD.py` | DFDC faces | DP-ViT (ViT-B/16 x2, 1536-d fused features) | EfficientNet-B2 (dual-path RGB + HF, gated fusion) | hard + soft + feature + intermediate + attention + RKD, staged unfreezing |
| `Diffusion KD.py` | Diffusion images | DP-ViT + gated fusion head (~175 M) | EfficientNet-B2 (dual-path, light fusion head, ~16 M) | KL (T = 4, alpha = 0.7) + BCE |

Common settings: 224 x 224 inputs, seed 42, distillation temperature T = 4, AdamW, gradient clipping at 1.0, mixed precision on CUDA.
The high-frequency input is a Laplacian-filtered copy of the RGB frame.

## Setup

```bash
git clone https://github.com/sabbir-just/Cross-Dataset-Generalization-of-Knowledge-Distilled-Deepfake-Detectors.git
cd Cross-Dataset-Generalization-of-Knowledge-Distilled-Deepfake-Detectors
pip install -r requirements.txt
```

Python 3.10+ and a CUDA GPU are recommended. All scripts were developed and run on **Kaggle GPU notebooks**.

## Data

The scripts expect face frames that were pre-extracted from the original videos. Datasets are **not** redistributed here; obtain them from their official sources:

- FaceForensics++: https://github.com/ondyari/FaceForensics
- Celeb-DF v2: https://github.com/yuezunli/celeb-deepdetect
- DFDC: https://ai.meta.com/datasets/dfdc/
- Diffusion set: DiffSwap / SDv15_DS0.3 image folders (see paper for provenance)

Expected layouts:

```text
FF++        <root>/{orginal, Face2Face, Deepfakes, NeuralTextures, FaceShifter, FaceSwap}/<video>/*.jpg
Celeb-DF    <root>/{real, fake}/<video_id>/*.jpg
DFDC        <root>/{real, fake}/*.jpg
Diffusion   <root>/{Real, DiffSwap, SDv15_DS0.3}/*.png|jpg
```

(The FF++ real folder name `orginal` is spelled that way in the script's config.)

## Usage

1. Open the script you want to run and edit the **CONFIGURATION** block at the top. The defaults are Kaggle paths such as `/kaggle/input/...`; point `FRAME_ROOT` / `DATASET_DIR` / `DATASET_FRAMES` / `BASE_PATH`, the teacher checkpoint path, and the output directory to your own locations.
2. Run it:

```bash
python "KD FF Dataset Teacher DPViT Std Effcnt.py"
python "KD Celeb DF Dataset Teacher DPViT Std Effcnt.py"   # needs a Phase-1 student checkpoint (STUDENT_CKPT)
python "DFDC KD.py"
python "Diffusion KD.py"
```

Outputs (checkpoints, metrics, figures) are written to each script's `OUT_DIR` (`/kaggle/working/...` by default).

### Teacher and student checkpoints

The scripts need a trained DP-ViT teacher for each dataset (`TEACHER_PATH` / `TEACHER_CHECKPOINT`). The Celeb-DF script additionally starts from a Phase-1 student (`STUDENT_CKPT`). Download the weights and set these paths before running. Checkpoint links: _add your Kaggle model links here_.

## Evaluation

Scripts report accuracy, F1, ROC-AUC and related metrics on a held-out split of their own dataset. The 4 x 4 transfer matrix is built by evaluating each trained student on the test data of the other three datasets.

## Citation

If you use this code, please cite the paper (see `CITATION.cff`):

```bibtex
@misc{hasan2026crossdatasetkd,
  title  = {Cross-Dataset Generalization of Knowledge-Distilled Deepfake Detectors},
  author = {Hasan, Syed Azmul and Bulbul, Md. Farhad},
  year   = {2026},
  note   = {Code: https://github.com/sabbir-just/Cross-Dataset-Generalization-of-Knowledge-Distilled-Deepfake-Detectors}
}
```

## License

Released under the MIT License. See `LICENSE`.

## Acknowledgements

Department of Mathematics, Jashore University of Science and Technology, Bangladesh. Supervisor: Dr. Md. Farhad Bulbul.

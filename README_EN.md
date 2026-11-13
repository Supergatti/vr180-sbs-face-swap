**English** · [中文](README.md)

<div align="center">

# 🎭 VR180 SBS Face-Swap Workbench

**Swap one face for another in 180° stereoscopic VR video.**

A workbench for face-swapping in 180° ERP side-by-side (SBS) VR video: a VR180-trained YOLOFace locates faces directly on the two eyes' ERP frames, a selected target is cropped into perspective tiles with a fixed framing, handed to FaceFusion for the swap, then inversely remapped back into the SBS output.

[![GitHub stars](https://img.shields.io/github/stars/Supergatti/vr180-sbs-face-swap?style=for-the-badge&logo=github&logoColor=white&labelColor=1c2128&color=2ea44f)](https://github.com/Supergatti/vr180-sbs-face-swap/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Supergatti/vr180-sbs-face-swap?style=for-the-badge&logo=github&logoColor=white&labelColor=1c2128&color=0969da)](https://github.com/Supergatti/vr180-sbs-face-swap/forks)
[![Python](https://img.shields.io/badge/Python-3.10-3776ab?style=for-the-badge&logo=python&logoColor=white&labelColor=1c2128)](https://www.python.org/)
[![Windows](https://img.shields.io/badge/Windows-0078d6?style=for-the-badge&logo=windows&logoColor=white&labelColor=1c2128)](#)

</div>

> **First version with "fixed dual-eye framing"**: the framing angle is determined by the target box on a reference frame and reused for the whole clip. Suited for shots where the target stays roughly inside the frame; faces will be missed when the target moves a lot, turns their head, or the camera moves. Per-frame tracking comes in a later version.

---

## ✨ Features

| Feature | Description |
| --- | --- |
| 🎯 **Native VR180 detection** | A project-specific `yoloface_12n-vr180.onnx` locates faces directly on the ERP spherical frames — no need to unfold into a flat image first |
| 👁️ **Independent per-eye processing** | Left and right eyes are located, framed, swapped, and back-blended independently; the same person uses the same source identity |
| 🖼️ **Perspective export / spherical back-fill** | Crop perspective tiles from the target box → FaceFusion swap → inverse remap back to SBS, preserving stereo layout, frame timing, and audio |
| 🔄 **Face swap + enhancement** | Optional face enhancement (when enabled, the pipeline goes from 3 steps to 4); turn it off for speed |
| 📊 **Live progress** | Step indicator, frame progress, percentage, speed, and live PNG preview; the console logs stage progress in sync |
| 🧑‍🤝‍🧑 **Target a specific person** | Pick the same person in both eyes to swap; manual boxing as a fallback when detection misses |
| 🧪 **Step-by-step test entry** | The `/test` page runs "read → perspective export → swap → back-fill" step by step for debugging |
| 🎚️ **Inference device switch** | One-click switch between CUDA · GPU and CPU |

## Layout

- `workbench/` — the workbench itself (HTTP server + frontend + perspective/swap/back-fill pipeline)
- `facefusion/` — one small patch for FaceFusion (progress/preview reporting, optional)
- `scripts/` — dependency install and model download scripts
- `docs/` — implementation plan and detection notes

## One-click setup (Windows)

Prerequisites: [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (or Anaconda), [Git](https://git-scm.com/), and `ffmpeg` / `ffprobe` on PATH.

Open PowerShell in the repository root and run:

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

The script will, in order:

1. Create (or reuse) the conda environment `facefusion` (Python 3.10);
2. Shallow-clone FaceFusion **3.1.2** into `.facefusion/` and install its dependencies (including onnxruntime-gpu);
3. Auto-download the workbench's dedicated VR180 detection model `yoloface_12n-vr180.onnx` (SHA-256 verified);
4. Apply the progress-reporting patch (optional; failures do not block).

FaceFusion's own swap/enhance models (inswapper, gfpgan, etc.) are **auto-downloaded by FaceFusion on the first swap** — no manual handling needed.

## Start

```powershell
powershell -ExecutionPolicy Bypass -File start.ps1
```

(Or double-click `start.bat` in Windows Explorer.)

Open http://127.0.0.1:8080/ in a browser:

1. Import the VR180 ERP SBS video and the source face (local path or file picker).
2. Pick a frame, detect, then select the same person in both eyes; draw a box manually if detection misses.
3. Review the left/right perspective crops and adjust the framing margin as needed. Defaults to processing the whole video.
4. Start the swap, then download the output video and the annotated PNG.

The step-by-step test entry at http://127.0.0.1:8080/test runs "read → perspective export → swap → back-fill" incrementally.

### Inference device

The "inference device" dropdown offers:

- **CUDA · GPU** — onnxruntime's CUDAExecutionProvider, most stable, default.
- **CPU** — fallback when no GPU is available.

## Manual setup (not recommended)

1. Install FaceFusion 3.1.2 in the same environment (or reuse an existing FaceFusion directory and write its root path to the `FACEFUSION_HOME` environment variable).
2. Copy `facefusion/workbench_progress.py` into FaceFusion's `facefusion/` directory, and apply `facefusion/workbench-progress.patch` to the 4 files (optional; only affects progress reporting).
3. Download `yoloface_12n-vr180.onnx` to `.assets/models/`, or run `python scripts/download-models.py`.

## Notes

- Only targets horizontal SBS with 180° ERP per eye; the browser preview is flat SBS, and headset auto-detection depends on the player and spatial metadata.
- `server.py` auto-switches to the `facefusion` conda environment and restarts when the current interpreter lacks onnxruntime.
- Perspective extraction / inverse back-fill remapping prefers `torch.cuda`, then `cv2.cuda`, then falls back to CPU (still swaps, just slower). Optionally install CUDA-enabled torch to enable GPU remapping.
- Generated files are written under `workbench/.workbench/jobs`; `.assets/`, `.workbench/`, and `.facefusion/` are excluded in `.gitignore`.

---

## ⭐ Star

Like this project? Give it a star:

[![Star this repo](https://img.shields.io/badge/Star-give%20it%20a%20star-2ea44f?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Supergatti/vr180-sbs-face-swap)

**Star history** (starts accumulating once the repository is made public):

[![Star History Chart](https://api.star-history.com/svg?repos=Supergatti/vr180-sbs-face-swap&type=Date)](https://star-history.com/#Supergatti/vr180-sbs-face-swap&Date)

# Project Change Log and Conflict Register

This file records project edits made during the current work session. For each
change, it describes the conflict or problem, the solution, and anything still
unresolved. Append a dated entry here whenever additional project files are
changed during this session.

## 2026-09-30: Containerization

### Conflicts and Problems

- **Base image/runtime mismatch (resolved):** The Dockerfile used end-of-life Python 3.8 on Debian Buster, while the project's environment uses Python 3.10.
- **Dependency version conflict (resolved for direct dependencies):** Requirements were mostly unpinned. The installed environment reports TensorFlow 2.15.0 alongside NumPy 2.2.6, but TensorFlow 2.15 requires NumPy below 2.0.
- **Container networking mismatch (resolved):** The app used loopback binding and port 8080 by default, while the Dockerfile exposed port 80. A container must bind on all interfaces for published ports to reach it.
- **Oversized build context (resolved):** No `.dockerignore` excluded virtual environments, Git/DVC data, datasets, or intermediate training output. At the same time, predictions need the trained model file in the image.

### Solutions Applied

- Replaced the base with `python:3.10-slim-bookworm`; removed the unused `awscli` OS package installation; enabled unbuffered Python output and disabled bytecode files; installed requirements without caching pip downloads.
- Pinned direct dependencies in `requirements.txt` and selected `numpy==1.26.4` for TensorFlow 2.15 compatibility. Added `Pillow==10.1.0` for the image-loading code.
- Changed Flask to bind to `0.0.0.0`; Docker now exposes the app's default port, 8080.
- Added `.dockerignore` exclusions for environments, Git/DVC caches, datasets, and intermediate outputs, with an exception to keep `artifacts/training/vgg16model.h5` in the image context.

### Outstanding Items and Verification

- **Transitive dependency versions (partially resolved):** Direct requirements are pinned, but their transitive dependencies are not fully locked. Builds may still resolve different transitive versions; a generated lock file would be needed for stronger reproducibility.
- **Model delivery (unresolved):** The image build still requires `artifacts/training/vgg16model.h5`. No repository-level DVC remote or tracked DVC pointer was found to fetch it on a clean checkout.
- **Docker image build (not verified):** The Docker CLI could not connect to the Docker Desktop Linux engine because it was not running. Pip's dependency dry run completed, and `webapp.py` passed Python compilation.
- **Existing Flask diagnostics (not changed):** The pre-existing `pass_mod()` code references undefined ONNX/image variables and has trailing whitespace. It is separate from the container changes and remains as-is.

## 2026-09-30: Model and Production Safety Research

### Problems Identified

- The reported 76.19% evaluation accuracy is not a clean held-out result: training uses a 20% validation split, while evaluation creates a 30% split from the same image directory.
- Training rescales pixels to `[0, 1]`, but prediction does not apply the same normalization. The model also uses a two-output sigmoid and binary cross-entropy with directory-generated categorical labels.
- The prepared dataset has 630 images across two labels (380 affected, 250 healthy), which does not establish general clinical performance.
- The Flask app exposes a training route that runs a shell command, writes incoming image data to a shared filename, has no explicit upload limits/validation, and enables broad CORS.

### Research and Recommendations Documented

- Added `MODEL_RESEARCH_AND_PRODUCTION_SAFETY.md` with project findings, model candidates (corrected VGG16, EfficientNetB0, MobileNetV3, ResNet50, ConvNeXtTiny, and dermatology-pretrained embeddings), a fair comparison protocol, dataset/evaluation requirements, production/security safeguards, regulatory caveats, and source links.
- The report recommends fixing splits, preprocessing, and label/loss alignment before comparing models. It explicitly treats candidate architectures as experiments, not guaranteed accuracy improvements, and marks the current project as research-only pending validation.

## 2026-09-30: Production Deployment Implementation Plan

### Requirement Conflicts and Decisions

- **FastAPI/Gunicorn wording:** FastAPI is a valid modern option, but is ASGI; Gunicorn's default worker is WSGI. The guide recommends Uvicorn and one process per container as an initial ECS baseline, then horizontal task scaling. Extra worker processes require measured memory because each may load its own model.
- **Gateway overlap:** Cloudflare, Nginx, and AWS ALB can duplicate responsibilities. The guide recommends one intentional path (Cloudflare optionally in front of ALB) and Nginx only when its specific proxy features are needed.
- **Queue complexity:** The project has not established production latency or load requiring asynchronous inference. The guide makes Celery/Redis conditional on benchmark results, describes SQS as a managed AWS broker alternative, and requires only object references in queue messages.
- **Inference speed claim:** ONNX, INT8, and TensorRT are not guaranteed to improve latency/cost. The guide requires parity, quality guardrails, and target-hardware benchmarks before making speedup claims; TensorRT is positioned for appropriate NVIDIA hardware, not CPU cost reduction.
- **AWS orchestration:** EKS adds Kubernetes operations that are not justified by the current project size. ECS Fargate is the recommended first AWS target; EKS remains an option only when Kubernetes-specific needs exist.

### Solution Documented

- Added `PRODUCTION_DEPLOYMENT_IMPLEMENTATION_GUIDE.md` with a proposed architecture, project-specific gaps, phased implementation plan, acceptance gates, network/API/queue/model/AWS safety steps, benchmark metrics, and official references.
- No application or cloud configuration was changed in this research/planning step.

## 2026-10-01: TensorFlow Startup Error

### Conflict

- `python webapp.py` failed while importing TensorFlow. The virtual environment had `numpy==2.2.6`, but the installed TensorFlow 2.15.0 binary was compiled against the NumPy 1.x API. The traceback included `_ARRAY_API not found` and `np.complex_ was removed in the NumPy 2.0 release`.
- The environment also had `ml-dtypes==0.6.0` and `protobuf==7.36.2`, which are outside the compatible TensorFlow 2.15/TensorBoard 2.15 dependency range.
- The failure occurred before Flask started; it was an environment dependency conflict, not an error in the HTTP route.

### Solution Applied

- Reinstalled `numpy==1.26.4`, `ml-dtypes==0.2.0`, and `protobuf==4.23.4` in the project virtual environment.
- Added the TensorFlow compatibility pins to `requirements.txt` so a fresh environment does not repeat this startup failure.
- Verified the repaired environment imports NumPy 1.26.4, TensorFlow 2.15.0, and protobuf 4.23.4 successfully.

### Remaining Verification

- A full app-start check was not run after the repair because the terminal start command was skipped. Run `python webapp.py` when you are ready to start the application.
- The environment contains an unrelated installed ONNX package whose newest version expects newer protobuf/ml-dtypes versions. ONNX is not part of the current pinned requirements; if ONNX conversion is added later, use a separate conversion environment or pin a compatible ONNX version rather than mixing incompatible TensorFlow and ONNX stacks.

## 2026-10-01: TensorFlow Version Decision

### Question

- Newer TensorFlow releases are available. TensorFlow's official release page currently shows newer 2.20/2.21 releases and a 2.22 release entry, so `2.15.0` is not the newest version.

### Why This Project Uses 2.15.0 For Now

- The project was already using TensorFlow 2.15.0, and the trained model, Keras APIs, Python 3.10 environment, and TensorFlow-compatible dependency pins were verified against that version.
- The recent startup failure was caused by incompatible NumPy/protobuf/ml-dtypes versions around TensorFlow 2.15, not by TensorFlow 2.15 itself. Pinning the compatible versions fixed the observed error without introducing a larger migration.
- Upgrading TensorFlow can change Keras behavior, serialization/loading behavior for the existing `.h5` model, numerical results, dependency requirements, and CPU/GPU runtime behavior. It should therefore be treated as a separate upgrade experiment, followed by retraining and full metric/parity validation.

### Upgrade Guidance

- Do not upgrade just because a newer number exists. Create a fresh environment, test the newest release supported by the target Python/OS/container, load or retrain the model, and compare predictions, metrics, calibration, latency, and memory against the 2.15 baseline.
- The official installation guide documents current Python/platform support and notes recent breaking changes, including TensorBoard no longer being bundled in newer releases. Check the exact release notes and PyPI resolver output before changing pins.
- Keep TensorFlow 2.15.0 for the current reproducible baseline until a newer-version test passes. Upgrade later through a separate branch/environment so the working baseline remains recoverable.

## 2026-10-01: Local Nginx Reverse Proxy Setup

### Problem

- The Flask app was listening on port 8080, but Nginx was not installed or available on Windows PATH. There was also no Nginx configuration in the project, so `nginx -s reload` could not be used reliably.

### Solution Applied

- Added `nginx.conf` with a minimal reverse proxy: Nginx listens on port 80 and forwards requests to `http://127.0.0.1:8080`.
- The configuration forwards the host and client/proxy headers needed by the Flask app.

### Remaining Setup

- Nginx must still be downloaded/extracted on the Windows machine, then started with this configuration. Ngrok should tunnel to port 80 if the goal is to expose Nginx, rather than bypassing it by tunneling directly to port 8080.
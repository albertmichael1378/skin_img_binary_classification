# Skin Lesion Image Classification (Research Prototype)

An experimental binary image-classification project using transfer learning
with VGG16. The current data loader expects images in two folders:
`cancerous/` and `non_cancerous/`. The repository also contains ISIC 2019
training images, ground-truth labels, and metadata.

> **Research use only.** This project is not a medical device and is not
> validated for diagnosis, triage, or treatment decisions. It must not be used
> to reassure someone that a lesion is benign or to delay professional care.

## Current state

- The Flask application serves image predictions; it does not expose a training
  route.
- The DVC training stage currently uses
  `artifacts/data_source/Binary_Classification`. The checked-in DVC lockfile
  still refers to a previous 630-image dataset and older parameter values, so
  it must be regenerated after the pipeline is validated.
- The observed binary folders contain 8,473 `cancerous` and 16,858
  `non_cancerous` images. Confirm the source, label mapping, and license before
  treating these as an ISIC-derived dataset.
- The current evaluation code resplits the same directory used for training;
  its accuracy is not an independent test-set result.
- Training writes `artifacts/training/run_metadata.json` with the run status,
  dataset manifest fingerprint, class mapping/counts, configuration, software
  versions, training history, and model checksum.

See [the ML improvement plan](docs/ML_IMPROVEMENT_PLAN.md) for the ordered
work needed to make experiments trustworthy and [the production safety
notes](MODEL_RESEARCH_AND_PRODUCTION_SAFETY.md) for deployment and clinical
limitations.

## Repository layout

```text
config/                         Runtime and artifact paths
docs/                           Project-specific ML improvement plan
metadata/                       Run metadata JSON Schema
src/vgg16classifier/
  components/                   Model, training, evaluation, callbacks
  config/                       Configuration loading
  entity/                       Configuration dataclasses
  pipeline/                     DVC stage entry points and prediction
  utils/                        Shared utilities
artifacts/                      Local datasets and generated model artifacts
dvc.yaml                        Reproducible stage graph
dvc.lock                        Resolved stage inputs/outputs (currently stale)
params.yaml                     Model and training parameters
```

Datasets and generated artifacts are local/DVC-managed data, not source files.
Do not commit raw images, patient-identifying metadata, model binaries, secrets,
or credentials to Git. Review the ISIC dataset terms and preserve applicable
attribution and usage restrictions.

## Environment setup (Windows PowerShell)

From the repository root:

```powershell
py -3.10 -m venv skinclassifier
.\skinclassifier\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The pinned requirements include DVC. Confirm it is available in the activated
environment. If the `dvc` executable is not on PowerShell's PATH, invoke the
installed module through that environment's Python:

```powershell
python -m dvc --version
python -m dvc status
```

A DVC remote must be configured separately for `dvc pull`/`dvc push`; this
repository does not include remote credentials.

## Running the pipeline

Do not start a full training run until the P0 data, split, and model checks in
the [ML improvement plan](docs/ML_IMPROVEMENT_PLAN.md) are complete. The current
pipeline is a prototype, not a validated model-release workflow.

When ready, from the activated environment:

```powershell
python -m dvc repro
python -m dvc status
```

Training metadata is written to
`artifacts/training/run_metadata.json`. The DVC lockfile should be regenerated
only after the stage dependencies, output paths, and evaluation protocol are
correct. The current lockfile is not evidence that the present dataset or
parameters produced the checked-in model.

## Current binary label contract

Keras directory loading sorts class-folder names alphabetically. With the
current folders, the expected mapping is:

| Class index | Folder |
| --- | --- |
| 0 | `cancerous` |
| 1 | `non_cancerous` |

For this binary model, the single sigmoid output represents the probability of
class index 1 (`non_cancerous`). Serving code thresholds that probability at
0.5. This is a software label mapping only; it is not a clinical decision
threshold.

The mapping from ISIC 2019's original labels to these two folders must be
recorded and reviewed. Keep unresolved/unknown labels out of training until
their handling is explicitly defined. Patient/lesion-aware splits are required
to avoid leakage.

## Experiment and model metadata

Each training attempt writes a versioned JSON record to
`artifacts/training/run_metadata.json`, including failures where metadata can
be persisted. The record follows
[`metadata/run_metadata.schema.json`](metadata/run_metadata.schema.json).
The dataset fingerprint hashes sorted relative file paths and file sizes; it is
a fast inventory fingerprint, **not** a cryptographic hash of every image.
Use DVC-tracked data and its content hashes as the authoritative data version.

The record is overwritten for each run because the model output path is also
overwritten. Archive or version artifacts through DVC before a new run when
history must be retained.

## Improving the project

Recommended order:

1. Verify label provenance and the binary mapping; inventory duplicates,
   patient/lesion identifiers, missing files, and class imbalance.
2. Build fixed patient/lesion-grouped train, validation, and untouched test
   splits. Do not derive a final test set from the training directory.
3. Make preprocessing identical in training, evaluation, and inference.
4. Validate the one-unit sigmoid/binary-label contract and add shape/mapping
   tests.
5. Evaluate with confusion matrix, sensitivity, specificity, balanced
   accuracy, ROC-AUC, PR-AUC, calibration, and confidence intervals—not
   accuracy alone.
6. Establish a corrected VGG16 baseline, then compare EfficientNetB0 and
   MobileNetV3 on the same splits and tuning budget.
7. Version data, code, parameters, metrics, model checksum, and environment
   together; only promote a model that passes predefined gates.

Details and acceptance criteria are in
[docs/ML_IMPROVEMENT_PLAN.md](docs/ML_IMPROVEMENT_PLAN.md).

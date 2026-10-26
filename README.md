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
- Evaluation uses the separate legacy `skin-disease-images` directory and its
  own validation split. Its independence from the training images is not
  established, so its results should not be treated as a verified held-out
  test result.
- Training writes a UTC-timestamped JSON file under
  `artifacts/training/run_metadata/` with the run status,
  dataset manifest fingerprint, class mapping/counts, configuration, software
  versions, training history, and model checksum.
- The VGG16 model applies ImageNet preprocessing internally and uses a
  `GlobalAveragePooling2D → Dense(256) → Dropout(0.5) → Dense` classification
  head. Training first fits the head with the pretrained backbone frozen, then
  fine-tunes VGG16 block 5 at a lower learning rate. This makes training,
  evaluation, and prediction use the same input preprocessing.
- Optional model-integrated augmentation (horizontal/vertical flips, up to 20°
  rotation, 10% zoom, and 20% contrast) is active during training and disabled
  automatically for evaluation and prediction. Each training phase creates
  fresh callbacks and a phase-specific best checkpoint, monitoring validation
  AUC.
- GitHub Actions is currently CI-only: it checks out the repo and compiles
  Python source/tests. AWS image publishing and deployment are disabled until
  AWS is configured. The previous AWS workflow is archived as
  `.github/workflows/main-aws-deployment.yaml.disabled`; GitHub does not run
  that extension. The active workflow does not train the model.

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

The current GitHub Actions workflow does not install DVC, fetch DVC-managed
data/artifacts, or run the training pipeline. Training should be run separately
on the configured local or Kaggle environment after its dataset and paths are
set up.

## Running the pipeline

Do not start a full training run until the P0 data, split, and model checks in
the [ML improvement plan](docs/ML_IMPROVEMENT_PLAN.md) are complete. The current
pipeline is a prototype, not a validated model-release workflow.

When ready, from the activated environment:

```powershell
python -m dvc repro
python -m dvc status
```

Training metadata is written as timestamped JSON files under
`artifacts/training/run_metadata/`. The DVC lockfile should be regenerated only
after the stage dependencies, output paths, and evaluation protocol are
correct. The current lockfile is not evidence that the present dataset or
parameters produced the checked-in model.

To run only reconstruction and evaluation from the configured ZIP archive,
activate the project environment and run:

```powershell
python src/vgg16classifier/pipeline/stage_04_evaluation.py
```

This stage reconstructs the configured architecture, restores the archived
weights into `evaluation.reconstructed_model_path`, and then evaluates that
model. It does not install or upgrade Keras. The existing
`reconstruct_and_evaluate` module remains available when you need to override
the configured archive or output path.

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

Each training attempt writes a uniquely named JSON record, for example
`artifacts/training/run_metadata/run_metadata_20261026T063015123456Z_12345678.json`.
The UTC timestamp and short run ID keep records distinct, including failed runs
where metadata can be persisted. The record follows
[`metadata/run_metadata.schema.json`](metadata/run_metadata.schema.json).
The dataset fingerprint hashes sorted relative file paths and file sizes; it is
a fast inventory fingerprint, **not** a cryptographic hash of every image.
Use DVC-tracked data and its content hashes as the authoritative data version.

The generated metadata directory is ignored by Git but declared as a DVC
training output. This keeps run records out of source commits while allowing
DVC to version and restore them: commit the updated `dvc.lock`, configure a DVC
remote, and run `dvc push` to retain artifacts remotely. To retrieve the files
for a committed experiment later, check out that Git revision and run
`dvc pull`; the JSON files will be restored under the directory above. Without
a configured remote and pushed DVC cache, ignored files remain only on the
machine where training ran.

Evaluation is configured under `evaluation` in `config/config.yaml`. It loads
the configured model and evaluates all images in the labeled test directory
without making a validation split. It writes a JSON report with accuracy,
class-specific precision/recall/F1, sensitivity, specificity, balanced
accuracy, ROC-AUC, PR-AUC, Brier score, a confusion matrix, and stratified
bootstrap confidence intervals. `bootstrap_samples` is the number of
class-stratified resamples used to estimate those intervals: larger values
usually make the estimates steadier but take longer. It does not add images or
improve model accuracy. These image-level intervals assume samples are
independent; if images are grouped by patient or lesion, confidence intervals
should resample those groups instead. A CSV contains each relative image path,
true and predicted labels, and the positive-class probability. The configured
`affected` class is positive by default, and the decision threshold is fixed
before evaluation rather than tuned on the evaluation set. The configuration
includes explicit `class_mapping_verified` and
`dataset_independence_verified` attestations; both default to `false` and
should only be changed after the label mapping and data provenance are checked.

The default directory is the separate legacy `skin-disease-images` dataset,
containing 380 `affected` and 250 `Healthy` images at last inspection. The
dataset's source, label mapping, overlap with training images, and
representativeness have not been verified. The report records the configured
verification flags (both false by default); these results must not be
presented as a verified held-out test result or clinical validation.

### Reconstruct and evaluate a Keras 3 weights archive

If a model archive cannot be loaded by the project's TensorFlow/Keras 2.15
runtime, reconstruct the configured VGG16 classifier and import its named
layer weights directly from the archive:

```powershell
python -m vgg16classifier.pipeline.reconstruct_and_evaluate `
  --archive artifacts/training/vgg16model..zip `
  --output artifacts/training/vgg16model_reconstructed.keras
```

The command validates the archive, refuses to overwrite an existing output,
saves a new native `.keras` model, and runs the regular evaluation against the
configured `evaluation.test_data_dir`. It writes the configured metrics JSON
and per-image predictions CSV. The original archive and model are left
unchanged. This does not establish that the evaluation images are independent
of training data; verify the class mapping and dataset provenance before
interpreting metrics as held-out performance.

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

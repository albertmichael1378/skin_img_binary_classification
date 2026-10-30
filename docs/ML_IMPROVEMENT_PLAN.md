# ML Improvement Plan

## Goal and scope

Make the current two-class skin-image experiment reproducible and honestly
evaluated before comparing architectures or making any deployment claims. The
selected initial task is `cancerous` versus `non_cancerous`; this plan does not
assume that the current folder labels are a clinically valid interpretation
of ISIC 2019 labels.

This is a research/educational workflow. It is not a clinical validation plan
and cannot establish safety for diagnosis or care decisions.

## Repository facts to resolve first

- The current loader and DVC stage point to
  `artifacts/data_source/Binary_Classification`.
- The observed directory contains 8,473 `cancerous` and 16,858
  `non_cancerous` images. Determine and document which original ISIC labels
  were mapped to each folder and how any ambiguous/unknown labels were handled.
- ISIC source files are present under `artifacts/data_source/imgs`, including
  ground-truth and metadata CSVs. The CSV metadata sample includes a
  `lesion_id`; confirm identifier coverage and whether patient identifiers are
  available before choosing a grouping key.
- Training applies `validation_split` to `Binary_Classification`, while
  evaluation uses a separate legacy `skin-disease-images` directory and its
  own validation split. Whether these folders overlap in image content or
  provenance is unknown, so evaluation is not yet established as an untouched
  independent test set.
- The current DVC lockfile records the previous `skin-disease-images` input
  (630 files) and previous parameter values. It does not describe the current
  binary dataset/parameters.
- The binary class folders are imbalanced. Any class weighting or resampling
  should be a documented experiment, not a hard-coded assumption.

## Phased plan and exit criteria

### P0 — Establish trustworthy data and labels

1. Record the exact dataset source/version, applicable license/terms, source
   CSV checksums, and transformation script that created the two binary
   folders.
2. Document an explicit mapping from source label columns to the two target
   labels. State how multi-label rows and `UNK`/unresolved labels are treated;
   exclude unresolved examples until a reviewed policy exists.
3. Audit image readability, duplicate files and perceptual duplicates,
   corrupted examples, missing labels, class counts, and identifier coverage.
4. Split by patient where available, otherwise by lesion/source identity.
   Freeze train, validation, and test manifests before model tuning. Keep the
   test set inaccessible to training and threshold selection.
5. Store split manifests and data versioning metadata; do not commit raw
   images or identifying data to Git.

**Exit gate:** A reviewed label mapping, a reproducible data-preparation
procedure, clean split manifests, and no known patient/lesion leakage between
splits.

### P1 — Repair model and serving correctness

1. Keep the binary formulation consistent: directory labels 0/1, one sigmoid
   output, and binary cross-entropy. Persist the class-index map rather than
   relying on implicit alphabetical ordering.
2. Choose one preprocessing contract for resize, color mode/channel order,
   dtype, scaling, and VGG16 preprocessing. Apply the same implementation in
   training, validation, evaluation, and inference.
3. Verify that the model emits shape `(batch, 1)` and that inference thresholds
   the scalar sigmoid probability; `argmax` on a one-value output always gives
   index 0.
4. Add tests for label mapping, preprocessing output, model output shape,
   prediction threshold behavior, and invalid image inputs.
5. Use a deterministic seed and record it, but do not claim bitwise determinism
   unless the hardware/runtime supports and verifies it.

**Exit gate:** Tests prove train/serve preprocessing equivalence and both
binary output classes can be returned.

### P2 — Make evaluation independent and informative

1. Evaluate once on the untouched, grouped test split after model and threshold
   selection are complete.
2. Report the confusion matrix, per-class precision/recall, sensitivity,
   specificity, balanced accuracy, ROC-AUC, PR-AUC, and calibration (for
   example Brier score). Include support/counts and confidence intervals.
3. Select any operating threshold on validation data according to a written
   use-case objective. Do not tune it against test data.
4. Analyze errors and performance across supported subgroups and acquisition
   sources when metadata quality/sample sizes permit. Include uncertainty and
   limitations.

**Exit gate:** A repeatable evaluation command produces versioned metrics
from a test manifest that was not used for fitting or tuning.

### P3 — Build fair model comparisons

1. Establish the corrected VGG16 baseline first.
2. Compare EfficientNetB0 and MobileNetV3Small as candidate backbones; add
   another model only when data size and compute justify it.
3. Keep splits, preprocessing contract, augmentation policy, tuning budget,
   stopping criteria, and test protocol fixed across candidates.
4. Compare discrimination, class-specific errors, calibration, model size,
   latency, and memory. Do not select a model on accuracy alone.
5. Treat class weighting, sampling, augmentation, and fine-tuning depth as
   separately tracked experiments.

**Exit gate:** A comparison report includes the same untouched test protocol
for every candidate and an explicit model-selection rationale.

### Conservative VGG16 regularization trial

The next VGG16 training run is configured as a single conservative trial:

- head learning rate `1e-4`, fine-tuning learning rate `3e-6`;
- fine-tune only `block5_conv3`, leaving the earlier VGG16 blocks frozen;
- head dropout `0.6` and dense-kernel L2 strength `5e-4`;
- use horizontal/vertical flips, up to 20-degree rotation, 10% zoom, and 20%
  contrast augmentation, plus configurable brightness augmentation
  (`AUGMENTATION_BRIGHTNESS_FACTOR: 0.1`);
- retain the 30-epoch head and 15-epoch fine-tune limits, with early-stopping
  patience increased to 7 epochs.

Training now saves the best `val_auc` checkpoint from each phase and exports
whichever phase has the higher validation AUC. The chosen phase and validation
scores are written to that run's metadata. These values are a testable starting
point, not evidence that overfitting is the only cause of poor performance or
that this configuration improves generalization. Compare it against the prior
configuration using identical patient/lesion-grouped splits and multiple seeds;
do not select settings or thresholds using the final test set.

### P4 — Reproducibility and release metadata

For each run, retain:

- unique run ID, UTC timestamps, status, source commit, and code/config version;
- data source and split-manifest fingerprint, class mapping and counts;
- model/backbone, preprocessing, seed, optimizer/loss, hyperparameters, and
  training history;
- Python/TensorFlow/DVC versions, model checksum, and evaluation metrics;
- decision threshold and the validation-only rationale for it.

The training stage now writes a UTC-timestamped metadata record under
`artifacts/training/run_metadata/` using
[`metadata/run_metadata.schema.json`](../metadata/run_metadata.schema.json).
Its path+size fingerprint detects many inventory changes but is not a content
hash. The metadata directory is a DVC output and ignored by Git; commit the
corresponding `dvc.lock` and push the DVC cache to a remote to preserve and
restore historical records. DVC content hashes and versioned split manifests
remain authoritative.

**Exit gate:** A clean checkout with the documented data version can reproduce
the selected run and trace its model to code, data, parameters, and metrics.

### P5 — Model release gates

Before considering any non-research use, define intended use, target population,
acceptable error trade-offs, uncertainty/abstention behavior, monitoring and
rollback. Require appropriate independent domain, privacy, legal, and
regulatory review. Do not describe a high test score on one dataset as clinical
validation.

## Suggested incremental source layout

Avoid a risky all-at-once move that breaks imports and DVC paths. Keep the
current package working while gradually extracting responsibilities:

```text
config/                         Committed non-secret configuration
data/                           DVC-managed data manifests/split definitions
docs/                           Design, evaluation protocol, model card
metadata/                       Schemas for run/evaluation records
src/vgg16classifier/
  data/                         Label mapping, validation, split loading
  models/                       Model construction and serialization
  training/                     Fit orchestration and callbacks
  evaluation/                   Metrics and reports
  inference/                    Shared preprocessing and prediction service
  tracking/                     Run metadata and artifact lineage
tests/                          Data, preprocessing, model, API tests
reports/                        DVC-tracked metrics and plots (no raw images)
```

Migrate one responsibility at a time with import compatibility tests. Keep
Flask serving separate from offline training; the current app no longer
contains a training route.

## Immediate next work

1. Verify the binary label mapping and source/license terms.
2. Generate patient/lesion-grouped split manifests and update both training and
   evaluation to consume those same explicit splits.
3. Add shared preprocessing and tests; retrain from scratch.
4. Regenerate `dvc.lock` only after the updated pipeline, input paths, and
   output/metric paths are correct.
5. Produce a test-set report and model card before comparing backbones.

# Model Research and Production Safety

**Research date:** 2026-09-30; **workspace recheck:** 2026-10-06
**Purpose:** Technical research and a project-specific improvement plan. This is not clinical, legal, or regulatory advice, and this model is not established as safe for diagnosis or treatment decisions.

## Executive Summary

Do not select a new architecture based on the current reported accuracy. First make the evaluation trustworthy and make training and inference use identical preprocessing. Then compare a small set of pretrained models on the same patient-separated splits and an untouched external test set.

The current loader points to `artifacts/data_source/Binary_Classification`,
which contains 8,473 `cancerous` and 16,858 `non_cancerous` images at the time
of the 2026-10-06 workspace check. ISIC 2019 source images, ground-truth, and
metadata files are also present. The exact provenance and mapping from ISIC
labels into these two folders still need to be reviewed and recorded.

The source code configures a 224 x 224 VGG16 transfer-learning model with one
sigmoid output for the two-class binary labels. The saved evaluation JSON
currently reports 60.85% accuracy, but it is produced from another validation
split of the training directory, not an independent test set. Neither this
result nor the project name's historical 79% claim establishes generalization.

The current app is a research prototype, not a safe clinical service. The
current Flask source no longer registers a training route, but shared-file
uploads, permissive CORS, and the development server still need attention
before any public deployment.

## What the Current Code Does

- Uses ImageNet VGG16 with its original classification head removed, then
  freezes the base and adds one sigmoid output for two classes. The current
  directory generators explicitly use binary labels, aligned with binary
  cross-entropy.
- The current training generator and inference path pass image values in the
  raw 0-255 range and do not call Keras VGG16 `preprocess_input`. This avoids a
  simple train/serve scaling difference but does not use the documented VGG16
  ImageNet preprocessing contract; choose a contract and retrain consistently.
- Maps the sorted directory classes to class indices. With the current folder
  names, `cancerous` is index 0 and `non_cancerous` is index 1; preserve and
  verify this mapping with the model artifact.
- Uses a 20% validation subset during training. Evaluation independently constructs a 30% validation subset from the same directory. These are not independent sets: their ranges overlap, and some images used for training can enter evaluation. The saved accuracy therefore is not a clean held-out result.
- Reports only loss and accuracy in `evaluation_scores_com726.json`; this hides class-specific errors and uncertainty.
- The binary folders are imbalanced (about 33% `cancerous` and 67%
  `non_cancerous`). Dataset count alone does not establish label quality,
  patient independence, or broad clinical performance.

The DVC lockfile still describes the previous 630-image `skin-disease-images`
input and older parameters. It is stale relative to the current DVC stage and
binary folder. Regenerate it only after the data split and evaluation pipeline
are corrected.

## Highest-Priority Model and Evaluation Fixes

1. **Make splits independent.** Create train, validation, and final test partitions before model tuning. Group by patient and lesion/source image where identifiers exist; never let crops or repeat images from one lesion cross partitions. Keep the final test set untouched until a model and threshold are selected. If the data comes from multiple clinics or devices, reserve at least one site/device for external evaluation.
2. **Unify preprocessing.** Implement one documented preprocessing path and use it in training, validation, testing, and serving. Evaluate Keras VGG16 `preprocess_input` for ImageNet weights; adopting it requires retraining and using it consistently everywhere. Do not combine preprocessing schemes accidentally.
3. **Align labels, output activation, and loss.** Choose a binary or categorical formulation deliberately. Persist the class-to-index map from training and return named class probabilities rather than relying on a hard-coded index. Validate that output probabilities and labels have the expected shape and meaning.
4. **Audit the dataset.** Check label quality, duplicates, source/patient grouping, acquisition method, lesion type, image quality, and representation across skin tones, age groups, body sites, and devices. Record exclusions and label provenance. A folder called `affected` does not define a clinically precise diagnosis.
5. **Report more than accuracy.** On the untouched test set, report confusion matrix, sensitivity/recall and specificity per class, balanced accuracy, ROC-AUC and precision-recall AUC, calibration (for example Brier score), and confidence intervals. Report results by relevant subgroups and image source. Pick any decision threshold on validation data based on the intended use; do not tune it against the final test set.

## Models Worth Benchmarking

All candidates should use a documented pretrained checkpoint and the preprocessing expected by that checkpoint. Compare them on exactly the same patient-separated splits and tuning budget. A newer architecture is a hypothesis to test, not evidence of improvement.

| Candidate | Why test it | Main trade-off for this project |
|---|---|---|
| Corrected VGG16 baseline | Establish a valid comparison against the existing 59 MB model. | Larger and older; current result is not a valid baseline until evaluation and preprocessing are fixed. |
| EfficientNetB0 | Strong first comparison: efficient image classification model with pretrained ImageNet weights, available in Keras Applications. | Still needs skin-image validation; model preprocessing and input handling must match the selected Keras version/checkpoint. |
| MobileNetV3Small or MobileNetV3Large | Good candidates when CPU latency, memory, or edge deployment matters. Benchmark actual container inference time and memory. | Smaller does not mean clinically safer or more accurate; validate sensitivity and calibration carefully. |
| ResNet50 | Familiar, well-supported transfer-learning reference point with useful ecosystem support. | May not beat EfficientNet or the current model; compare rather than assume. |
| ConvNeXtTiny | Modern convolutional alternative worth testing if adequate compute is available. | Higher capacity and overfitting risk on the current dataset; use careful regularization and external testing. |
| Dermatology foundation-model embeddings (for example, investigate Derm Foundation) | Domain-pretrained features may transfer better than generic ImageNet features. Test frozen embeddings plus a simple classifier before end-to-end fine-tuning. | Confirm actual access, license, input requirements, population/task match, and independent evidence. Do not treat a research model or vendor claim as clinical validation. |

Recommended first experiment: fix the current VGG16 pipeline, then compare EfficientNetB0 and MobileNetV3Small. Add ConvNeXtTiny or a dermatology-pretrained encoder only if the first comparison is sound and compute/data allow. Avoid large ensembles until each component is independently validated; ensembles complicate latency, calibration, and failure analysis.

## Production Deployment Checklist

### Intended use and clinical boundaries

- Write an intended-use statement: user, setting, image type, target population, task, and what action a result may inform. The current `cancerous` versus `non_cancerous` folders do not identify a specific disease, establish malignancy, or indicate urgency.
- Until clinical validation and applicable authorization are complete, label the system research/educational only. Do not tell users that a lesion is safe or that professional assessment is unnecessary. Provide clear escalation language for concerning, uncertain, or worsening lesions.
- Use an appropriately qualified clinician in any clinical workflow. The exact medical-device and clinical decision-support rules depend on jurisdiction, claims, users, and workflow; obtain specialist regulatory advice before clinical use.

### Image and API security

The current Flask code has several concrete risks:

- The current Flask source no longer registers `/train`; keep training offline in a controlled pipeline and do not reintroduce it as a public inference route.
- `decodeImage()` base64-decodes the request and writes to the fixed shared `inputImage.jpg`. Concurrent requests can overwrite each other's images. Use validated in-memory decoding or unique temporary files, clean them up, and never use a shared request filename.
- The API does not enforce request size, strict image format, dimensions, or pixel limits before decoding. Apply a small maximum request size, validate the content type and actual decoded image, limit dimensions/decompression, normalize to RGB, and return safe 4xx errors for invalid files.
- `CORS(app)` enables broad cross-origin access. Restrict origins to the actual UI domains; add authentication/authorization if the service or results are not intentionally public. Add per-client rate limits and abuse monitoring.
- Avoid logging raw images, base64 payloads, identifiers, or sensitive health information. Define consent, retention, deletion, access controls, encryption in transit/at rest, and incident response. Skin images may contain identifying details.

### Runtime, model behavior, and operations

- Use a production WSGI server (for example, Waitress or Gunicorn), not Flask's development server. Place it behind TLS and an appropriately configured reverse proxy/load balancer.
- Load the model once at process startup rather than reloading it for every prediction. Make model/version/preprocessing metadata explicit and immutable for each release.
- Add input-quality and out-of-distribution checks. If image quality is poor, the input is unlike training data, or confidence is insufficient, abstain and request another image or human review; do not force a binary answer.
- Run concurrency tests. Keep request data isolated; set worker counts based on measured TensorFlow memory use. Add health/readiness probes, timeouts, graceful shutdown, and capacity limits.
- Harden the container: run as a non-root user, use a read-only filesystem where possible, keep secrets out of the image, scan dependencies and image layers, restrict network access, and produce an SBOM. Pin or lock the full dependency graph for repeatable builds.
- Monitor latency, error rates, abstention rates, input quality, calibration, and data drift without collecting unnecessary patient data. Define rollback and incident procedures. Do not automatically retrain or change thresholds in production without review and revalidation.

## Safe Validation and Release Gates

1. Freeze a versioned dataset manifest with provenance, labels, duplicate checks, patient/lesion groups, and permission/consent status.
2. Fix the split, preprocessing, label encoding, and loss/activation; rerun training and evaluation from scratch with recorded seeds and versions.
3. Compare candidate models on validation data, then report final metrics once on a held-out patient-level test set. Repeat on an external site/device/population before making generalization claims.
4. Have domain experts review false negatives, false positives, subgroup results, and failure examples. Set a clinically defensible operating threshold and abstention policy before any clinical workflow.
5. Test API abuse, malformed/oversized images, concurrency, privacy controls, container scanning, recovery, and rollback. Run a silent prospective evaluation before any user-facing or clinical use.
6. Complete jurisdiction-specific legal/regulatory review and establish documented change control, monitoring, and human oversight.

**Current disposition:** research prototype only. The current binary data mapping,
preprocessing contract, and evaluation split are not sufficiently validated to
establish diagnostic performance or production safety.

## Sources and Further Reading

Official model documentation:

- [Keras VGG16 and VGG19](https://keras.io/api/applications/vgg/) - model and preprocessing APIs.
- [Keras EfficientNet B0-B7](https://keras.io/api/applications/efficientnet/) - model family and preprocessing details.
- [Keras MobileNet, MobileNetV2, and MobileNetV3](https://keras.io/api/applications/mobilenet/) - model family and preprocessing APIs.
- [Keras ConvNeXt](https://keras.io/api/applications/convnext/) - model family and preprocessing APIs.
- [TensorFlow transfer learning tutorial](https://www.tensorflow.org/tutorials/images/transfer_learning) - staged feature extraction and fine-tuning.
- [ISIC Challenge datasets](https://challenge.isic-archive.com/data/) - dataset descriptions, citations, and terms.
- [Google Research: A foundation model for dermatology](https://research.google/blog/a-foundation-model-for-dermatology/) - research candidate to investigate; independently verify current availability, terms, and task/population fit.

Safety, regulation, and production:

- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework) - voluntary AI risk management framework.
- [FDA Clinical Decision Support Software guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software) - current US FDA guidance; applicability depends on the product's intended use and context.
- [FDA AI/ML-enabled medical devices](https://www.fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-aiml-enabled-medical-devices) - FDA information and authorized-device listings.
- [FDA Good Machine Learning Practice guiding principles](https://www.fda.gov/medical-devices/software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles) - lifecycle quality principles for medical-device ML development.
- [WHO: Ethics and governance of artificial intelligence for health](https://www.who.int/publications/i/item/9789240029200) - health AI ethics and governance considerations.
- [Flask: Deploying to Production](https://flask.palletsprojects.com/en/stable/deploying/) - use a production WSGI server rather than Flask's development server.
- [OWASP API Security Project](https://owasp.org/API-Security/) - API security risks and controls.
- [Daneshjou et al., Disparities in dermatology AI performance on a diverse, curated clinical image set](https://doi.org/10.1126/sciadv.abq6147) - evidence motivating diverse, subgroup-aware evaluation.
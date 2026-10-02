# Model Research and Production Safety

**Research date:** 2026-09-30  
**Purpose:** Technical research and a project-specific improvement plan. This is not clinical, legal, or regulatory advice, and this model is not established as safe for diagnosis or treatment decisions.

## Executive Summary

Do not select a new architecture based on the current reported accuracy. First make the evaluation trustworthy and make training and inference use identical preprocessing. Then compare a small set of pretrained models on the same patient-separated splits and an untouched external test set.

This workspace currently contains 630 images in two folders: 380 `affected` and 250 `Healthy`. The model is a 224 x 224 VGG16 transfer-learning model. The evaluation JSON reports 76.19% accuracy, while the project name suggests 79%; neither number should be treated as a reliable estimate of real-world performance until the split and preprocessing issues below are resolved.

The current app is a research prototype, not a safe clinical service. Its upload handling, shared image filename, open training route, and development server need attention before any public deployment.

## What the Current Code Does

- Uses ImageNet VGG16 with its original classification head removed, then freezes the base and adds a two-unit sigmoid output layer.
- Trains with `BinaryCrossentropy`, while `flow_from_directory` defaults to categorical one-hot labels. For mutually exclusive classes, use either a two-unit softmax with categorical cross-entropy, or one sigmoid output with binary labels and binary cross-entropy.
- Rescales training and validation images to `[0, 1]`, but `PredictionPipeline.predict()` passes pixel values through without the same rescaling. This is a train/serve preprocessing mismatch. It can invalidate predictions regardless of which backbone is chosen.
- Uses a 20% validation subset during training. Evaluation independently constructs a 30% validation subset from the same directory. These are not independent sets: their ranges overlap, and some images used for training can enter evaluation. The saved accuracy therefore is not a clean held-out result.
- Reports only loss and accuracy in `evaluation_scores_com726.json`; this hides class-specific errors and uncertainty.
- Has 59 MB model weights. The input folders are moderately imbalanced (about 60% affected, 40% healthy), and the total of 630 images is small for establishing broad clinical performance.

The workspace also contains ISIC 2019 source files. Verify whether the 630-image folders were derived from those files, how labels were created, and which ISIC terms apply. Do not assume the source dataset and the prepared folders have identical scope or permitted uses.

## Highest-Priority Model and Evaluation Fixes

1. **Make splits independent.** Create train, validation, and final test partitions before model tuning. Group by patient and lesion/source image where identifiers exist; never let crops or repeat images from one lesion cross partitions. Keep the final test set untouched until a model and threshold are selected. If the data comes from multiple clinics or devices, reserve at least one site/device for external evaluation.
2. **Unify preprocessing.** Implement one documented preprocessing path and use it in training, validation, testing, and serving. For the existing model, matching the current `[0, 1]` rescale at inference is the minimum correction. For ImageNet VGG16 weights, also evaluate Keras VGG16 `preprocess_input` (RGB-to-BGR and channel mean adjustment); switching to it requires retraining and using it consistently everywhere. Do not combine preprocessing schemes accidentally.
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
| ConvNeXtTiny | Modern convolutional alternative worth testing if adequate compute is available. | Higher capacity and overfitting risk on only 630 images; use careful regularization and external testing. |
| Dermatology foundation-model embeddings (for example, investigate Derm Foundation) | Domain-pretrained features may transfer better than generic ImageNet features. Test frozen embeddings plus a simple classifier before end-to-end fine-tuning. | Confirm actual access, license, input requirements, population/task match, and independent evidence. Do not treat a research model or vendor claim as clinical validation. |

Recommended first experiment: fix the current VGG16 pipeline, then compare EfficientNetB0 and MobileNetV3Small. Add ConvNeXtTiny or a dermatology-pretrained encoder only if the first comparison is sound and compute/data allow. Avoid large ensembles until each component is independently validated; ensembles complicate latency, calibration, and failure analysis.

## Production Deployment Checklist

### Intended use and clinical boundaries

- Write an intended-use statement: user, setting, image type, target population, task, and what action a result may inform. This model has only `Healthy` versus `affected` labels; it cannot identify a specific disease, malignancy, or urgency from those labels.
- Until clinical validation and applicable authorization are complete, label the system research/educational only. Do not tell users that a lesion is safe or that professional assessment is unnecessary. Provide clear escalation language for concerning, uncertain, or worsening lesions.
- Use an appropriately qualified clinician in any clinical workflow. The exact medical-device and clinical decision-support rules depend on jurisdiction, claims, users, and workflow; obtain specialist regulatory advice before clinical use.

### Image and API security

The current Flask code has several concrete risks:

- `/train` runs `python main.py` through `os.system()` and accepts GET as well as POST. Remove this route from public production service. Run training only in a controlled, authenticated offline job/pipeline.
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

**Current disposition:** research prototype only. The available 630-image two-class dataset and current evaluation do not establish diagnostic performance or production safety.

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
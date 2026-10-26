import os
import hashlib
import json
import platform
import subprocess
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _git_revision(
    repo_root: Path,
) -> tuple[str | None, bool | None, str | None]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            check=False,
            text=True,
        )
        if revision.returncode:
            return None, None, revision.stderr.strip() or "git rev-parse failed"

        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            check=False,
            text=True,
        )
    except FileNotFoundError as error:
        return None, None, str(error)
    if status.returncode:
        return (
            revision.stdout.strip(),
            None,
            status.stderr.strip() or "git status failed",
        )
    return revision.stdout.strip(), not status.stdout.strip(), None


def _labels_count(
    labels: Iterable[int], class_indices: Mapping[str, int]
) -> dict[str, int]:
    counts = {class_name: 0 for class_name in class_indices}
    index_to_name = {index: name for name, index in class_indices.items()}
    for label in labels:
        class_index = int(label)
        if class_index not in index_to_name:
            raise ValueError(f"Dataset label index {class_index} has no class name")
        counts[index_to_name[class_index]] += 1
    return counts


def _dataset_manifest(
    dataset_root: Path, relative_paths: Sequence[str]
) -> tuple[str, int]:
    digest = hashlib.sha256()
    # Normalize path string representation without resolving symlinks
    normalized_root = Path(os.path.abspath(dataset_root))
    unique_paths = sorted(set(relative_paths))
    for filename in unique_paths:
        image_path = Path(filename)
        if not image_path.is_absolute():
            image_path = normalized_root / image_path
        else:
            image_path = Path(os.path.abspath(image_path))

        try:
            relative_path = image_path.relative_to(normalized_root).as_posix()
        except ValueError:
            # Fallback for paths that cannot be made relative to root directly
            relative_path = Path(filename).as_posix()

        # Follow symlink to get actual file size for hashing
        size = image_path.stat().st_size
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), len(unique_paths)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_training_metadata(
    *,
    repo_root: Path,
    dataset_root: Path,
    train_filenames: Sequence[str],
    train_labels: Iterable[int],
    validation_filenames: Sequence[str],
    validation_labels: Iterable[int],
    class_indices: Mapping[str, int],
    parameters: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
    optimizer: str | None,
    loss: str | None,
    model_path: Path,
) -> dict[str, Any]:
    git_commit, git_worktree_clean, git_revision_error = _git_revision(repo_root)
    all_filenames = list(train_filenames) + list(validation_filenames)
    manifest_sha256, sample_count = _dataset_manifest(dataset_root, all_filenames)
    train_class_counts = _labels_count(train_labels, class_indices)
    validation_counts = _labels_count(validation_labels, class_indices)
    class_counts = dict(train_class_counts)
    for class_name, count in validation_counts.items():
        class_counts[class_name] += count

    return {
        "schema_version": 1,
        "run_id": str(uuid.uuid4()),
        "status": "running",
        "started_at_utc": utc_now(),
        "finished_at_utc": None,
        "source": {
            "git_commit": git_commit,
            "git_worktree_clean": git_worktree_clean,
            "git_revision_error": git_revision_error,
        },
        "dataset": {
            "root": str(dataset_root),
            "manifest_algorithm": "sha256(relative_path_and_file_size)",
            "manifest_sha256": manifest_sha256,
            "sample_count": sample_count,
            "class_indices": dict(class_indices),
            "class_counts": class_counts,
            "split_class_counts": {
                "training": train_class_counts,
                "validation": validation_counts,
            },
        },
        "training": {
            "parameters": dict(parameters),
            "preprocessing": dict(preprocessing),
            "optimizer": optimizer,
            "loss": loss,
            "history": {},
            "error": None,
        },
        "environment": {
            "python": platform.python_version(),
            "tensorflow": _package_version("tensorflow"),
            "numpy": _package_version("numpy"),
            "dvc": _package_version("dvc"),
        },
        "model": {
            "path": str(model_path),
            "sha256": None,
        },
    }


def write_run_metadata(path: Path, metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def run_metadata_path(metadata_dir: Path, metadata: Mapping[str, Any]) -> Path:
    """Build a stable, timestamped path for one training run's metadata record."""
    started_at = metadata.get("started_at_utc")
    run_id = metadata.get("run_id")
    if not isinstance(started_at, str) or not isinstance(run_id, str):
        raise ValueError("Run metadata must include started_at_utc and run_id")

    timestamp = datetime.fromisoformat(
        started_at.replace("Z", "+00:00")
    ).astimezone(timezone.utc)
    filename_timestamp = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
    return metadata_dir / f"run_metadata_{filename_timestamp}_{run_id[:8]}.json"


def finish_training_metadata(
    metadata: dict[str, Any],
    *,
    status: str,
    history: Mapping[str, Sequence[float]] | None = None,
    model_path: Path | None = None,
    error: Exception | None = None,
) -> None:
    if status not in {"completed", "failed"}:
        raise ValueError(f"Unsupported terminal run status: {status}")
    metadata["status"] = status
    metadata["finished_at_utc"] = utc_now()
    if history is not None:
        metadata["training"]["history"] = {
            name: [float(value) for value in values]
            for name, values in history.items()
        }
    if model_path is not None and model_path.is_file():
        metadata["model"]["sha256"] = _sha256_file(model_path)
    if error is not None:
        metadata["training"]["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }

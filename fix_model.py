"""Safely rebuild a Keras archive after validating the target runtime module."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any


REQUIRED_FILES = ("config.json", "metadata.json", "model.weights.h5")
DEFAULT_INPUT = Path("artifacts/training/vgg16model.keras")
DEFAULT_OUTPUT = Path("artifacts/training/vgg16model_fixed.keras")
DEFAULT_FUNCTIONAL_MODULE = "keras.engine.functional"


def _load_config(archive: zipfile.ZipFile) -> dict[str, Any]:
    names = archive.namelist()
    if len(names) != len(REQUIRED_FILES) or set(names) != set(REQUIRED_FILES):
        raise ValueError(
            "The input archive must contain exactly these files at its root: "
            f"{', '.join(REQUIRED_FILES)}. Found: {names}"
        )
    if archive.testzip() is not None:
        raise ValueError("The input archive failed its ZIP integrity check")

    config = json.loads(archive.read("config.json"))
    if not isinstance(config, dict):
        raise ValueError("config.json must contain a JSON object")
    return config


def _validate_functional_module(module_name: str) -> None:
    """Fail early unless the installed Keras can resolve the requested class."""
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ValueError(
            f"The configured Keras runtime cannot import {module_name!r}; "
            "do not patch the model config to an unavailable module."
        ) from error

    if not isinstance(getattr(module, "Functional", None), type):
        raise ValueError(
            f"{module_name!r} does not expose the Functional model class"
        )


def _sha256_member(archive: zipfile.ZipFile, name: str) -> str:
    digest = hashlib.sha256()
    with archive.open(name) as member:
        for chunk in iter(lambda: member.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rebuild_archive(
    input_path: Path,
    output_path: Path,
    functional_module: str = DEFAULT_FUNCTIONAL_MODULE,
    *,
    force: bool = False,
) -> None:
    """Rebuild the archive with root-level files and a validated module path."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Input model archive does not exist: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different")
    if output_path.exists() and not force:
        raise FileExistsError(
            f"Output already exists; pass --force to replace it: {output_path}"
        )

    # The proposed config edit is safe only if the active runtime can resolve it.
    _validate_functional_module(functional_module)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(input_path, "r") as source:
        config = _load_config(source)
        original_module = config.get("module")
        config["module"] = functional_module
        updated_config = json.dumps(
            config, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.stem}-",
            suffix=".tmp",
            dir=output_path.parent,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            with zipfile.ZipFile(temporary_path, "w") as rebuilt:
                rebuilt.writestr(
                    "config.json", updated_config, compress_type=zipfile.ZIP_DEFLATED
                )
                for name in ("metadata.json", "model.weights.h5"):
                    # HDF5 weights are already compressed; storing avoids wasteful
                    # recompression while preserving their exact bytes.
                    with source.open(name) as source_member:
                        with rebuilt.open(
                            name, "w", force_zip64=True
                        ) as output_member:
                            for chunk in iter(
                                lambda: source_member.read(1024 * 1024), b""
                            ):
                                output_member.write(chunk)

            with zipfile.ZipFile(temporary_path, "r") as candidate:
                rebuilt_config = _load_config(candidate)
                if rebuilt_config.get("module") != functional_module:
                    raise ValueError("The rebuilt config has the wrong module path")
                for name in ("metadata.json", "model.weights.h5"):
                    if _sha256_member(source, name) != _sha256_member(
                        candidate, name
                    ):
                        raise ValueError(f"Model payload changed unexpectedly: {name}")

            if force:
                os.replace(temporary_path, output_path)
            else:
                # Creating a hard link publishes the complete file atomically and
                # refuses to overwrite an output created by another process.
                os.link(temporary_path, output_path)
                temporary_path.unlink()
        finally:
            temporary_path.unlink(missing_ok=True)

    print(f"Created validated archive: {output_path}")
    print(f"Config module: {original_module!r} -> {functional_module!r}")
    print(f"Root entries: {', '.join(REQUIRED_FILES)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild a .keras ZIP with root-level files and a Keras Functional "
            "module that exists in the active Python environment."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--module", default=DEFAULT_FUNCTIONAL_MODULE, dest="functional_module"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the output archive if it already exists.",
    )
    args = parser.parse_args()

    try:
        rebuild_archive(
            args.input,
            args.output,
            args.functional_module,
            force=args.force,
        )
    except (FileNotFoundError, FileExistsError, ValueError, OSError, zipfile.BadZipFile) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

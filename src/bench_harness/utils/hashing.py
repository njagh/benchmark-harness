"""Artifact fingerprinting utilities.

Computes hashes and metadata summaries for model artifacts to enable
distinguishing between runs against different builds.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def compute_file_hash(path: Path, algorithm: str = "sha256") -> str:
    """Compute hash of a file."""
    h = hashlib.new(algorithm)
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def compute_config_hash(config_path: Path) -> str | None:
    """Compute SHA-256 of a config file."""
    if not config_path.exists():
        return None
    return compute_file_hash(config_path)


def compute_weight_manifest_hash(artifacts_dir: Path) -> str:
    """Compute a hash of weight file paths + sizes. Fast, no actual weight hashing."""
    h = hashlib.sha256()
    for root, dirs, files in os.walk(artifacts_dir):
        dirs.sort()
        for f in sorted(files):
            fp = Path(root) / f
            try:
                size = fp.stat().st_size
            except OSError:
                continue
            rel = str(fp.relative_to(artifacts_dir))
            entry = f"{rel}:{size}"
            h.update(entry.encode())
    return h.hexdigest()


def scan_artifact_path(path: Path) -> dict:
    """Scan an artifact directory and return metadata summary."""
    total_size = 0
    file_count = 0
    files = {}
    config_hashes = {}

    config_names = {
        'config.json',
        'model.safetensors.index.json',
        'generation_config.json',
        'config.yaml',
        'tokenizer.json',
        'tokenizer_config.json',
        'special_tokens_map.json',
        'added_tokens.json',
    }

    for root, dirs, fnames in os.walk(path):
        dirs.sort()
        for fname in fnames:
            fp = Path(root) / fname
            try:
                size = fp.stat().st_size
            except OSError:
                continue
            total_size += size
            file_count += 1
            rel = str(fp.relative_to(path))
            files[rel] = size
            if fname in config_names:
                config_hashes[rel] = compute_file_hash(fp)

    return {
        "total_size_bytes": total_size,
        "file_count": file_count,
        "files": files,
        "config_hashes": config_hashes,
    }


def compute_artifact_fingerprint(artifact: Any, scan_path: Path,
                                  hash_weights: bool = False) -> dict:
    """Compute a fingerprint dict for an artifact.

    Returns dict with config_hash, manifest_hash, size, file_count, etc.
    """
    scan = scan_artifact_path(scan_path)

    fingerprint = {
        "config_file_hash": list(scan["config_hashes"].values())[0] if scan["config_hashes"] else None,
        "weight_manifest_hash": compute_weight_manifest_hash(scan_path),
        "total_size_bytes": scan["total_size_bytes"],
        "file_count": scan["file_count"],
    }

    # Try to detect model ID from config files
    for config_name, config_hash in scan["config_hashes"].items():
        config_file = scan_path / config_name
        if config_name == "config.json" and config_file.exists():
            try:
                with open(config_file) as f:
                    config_data = json.load(f)
                fingerprint["detected_model_id"] = config_data.get("_name_or_path") or config_data.get("model_id")
                fingerprint["detected_dtype"] = str(config_data.get("torch_dtype"))
                fingerprint["parameter_class"] = config_data.get("architectures", [None])[0]
            except Exception:
                pass

    if hash_weights:
        # Full weight file hashing (slow, only for final releases)
        weight_hashes = {}
        for root, dirs, fnames in os.walk(scan_path):
            dirs.sort()
            for fname in fnames:
                fp = Path(root) / fname
                try:
                    size = fp.stat().st_size
                except OSError:
                    continue
                if size > 100 * 1024 * 1024:  # Only hash files > 100MB
                    try:
                        weight_hashes[str(fp.relative_to(scan_path))] = compute_file_hash(fp)
                    except OSError:
                        pass
        fingerprint["weight_file_hashes"] = weight_hashes

    return fingerprint

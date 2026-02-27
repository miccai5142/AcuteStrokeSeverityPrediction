from __future__ import annotations

import importlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aispycore.config.schema import PipelineConfig


def save_run_manifest(
    output_dir: Path,
    config: PipelineConfig,
    cli_args: dict[str, Any],
) -> Path:
    """
    Write a machine-readable run manifest to output_dir/run_manifest.json.

    Args:
        output_dir: Root output directory for this run.
        config:     The base (pre-tuning) PipelineConfig.
        cli_args:   dict of CLI arguments as parsed by argparse (vars(args)).

    Returns:
        Path to the written manifest file.
    """
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cli_args": cli_args,
        "config": config.to_dict(),
        "software_versions": _collect_versions(),
        "git_commit": _get_git_commit(),
        "python_version": sys.version,
    }

    manifest_path = output_dir / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    return manifest_path


def _collect_versions() -> dict[str, str]:
    packages = ["tensorflow", "keras", "numpy", "pandas", "sklearn", "nibabel", "aispycore"]
    versions = {}
    for pkg in packages:
        try:
            mod = importlib.import_module(pkg if pkg != "sklearn" else "sklearn")
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not installed"
    return versions


def _get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unavailable"



import os 


def setup_logging(output_dir, model_name):
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, "logs", f"{model_name}_{timestamp}.log")
    return log_file


def write_log(log_file, msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{timestamp} - {msg}\n"
    with open(log_file, "a") as f:
        f.write(line)
    print(line)

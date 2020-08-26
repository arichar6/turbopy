import pytest
from pathlib import Path
import shutil


def pytest_unconfigure(config):
    """Finds the parent directory (turbopy) and removes the specified folders"""
    parent = Path(__file__).parents[0]
    if Path(f"{parent}/tmp").is_dir():
        shutil.rmtree(Path(f"{parent}/tmp").resolve())
    if Path(f"{parent}/default_output").is_dir():
        shutil.rmtree(Path(f"{parent}/default_output").resolve())

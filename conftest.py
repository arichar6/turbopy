from pathlib import Path
import shutil
import pytest


def pytest_unconfigure():
    """Locates and removes the specified directories after testing"""
    directories = ["tmp", "default_output"]
    parent = Path(__file__).parents[0]
    for file in directories:
        if Path(f"{parent}/{file}").is_dir():
            shutil.rmtree(Path(f"{parent}/{file}").resolve())

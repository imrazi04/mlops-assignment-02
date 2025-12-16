import os
import pytest
import sys

def test_data_file_exists():
    assert os.path.exists("data/dataset.csv")

def test_model_training_runs():
    import subprocess
    result = subprocess.run([sys.executable, "src/train.py"], capture_output=True, text=True)
    assert "Model saved" in result.stdout

def test_model_file_exists():
    assert os.path.exists("models/model.pkl")

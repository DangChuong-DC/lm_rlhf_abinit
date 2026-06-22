from pathlib import Path
from runpy import run_path


def load_python_config(config_path):
    config_path = Path(config_path)
    config = run_path(str(config_path))
    return {
        key: value
        for key, value in config.items()
        if not key.startswith("__")
    }

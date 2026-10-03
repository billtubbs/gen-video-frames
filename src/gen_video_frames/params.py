"""Loading the YAML parameters file that specifies a frame sequence."""

import yaml


def load_params(params_file):
    """Loads a YAML parameters file into a dict (see README.md for keys)."""
    with open(params_file) as f:
        return yaml.safe_load(f)

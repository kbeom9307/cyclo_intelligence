"""Versioned observation/model contracts, importable without a GPU framework."""
from pathlib import Path
import os
import sys


def prepare_jax_libraries():
    """Select the isolated venv's cuDNN before legacy torch modules are imported.

    The ROS base image bundles cuDNN 9.7; JAX's Blackwell build needs >=9.8.
    This affects only this venv, never the base interpreter or Isaac process.
    """
    if Path(sys.prefix).name != 'hil-serl-venv':
        return
    import ctypes
    libraries = list(Path(sys.prefix).glob('lib/python*/site-packages/nvidia/cudnn/lib/libcudnn.so.9'))
    if libraries:
        ctypes.CDLL(str(libraries[0]), mode=ctypes.RTLD_GLOBAL)

UPSTREAM = 'hil_serl'
LEGACY = 'legacy_torch'
UPSTREAM_SCHEMA = 'nist-sg2-rgb3x128-proprio22-delta6-fixed-hilserl-v1'
UPSTREAM_CLASSIFIER = 'nist-sg2-success-rgb3x128-resnet10-v1'
LEGACY_SCHEMA = 'nist-sg2-rgb3x64-proprio22-delta6-fixed-v3'
CAMERAS = ['head', 'left_wrist', 'right_wrist']


def upstream(config):
    # Configurations saved before this migration retain their original contract.
    return config.get('learning_backend', LEGACY) == UPSTREAM


def image_shape(config):
    size = 128 if upstream(config) else 64
    return (9, size, size)


def pixel_schema(config):
    return UPSTREAM_SCHEMA if upstream(config) else LEGACY_SCHEMA


def runtime_python(config):
    if upstream(config) and config.get('environment', 'isaac_sim') == 'isaac_sim':
        executable = Path(os.environ.get('CYCLO_HILSERL_PYTHON', '/opt/hil-serl-venv/bin/python'))
        if not executable.is_file():
            raise ValueError('Install the pinned HIL-SERL runtime before using the restored engine')
        return str(executable)
    import sys
    return sys.executable

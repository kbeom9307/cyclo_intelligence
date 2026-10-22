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


def shared_mujoco_gpu(config):
    actor = config.get('actor_device', 'cuda:0')
    return (config.get('environment') == 'mujoco'
            and config.get('learner_placement', 'local') == 'local'
            and actor.startswith('cuda:')
            and actor == config.get('learner_device', 'cuda:0'))


def jax_memory_fraction(config):
    # The combined MuJoCo/actor/classifier/learner benchmark uses an 80% pool.
    # This limits allocator caching, without reducing model size or batch size.
    return os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION',
                          '.80' if shared_mujoco_gpu(config) else '.90')


def image_shape(config):
    size = config.get('image_size', 128) if upstream(config) else 64
    if size not in (64, 128, 256):
        raise ValueError('Training image size must be 64, 128 or 256')
    return (9, size, size)


def pixel_schema(config):
    return UPSTREAM_SCHEMA.replace('3x128', f'3x{image_shape(config)[1]}') if upstream(config) else LEGACY_SCHEMA


def runtime_python(config):
    if upstream(config) and config.get('environment', 'isaac_sim') in ('isaac_sim', 'mujoco'):
        executable = Path(os.environ.get('CYCLO_HILSERL_PYTHON', '/opt/hil-serl-venv/bin/python'))
        if not executable.is_file():
            raise ValueError('Install the pinned HIL-SERL runtime before using the restored engine')
        return str(executable)
    import sys
    return sys.executable


def classifier_environment(device):
    env = dict(os.environ)
    env.pop('JAX_PLATFORM_NAME', None)
    env['JAX_PLATFORMS'] = 'cpu' if device == 'cpu' else 'cuda,cpu'
    env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    if device != 'cpu' and '--xla_gpu_autotune_level' not in env.get('XLA_FLAGS', ''):
        # Training needs tuned convolution kernels. Level 0 selected kernels
        # about 30x slower for batch 256 on the local RTX 5070 Laptop GPU.
        env['XLA_FLAGS'] = (env.get('XLA_FLAGS', '') + ' --xla_gpu_autotune_level=2').strip()
    return env


def classifier_compilation_cache(folder):
    """Reuse compilation across classifier jobs; no policy/optimizer data is cached."""
    import jax
    cache = Path(os.environ.get('CYCLO_HILSERL_CLASSIFIER_CACHE_DIR', str(Path(folder).parent / '.jax_cache')))
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    jax.config.update('jax_compilation_cache_dir', str(cache))
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.1)
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
    jax.config.update('jax_compilation_cache_max_size', 1024 ** 3)
    return str(cache)


def classifier_schema(config):
    return UPSTREAM_CLASSIFIER.replace('3x128', f'3x{image_shape(config)[1]}')

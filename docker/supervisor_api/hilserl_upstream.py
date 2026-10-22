"""Cyclo adapter for the pinned, vendored HIL-SERL JAX/Flax implementation.

The actor holds only upstream actor parameters. The learner runs the original
SAC losses, frozen ImageNet ResNet-10, spatial pooling, and random crop pipeline.
No TensorFlow or PyTorch CUDA context is created for this engine.
"""
import os
import json
from pathlib import Path
import sys

# JAX otherwise reserves 75% of the GPU before inference. Allocate on demand;
# this is not a model/batch reduction or a hard VRAM allocation.
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
sys.path.insert(0, str(Path(__file__).parent / 'third_party' / 'hil_serl'))

import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization, traverse_util
from flax.core import freeze, unfreeze
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.vision.data_augmentations import batched_random_crop
from hilserl_profiles import CAMERAS, UPSTREAM_CLASSIFIER, image_shape, classifier_schema
from hilserl_cameras import selected


def device_for(name):
    platform, _, index = str(name).partition(':')
    if platform not in ('cpu', 'cuda'):
        raise ValueError('Expected cpu or cuda:N')
    try:
        devices = jax.devices('gpu' if platform == 'cuda' else 'cpu')
        return devices[int(index or 0)]
    except (RuntimeError, IndexError) as exc:
        raise RuntimeError(f'JAX device {name} is unavailable; no CPU fallback') from exc


def memory(device):
    stats = device.memory_stats() or {}
    result = {'device': 'cpu' if device.platform == 'cpu' else f'cuda:{device.id}',
              'framework': 'JAX', 'name': device.device_kind}
    for source, dest in [('bytes_in_use', 'allocated_mib'), ('pool_bytes', 'reserved_mib'),
                         ('peak_bytes_in_use', 'peak_allocated_mib'), ('bytes_limit', 'allocator_limit_mib')]:
        if source in stats:
            result[dest] = round(stats[source] / 1024 ** 2, 1)
    if device.platform != 'cpu':
        from hilserl_devices import hardware_status
        gpu = next((g for g in hardware_status()['gpus'] if g['device'] == result['device']), {})
        result.update({k: gpu[k] for k in ('free_mib', 'total_mib') if k in gpu})
    return result


def export_tree(tree):
    return {key: np.asarray(value).copy() for key, value in
            traverse_util.flatten_dict(unfreeze(tree), sep='/').items()}


def restore_tree(template, arrays, device):
    expected = export_tree(template)
    if set(expected) != set(arrays) or any(expected[k].shape != np.shape(arrays[k]) for k in expected):
        raise ValueError('Checkpoint does not match the HIL-SERL ResNet-10 model/camera selection')
    if any(np.asarray(a).dtype.kind not in 'fiu' or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Invalid model parameters')
    return jax.device_put(traverse_util.unflatten_dict(dict(arrays), sep='/'), device)


def observations(states, images, cameras, size=128):
    states, images = np.asarray(states, np.float32), np.asarray(images, np.uint8)
    if states.ndim == 1:
        states, images = states[None], images[None]
    if images.shape != (len(states), 9, size, size):
        raise ValueError(f'HIL-SERL requires {size}×{size} camera observations; use matching datasets')
    result = {'state': states[:, None, :]}
    for camera in cameras:
        index = CAMERAS.index(camera)
        result[camera] = images[:, index * 3:index * 3 + 3].transpose(0, 2, 3, 1)[:, None]
    return freeze(result)


def augmentation(cameras):
    def augment(batch, rng):
        rng, obs_rng, next_rng = jax.random.split(rng, 3)
        replacements = {}
        for field, key in [('observations', obs_rng), ('next_observations', next_rng)]:
            value = batch[field]
            for camera in cameras:
                value = value.copy(add_or_replace={
                    camera: batched_random_crop(value[camera], key, padding=4, num_batch_dims=2)})
            replacements[field] = value
        return batch.copy(add_or_replace=replacements)
    return augment


def make_agent(config, observation_dim=22, action_dim=6):
    """Exact fixed-gripper launcher settings, initialized on CPU before placement."""
    cameras = tuple(selected(config, 'observation'))
    with jax.default_device(jax.devices('cpu')[0]):
        sample = observations(np.zeros((1, observation_dim), np.float32),
                              np.zeros((1, *image_shape(config)), np.uint8), cameras, image_shape(config)[1])
        network = {'activations': jax.nn.tanh, 'use_layer_norm': True, 'hidden_dims': [256, 256]}
        optimizer = {'learning_rate': config.get('learning_rate', 3e-4)}
        agent = SACAgent.create_pixels(
            jax.random.PRNGKey(config.get('seed', 0)), sample, np.zeros((1, action_dim), np.float32),
            encoder_type='resnet-pretrained', use_proprio=True, image_keys=cameras,
            policy_kwargs={'tanh_squash_distribution': True, 'std_parameterization': 'exp',
                           'std_min': 1e-5, 'std_max': 5},
            critic_network_kwargs=dict(network), policy_network_kwargs=dict(network),
            actor_optimizer_kwargs=optimizer, critic_optimizer_kwargs=optimizer,
            temperature_optimizer_kwargs=optimizer, temperature_init=1e-2,
            discount=config.get('discount', .97), backup_entropy=False,
            critic_ensemble_size=2, critic_subsample_size=None,
            augmentation_function=augmentation(cameras))
        # Upstream loads pretraining after create(); keep target initialization
        # consistent with the loaded frozen backbone as well.
        agent = agent.replace(state=agent.state.replace(target_params=agent.state.params))
    return agent


class UpstreamPolicy:
    def __init__(self, config, observation_dim=22, action_dim=6, *, agent=None, device=None):
        self.config, self.cameras = config, tuple(selected(config, 'observation'))
        self.device = device or device_for(config.get('actor_device', 'cuda:0'))
        agent = agent or make_agent(config, observation_dim, action_dim)
        self.params = jax.device_put({'modules_actor': agent.state.params['modules_actor']}, self.device)
        apply_fn = agent.state.apply_fn
        # Do not capture the agent/train state: actor needs no critics or optimizer.
        def apply(params, obs, rng, deterministic):
            dist = apply_fn({'params': params}, obs, name='actor', train=False)
            return dist.mode() if deterministic else dist.sample(seed=rng)
        self.apply = jax.jit(apply, static_argnums=(3,))
        self.rng = jax.device_put(jax.random.PRNGKey(config.get('seed', 0)), self.device)

    def action(self, state, images, deterministic=False):
        self.rng, key = jax.random.split(self.rng)
        obs = jax.device_put(observations(state, images, self.cameras, image_shape(self.config)[1]), self.device)
        return np.asarray(self.apply(self.params, obs, key, deterministic))[0]

    def export(self):
        return export_tree(self.params)

    def restore(self, arrays):
        self.params = restore_tree(self.params, arrays, self.device)

    def memory(self):
        return memory(self.device)


class UpstreamSAC:
    def __init__(self, config, observation_dim=22, action_dim=6):
        self.config = config
        self.device = device_for(config.get('learner_device', 'cuda:0'))
        self.agent = jax.device_put(make_agent(config, observation_dim, action_dim), self.device)
        self.policy = UpstreamPolicy(config, agent=self.agent, device=self.device)
        self.steps = 0
        self.critic_steps = 0

    def batch(self, rows):
        values = {key: np.stack([r[key] for r in rows]) for key in
                  ('observation', 'action', 'reward', 'next_observation', 'terminal', 'images', 'next_images')}
        return jax.device_put(freeze({
            'observations': observations(values['observation'], values['images'], self.policy.cameras, image_shape(self.config)[1]),
            'next_observations': observations(values['next_observation'], values['next_images'], self.policy.cameras, image_shape(self.config)[1]),
            'actions': values['action'].astype(np.float32),
            'rewards': values['reward'].astype(np.float32),
            'masks': 1. - values['terminal'].astype(np.float32),
        }), self.device)

    def update(self, rows, *, critic_only=False):
        networks = frozenset({'critic'}) if critic_only else frozenset({'critic', 'actor', 'temperature'})
        self.agent, metrics = self.agent.update(self.batch(rows), networks_to_update=networks)
        # Synchronize to bound asynchronous dispatch and report true peak usage.
        metrics = jax.device_get(metrics)
        self.critic_steps += 1
        if not critic_only:
            self.steps += 1
            self.policy.params = {'modules_actor': self.agent.state.params['modules_actor']}
        return {**{key: float(value) for key, value in metrics['critic'].items()},
                **({} if critic_only else {
                    **{key: float(value) for key, value in metrics['actor'].items()},
                    **{key: float(value) for key, value in metrics['temperature'].items()}}),
                'critic_steps': self.critic_steps}

    def restore_policy(self, arrays):
        self.policy.restore(arrays)
        params = {**self.agent.state.params, **self.policy.params}
        self.agent = self.agent.replace(state=self.agent.state.replace(params=params, target_params=params))

    def save(self, folder):
        folder = Path(folder)
        temporary = folder / 'learner.msgpack.tmp'
        temporary.write_bytes(serialization.to_bytes(jax.device_get(self.agent.state)))
        temporary.replace(folder / 'learner.msgpack')
        (folder / 'learner-state.json').write_text(json.dumps({'learner_steps': self.steps, 'critic_steps': self.critic_steps}))

    def restore_training(self, folder, counts):
        folder = Path(folder)
        state = serialization.from_bytes(self.agent.state, (folder / 'learner.msgpack').read_bytes())
        self.agent = self.agent.replace(state=jax.device_put(state, self.device))
        self.policy.params = {'modules_actor': self.agent.state.params['modules_actor']}
        saved = folder / 'learner-state.json'
        counts = json.loads(saved.read_text()) if saved.exists() else counts
        self.steps = int(counts['learner_steps'])
        self.critic_steps = int(counts['critic_steps'])

    def memory(self):
        return memory(self.device)


class UpstreamClassifier:
    def __init__(self, cameras=CAMERAS, device='cpu', seed=0, image_size=128):
        self.image_size = image_size
        from serl_launcher.networks.reward_classifier import create_classifier
        self.cameras, self.device = tuple(cameras), device_for(device)
        with jax.default_device(jax.devices('cpu')[0]):
            sample = observations(np.zeros((1, 22)), np.zeros((1, 9, image_size, image_size), np.uint8), self.cameras, image_size)
            self.state = create_classifier(jax.random.PRNGKey(seed), sample, self.cameras)
        self.state = jax.device_put(self.state, self.device)
        self.rng = jax.device_put(jax.random.PRNGKey(seed), self.device)
        self.infer = jax.jit(lambda params, obs: self.state.apply_fn({'params': params}, obs, train=False))

    def logits(self, images):
        images = np.asarray(images)
        batch = observations(np.zeros((len(images), 22)), images, self.cameras, self.image_size)
        return np.asarray(self.infer(self.state.params, jax.device_put(batch, self.device))).reshape(-1)

    def predict(self, images):
        logit = float(self.logits(np.asarray(images)[None])[0])
        return logit, float(1. / (1. + np.exp(-np.clip(logit, -80, 80))))

    def train_batch(self, images, labels):
        self.rng, key = jax.random.split(self.rng)
        batch = observations(np.zeros((len(images), 22)), images, self.cameras, self.image_size)
        self.state, loss = classifier_step(self.state, jax.device_put(batch, self.device),
                                          jax.device_put(np.asarray(labels, np.float32), self.device), key, self.cameras)
        return float(loss)

    def export(self):
        return export_tree(self.state.params)

    def restore(self, weights):
        self.state = self.state.replace(params=restore_tree(self.state.params, weights, self.device))

    @classmethod
    def load(cls, path, device='cpu'):
        import json
        with np.load(path, allow_pickle=False) as data:
            size = next((n for n in (64, 128, 256) if str(data['__schema__']) == classifier_schema({'learning_backend':'hil_serl', 'image_size':n})), None)
            if size is None:
                raise ValueError('Select a supported ResNet-10 classifier checkpoint; legacy CNN checkpoints require retraining')
            model = cls(json.loads(str(data['__cameras__'])), device, image_size=size)
            model.restore({k: data[k] for k in data.files if not k.startswith('__')})
        return model

    def memory(self):
        return memory(self.device)


from functools import partial
import optax


@partial(jax.jit, static_argnums=(4,))
def classifier_step(state, obs, labels, rng, cameras):
    rng, crop_rng, dropout_rng = jax.random.split(rng, 3)
    for camera in cameras:
        obs = obs.copy(add_or_replace={
            camera: batched_random_crop(obs[camera], crop_rng, padding=4, num_batch_dims=2)})
    def loss(params):
        logits = state.apply_fn({'params': params}, obs, train=True, rngs={'dropout': dropout_rng}).squeeze(-1)
        return optax.sigmoid_binary_cross_entropy(logits, labels).mean()
    value, grads = jax.value_and_grad(loss)(state.params)
    return state.apply_gradients(grads=grads), value

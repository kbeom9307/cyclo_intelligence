"""Opt-in live simulator test; temporary data only, no user collection changes."""
import json
import os
import time
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get('CYCLO_TEST_UPSTREAM_ISAAC') != '1',
                               reason='requires an idle running Isaac bridge')


def test_native_isaac_observations_actor_classifier_and_collection(tmp_path):
    from hilserl import ExperimentConfig
    from hilserl_isaac import IsaacNistEnv, health
    from hilserl_data import Collection
    from hilserl_profiles import UPSTREAM_SCHEMA
    from hilserl_upstream import UpstreamPolicy, UpstreamClassifier
    from hilserl_devices import hardware_status
    bridge = health()
    assert bridge['ready'] and not bridge['busy'] and 128 in bridge['pixel_sizes']
    c = ExperimentConfig(environment='isaac_sim', robot_type='ffw_sg2_rev1', control_hz=20,
                         learning_backend='hil_serl', learner_placement='remote',
                         translation_limit_m=.001, rotation_limit_rad=.01).model_dump()
    actor = UpstreamPolicy(c)
    classifier = UpstreamClassifier(c['classifier_cameras'], 'cuda:0')
    env = IsaacNistEnv(c)
    try:
        assert env.schema == UPSTREAM_SCHEMA
        assert env.images.shape == (9, 128, 128) and env.images.dtype == np.uint8
        assert env.observation().shape == (22,)
        assert all(env.images[start:start+3].std() > 1 for start in (0, 3, 6))
        sequence = env.capture_metadata['sequence']
        env.step(np.zeros(6, np.float32))
        assert env.capture_metadata['sequence'] > sequence
        times = []
        for _ in range(4):
            before = time.monotonic()
            action = actor.action(env.observation(), env.images)
            logit, probability = classifier.predict(env.classifier_images)
            times.append(time.monotonic() - before)
        assert action.shape == (6,) and np.isfinite(action).all() and np.abs(action).max() <= 1
        assert np.isfinite(logit) and 0 <= probability <= 1
        collection = Collection(tmp_path, c, 'classifier', env.schema)
        collection.capture(env, 0, 1, fresh=False)
        path = tmp_path / 'classifier_samples' / f"{collection.samples[0]['id']}.npz"
        with np.load(path, allow_pickle=False) as data:
            assert data['images'].shape == (9, 128, 128)
        manifest = json.loads((path.parent / 'manifest.json').read_text())
        assert manifest['learning_backend'] == 'hil_serl'
        result = {'schema': env.schema, 'image_shape': list(env.images.shape),
                  'warm_actor_and_classifier_ms': float(np.median(times[1:]) * 1000),
                  'actor_memory': actor.memory(), 'hardware': hardware_status(),
                  'overview_rendering': bridge['visualization']['overview_rendering']}
        from pathlib import Path
        Path('/tmp/hil-restored-isaac-integration.json').write_text(json.dumps(result, indent=2))
    finally:
        env.close()
    assert not health()['busy']

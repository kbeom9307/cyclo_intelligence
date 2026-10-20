import json
from pathlib import Path
import numpy as np
import pytest
import torch
from hilserl_pixels import PixelSAC, PixelPolicy, IMAGE_SHAPE, PIXEL_SCHEMA, SuccessClassifier
from hilserl_data import Collection, save_rows
from hilserl_classifier import train, split_groups

CONFIG = dict(environment='isaac_sim', robot_type='ffw_sg2_rev1', arm='right', target_part='peg_round_8', learning_rate=.0003, discount=.99)


def test_pixel_policy_and_encoder_learn_from_rgb():
    torch.set_num_threads(1); torch.manual_seed(7)
    model = PixelSAC(CONFIG)
    black = torch.zeros((1, *IMAGE_SHAPE), dtype=torch.uint8)
    white = torch.full_like(black, 255)
    obs = torch.zeros(1, 22)
    with torch.no_grad():
        a = model.policy.sample(obs, deterministic=True, images=black)[0]
        b = model.policy.sample(obs, deterministic=True, images=white)[0]
    assert not torch.allclose(a, b)
    before = model.policy.encoder.net[0].weight.detach().clone()
    rows = [dict(observation=np.zeros(22), action=np.zeros(6), reward=1., next_observation=np.ones(22)*.01,
                 terminal=False, images=white[0].numpy(), next_images=black[0].numpy()) for _ in range(4)]
    losses = model.update(rows)
    assert all(np.isfinite(value) for value in losses.values())
    assert not torch.equal(before, model.policy.encoder.net[0].weight)
    restored = PixelPolicy(); restored.restore(model.policy.export())
    with torch.no_grad():
        assert torch.equal(restored.sample(obs, True, white)[0], model.policy.sample(obs, True, white)[0])


def test_dataset_labels_and_demo_review(tmp_path):
    class Env:
        info = {'physics_steps': 1}
        images = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
        capture_metadata = {'sequence': 1}
        def capture(self): pass
        def observation(self): return np.zeros(22)
    c = Collection(tmp_path, CONFIG, 'classifier', PIXEL_SCHEMA)
    c.capture(Env(), 0, 1)
    with pytest.raises(ValueError, match='already labeled'): c.capture(Env(), 1, 1)
    d = Collection(tmp_path, CONFIG, 'demo', PIXEL_SCHEMA)
    d.rows = [dict(observation=np.zeros(22), next_observation=np.ones(22), action=np.ones(6)*.1,
                   images=np.zeros(IMAGE_SHAPE,dtype=np.uint8), next_images=np.ones(IMAGE_SHAPE,dtype=np.uint8),
                   reward=0., terminal=False, truncated=False, human=True)]
    d.save_demo(1); path = d.export()
    with np.load(path,allow_pickle=False) as data:
        assert data['images'].shape == (1,*IMAGE_SHAPE)
        assert data['terminal'][0] and data['reward'][0] == 10.
        assert str(data['__schema__']) == PIXEL_SCHEMA


def test_classifier_training_has_episode_separated_validation(tmp_path):
    # Synthetic fixtures test mechanics only, never stored in the user's HIL library.
    samples = tmp_path/'collection'; c = Collection(samples, CONFIG, 'classifier', PIXEL_SCHEMA)
    class Env:
        capture_metadata = {'sequence': 1}
        def capture(self): pass
        def observation(self): return np.zeros(22)
    env = Env(); rng = np.random.default_rng(9)
    for episode in range(4):
        for i in range(10):
            label = i % 2
            env.info = {'physics_steps': i}
            env.images = rng.integers(0, 20, IMAGE_SHAPE, dtype=np.uint8) + label * 220
            c.capture(env, label, episode)
    folder = tmp_path/'job'; folder.mkdir()
    (folder/'request.json').write_text(json.dumps(dict(collections=[str(samples)],task=CONFIG | {},seed=0,epochs=2,threshold=.5)))
    request = json.loads((folder/'request.json').read_text());request['task']={k:CONFIG[k] for k in ('environment','robot_type','arm','target_part')}
    (folder/'request.json').write_text(json.dumps(request));train(folder)
    status = json.loads((folder/'status.json').read_text())
    assert status['state']=='complete'
    assert not set(status['train_groups']) & set(status['validation_groups'])
    model = SuccessClassifier.load(folder/'classifier.npz')
    assert 0 <= model.probability(env.images) <= 1
    validation_images, validation_labels = [], []
    manifest = json.loads((samples/'classifier_samples/manifest.json').read_text())
    for sample in manifest['samples']:
        if f"{samples.name}:{sample['episode']}" in status['validation_groups']:
            with np.load(samples/'classifier_samples'/f"{sample['id']}.npz") as data:
                validation_images.append(data['images'])
                validation_labels.append(sample['label'])
    from hilserl_classifier import metrics
    with torch.no_grad():
        measured = metrics(model(torch.as_tensor(np.stack(validation_images))).sigmoid().numpy(), np.array(validation_labels), .5)
    assert measured['balanced_accuracy'] == status['checkpoint_metrics']['balanced_accuracy']
    assert 1 <= status['best_epoch'] <= 2
    with pytest.raises(ValueError): split_groups(np.array([0,1]), ['same','same'],0)


def test_isaac_pixel_input_excludes_privileged_state_and_rejects_stale_frames():
    import base64, zlib, time
    from hilserl_isaac import IsaacNistEnv
    from hilserl_pixels import CAMERAS
    env=IsaacNistEnv.__new__(IsaacNistEnv)
    env.pixel_mode=True; env.capture_metadata=None
    raw=np.arange(39,dtype=np.float32)
    image=np.arange(np.prod(IMAGE_SHAPE),dtype=np.uint8).reshape(IMAGE_SHAPE)
    result={'observation':raw.tolist(),'info':{'position_m':[0,0,0]},'pixels':{
        'encoding':'zlib-uint8','shape':list(IMAGE_SHAPE),'cameras':CAMERAS,'sequence':1,'timestamp':time.time(),
        'data':base64.b64encode(zlib.compress(image.tobytes())).decode()}}
    obs=env._accept(result)
    np.testing.assert_array_equal(obs,np.r_[raw[:21],raw[37:38]])
    np.testing.assert_array_equal(env.images,image)
    with pytest.raises(ValueError,match='Stale pixel'):env._accept(result)


def test_independent_camera_routing_and_artifact_compatibility(tmp_path):
    from hilserl_cameras import selected, route, validate_artifact
    config = {**CONFIG, 'observation_cameras': ['head'], 'classifier_cameras': ['right_wrist']}
    raw = np.arange(np.prod(IMAGE_SHAPE), dtype=np.uint8).reshape(IMAGE_SHAPE)
    policy_images = route(raw, selected(config, 'observation'))
    classification = route(raw, selected(config, 'classifier'))
    assert policy_images[:3].any() and not policy_images[3:].any()
    assert not classification[:6].any() and classification[6:].any()
    class Env:
        images = policy_images
        classifier_images = classification
        info = {'physics_steps': 1}
        capture_metadata = {'sequence': 1}
        def capture(self): pass
        def observation(self): return np.zeros(22)
    collection = Collection(tmp_path, config, 'classifier', PIXEL_SCHEMA)
    collection.capture(Env(), 1, 1)
    manifest = json.loads((tmp_path/'classifier_samples/manifest.json').read_text())
    assert manifest['cameras'] == ['right_wrist']
    with np.load(tmp_path/'classifier_samples'/f"{collection.last_capture['id']}.npz") as sample:
        np.testing.assert_array_equal(sample['images'], classification)
    validate_artifact({}, ['head', 'left_wrist', 'right_wrist'])
    with pytest.raises(ValueError, match='Artifact cameras'):
        validate_artifact({}, ['head'])
    validate_artifact({'__cameras__': np.array('["head"]')}, ['head'])


def test_camera_config_validation():
    from hilserl import ExperimentConfig
    from pydantic import ValidationError
    config = ExperimentConfig(observation_cameras=['right_wrist', 'head'], classifier_cameras=['left_wrist'])
    assert config.observation_cameras == ['head', 'right_wrist']
    assert config.classifier_cameras == ['left_wrist']
    for value in ([], ['head', 'head'], ['spectator']):
        with pytest.raises(ValidationError): ExperimentConfig(classifier_cameras=value)


def test_automatic_capture_reuses_step_images_and_keeps_each_step(tmp_path):
    class Env:
        images = np.ones(IMAGE_SHAPE, dtype=np.uint8)
        info = {'physics_steps': 1}
        capture_metadata = {'sequence': 1}
        def capture(self): raise AssertionError('Automatic collection must not render or step again')
        def observation(self): return np.zeros(22)
    env = Env(); c = Collection(tmp_path, CONFIG, 'classifier', PIXEL_SCHEMA)
    for step, label in enumerate([0, 1, 1, 0], 1):
        env.info = {'physics_steps': step}
        c.capture(env, label, 1, fresh=False, automatic=True)
    assert [s['label'] for s in c.samples] == [0, 1, 1, 0]
    assert all(s['source'] == 'step' for s in c.samples)
    assert len(list((tmp_path/'classifier_samples').glob('*.npz'))) == 4


def test_success_target_counts_only_positive_samples_and_prevents_overflow(tmp_path):
    from hilserl import ExperimentConfig
    from pydantic import ValidationError
    assert ExperimentConfig().classifier_success_target == 200
    for target in [0, 10001]:
        with pytest.raises(ValidationError): ExperimentConfig(classifier_success_target=target)
    class Env:
        images = np.ones(IMAGE_SHAPE, dtype=np.uint8)
        capture_metadata = {'sequence': 1}
        def capture(self): pass
        def observation(self): return np.zeros(22)
    env=Env(); c=Collection(tmp_path,{**CONFIG,'classifier_success_target':2},'classifier',PIXEL_SCHEMA)
    for step,label in enumerate([0,1,0,1]):
        env.info={'physics_steps':step}
        c.capture(env,label,1,fresh=False,automatic=True)
        assert c.target_reached == (step==3)
    assert c.state()['positive_samples']==2 and c.state()['success_target']==2
    manifest=json.loads((tmp_path/'classifier_samples/manifest.json').read_text())
    assert manifest['target_reached'] and len(manifest['samples'])==4
    env.info={'physics_steps':5}
    with pytest.raises(ValueError,match='target reached'):c.capture(env,1,1,fresh=False,automatic=True)

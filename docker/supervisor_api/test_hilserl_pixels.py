import json
from pathlib import Path
import numpy as np
import pytest
import torch
from hilserl_pixels import PixelSAC, PixelPolicy, IMAGE_SHAPE, PIXEL_SCHEMA, SuccessClassifier
from hilserl_data import Collection, ClassifierSamples, save_rows
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


@pytest.mark.parametrize('conflict_scope', [None, 'same_collection', 'other_collection'])
def test_classifier_training_has_global_label_stratified_validation(tmp_path, conflict_scope):
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
    (folder/'request.json').write_text(json.dumps(dict(collections=[str(samples)],task=CONFIG | {},seed=0,epochs=2,threshold=.5,device='cpu')))
    request = json.loads((folder/'request.json').read_text());request['task']={k:CONFIG[k] for k in ('environment','robot_type','arm','target_part')}
    originals = {}
    if conflict_scope:
        import hashlib
        from hilserl_data import write_npz
        target = samples if conflict_scope == 'same_collection' else tmp_path / 'other'
        if target != samples:
            Collection(target, CONFIG, 'classifier', PIXEL_SCHEMA)
            request['collections'].append(str(target))
        manifest_path = target / 'classifier_samples' / 'manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
        else:
            manifest = json.loads((samples / 'classifier_samples' / 'manifest.json').read_text())
            manifest['samples'] = []
        value = np.full(IMAGE_SHAPE, 123, dtype=np.uint8)
        key = hashlib.sha256(value.tobytes()).hexdigest()
        # A late opposite label must exclude earlier copies as well.
        for i, label in enumerate([0, 0, 1]):
            sample_id = f'conflict-{i}'
            path = target / 'classifier_samples' / f'{sample_id}.npz'
            path.parent.mkdir(parents=True, exist_ok=True)
            write_npz(path, images=value, label=np.array(label))
            manifest['samples'].append(dict(id=sample_id, label=label, image_hash=key, episode=5))
        manifest_path.write_text(json.dumps(manifest))
        for directory in request['collections']:
            for path in (Path(directory) / 'classifier_samples').iterdir():
                if path.is_file():
                    originals[path] = path.read_bytes()
    (folder/'request.json').write_text(json.dumps(request));train(folder)
    if conflict_scope:
        report = json.loads((folder / 'excluded_samples.json').read_text())
        assert len(report['samples']) == 3
        assert {item['sample_id'] for item in report['samples']} == {'conflict-0', 'conflict-1', 'conflict-2'}
        assert all(path.read_bytes() == data for path, data in originals.items())
    status = json.loads((folder/'status.json').read_text())
    assert status['state']=='complete'
    from hilserl_classifier import checkpoint_score
    assert status['checkpoint_selection'] == 'validation_f0_5_precision_loss_v1'
    assert status['best_epoch'] == max(status['history'], key=checkpoint_score)['epoch']
    assert status['checkpoint_metrics']['f0_5'] == status['best_metrics']['f0_5']
    assert status['excluded_conflicting_samples'] == (3 if conflict_scope else 0)
    if conflict_scope:
        assert any('Automatically excluded 3 samples' in warning for warning in status['warnings'])
    assert status['phase'] == 'Complete'
    assert status['loading_seconds'] >= 0 and status['initialization_seconds'] >= 0
    assert status['first_batch_seconds'] >= 0 and status['epoch_seconds'] > 0
    assert status['train_samples'] + status['validation_samples'] == 40
    assert status['validation_split'] == 'global_label_stratified_v1'
    model = SuccessClassifier.load(folder/'classifier.npz')
    assert 0 <= model.probability(env.images) <= 1
    validation_images, validation_labels = [], []
    manifest = json.loads((samples/'classifier_samples/manifest.json').read_text())
    split = json.loads((folder/'split.json').read_text())
    validation_hashes = {r['image_hash'] for r in split['samples'] if r['split'] == 'validation'}
    with ClassifierSamples(samples/'classifier_samples') as source:
        for sample in manifest['samples']:
            if sample['image_hash'] in validation_hashes:
                validation_images.append(source.read(sample)[0])
                validation_labels.append(sample['label'])
    from hilserl_classifier import metrics
    with torch.no_grad():
        measured = metrics(model(torch.as_tensor(np.stack(validation_images))).sigmoid().numpy(), np.array(validation_labels), .5)
    assert measured['balanced_accuracy'] == status['checkpoint_metrics']['balanced_accuracy']
    assert 1 <= status['best_epoch'] <= 2
    with pytest.raises(ValueError): split_groups(np.array([0,1]), ['same','same'],0)


def test_classifier_evaluation_padding_preserves_sample_order_and_metrics():
    from hilserl_classifier import fixed_batch_logits
    images = np.arange(15).reshape(5, 3)
    shapes = []
    class Model:
        def logits(self, batch):
            shapes.append(batch.shape)
            return batch[:, 0].astype(float)
    completed = []
    result = fixed_batch_logits(Model(), images, np.array([3, 0, 4, 1, 2]), 4, completed.append)
    np.testing.assert_array_equal(result, [9, 0, 12, 3, 6])
    assert shapes == [(4, 3), (4, 3)]
    assert completed == [4, 5]


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


def test_classifier_global_split_ignores_episodes_and_preserves_label_ratio():
    labels = np.array([0] * 52 + [1] * 24)
    groups = [f'episode-{i}' for i in range(len(labels))]
    keys = [f'image-{i:04d}' for i in range(len(labels))]
    train_ids, val_ids = split_groups(labels, groups, 0, keys)
    assert not set(train_ids) & set(val_ids)
    assert set(train_ids) | set(val_ids) == set(range(len(labels)))
    assert (labels[val_ids] == 0).sum() == 13
    assert (labels[val_ids] == 1).sum() == 6
    # All episodes contain only one label, even one step: all still participate.
    same_train, same_val = split_groups(labels, ['one-episode'] * len(labels), 0, keys)
    np.testing.assert_array_equal(train_ids, same_train)
    np.testing.assert_array_equal(val_ids, same_val)
    order = np.random.default_rng(5).permutation(len(labels))
    _, reordered = split_groups(labels[order], [groups[i] for i in order], 0, [keys[i] for i in order])
    assert {keys[order[i]] for i in reordered} == {keys[i] for i in val_ids}
    _, other_seed = split_groups(labels, groups, 1, keys)
    assert set(other_seed) != set(val_ids)


def test_classifier_global_split_small_dataset():
    labels = np.array([0, 0, 1, 1])
    train_ids, valid_ids = split_groups(labels, ['a', 'b', 'c', 'd'], 0)
    assert sorted(labels[train_ids]) == [0, 1]
    assert sorted(labels[valid_ids]) == [0, 1]
    with pytest.raises(ValueError, match='two unique images per label'):
        split_groups(np.array([0, 0, 1]), ['a', 'b', 'c'], 0)


def test_conflicting_images_excludes_all_copies_across_collections():
    from hilserl_classifier import conflicting_samples
    def sample(identifier, key, label):
        return dict(id=identifier, image_hash=key, label=label, episode=1)
    manifests = [(Path('a'), {'samples': [sample('a0', 'bad', 0), sample('a1', 'good', 1)]}),
                 (Path('b'), {'samples': [sample('b0', 'bad', 1), sample('b1', 'bad', 0), sample('b2', 'good', 1)]})]
    for ordered in (manifests, manifests[::-1]):
        conflicts, excluded = conflicting_samples(ordered)
        assert conflicts == {'bad'}
        assert {item['sample_id'] for item in excluded} == {'a0', 'b0', 'b1'}
    with pytest.raises(ValueError, match='Classifier labels'):
        conflicting_samples([(Path('a'), {'samples': [sample('invalid', 'bad', 2)]})])


def test_conflict_exclusion_reports_insufficient_remaining_data(tmp_path):
    collection = tmp_path / 'collection' / 'classifier_samples'
    collection.mkdir(parents=True)
    task = {key: CONFIG[key] for key in ('environment', 'robot_type', 'arm', 'target_part')}
    (collection / 'manifest.json').write_text(json.dumps(dict(task=task, samples=[
        dict(id=str(label), image_hash='ambiguous', label=label, episode=1) for label in (0, 1)])))
    folder = tmp_path / 'job'
    folder.mkdir()
    (folder / 'request.json').write_text(json.dumps(dict(collections=[str(collection.parent)], task=task,
                                                        seed=0, epochs=1, threshold=.5, device='cpu')))
    with pytest.raises(ValueError, match='After excluding conflicting labels and duplicates'):
        train(folder)
    status = json.loads((folder / 'status.json').read_text())
    assert status['excluded_conflicting_samples'] == 2
    assert len(json.loads((folder / 'excluded_samples.json').read_text())['samples']) == 2
    assert not (folder / 'classifier.npz').exists()


def test_f0_5_checkpoint_prefers_precision_over_balanced_accuracy():
    from hilserl_classifier import metrics, checkpoint_score
    labels = np.array([1] * 10 + [0] * 10)
    # Broad detector: TP=10, FP=2, BA=.9; conservative: TP=7, FP=0, BA=.85.
    broad = metrics(np.array([.9] * 12 + [.1] * 8), labels, .5)
    conservative = metrics(np.array([.9] * 7 + [.1] * 13), labels, .5)
    assert conservative['balanced_accuracy'] < broad['balanced_accuracy']
    assert conservative['f0_5'] == pytest.approx(35 / 38)
    assert checkpoint_score(conservative) > checkpoint_score(broad)
    empty = metrics(np.zeros(20), labels, .5)
    assert empty['f0_5'] == 0
    base = dict(f0_5=.9, precision=.95, validation_loss=.2)
    assert checkpoint_score(base) > checkpoint_score({**base, 'precision': .9, 'validation_loss': .1})
    assert checkpoint_score(base) > checkpoint_score({**base, 'validation_loss': .3})

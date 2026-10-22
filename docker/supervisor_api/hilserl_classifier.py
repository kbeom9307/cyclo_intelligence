"""Offline success CNN training job with globally label-stratified image validation."""
import json
import hashlib
from pathlib import Path
import sys
import time
import numpy as np
from hilserl_profiles import prepare_jax_libraries
prepare_jax_libraries()
import torch
from hilserl_pixels import SuccessClassifier, CLASSIFIER_SCHEMA
from hilserl_data import write_npz, ClassifierSamples
from hilserl_cameras import CAMERAS, route
from hilserl_profiles import upstream, image_shape, classifier_schema
from hilserl_classifier_jobs import status, TrainingProgress


def split_groups(labels, groups, seed, sample_keys=None):
    """Pool all episodes and hold out ~25% of each label's unique images.

    Groups are metadata only. Hash ordering makes identical inputs reproducible
    regardless of manifest order; changing a label's pool may change its split.
    """
    labels = np.asarray(labels)
    if len(labels) != len(groups) or (sample_keys is not None and len(sample_keys) != len(labels)):
        raise ValueError('Classifier split inputs have different lengths')
    buckets = {}
    for i, label in enumerate(labels):
        if label not in (0, 1):
            raise ValueError('Classifier labels must be success (1) or failure (0)')
        buckets.setdefault(int(label), []).append(i)
    mask = np.zeros(len(labels), dtype=bool)
    for label, ids in sorted(buckets.items()):
        if len(ids) < 2:
            continue
        if sample_keys is not None:
            ids.sort(key=lambda i: sample_keys[i])
        digest = hashlib.sha256(json.dumps([int(seed), label]).encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:16], 'big'))
        count = min(len(ids) - 1, max(1, round(len(ids) * .25)))
        mask[rng.permutation(ids)[:count]] = True
    if len(set(labels[mask])) != 2 or len(set(labels[~mask])) != 2:
        raise ValueError('Need at least two unique images per label across the selected datasets to split training and validation')
    return np.flatnonzero(~mask), np.flatnonzero(mask)


def metrics(probabilities, labels, threshold):
    predicted, truth = probabilities >= threshold, labels.astype(bool)
    tp, tn = int((predicted & truth).sum()), int((~predicted & ~truth).sum())
    fp, fn = int((predicted & ~truth).sum()), int((~predicted & truth).sum())
    recall = tp / max(1, tp + fn)
    p = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return {'precision': tp / max(1, tp + fp), 'recall': recall,
            'balanced_accuracy': .5 * (recall + tn / max(1, tn + fp)),
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'validation_loss': float(-(labels * np.log(p) + (1-labels) * np.log(1-p)).mean())}


def fixed_batch_logits(model, images, ids, batch_size, progress=None):
    """One compiled inference shape; discard padding before computing metrics."""
    logits = []
    for start in range(0, len(ids), batch_size):
        batch = images[ids[start:start + batch_size]]
        count = len(batch)
        if count < batch_size:
            batch = np.concatenate([batch, np.repeat(batch[-1:], batch_size - count, axis=0)])
        logits.append(model.logits(batch)[:count])
        if progress:
            progress(min(start + count, len(ids)))
    return np.concatenate(logits)


def conflicting_samples(manifests):
    """Exclude every occurrence of an ambiguous image, regardless of order."""
    labels_by_hash = {}
    for _, manifest in manifests:
        for sample in manifest['samples']:
            if sample['label'] not in (0, 1):
                raise ValueError('Classifier labels must be success (1) or failure (0)')
            labels_by_hash.setdefault(sample['image_hash'], set()).add(sample['label'])
    conflicts = {key for key, labels in labels_by_hash.items() if len(labels) > 1}
    excluded = [dict(collection=str(directory), sample_id=sample['id'],
                     image_hash=sample['image_hash'], label=sample['label'],
                     episode=sample['episode'])
                for directory, manifest in manifests for sample in manifest['samples']
                if sample['image_hash'] in conflicts]
    return conflicts, excluded


def train(folder):
    progress = TrainingProgress(folder)
    started = time.perf_counter()
    torch.set_num_threads(1)
    request = json.loads((folder / 'request.json').read_text())
    torch.manual_seed(request['seed'])
    rng = np.random.default_rng(request['seed'])
    cameras = request.get('cameras', CAMERAS)
    restored = upstream(request)
    device_name = request.get('device', 'cuda:0')
    expected_shape = image_shape(request)
    manifests = [(Path(directory), json.loads((Path(directory) / 'classifier_samples' / 'manifest.json').read_text())) for directory in request['collections']]
    total_samples = sum(len(manifest['samples']) for _, manifest in manifests)
    progress.update('Loading images', 0, total_samples, device=device_name)
    conflicts, excluded = conflicting_samples(manifests)
    warnings = []
    if excluded:
        warnings.append(f'Automatically excluded {len(excluded)} samples with conflicting labels '
                        f'({len(conflicts)} identical image groups) from training and validation. '
                        'Original collection data is unchanged.')
    exclusion_path = folder / 'excluded_samples.json'
    exclusion_path.write_text(json.dumps({'reason': 'identical_images_conflicting_labels',
                                         'samples': excluded}))
    status(folder, excluded_conflicting_samples=len(excluded),
           conflicting_image_groups=len(conflicts),
           excluded_samples_path=str(exclusion_path), warnings=warnings)
    images, labels, groups, hashes, sample_keys = [], [], [], {}, []
    loaded = 0
    for directory, manifest in manifests:
        if manifest['task'] != request['task']:
            raise ValueError('Classifier datasets contain different robot/arm/target tasks')
        if manifest.get('cameras', CAMERAS) != cameras:
            raise ValueError('Classifier datasets have different camera selections')
        with ClassifierSamples(directory / 'classifier_samples') as source:
            for sample in manifest['samples']:
                loaded += 1
                if loaded % 100 == 0:
                    progress.update('Loading images', loaded, total_samples)
                if sample['label'] not in (0, 1):
                    raise ValueError('Classifier labels must be success (1) or failure (0)')
                if sample['image_hash'] in conflicts or sample['image_hash'] in hashes:
                    continue
                hashes[sample['image_hash']] = sample['label']
                value, label = source.read(sample)
                if value.shape != expected_shape or value.dtype != np.uint8 or label != sample['label']:
                    raise ValueError('Invalid classifier sample')
                images.append(route(value, cameras))
                sample_keys.append(sample['image_hash'])
                labels.append(sample['label'])
                groups.append(f"{directory.name}:{sample['episode']}")
    labels = np.array(labels, dtype=np.int64)
    if any(int((labels == value).sum()) < 10 for value in (0, 1)):
        raise ValueError('After excluding conflicting labels and duplicates, at least 10 unique success and 10 unique failure images are required for training')
    train_ids, valid_ids = split_groups(labels, groups, request['seed'], sample_keys)
    split = {'method': 'global_label_stratified_v1', 'validation_fraction': .25,
             'seed': request['seed'], 'scope': 'all_selected_datasets'}
    validation_ids = set(valid_ids.tolist())
    split['samples'] = [{'image_hash': key, 'group': group, 'label': int(label),
                         'split': 'validation' if i in validation_ids else 'train'}
                        for i, (key, group, label) in enumerate(zip(sample_keys, groups, labels))]
    (folder / 'split.json').write_text(json.dumps(split))
    images = np.stack(images)
    loading_seconds = time.perf_counter() - started
    progress.update('Initializing model', 0, 1, loading_seconds=round(loading_seconds, 3))
    model_started = time.perf_counter()
    if restored:
        from hilserl_profiles import classifier_compilation_cache
        cache_directory = classifier_compilation_cache(folder)
        status(folder, compilation_cache=cache_directory)
        from hilserl_upstream import UpstreamClassifier
        model = UpstreamClassifier(cameras, device_name, request['seed'], image_size=expected_shape[1])
        device_name = 'cpu' if model.device.platform == 'cpu' else f'cuda:{model.device.id}'
        device_memory = model.memory
    else:
        from hilserl_devices import resolve_device, memory_status
        device = resolve_device(device_name)
        device_name = str(device)
        model = SuccessClassifier().to(device)
        device_memory = lambda: memory_status(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
    batch_size = 256 if restored else 32
    initialization_seconds = time.perf_counter() - model_started
    def predict(ids, phase):
        progress.update(phase, 0, len(ids))
        if restored:
            logits = fixed_batch_logits(model, images, ids, batch_size, lambda done: progress.update(phase, done, len(ids)))
            return 1. / (1. + np.exp(-np.clip(logits, -80, 80)))
        with torch.no_grad():
            values = []
            for i in range(0, len(ids), batch_size):
                values.append(model(torch.as_tensor(images[ids[i:i+batch_size]], device=device)).sigmoid())
                progress.update(phase, min(i + batch_size, len(ids)), len(ids))
            return torch.cat(values).cpu().numpy()
    classes = [train_ids[labels[train_ids] == value] for value in (0, 1)]
    status(folder, state='training', image_size=expected_shape[1], learning_backend=request.get('learning_backend', 'legacy_torch'), batch_size=batch_size, device=device_name, memory=device_memory(), train_samples=len(train_ids), validation_samples=len(valid_ids),
           initialization_seconds=round(initialization_seconds, 3),
           positive_samples=int(labels.sum()), negative_samples=int(len(labels) - labels.sum()),
           validation_split=split['method'], split_path=str(folder / 'split.json'),
           train_positive_samples=int(labels[train_ids].sum()), validation_positive_samples=int(labels[valid_ids].sum()),
           train_groups=sorted(set(groups[i] for i in train_ids)), validation_groups=sorted(set(groups[i] for i in valid_ids)))
    best_score = (-1., float('-inf'))
    best = None
    history = []
    path = folder / 'classifier.npz'
    for epoch in range(request['epochs']):
        epoch_started = time.perf_counter()
        if not restored:
            model.train()
        losses = []
        batches = max(1, int(np.ceil(len(train_ids) / batch_size)))
        progress.update('Preparing first GPU batch' if epoch == 0 and device_name.startswith('cuda:') else 'Training batches', 0, batches, current_epoch=epoch + 1)
        for batch in range(batches):
            batch_started = time.perf_counter()
            ids = np.r_[rng.choice(classes[0], batch_size // 2), rng.choice(classes[1], batch_size // 2)]
            if restored:
                losses.append(model.train_batch(images[ids], labels[ids]))
            else:
                logits = model(torch.as_tensor(images[ids], device=device))
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.as_tensor(labels[ids], dtype=torch.float32, device=device))
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                losses.append(float(loss.detach()))
            if epoch == 0 and batch == 0:
                status(folder, first_batch_seconds=round(time.perf_counter() - batch_started, 3))
            progress.update('Training batches', batch + 1, batches)
        training_seconds = time.perf_counter() - epoch_started
        if not restored:
            model.eval()
        validation_started = time.perf_counter()
        probabilities = predict(valid_ids, 'Validating held-out images')
        measured = metrics(probabilities, labels[valid_ids], request['threshold'])
        train_probabilities = predict(train_ids, 'Evaluating training images')
        train_metrics = metrics(train_probabilities, labels[train_ids], request['threshold'])
        validation_seconds = time.perf_counter() - validation_started
        record = {'epoch': epoch + 1, 'loss': float(np.mean(losses)), **measured,
                  'train_balanced_accuracy': train_metrics['balanced_accuracy'],
                  'training_seconds': round(training_seconds, 3), 'evaluation_seconds': round(validation_seconds, 3)}
        history.append(record)
        score = (measured['balanced_accuracy'], -measured['validation_loss'])
        if score > best_score:
            progress.update('Saving checkpoint', 0, 1)
            best_score = score
            best = dict(record)
            write_npz(path, __schema__=np.array(classifier_schema(request) if restored else CLASSIFIER_SCHEMA), __task__=np.array(json.dumps(request['task'], sort_keys=True)),
                      __cameras__=np.array(json.dumps(cameras)),
                      __crops__=np.array(json.dumps(request.get('camera_crops', {}), sort_keys=True)),
                      __training_metrics__=np.array(json.dumps(record, allow_nan=False)),
                      **(model.export() if restored else {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}))
        status(folder, **record, threshold=request['threshold'], best_epoch=best['epoch'],
               best_metrics=best, history=history, memory=device_memory(),
               epoch_seconds=round(time.perf_counter() - epoch_started, 3),
               elapsed_seconds=round(time.perf_counter() - started, 3))
    if best['recall'] == 0:
        warnings.append('Validation detected no successes. This checkpoint is not ready for reward use.')
    if best['train_balanced_accuracy'] - best['balanced_accuracy'] > .15:
        warnings.append('Training performance exceeds validation: collect more diverse episodes and review labels/camera visibility.')
    if best['balanced_accuracy'] <= .55:
        warnings.append('Validation is near chance. Review success/failure labels and selected camera views before more training.')
    status(folder, state='complete', phase='Complete', phase_completed=request['epochs'], phase_total=request['epochs'], checkpoint_path=str(path), warnings=warnings,
           checkpoint_metrics=best, best_epoch=best['epoch'])



if __name__ == '__main__':
    folder = Path(sys.argv[1])
    try:
        train(folder)
    except Exception as exc:
        status(folder, state='failed', error=str(exc))
        raise

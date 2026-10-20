"""Offline success CNN training job with episode-separated validation."""
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
from hilserl_pixels import SuccessClassifier, CLASSIFIER_SCHEMA, IMAGE_SHAPE
from hilserl_data import write_npz
from hilserl_cameras import CAMERAS, route


def status(folder, **values):
    path = folder / 'status.json'
    previous = json.loads(path.read_text()) if path.exists() else {}
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps({**previous, **values, 'updated_at': time.time()}, allow_nan=False))
    temporary.replace(path)


def split_groups(labels, groups, seed):
    unique = sorted(set(groups))
    rng = np.random.default_rng(seed)
    for _ in range(200):
        rng.shuffle(unique)
        validation = set(unique[:max(1, round(len(unique) * .25))])
        mask = np.array([g in validation for g in groups])
        if len(set(labels[mask])) == 2 and len(set(labels[~mask])) == 2:
            return np.flatnonzero(~mask), np.flatnonzero(mask)
    raise ValueError('Need separate episodes with both classes in training and validation; collect success and failure views across more episodes')


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


def train(folder):
    torch.set_num_threads(1)
    request = json.loads((folder / 'request.json').read_text())
    torch.manual_seed(request['seed'])
    rng = np.random.default_rng(request['seed'])
    cameras = request.get('cameras', CAMERAS)
    images, labels, groups, hashes = [], [], [], {}
    for directory in request['collections']:
        directory = Path(directory)
        manifest = json.loads((directory / 'classifier_samples' / 'manifest.json').read_text())
        if manifest['task'] != request['task']:
            raise ValueError('Classifier datasets contain different robot/arm/target tasks')
        if manifest.get('cameras', CAMERAS) != cameras:
            raise ValueError('Classifier datasets have different camera selections')
        for sample in manifest['samples']:
            if sample['label'] not in (0, 1):
                raise ValueError('Classifier labels must be success (1) or failure (0)')
            if sample['image_hash'] in hashes:
                if hashes[sample['image_hash']] != sample['label']:
                    raise ValueError('Identical images have conflicting labels across collections')
                continue
            hashes[sample['image_hash']] = sample['label']
            with np.load(directory / 'classifier_samples' / f"{sample['id']}.npz", allow_pickle=False) as data:
                value = data['images']
                if value.shape != IMAGE_SHAPE or value.dtype != np.uint8 or int(data['label']) != sample['label']:
                    raise ValueError('Invalid classifier sample')
                images.append(route(value, cameras))
            labels.append(sample['label'])
            groups.append(f"{directory.name}:{sample['episode']}")
    labels = np.array(labels, dtype=np.int64)
    if any(int((labels == value).sum()) < 10 for value in (0, 1)):
        raise ValueError('Collect at least 10 unique success and 10 unique failure images before training')
    train_ids, valid_ids = split_groups(labels, groups, request['seed'])
    images = np.stack(images)
    model = SuccessClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
    classes = [train_ids[labels[train_ids] == value] for value in (0, 1)]
    status(folder, state='training', train_samples=len(train_ids), validation_samples=len(valid_ids),
           positive_samples=int(labels.sum()), negative_samples=int(len(labels) - labels.sum()),
           train_groups=sorted(set(groups[i] for i in train_ids)), validation_groups=sorted(set(groups[i] for i in valid_ids)))
    best_score = (-1., float('-inf'))
    best = None
    history = []
    path = folder / 'classifier.npz'
    for epoch in range(request['epochs']):
        model.train()
        losses = []
        for _ in range(max(1, int(np.ceil(len(train_ids) / 32)))):
            ids = np.r_[rng.choice(classes[0], 16), rng.choice(classes[1], 16)]
            logits = model(torch.as_tensor(images[ids]))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.as_tensor(labels[ids], dtype=torch.float32))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            probabilities = torch.cat([model(torch.as_tensor(images[valid_ids[i:i+32]])).sigmoid() for i in range(0, len(valid_ids), 32)]).numpy()
        measured = metrics(probabilities, labels[valid_ids], request['threshold'])
        with torch.no_grad():
            train_probabilities = torch.cat([model(torch.as_tensor(images[train_ids[i:i+32]])).sigmoid() for i in range(0, len(train_ids), 32)]).numpy()
        train_metrics = metrics(train_probabilities, labels[train_ids], request['threshold'])
        record = {'epoch': epoch + 1, 'loss': float(np.mean(losses)), **measured,
                  'train_balanced_accuracy': train_metrics['balanced_accuracy']}
        history.append(record)
        score = (measured['balanced_accuracy'], -measured['validation_loss'])
        if score > best_score:
            best_score = score
            best = dict(record)
            write_npz(path, __schema__=np.array(CLASSIFIER_SCHEMA), __task__=np.array(json.dumps(request['task'], sort_keys=True)),
                      __cameras__=np.array(json.dumps(cameras)),
                      **{k: v.detach().numpy() for k, v in model.state_dict().items()})
        status(folder, **record, threshold=request['threshold'], best_epoch=best['epoch'],
               best_metrics=best, history=history)
    warnings = []
    if best['recall'] == 0:
        warnings.append('Validation detected no successes. This checkpoint is not ready for reward use.')
    if best['train_balanced_accuracy'] - best['balanced_accuracy'] > .15:
        warnings.append('Training performance exceeds validation: collect more diverse episodes and review labels/camera visibility.')
    if best['balanced_accuracy'] <= .55:
        warnings.append('Validation is near chance. Review success/failure labels and selected camera views before more training.')
    status(folder, state='complete', checkpoint_path=str(path), warnings=warnings,
           checkpoint_metrics=best, best_epoch=best['epoch'])



if __name__ == '__main__':
    folder = Path(sys.argv[1])
    try:
        train(folder)
    except Exception as exc:
        status(folder, state='failed', error=str(exc))
        raise

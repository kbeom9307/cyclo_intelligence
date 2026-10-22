"""Classifier job metadata, independent of the training frameworks."""
import json
import time


class TrainingProgress:
    def __init__(self, folder):
        self.folder = folder
        self.started = time.perf_counter()
        self.phase = None
        self.phase_started_at = time.time()

    def update(self, phase, completed, total, **values):
        if phase != self.phase or completed == 0:
            self.phase_started_at = time.time()
        self.phase = phase
        return status(self.folder, state='training', phase=phase, phase_completed=completed,
                      phase_total=total, phase_started_at=self.phase_started_at,
                      elapsed_seconds=round(time.perf_counter() - self.started, 3), **values)


def status(folder, **values):
    path = folder / 'status.json'
    previous = json.loads(path.read_text()) if path.exists() else {}
    value = {**previous, **values, 'updated_at': time.time()}
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False))
    temporary.replace(path)
    return value


def stopped_status(folder):
    """Call only after the worker exits; retain the last atomic checkpoint."""
    previous = json.loads((folder / 'status.json').read_text())
    checkpoint = folder / 'classifier.npz'
    values = {'state': 'stopped', 'stopped_at': time.time(), 'stop_reason': 'user'}
    if checkpoint.is_file():
        import numpy as np
        with np.load(checkpoint, allow_pickle=False) as data:
            # A stop can land between publishing the checkpoint and updating
            # the epoch status. Recover metrics from the same atomic artifact.
            metrics = (json.loads(str(data['__training_metrics__'])) if '__training_metrics__' in data
                       else previous.get('best_metrics'))
        values['checkpoint_path'] = str(checkpoint)
        if metrics:
            values.update(checkpoint_metrics=metrics, best_metrics=metrics, best_epoch=metrics['epoch'])
    return status(folder, **values)

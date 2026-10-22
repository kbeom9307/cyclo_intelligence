"""Validate immutable stopped-run snapshots before resuming into a new run."""
import json
from pathlib import Path


def validate_snapshot(folder, spec):
    folder = Path(folder)
    saved = json.loads((folder / 'spec.json').read_text())
    if any(saved[k] != spec[k] for k in ('pixel', 'schema', 'observation_dim', 'action_dim')):
        raise ValueError('Resume observation/action schema does not match')
    # Resume preserves the original optimizer, architecture and replay configuration.
    def normalized(config):
        return {'image_size': 128, **{k: v for k, v in config.items() if k != 'learner_device'}}
    if normalized(saved['config']) != normalized(spec['config']):
        raise ValueError('Resume learner configuration does not match the saved run')
    if not spec['pixel'] or spec['config'].get('learning_backend') != 'hil_serl':
        raise ValueError('Full resume requires HIL-SERL pixel training')
    for name in ('learner.msgpack', 'replay/online/index.json', 'replay/demonstrations/index.json'):
        if not (folder / name).is_file():
            raise ValueError(f'Resume checkpoint is incomplete: {name}')


def counters(status):
    metrics = status.get('metrics', {})
    completed = int(metrics.get('completed_episodes') or 0)
    return {'environment_steps': int(metrics.get('environment_steps') or 0),
            'human_steps': int(metrics.get('human_steps') or 0),
            'completed_episodes': completed,
            'successful_episodes': int(metrics.get('successful_episodes', round((metrics.get('success_rate') or 0) * completed))),
            'learner_steps': int(metrics.get('learner_steps') or 0),
            'critic_steps': int(metrics.get('critic_steps') or 0),
            'episode': int(status.get('episode') or 1) + 1}


def elapsed_active(session, status):
    from datetime import datetime
    total = session.get('resume', {}).get('active_seconds', 0.)
    start = None
    for event in session.get('events', []):
        at = datetime.fromisoformat(event['at'].replace('Z', '+00:00')).timestamp()
        if event['action'] in ('start', 'resume') and start is None:
            start = at
        if event['action'] in ('pause', 'stop', 'success', 'failure', 'truncate') and start is not None:
            total += max(0., at - start)
            start = None
    if start is not None:
        total += max(0., status.get('heartbeat', start) - start)
    return total

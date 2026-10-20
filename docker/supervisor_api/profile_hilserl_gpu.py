"""Reproducible upstream model VRAM probe. Run with Isaac stopped for isolation.

/opt/hil-serl-venv/bin/python profile_hilserl_gpu.py actor --output /tmp/actor.json
/opt/hil-serl-venv/bin/python profile_hilserl_gpu.py learner --batch 256 --output /tmp/learner.json
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import numpy as np


def probe(args):
    def used():
        return float(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                     '--format=csv,noheader,nounits'], text=True).splitlines()[0])
    baseline = used()
    from hilserl_upstream import UpstreamPolicy, UpstreamSAC, UpstreamClassifier, CAMERAS, jax
    config = {'learning_backend': 'hil_serl', 'actor_device': 'cuda:0', 'learner_device': 'cuda:0',
              'observation_cameras': CAMERAS[:args.cameras], 'seed': 0, 'discount': .97, 'learning_rate': .0003}
    samples, done = [], threading.Event()
    def monitor():
        while not done.wait(.1):
            samples.append(used())
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    result = {'mode': args.mode, 'batch': args.batch if args.mode != 'actor' else 1,
              'cameras': config['observation_cameras'], 'image_size': 128, 'baseline_mib': baseline}
    start = time.monotonic()
    rng = np.random.default_rng(0)
    try:
        if args.mode == 'actor':
            model = UpstreamPolicy(config)
            classifier = UpstreamClassifier(config['observation_cameras'], 'cuda:0') if args.with_classifier else None
            image = rng.integers(0, 256, (9, 128, 128), dtype=np.uint8)
            model.action(np.zeros(22), image)
            if classifier:
                classifier.predict(image)
            timings = []
            for _ in range(30):
                before = time.monotonic()
                model.action(np.zeros(22), image)
                if classifier:
                    classifier.predict(image)
                timings.append(time.monotonic() - before)
            result.update(inference_ms=float(np.median(timings) * 1000),
                          parameters=sum(x.size for x in model.export().values()), classifier=bool(classifier))
        elif args.mode == 'learner':
            model = UpstreamSAC(config)
            rows = [{'observation': rng.normal(size=22).astype(np.float32), 'action': np.zeros(6, np.float32),
                     'next_observation': rng.normal(size=22).astype(np.float32),
                     'reward': float(i % 2), 'terminal': bool(i % 2),
                     'images': rng.integers(0, 256, (9, 128, 128), dtype=np.uint8),
                     'next_images': rng.integers(0, 256, (9, 128, 128), dtype=np.uint8)} for i in range(args.batch)]
            batch = model.batch(rows)
            compiled = model.agent.update.lower(model.agent, batch).compile()
            analysis = compiled.memory_analysis()
            result['compiled_memory_mib'] = {k: getattr(analysis, k) / 1024 ** 2 for k in
                ('argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes', 'alias_size_in_bytes')}
            Path(args.output).write_text(json.dumps({**result, 'state': 'compiled'}, indent=2))
            if not args.compile_only:
                timings = []
                for _ in range(3):
                    before = time.monotonic()
                    model.update(rows, critic_only=True)
                    result['metrics'] = model.update(rows)
                    timings.append(time.monotonic() - before)
                result['warm_update_pair_ms'] = float(np.median(timings[1:]) * 1000)
        else:
            model = UpstreamClassifier(config['observation_cameras'], 'cuda:0')
            images = rng.integers(0, 256, (args.batch, 9, 128, 128), dtype=np.uint8)
            for _ in range(2):
                result['loss'] = model.train_batch(images, np.arange(args.batch) % 2)
        result.update(state='complete', device_memory=model.memory(), elapsed_s=time.monotonic() - start)
    except Exception as exc:
        result.update(state='failed', error=f'{type(exc).__name__}: {exc}', elapsed_s=time.monotonic() - start)
        raise
    finally:
        done.set()
        thread.join(timeout=2)
        result.update(steady_gpu_delta_mib=used() - baseline,
                      peak_gpu_delta_mib=max(samples or [baseline]) - baseline)
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['actor', 'learner', 'classifier'])
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--cameras', type=int, choices=[1, 2, 3], default=3)
    parser.add_argument('--with-classifier', action='store_true')
    parser.add_argument('--compile-only', action='store_true')
    parser.add_argument('--output', required=True)
    probe(parser.parse_args())

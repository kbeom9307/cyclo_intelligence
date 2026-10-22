"""Local replay reception must continue during slow GPU compilation/updates."""
import threading
import time

from hilserl_learner import AsyncLearner


def test_local_ingest_continues_during_update_and_pause_waits(tmp_path, monkeypatch):
    import hilserl_learner as module
    updating, finish_update, paused = threading.Event(), threading.Event(), threading.Event()
    received = []

    class Engine:
        def __init__(self, *args, **kwargs):
            self.steps = 0

        def ingest(self, rows):
            received.extend(rows)

        def replay_status(self):
            return bool(received), {'replay_size': len(received), 'demo_replay_size': 1,
                                    'replay_ready': bool(received)}

        def snapshot(self):
            return {'version': self.steps, 'metrics': {'learner_steps': self.steps}}, {'weight': self.steps}

        def exchange(self, rows, *args, **kwargs):
            self.ingest(rows)
            return self.snapshot()

        def learn_and_snapshot(self, version):
            updating.set()
            assert finish_update.wait(10)
            self.steps += 1
            return self.snapshot()

        def close(self):
            assert finish_update.is_set()
            return self.snapshot()

    monkeypatch.setattr(module, 'LearnerEngine', Engine)
    config = {'learner_placement': 'local', 'learning_backend': 'hil_serl'}
    spec = {'pixel': True, 'schema': 'test'}
    backend = AsyncLearner(config, spec, tmp_path, None, [])
    try:
        backend.insert({'id': 0})
        backend.tick(True)
        assert updating.wait(5)
        for start in range(1, 401, 100):
            for index in range(start, start+100):
                backend.insert({'id': index})
            deadline = time.monotonic()+5
            while len(received) < start+100 and time.monotonic() < deadline:
                time.sleep(.01)
            assert len(received) == start+100
        assert backend.snapshot()[0]['metrics']['learner_steps'] == 0
        waiter = threading.Thread(target=lambda: (backend.pause(), paused.set()))
        waiter.start()
        assert not paused.wait(.1)
        finish_update.set()
        assert paused.wait(5)
        waiter.join()
        metadata, weights = backend.close()
        assert metadata['metrics']['learner_steps'] >= 1
        assert weights['weight'] >= 1
        assert [row['id'] for row in received] == list(range(401))
    finally:
        finish_update.set()
        backend.close()

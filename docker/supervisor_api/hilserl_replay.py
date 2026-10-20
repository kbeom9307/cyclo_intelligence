"""Disk-backed CPU replay with the upstream 50% online / 50% demonstration mix."""
import json
from pathlib import Path
import threading
import numpy as np
from agentlace.data.data_store import DataStoreBase
from hilserl_profiles import image_shape


class DiskBank:
    def __init__(self, folder, capacity, observation_dim, action_dim, config):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.capacity, self.size, self.cursor = capacity, 0, 0
        shapes = {'observation': ((observation_dim,), np.float32),
                  'next_observation': ((observation_dim,), np.float32),
                  'action': ((action_dim,), np.float32), 'reward': ((), np.float32),
                  **{k: ((), np.bool_) for k in ('terminal', 'truncated', 'human')},
                  **{k: (image_shape(config), np.uint8) for k in ('images', 'next_images')}}
        self.arrays = {k: np.lib.format.open_memmap(self.folder / f'{k}.npy', mode='w+', dtype=dtype,
                                                   shape=(capacity, *shape))
                       for k, (shape, dtype) in shapes.items()}

    def insert(self, row):
        for key, array in self.arrays.items():
            array[self.cursor] = row.get(key, False)
        self.size = min(self.size + 1, self.capacity)
        self.cursor = (self.cursor + 1) % self.capacity

    def sample(self, count, rng):
        if not self.size:
            raise ValueError('Both online and demonstration replay must contain transitions')
        indices = rng.integers(0, self.size, size=count)
        return [{key: np.array(value[i], copy=True) for key, value in self.arrays.items()} for i in indices]

    def save(self):
        for array in self.arrays.values():
            array.flush()
        (self.folder / 'index.json').write_text(json.dumps({
            'capacity': self.capacity, 'size': self.size, 'cursor': self.cursor,
            'format': 'cyclo-memmap-replay-v1'}))


class DiskReplay(DataStoreBase):
    def __init__(self, folder, spec):
        capacity = spec['config']['pixel_replay_capacity']
        super().__init__(capacity)
        self.sample_lock = threading.Lock()
        self.latest_seq_id = -1
        self.rng = np.random.default_rng(spec['config'].get('seed', 0))
        args = (capacity, spec['observation_dim'], spec['action_dim'], spec['config'])
        self.online = DiskBank(Path(folder) / 'online', *args)
        self.demo = DiskBank(Path(folder) / 'demonstrations', *args)

    def insert(self, row):
        with self.sample_lock:
            self.latest_seq_id += 1
            self.online.insert(row)
            if row.get('human', False):
                self.demo.insert(row)

    def insert_demo(self, row):
        with self.sample_lock:
            self.demo.insert(row)

    def sample(self, size):
        with self.sample_lock:
            return self.online.sample(size // 2, self.rng) + self.demo.sample(size // 2, self.rng)

    def latest_data_id(self):
        return self.latest_seq_id

    def __len__(self):
        return self.online.size

    def ready(self, training_starts):
        return self.online.size >= training_starts and self.demo.size > 0

    def save(self):
        with self.sample_lock:
            self.online.save()
            self.demo.save()

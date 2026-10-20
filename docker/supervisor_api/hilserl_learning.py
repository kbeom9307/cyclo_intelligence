"""SAC and loopback-only Agentlace replay transport shared by both learner placements."""
import copy
import random
import threading
from collections import deque
import numpy as np
from hilserl_profiles import prepare_jax_libraries
prepare_jax_libraries()
import torch
from torch import nn
from torch.distributions import Normal
from agentlace import trainer
from agentlace.data.data_store import QueuedDataStore
from agentlace.internal.utils import make_compression_method
import zmq
from hilserl_devices import resolve_device

def mlp(inputs, outputs):
    return nn.Sequential(nn.Linear(inputs, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, outputs))


class Policy(nn.Module):
    def __init__(self, observation_dim=4, action_dim=3):
        super().__init__()
        self.net = mlp(observation_dim, action_dim * 2)

    def sample(self, obs, deterministic=False):
        mean, logstd = self.net(obs).chunk(2, dim=-1)
        dist = Normal(mean, logstd.clamp(-5, 2).exp())
        raw = mean if deterministic else dist.rsample()
        action = raw.tanh()
        logp = (dist.log_prob(raw) - torch.log(1 - action.square() + 1e-6)).sum(-1, keepdim=True)
        return action, logp

    def export(self):
        return {key: value.detach().cpu().numpy().copy() for key, value in self.state_dict().items()}

    def restore(self, weights):
        self.load_state_dict({key: torch.as_tensor(value.copy()) for key, value in weights.items()}, strict=True)


class SAC:
    def __init__(self, config, observation_dim=4, action_dim=3):
        self.config = config
        self.device = resolve_device(config.get('learner_device', 'cpu'))
        self.policy = Policy(observation_dim, action_dim)
        self.q1, self.q2 = mlp(observation_dim + action_dim, 1), mlp(observation_dim + action_dim, 1)
        self.policy.to(self.device)
        self.q1.to(self.device); self.q2.to(self.device)
        self.targets = [copy.deepcopy(self.q1), copy.deepcopy(self.q2)]
        self.qopt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=config['learning_rate'])
        self.popt = torch.optim.Adam(self.policy.parameters(), lr=config['learning_rate'])
        self.steps = 0

    def update(self, rows):
        obs, act, reward, nxt, terminal = [torch.as_tensor(np.stack([r[key] for r in rows]), dtype=torch.float32, device=self.device) for key in ('observation', 'action', 'reward', 'next_observation', 'terminal')]
        reward, terminal = reward[:, None], terminal[:, None]
        with torch.no_grad():
            next_action, logp = self.policy.sample(nxt)
            pair = torch.cat([nxt, next_action], -1)
            target = reward + self.config['discount'] * (1 - terminal) * (torch.minimum(*(q(pair) for q in self.targets)) - .2 * logp)
        pair = torch.cat([obs, act], -1)
        qloss = sum((q(pair) - target).square().mean() for q in (self.q1, self.q2))
        self.qopt.zero_grad(); qloss.backward(); self.qopt.step()
        for q in (self.q1, self.q2):
            q.requires_grad_(False)
        sampled, logp = self.policy.sample(obs)
        pair = torch.cat([obs, sampled], -1)
        ploss = (.2 * logp - torch.minimum(self.q1(pair), self.q2(pair))).mean()
        self.popt.zero_grad(); ploss.backward(); self.popt.step()
        for q, target_q in zip((self.q1, self.q2), self.targets):
            q.requires_grad_(True)
            with torch.no_grad():
                for target_p, param in zip(target_q.parameters(), q.parameters()):
                    target_p.lerp_(param, .005)
        self.steps += 1
        return {'critic_loss': float(qloss.detach()), 'actor_loss': float(ploss.detach())}


class Replay(QueuedDataStore):
    def __init__(self, capacity):
        super().__init__(capacity)
        self.interventions = deque(maxlen=capacity)
        self.sample_lock = threading.Lock()

    def insert(self, row):
        with self.sample_lock:
            super().insert(row)
            if row.get('human', False):
                self.interventions.append(row)

    def sample(self, size):
        with self.sample_lock:
            rows = list(self._data_queue)
            demos = list(self.interventions)
            count = size // 2 if demos else 0
            return random.choices(demos, k=count) + random.choices(rows, k=size-count)


# Upstream 0.1.4 binds all interfaces. Keep its protocol, but bind exclusively
# to ephemeral loopback ports; Agentlace uses pickle and must remain local.
class LocalReply(trainer.ReqRepServer):
    def reset(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.port = self.socket.bind_to_random_port('tcp://127.0.0.1')
        self.socket.setsockopt(zmq.RCVTIMEO, 250)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.is_kill = False


class LocalBroadcast(trainer.BroadcastServer):
    def __init__(self, port, **kwargs):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 3)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.port = self.socket.bind_to_random_port('tcp://127.0.0.1')
        self.compress, _ = make_compression_method('lz4')

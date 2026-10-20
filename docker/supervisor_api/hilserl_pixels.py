"""Small trainable multi-view CNNs for pixel SAC and success classification.

This is the local PyTorch SAC implementation, not upstream HIL-SERL/DrQ.
"""
import copy
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

CAMERAS = ['head', 'left_wrist', 'right_wrist']
IMAGE_SHAPE = (9, 64, 64)
PIXEL_SCHEMA = 'nist-sg2-rgb3x64-proprio22-delta6-fixed-v3'
CLASSIFIER_SCHEMA = 'nist-sg2-success-rgb3x64-v1'


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(9, 16, 5, stride=2), nn.ReLU(),
                                 nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
                                 nn.Conv2d(32, 32, 3, stride=2), nn.ReLU(),
                                 nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
                                 nn.Linear(512, 64), nn.LayerNorm(64), nn.Tanh())

    def forward(self, images):
        return self.net(images.float() / 255.)


class PixelPolicy(nn.Module):
    def __init__(self, observation_dim=22, action_dim=6):
        super().__init__()
        self.encoder = Encoder()
        self.net = nn.Sequential(nn.Linear(64 + observation_dim, 128), nn.ReLU(), nn.Linear(128, action_dim * 2))

    def sample(self, obs, deterministic=False, images=None):
        if images is None:
            raise ValueError('Pixel policy requires current camera images')
        mean, logstd = self.net(torch.cat((obs, self.encoder(images)), -1)).chunk(2, -1)
        dist = Normal(mean, logstd.clamp(-5, 1).exp())
        raw = mean if deterministic else dist.rsample()
        action = raw.tanh()
        logp = (dist.log_prob(raw) - torch.log(1 - action.square() + 1e-6)).sum(-1, keepdim=True)
        return action, logp

    def export(self):
        return {k: v.detach().cpu().numpy().copy() for k, v in self.state_dict().items()}

    def restore(self, weights):
        self.load_state_dict({k: torch.as_tensor(v.copy()) for k, v in weights.items()}, strict=True)


class PixelQ(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super().__init__()
        self.encoder = Encoder()
        self.net = nn.Sequential(nn.Linear(64 + observation_dim + action_dim, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, obs, action, images):
        return self.net(torch.cat((obs, action, self.encoder(images)), -1))


class PixelSAC:
    def __init__(self, config, observation_dim=22, action_dim=6):
        self.config = config
        self.policy = PixelPolicy(observation_dim, action_dim)
        self.q1, self.q2 = [PixelQ(observation_dim, action_dim) for _ in range(2)]
        self.targets = [copy.deepcopy(q) for q in (self.q1, self.q2)]
        self.qopt = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=config['learning_rate'])
        self.popt = torch.optim.Adam(self.policy.parameters(), lr=config['learning_rate'])
        self.steps = 0

    def update(self, rows):
        obs, act, reward, nxt, terminal = [torch.as_tensor(np.stack([r[k] for r in rows]), dtype=torch.float32)
                                          for k in ('observation', 'action', 'reward', 'next_observation', 'terminal')]
        images, next_images = [torch.as_tensor(np.stack([r[k] for r in rows])) for k in ('images', 'next_images')]
        with torch.no_grad():
            next_action, logp = self.policy.sample(nxt, images=next_images)
            target = reward[:, None] + self.config['discount'] * (1 - terminal[:, None]) * (
                torch.minimum(*(q(nxt, next_action, next_images) for q in self.targets)) - .2 * logp)
        qloss = sum((q(obs, act, images) - target).square().mean() for q in (self.q1, self.q2))
        self.qopt.zero_grad(); qloss.backward(); self.qopt.step()
        for q in (self.q1, self.q2): q.requires_grad_(False)
        action, logp = self.policy.sample(obs, images=images)
        ploss = (.2 * logp - torch.minimum(self.q1(obs, action, images), self.q2(obs, action, images))).mean()
        self.popt.zero_grad(); ploss.backward(); self.popt.step()
        for q, target_q in zip((self.q1, self.q2), self.targets):
            q.requires_grad_(True)
            with torch.no_grad():
                for target_p, p in zip(target_q.parameters(), q.parameters()): target_p.lerp_(p, .005)
        self.steps += 1
        return {'critic_loss': float(qloss.detach()), 'actor_loss': float(ploss.detach())}


class SuccessClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.head = nn.Linear(64, 1)

    def forward(self, images):
        return self.head(self.encoder(images)).squeeze(-1)

    @classmethod
    def load(cls, path):
        model = cls()
        with np.load(path, allow_pickle=False) as data:
            if str(data['__schema__']) != CLASSIFIER_SCHEMA:
                raise ValueError('Incompatible success classifier schema')
            weights = {k: torch.as_tensor(data[k].copy()) for k in data.files if not k.startswith('__')}
            if not all(torch.isfinite(v).all() for v in weights.values()):
                raise ValueError('Non-finite classifier weights')
            model.load_state_dict(weights, strict=True)
        return model.eval()

    def probability(self, images):
        with torch.no_grad():
            return float(self(torch.as_tensor(images)[None]).sigmoid()[0])

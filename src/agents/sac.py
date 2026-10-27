"""Soft Actor-Critic for every home at once, built so the federated methods can reuse it.

Every network has a leading agent dimension: agent i's weights are W[i], and all agents train in one batched step.
  local SAC    one agent per home, trained only on its own data
  central SAC  one agent trained on every home's data (no privacy), the learning upper reference
Each network is split into shared layers, which federated learning will average across homes, and a final local
layer that stays in the home.

Run: .venv/bin/python src/agents/sac.py [local|central] [training days] [seed]
"""
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from env.battery_env import TEST, TRAIN, BatteryStreet, evaluate


class Linear(nn.Module):
    """n independent linear layers in one batched matrix multiply. Input shape (n, batch, i)."""

    def __init__(self, n, i, o):
        super().__init__()
        bound = i ** -0.5  # same range as PyTorch's nn.Linear default
        self.w = nn.Parameter(torch.empty(n, i, o).uniform_(-bound, bound))
        self.b = nn.Parameter(torch.empty(n, 1, o).uniform_(-bound, bound))

    def forward(self, x):
        return torch.baddbmm(self.b, x, self.w)


class Net(nn.Module):
    def __init__(self, n, i, o, hidden):
        super().__init__()
        self.shared = nn.Sequential(Linear(n, i, hidden), nn.ReLU(), Linear(n, hidden, hidden), nn.ReLU())
        self.local = Linear(n, hidden, o)

    def forward(self, x):
        return self.local(self.shared(x))


class SAC:
    # ponytail: hidden=128 and batch=128 keep 36 agents trainable on a laptop CPU. Try 256 on a bigger machine.
    def __init__(self, n, obs_dim, hidden=128, lr=3e-4, gamma=0.99, tau=0.005, batch=128):
        self.actor = Net(n, obs_dim, 2, hidden)  # mean and log std of one action
        self.q = nn.ModuleList([Net(n, obs_dim + 1, 1, hidden) for _ in range(2)])
        self.q_target = copy.deepcopy(self.q)
        self.log_alpha = torch.zeros(n, 1, 1, requires_grad=True)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr)
        self.opt_q = torch.optim.Adam(self.q.parameters(), lr)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr)
        self.gamma, self.tau, self.batch = gamma, tau, batch
        self.target_entropy = -1.0  # one action per home
        self.mu, self.anchor = 0.0, None  # FedProx: pull on the shared layers towards the last global model

    def prox(self, nets):
        if not self.mu:
            return 0.0
        return self.mu / 2 * sum(((p - a) ** 2).sum()
                                 for net in nets for p, a in zip(net.shared.parameters(), self.anchor[net]))

    def sample(self, o):
        mu, log_std = self.actor(o).chunk(2, -1)
        std = log_std.clamp(-5, 2).exp()
        x = mu + std * torch.randn_like(mu)
        a = torch.tanh(x)
        logp = -0.5 * ((x - mu) / std) ** 2 - std.log() - 0.5 * np.log(2 * np.pi) - torch.log(1 - a ** 2 + 1e-6)
        return a, logp

    @torch.no_grad()
    def act(self, o, deterministic=False):
        return torch.tanh(self.actor(o).chunk(2, -1)[0]) if deterministic else self.sample(o)[0]

    @staticmethod
    def q_min(nets, o, a):
        oa = torch.cat([o, a], -1)
        return torch.min(nets[0](oa), nets[1](oa))

    def update(self, o, a, r, o2, done):
        """One gradient step for every agent. Losses are averaged over each agent's batch, then summed over agents,
        so each agent's gradients come only from its own data."""
        alpha = self.log_alpha.exp()
        with torch.no_grad():
            a2, logp2 = self.sample(o2)
            y = r + self.gamma * (1 - done) * (self.q_min(self.q_target, o2, a2) - alpha * logp2)
        oa = torch.cat([o, a], -1)
        loss_q = sum(F.mse_loss(q(oa), y, reduction="none").mean((1, 2)).sum() for q in self.q) + self.prox(self.q)
        self.opt_q.zero_grad()
        loss_q.backward()
        self.opt_q.step()

        a_new, logp = self.sample(o)
        loss_actor = (alpha.detach() * logp - self.q_min(self.q, o, a_new)).mean((1, 2)).sum() + self.prox([self.actor])
        self.opt_actor.zero_grad()
        loss_actor.backward()
        self.opt_actor.step()

        loss_alpha = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean((1, 2)).sum()
        self.opt_alpha.zero_grad()
        loss_alpha.backward()
        self.opt_alpha.step()

        with torch.no_grad():
            for p, p_target in zip(self.q.parameters(), self.q_target.parameters()):
                p_target.lerp_(p, self.tau)


class Buffer:
    """Each agent's own transitions, one tensor per field with a leading agent dimension."""

    def __init__(self, n, size, obs_dim):
        self.o, self.o2 = torch.zeros(n, size, obs_dim), torch.zeros(n, size, obs_dim)
        self.a, self.r, self.d = torch.zeros(n, size, 1), torch.zeros(n, size, 1), torch.zeros(n, size, 1)
        self.size, self.i, self.count = size, 0, 0

    def add(self, o, a, r, o2, d):  # each shaped (n, k, ...), k transitions per agent
        idx = (self.i + torch.arange(o.shape[1])) % self.size
        for store, x in ((self.o, o), (self.a, a), (self.r, r), (self.o2, o2), (self.d, d)):
            store[:, idx] = x
        self.i = (self.i + o.shape[1]) % self.size
        self.count = min(self.count + o.shape[1], self.size)

    def sample(self, batch):
        idx = torch.randint(0, self.count, (batch,))
        return self.o[:, idx], self.a[:, idx], self.r[:, idx], self.o2[:, idx], self.d[:, idx]


def layout(env, mode):
    """(agents, homes per agent): local has one agent per home, central has one agent for all homes."""
    return (env.n, 1) if mode == "local" else (1, env.n)


def train(env, mode="local", days=60, warmup_days=5, block_days=7, reward_scale=10.0, seed=0, server=None):
    """Train on random 7 day blocks of the training years, carrying battery charge across midnight.
    A federated server (src/agents/federated.py) gets the agents at the start and at the end of every day."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    n, per = layout(env, mode)
    shape = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32).reshape(n, per, -1)
    train_days = pd.date_range(*TRAIN, freq="D", inclusive="left")
    day = train_days[rng.integers(len(train_days) - block_days)]
    obs = env.reset(day)
    agent, buf = SAC(n, obs.shape[1]), Buffer(n, days * 48 * per, obs.shape[1])
    if server:
        server.start(agent)
    for d in range(days):
        for _ in range(48):
            o = shape(obs)
            a = torch.rand(n, per, 1) * 2 - 1 if d < warmup_days else agent.act(o)
            obs2, r, done, _ = env.step(a.reshape(-1).numpy())
            jump = 0.0
            if done and (d + 1) % block_days:  # same block: the next day follows on
                day += pd.Timedelta(days=1)
                obs2 = env.reset(day, env.soc)
            elif done:  # new block on a random day, so don't bootstrap across the jump
                jump = 1.0
                day = train_days[rng.integers(len(train_days) - block_days)]
                obs2 = env.reset(day)
            buf.add(o, a, shape(reward_scale * r), shape(obs2), torch.full((n, per, 1), jump))
            if d >= warmup_days:
                agent.update(*buf.sample(agent.batch))
            obs = obs2
        if server:
            server.end_of_day(agent, d)
    return agent


def policy(agent, env, mode):
    n, per = layout(env, mode)

    def act(env):
        o = torch.as_tensor(env.observe(), dtype=torch.float32).reshape(n, per, -1)
        return agent.act(o, deterministic=True).reshape(-1).numpy()
    return act


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "local"
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0  # training randomness only, the street stays the same
    env = BatteryStreet()
    start = time.time()
    agent = train(env, mode, days, seed=seed)
    print(f"{mode} SAC, seed {seed}: trained on {days} days in {time.time() - start:.0f} s")
    print(pd.Series(evaluate(env, policy(agent, env, mode), *TEST)).round(2).to_string())

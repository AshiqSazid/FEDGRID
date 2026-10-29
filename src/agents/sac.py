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
from env.battery_env import TEST, TRAIN, BatteryStreet, evaluate, self_consumption

# Networks and replay buffers go on a GPU when there is one. The street simulation always runs on the CPU.
if torch.cuda.is_available():
    torch.set_default_device("cuda")


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
        nn.init.zeros_(self.actor.local.w)  # starts at mean 0, which is the self consumption rule
        nn.init.zeros_(self.actor.local.b)
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


# The training years in 7 day blocks. Every 8th block (13 weeks across all seasons) is held out to pick the best policy,
# so the test year is never used for any choice.
WEEKS = pd.date_range(TRAIN[0], pd.Timestamp(TRAIN[1]) - pd.Timedelta(days=7), freq="7D")
VALID_WEEKS = WEEKS[4::8]
TRAIN_WEEKS = WEEKS.drop(VALID_WEEKS)


def battery_action(env, a):
    """The agent learns a correction to the self consumption rule (residual policy learning, Silver et al. 2018).
    Zero means follow the rule. With idle as the zero action, some seeds never learned to charge."""
    return np.clip(self_consumption(env) + a, -1, 1)


def layout(env, mode):
    """(agents, homes per agent): local has one agent per home, central has one agent for all homes."""
    return (env.n, 1) if mode == "local" else (1, env.n)


def validate(env, agent, mode):
    """Bill plus the curtailment penalty per home over the held out weeks, each starting half full. Lower is better.
    In a real federation each home would score its own held out weeks and send back one number."""
    act, score = policy(agent, env, mode), 0.0
    for week in VALID_WEEKS:
        r = evaluate(env, act, week, week + pd.Timedelta(days=7))
        score += r["bill $/home/yr"] + env.w_curtail * r["curtailed kWh/home/yr"]
    return score


class Learner:
    """SAC learning on random 7 day blocks of the training years, one day at a time, carrying battery charge across
    midnight. A federated server also runs one on its own reference street (src/agents/federated.py).

    poison = (homes, "meter" or "reward") for the data poisoning attacks, local mode only: those homes' meters report
    load and solar swapped, or their rewards have the wrong sign. Only what their agents see changes, not the street."""

    def __init__(self, env, mode, days, seed, warmup_days=5, reward_scale=10.0, poison=None):
        assert poison is None or mode == "local"
        self.env, self.warmup_days, self.reward_scale = env, warmup_days, reward_scale
        self.rng = np.random.default_rng(seed)
        self.n, self.per = layout(env, mode)
        self.day = TRAIN_WEEKS[self.rng.integers(len(TRAIN_WEEKS))]
        self.obs = env.reset(self.day)
        self.agent = SAC(self.n, self.obs.shape[1])
        self.buf = Buffer(self.n, days * 48 * self.per, self.obs.shape[1])
        self.bad, self.poison = (torch.as_tensor(poison[0]), poison[1]) if poison else (None, None)
        self.d = 0

    def shape(self, x):
        return torch.as_tensor(np.asarray(x), dtype=torch.float32).reshape(self.n, self.per, -1)

    def seen(self, obs):
        o = self.shape(obs)
        if self.poison == "meter":
            o[self.bad, :, 1:3] = o[self.bad, :, 1:3].flip(-1)  # load and solar readings swapped
        return o

    def train_day(self):
        env, agent, n, per, d = self.env, self.agent, self.n, self.per, self.d
        for _ in range(48):
            o = self.seen(self.obs)
            a = torch.rand(n, per, 1) * 2 - 1 if d < self.warmup_days else agent.act(o)
            obs2, r, done, _ = env.step(battery_action(env, a.reshape(-1).cpu().numpy()))
            jump = 0.0
            if done and (d + 1) % 7:  # same block: the next day follows on
                self.day += pd.Timedelta(days=1)
                obs2 = env.reset(self.day, env.soc)
            elif done:  # new block on a random day, so don't bootstrap across the jump
                jump = 1.0
                self.day = TRAIN_WEEKS[self.rng.integers(len(TRAIN_WEEKS))]
                obs2 = env.reset(self.day)
            r = self.shape(self.reward_scale * r)
            if self.poison == "reward":
                r[self.bad] = -r[self.bad]
            self.buf.add(o, a, r, self.seen(obs2), torch.full((n, per, 1), jump))
            if d >= self.warmup_days:
                agent.update(*self.buf.sample(agent.batch))
            self.obs = obs2
        self.d += 1


def train(env, mode="local", days=60, warmup_days=5, reward_scale=10.0, seed=0, server=None, valid_every=25,
          keep_best=True, poison=None):
    """Train on random 7 day blocks of the training years (see Learner).
    A federated server (src/agents/federated.py) gets the agents at the start and at the end of every day.
    Every valid_every days the policy is scored on the held out weeks, and the best policy is kept (the last one if
    keep_best is False, as in the attack study, where keeping the best would hide damage done late in training)."""
    torch.manual_seed(seed)
    learner = Learner(env, mode, days, seed, warmup_days, reward_scale, poison)
    agent = learner.agent
    if server:
        server.start(agent)
    best = (np.inf, 0, None)
    for d in range(days):
        learner.train_day()
        if server:
            server.end_of_day(agent, d)
        # Day 1 is warmup, so the first score is the rule itself and the kept policy never scores worse than the rule
        if d == 0 or (d + 1) % valid_every == 0:
            soc = env.soc.copy()
            score = validate(env, agent, mode)
            if score < best[0]:
                best = (score, d + 1, copy.deepcopy(agent.actor.state_dict()))
            print(f"day {d + 1}: validation {score:.2f} $/home, alpha {agent.log_alpha.exp().mean():.3f}", flush=True)
            learner.obs = env.reset(learner.day, soc)  # back to where training was
    if keep_best and best[2]:
        agent.actor.load_state_dict(best[2])
        print(f"kept the policy from day {best[1]}", flush=True)
    return agent


def policy(agent, env, mode):
    n, per = layout(env, mode)

    def act(env):
        o = torch.as_tensor(env.observe(), dtype=torch.float32).reshape(n, per, -1)
        return battery_action(env, agent.act(o, deterministic=True).reshape(-1).cpu().numpy())
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

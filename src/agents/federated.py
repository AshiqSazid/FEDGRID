"""Federated SAC: each home trains its own agent, and only the shared layers are combined across homes.

Each round (one day by default), every home sends the change in its shared layers since the last global model.
The server combines the changes with an aggregation rule, adds the result to the global model, and sends the new
shared layers back to every home. Each home's local layer, replay buffer and meter data never leave it.

  FedAvg   aggregation rule = mean of the changes
  FedProx  same, plus a penalty in local training that keeps the shared layers near the global model
  median, trimmed mean, Krum   standard robust aggregation rules, for the attack study
  FLTrust  trusts each change by how well it points the same way as the change of a few trusted homes (Cao et al., 2021)
  FLAME    keeps the majority cluster of models, clips the changes and adds noise (Nguyen et al., 2022)

Attacks: a fixed set of bad homes train honestly but send a corrupted change instead of their own.
  flip        minus their own change
  scale       their own change times the number of homes, so one home outweighs the whole street
  noise       random values, variance 1000 each (the random noise attack of Fang et al., 2025)
  normalized  the Normalized attack of Fang et al. (2025), which turns the aggregate's direction away from the honest
              one. It knows every home's change and the aggregation rule, the paper's default (full knowledge).

Run: .venv/bin/python src/agents/federated.py fedavg|fedprox|median|trimmed|krum|fltrust|flame [training days] [seed]
         [--attack none|flip|scale|noise|normalized] [--share 0.3] [--last]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import HDBSCAN

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.sac import policy, train
from env.battery_env import TEST, BatteryStreet, evaluate


# Aggregation rules. changes: (homes, parameters). Each returns the change to apply to the global model.
# The robust rules are told the share of bad homes to expect, as the server knows an upper bound in Fang et al. (2025).
def fedavg(changes):
    return changes.mean(0)


def median(changes):
    return changes.median(0).values


def trimmed_mean(changes, trim=0.2):
    k = round(trim * len(changes))
    return changes.sort(0).values[k:len(changes) - k].mean(0)


def krum(changes, assumed_bad=0.2):
    """The one change closest to its n - f - 2 nearest neighbours (Blanchard et al., 2017)."""
    n = len(changes)
    f = round(assumed_bad * n)
    distance = torch.cdist(changes, changes) ** 2
    return changes[distance.sort(1).values[:, 1:n - f - 1].sum(1).argmin()]  # column 0 is the distance to itself


def fltrust(changes, trusted):
    """Cao et al. (2021). The server's own change g0 comes from homes it trusts, which play the part of the paper's
    small clean root dataset. Each change is weighted by ReLU(cosine to g0) and rescaled to the length of g0."""
    g0 = changes[trusted].mean(0)
    if not g0.any():  # warmup days, before any learning
        return g0
    trust = torch.relu(torch.cosine_similarity(changes, g0[None], dim=1))
    scaled = changes * (g0.norm() / changes.norm(dim=1, keepdim=True).clamp_min(1e-12))
    return (trust[:, None] * scaled).sum(0) / trust.sum().clamp_min(1e-12)


# ponytail: the paper's lambda for image classification and NLP (0.01 for its intrusion detection case). Our task is
# none of these, so sweep it if FLAME's noise looks too strong or too weak.
FLAME_NOISE = 0.001


def flame(changes, global_model):
    """Nguyen et al. (2022), Algorithm 1. HDBSCAN on the cosine distances between local models, with minimum cluster
    size n/2 + 1 and min_samples 1, keeps the majority cluster. Admitted changes are clipped to the median change
    length over all homes, averaged, and Gaussian noise with sd lambda * that median is added."""
    n = len(changes)
    unit = F.normalize((global_model + changes).double(), dim=1)  # float64: models differ only slightly in angle
    distance = (1 - unit @ unit.T).clamp_min(0).numpy()
    labels = HDBSCAN(min_cluster_size=n // 2 + 1, min_samples=1, metric="precomputed",
                     allow_single_cluster=True, copy=False).fit_predict(distance)
    admitted = torch.as_tensor(labels >= 0)
    if not admitted.any():  # ponytail: the paper doesn't say what happens with no majority cluster, so keep them all
        admitted[:] = True
    norms = changes.norm(dim=1, keepdim=True)
    bound = norms.median()
    clipped = changes[admitted] * (bound / norms[admitted].clamp_min(1e-12)).clamp(max=1)
    return clipped.mean(0) + FLAME_NOISE * bound * torch.randn(changes.shape[1])


def aggregator(method, share, trusted=(), server=None):
    return {"fedavg": fedavg, "fedprox": fedavg, "median": median,
            "trimmed": lambda c: trimmed_mean(c, share), "krum": lambda c: krum(c, share),
            "fltrust": lambda c: fltrust(c, trusted), "flame": lambda c: flame(c, server.global_model)}[method]


# Attacks. Each gets every home's honest change, the bad homes and the aggregation rule, and returns the bad rows.
def sign_flip(changes, bad, aggregate):
    return -changes[bad]


def scaling(changes, bad, aggregate):
    return len(changes) * changes[bad]


def noise(changes, bad, aggregate):
    return 1000 ** 0.5 * torch.randn(len(bad), changes.shape[1])


# ponytail: the paper adjusts lambda and zeta with a step rule from starting values it doesn't give. This searches the
# same two quantities on fixed grids instead. Finer grids if the attack looks weak.
LAMBDAS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)  # weight of the push direction, relative to the honest direction
ZETAS = (0.5, 1.0, 2.0, 5.0, 10.0, 100.0)  # size of the bad change, relative to the median honest change


def normalized(changes, bad, aggregate):
    """Fang et al. (2025), Section 4: maximise || AR(honest)/|AR(honest)| - AR(attacked)/|AR(attacked)| ||.
    Stage 1 picks the direction honest/|honest| + lambda * delta with delta = -sign(mean change), stage 2 its size.
    Every bad home sends the same change."""
    before = aggregate(changes)
    if not before.any():  # nothing to turn yet (warmup days, before any learning)
        return changes[bad]
    before = before / before.norm()
    delta = -torch.sign(changes.mean(0))
    delta = delta / delta.norm()

    def deviation(row):
        attacked = changes.clone()
        attacked[bad] = row
        after = aggregate(attacked)
        return (before - after / (after.norm() + 1e-12)).norm()

    row = max((before + lam * delta for lam in LAMBDAS), key=deviation)
    size = changes.norm(dim=1).median()
    row = max((row / row.norm() * size * zeta for zeta in ZETAS), key=deviation)
    return row.expand(len(bad), -1)


ATTACKS = {"none": None, "flip": sign_flip, "scale": scaling, "noise": noise, "normalized": normalized}


class Server:
    def __init__(self, aggregate=fedavg, round_days=1, mu=0.0, attack=None, bad=()):
        self.aggregate, self.round_days, self.mu = aggregate, round_days, mu
        self.attack, self.bad = attack, torch.as_tensor(bad, dtype=torch.long)
        self.rounds = 0

    @staticmethod
    def nets(agent):
        return [agent.actor, *agent.q]

    def shared(self, agent):
        return [p for net in self.nets(agent) for p in net.shared.parameters()]

    def flat(self, agent):
        """Shared layers of every home, one row per home."""
        return torch.cat([p.detach().reshape(p.shape[0], -1) for p in self.shared(agent)], 1)

    def send(self, agent):
        """Copy the global shared layers into every home, and anchor FedProx on them."""
        i = 0
        with torch.no_grad():
            for p in self.shared(agent):
                k = p[0].numel()
                p.copy_(self.global_model[i:i + k].reshape(p.shape[1:]).expand_as(p))
                i += k
        agent.mu = self.mu
        agent.anchor = {net: [p.detach().clone() for p in net.shared.parameters()] for net in self.nets(agent)}

    def start(self, agent):
        self.global_model = self.flat(agent)[0]  # every home starts from the same shared layers
        self.send(agent)

    def end_of_day(self, agent, day):
        if (day + 1) % self.round_days:
            return
        changes = self.flat(agent) - self.global_model
        if self.attack:
            changes[self.bad] = self.attack(changes, self.bad, self.aggregate)
        self.global_model = self.global_model + self.aggregate(changes)
        self.send(agent)
        self.rounds += 1


def check():
    """Made up rounds: 29 honest homes near +1, 7 bad homes."""
    torch.manual_seed(0)
    changes = 1 + 0.1 * torch.randn(36, 100)
    attacked = changes.clone()
    attacked[:7] = -10
    assert fedavg(attacked).mean() < 0  # the mean is pulled the wrong way
    rules = {"median": median, "trimmed": trimmed_mean, "krum": krum,
             "fltrust": lambda c: fltrust(c, [35]), "flame": lambda c: flame(c, torch.ones(100))}
    for name, rule in rules.items():
        assert (rule(attacked) - 1).abs().mean() < 0.15, name
    # Warmup: honest changes are all zero, bad homes send noise. No rule may return NaN.
    warmup = torch.zeros(36, 100)
    warmup[:7] = torch.randn(7, 100)
    for name, rule in rules.items():
        assert rule(warmup).isfinite().all(), name
    # The Normalized attack turns the mean around with only 7 of 36 homes.
    bad = torch.arange(7)
    attacked[bad] = normalized(changes, bad, fedavg)
    assert torch.cosine_similarity(fedavg(attacked), fedavg(changes), 0) < 0


if __name__ == "__main__":
    check()
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=["fedavg", "fedprox", "median", "trimmed", "krum", "fltrust", "flame"])
    ap.add_argument("days", type=int, nargs="?", default=60)
    ap.add_argument("seed", type=int, nargs="?", default=0, help="training randomness and bad homes, not the street")
    ap.add_argument("--attack", choices=ATTACKS, default="none")
    ap.add_argument("--share", type=float, default=0.3, help="share of bad homes, and the share robust rules expect")
    ap.add_argument("--last", action="store_true", help="keep the last policy, not the best on the held out weeks")
    args = ap.parse_args()

    env = BatteryStreet()
    attack = ATTACKS[args.attack]
    bad = np.array([], int)
    if attack:
        bad = np.sort(np.random.default_rng(args.seed).choice(env.n, round(args.share * env.n), replace=False))
    # FLTrust's trusted homes are never bad: the operator controls them. Taking the lowest numbered honest homes keeps
    # the bad homes the same as for every other rule.
    # ponytail: 2 of 36 homes, as the paper's root dataset is small. Sweep 1 to 4 if FLTrust looks sensitive to it.
    trusted = np.setdiff1d(np.arange(env.n), bad)[:2]
    # ponytail: mu = 0.01 is a common FedProx starting value. Tune it with a small sweep.
    server = Server(mu=0.01 if args.method == "fedprox" else 0.0, attack=attack, bad=bad)
    server.aggregate = aggregator(args.method, args.share, trusted, server)
    start = time.time()
    agent = train(env, "local", args.days, seed=args.seed, server=server, keep_best=not args.last)
    size = server.global_model.numel()
    print(f"{args.method} SAC, seed {args.seed}, attack {args.attack} from homes {bad.tolist()}"
          f"{', last policy' if args.last else ''}{f', trusted homes {trusted.tolist()}' if args.method == 'fltrust' else ''}: trained on {args.days} days in {time.time() - start:.0f} s, "
          f"{server.rounds} rounds, {size} shared values per home per round ({size * 4 / 1024:.0f} KB)")

    # Check: every home holds the same shared layers (the kept policy may be from an earlier round than the critics).
    assert (server.flat(agent) == server.flat(agent)[0]).all()
    print(pd.Series(evaluate(env, policy(agent, env, "local"), *TEST)).round(2).to_string())

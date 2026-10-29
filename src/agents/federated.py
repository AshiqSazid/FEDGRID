"""Federated SAC: each home trains its own agent, and only the shared layers are combined across homes.

Each round (one day by default), every home sends the change in its shared layers since the last global model.
The server combines the changes with an aggregation rule, adds the result to the global model, and sends the new
shared layers back to every home. Each home's local layer, replay buffer and meter data never leave it.

  FedAvg   aggregation rule = mean of the changes
  FedProx  same, plus a penalty in local training that keeps the shared layers near the global model
  median, trimmed mean, Krum   standard robust aggregation rules
  FLTrust  weights each change by how well it points the same way as the server's own change (Cao et al., 2021)
  FLAME    keeps the majority cluster of models, clips the changes and adds noise (Nguyen et al., 2022)
  TAFDRL   style trust filter: Mahalanobis distance flags, Bayesian trust per home (Sepehrzad et al., 2026)
  FedGrid  this project's rule, the three steps of the research plan: clip, filter within solar size groups, and
           weight each group by agreement with the server's own change

The server's own change (FLTrust, FedGrid) comes from a reference street of other Ausgrid homes outside the
federation, trained from the global model every round, so the server never needs data from participating homes.

Attacks: a fixed set of bad homes. In the update attacks they train honestly but send a corrupted change.
  flip        minus their own change
  scale       their own change times the number of homes, so one home outweighs the whole street
  noise       random values, variance 1000 each (the random noise attack of Fang et al., 2025)
  normalized  the Normalized attack of Fang et al. (2025), which turns the aggregate's direction away from the honest
              one. It knows every home's change and the aggregation rule (full knowledge), so it adapts to each
              defence: it is also the adaptive attacker of the research plan.
In the data attacks they send honest changes learned from false data (src/agents/sac.py, Learner):
  meter       their meters report load and solar swapped
  reward      their rewards have the wrong sign
Message faults, for honest homes: --lost drops a change, --delayed sends last round's change instead.

Run: .venv/bin/python src/agents/federated.py fedavg|fedprox|median|trimmed|krum|fltrust|flame|tafdrl|fedgrid
         [training days] [seed] [--attack none|flip|scale|noise|normalized|meter|reward] [--share 0.3] [--last]
         [--mu 0.01] [--without clip|groups|trust] [--lost 0.1] [--delayed 0.1] [--feeder L]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import chi2
from sklearn.cluster import HDBSCAN

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.sac import Learner, policy, train
from env.battery_env import TEST, BatteryStreet, evaluate


# Aggregation rules. changes: (homes, parameters). Each returns the change to apply to the global model, and which
# homes' changes it used (None for rules that don't accept or reject whole changes).
# The robust rules are told the share of bad homes to expect, as the server knows an upper bound in Fang et al. (2025).
def fedavg(changes):
    return changes.mean(0), None


def median(changes):
    return changes.median(0).values, None


def trimmed_mean(changes, trim=0.2):
    k = round(trim * len(changes))
    return changes.sort(0).values[k:len(changes) - k].mean(0), None


def krum(changes, assumed_bad=0.2):
    """The one change closest to its n - f - 2 nearest neighbours (Blanchard et al., 2017)."""
    n = len(changes)
    f = round(assumed_bad * n)
    distance = torch.cdist(changes, changes) ** 2
    best = distance.sort(1).values[:, 1:n - f - 1].sum(1).argmin()  # column 0 is the distance to itself
    kept = torch.zeros(n, dtype=torch.bool)
    kept[best] = True
    return changes[best], kept


def fltrust(changes, g0):
    """Cao et al. (2021), Algorithm 2. g0 is the server's own change on its root data, here the reference street.
    Each change is weighted by ReLU(cosine to g0) and rescaled to the length of g0."""
    if not g0.any():  # warmup days, before any learning
        return g0, None
    trust = torch.relu(torch.cosine_similarity(changes, g0[None], dim=1))
    scaled = changes * (g0.norm() / changes.norm(dim=1, keepdim=True).clamp_min(1e-12))
    return (trust[:, None] * scaled).sum(0) / trust.sum().clamp_min(1e-12), trust > 0


# ponytail: the paper's lambda for image classification and NLP (0.01 for its intrusion detection case). Our task is
# none of these, so sweep it if FLAME's noise looks too strong or too weak.
FLAME_NOISE = 0.001


def flame(changes, global_model):
    """Nguyen et al. (2022), Algorithm 1. HDBSCAN on the cosine distances between local models, with minimum cluster
    size n/2 + 1 and min_samples 1, keeps the majority cluster. Admitted changes are clipped to the median change
    length over all homes, averaged, and Gaussian noise with sd lambda * that median is added."""
    n = len(changes)
    unit = F.normalize((global_model + changes).double(), dim=1)  # float64: models differ only slightly in angle
    distance = (1 - unit @ unit.T).clamp_min(0).cpu().numpy()
    labels = HDBSCAN(min_cluster_size=n // 2 + 1, min_samples=1, metric="precomputed",
                     allow_single_cluster=True, copy=False).fit_predict(distance)
    admitted = torch.as_tensor(labels >= 0)
    if not admitted.any():  # ponytail: the paper doesn't say what happens with no majority cluster, so keep them all
        admitted[:] = True
    norms = changes.norm(dim=1, keepdim=True)
    bound = norms.median()
    clipped = changes[admitted] * (bound / norms[admitted].clamp_min(1e-12)).clamp(max=1)
    return clipped.mean(0) + FLAME_NOISE * bound * torch.randn(changes.shape[1]), admitted


class TrustFilter:
    """TAFDRL-style trust filter (Sepehrzad et al., Applied Energy 2026), rebuilt from the paper's abstract since the
    full text was not available, so the embedding and threshold are our choices. Each round's changes are embedded in
    their first principal components. A change whose robust Mahalanobis distance from the median is beyond the
    chi-square 99% point is flagged. Each home keeps a Bayesian trust, the mean of a Beta posterior over its history
    of flags, and the unflagged changes are averaged with trust weights."""

    def __init__(self, n, dims=3, level=0.99):
        self.passed, self.flagged = torch.ones(n), torch.ones(n)  # Beta(1, 1) prior
        self.dims, self.limit = dims, chi2.ppf(level, dims)

    def __call__(self, changes, homes, g0=None):
        if not changes.any():  # warmup days, before any learning
            return changes.mean(0), None
        centred = changes - changes.median(0).values
        z = centred @ torch.pca_lowrank(centred, q=self.dims, center=False)[2]
        z = z - z.median(0).values
        # Principal components are close to uncorrelated, so a diagonal robust covariance (median absolute deviation)
        spread = 1.4826 * z.abs().median(0).values
        kept = ((z / spread.clamp_min(1e-12)) ** 2).sum(1) <= self.limit
        w = kept * self.passed[homes] / (self.passed[homes] + self.flagged[homes])
        return (w[:, None] * changes).sum(0) / w.sum().clamp_min(1e-12), kept

    def commit(self, kept, homes):
        """Called after the real aggregation only, so an attacker's trial runs of the rule change no trust."""
        if kept is not None:
            self.passed[homes] += kept
            self.flagged[homes] += ~kept


def merge_small(groups, min_group):
    """Merge any group smaller than min_group into the nearest band, until all are big enough or one is left."""
    groups = groups.clone()
    while True:
        values, counts = groups.unique(return_counts=True)
        if len(values) == 1 or counts.min() >= min_group:
            return groups
        small = values[counts.argmin()]
        others = values[values != small]
        groups[groups == small] = others[(others - small).abs().argmin()]


# ponytail: min_group and z_max are starting values, not tuned.
def fedgrid(changes, bands, g0=None, without=None, min_group=5, z_max=2.5):
    """This project's rule, the three steps of the research plan.
    1. Clip every change to the median change length, so no single home can dominate.
    2. Group homes by solar size band (given at sign up). Within each group, drop changes far from the group's
       median: robust z score of the distance above z_max. Small groups merge into the nearest band.
    3. Weight each group by its size times ReLU(cosine of its mean change to the server's own change g0).
    without switches one step off for the ablation: clip, groups (filter across all homes at once) or trust."""
    if without != "clip":
        norms = changes.norm(dim=1, keepdim=True)
        changes = changes * (norms.median() / norms.clamp_min(1e-12)).clamp(max=1)
    groups = torch.zeros_like(bands) if without == "groups" else merge_small(bands, min_group)
    kept = torch.ones(len(changes), dtype=torch.bool)
    total, weight = torch.zeros_like(changes[0]), torch.zeros(())
    for g in groups.unique():
        members = (groups == g).nonzero().squeeze(1)
        x = changes[members]
        d = (x - x.median(0).values).norm(dim=1)
        out = d - d.median() > z_max * 1.4826 * (d - d.median()).abs().median().clamp_min(1e-12)
        kept[members[out]] = False
        mean = x[~out].mean(0)
        w = (~out).sum() * (torch.relu(torch.cosine_similarity(mean, g0, 0)) if without != "trust" else 1.0)
        total, weight = total + w * mean, weight + w
    return total / weight.clamp_min(1e-12), kept


def aggregator(method, share, server, bands=None, without=None):
    """The rule as the server calls it: rule(changes, homes the changes came from, the server's own change)."""
    if method == "tafdrl":
        return TrustFilter(server.n)
    return {"fedavg": lambda c, h, g0: fedavg(c), "fedprox": lambda c, h, g0: fedavg(c),
            "median": lambda c, h, g0: median(c), "trimmed": lambda c, h, g0: trimmed_mean(c, share),
            "krum": lambda c, h, g0: krum(c, share), "fltrust": lambda c, h, g0: fltrust(c, g0),
            "flame": lambda c, h, g0: flame(c, server.global_model),
            "fedgrid": lambda c, h, g0: fedgrid(c, bands[h], g0, without)}[method]


# Update attacks. Each gets every home's honest change, the bad homes and the aggregation rule, and returns the bad rows.
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
DATA_ATTACKS = ("meter", "reward")


class Server:
    def __init__(self, n, aggregate=None, round_days=1, mu=0.0, attack=None, bad=(), reference=None, lost=0.0,
                 delayed=0.0, seed=0):
        self.n, self.aggregate, self.round_days, self.mu = n, aggregate, round_days, mu
        self.attack, self.bad = attack, torch.as_tensor(bad, dtype=torch.long)
        self.reference = reference  # Learner on the reference street, for the server's own change
        self.lost, self.delayed, self.rng, self.previous = lost, delayed, np.random.default_rng(seed), None
        self.rounds = 0
        # For rules that accept or reject whole changes: changes received and rejected per home after warmup, and
        # delayed honest changes received and rejected
        self.received, self.rejected, self.late = torch.zeros(n), torch.zeros(n), torch.zeros(2)

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
        learning = changes.any()  # false in the warmup days
        g0 = None
        if self.reference:  # the server's own change, from the same global model, on the reference street
            self.send(self.reference.agent)
            self.reference.train_day()
            g0 = self.flat(self.reference.agent)[0] - self.global_model
        homes = torch.arange(self.n)

        def rule(c, h=homes):
            return self.aggregate(c, h, g0)

        if self.attack:
            changes[self.bad] = self.attack(changes, self.bad, lambda c: rule(c)[0])
        late = torch.zeros(self.n, dtype=torch.bool)
        if self.lost or self.delayed:  # message faults, for honest homes only
            u = torch.as_tensor(self.rng.random(self.n))
            honest = torch.ones(self.n, dtype=torch.bool)
            honest[self.bad] = False
            sent = changes.clone()
            if self.previous is not None:
                late = honest & (u < self.delayed)
                changes[late] = self.previous[late]
            self.previous = sent
            arrived = ~(honest & (u >= self.delayed) & (u < self.delayed + self.lost))
            changes, homes, late = changes[arrived], homes[arrived], late[arrived]
        change, kept = rule(changes, homes)
        if hasattr(self.aggregate, "commit"):
            self.aggregate.commit(kept, homes)
        if kept is not None and learning:
            self.received[homes] += 1
            self.rejected[homes] += ~kept
            self.late += torch.stack([late.sum(), (late & ~kept).sum()])
        self.global_model = self.global_model + change
        self.send(agent)
        self.rounds += 1


def check():
    """Made up rounds: 29 honest homes near +1, 7 bad homes."""
    torch.manual_seed(0)
    n = 36
    changes = 1 + 0.1 * torch.randn(n, 100)
    attacked = changes.clone()
    attacked[:7] = -10
    homes, g0, bands = torch.arange(n), torch.ones(100), torch.arange(n) % 3
    assert fedavg(attacked)[0].mean() < 0  # the mean is pulled the wrong way
    rules = {"median": median, "trimmed": trimmed_mean, "krum": krum, "fltrust": lambda c: fltrust(c, g0),
             "flame": lambda c: flame(c, torch.ones(100)), "tafdrl": lambda c: TrustFilter(n)(c, homes),
             **{f"fedgrid without {w}": lambda c, w=w: fedgrid(c, bands, g0, w) for w in (None, "clip", "groups",
                                                                                         "trust")}}
    for name, rule in rules.items():
        change, kept = rule(attacked)
        assert (change - 1).abs().mean() < 0.15, name
        if kept is not None:  # bad homes rejected and honest homes kept: Krum keeps only one home, and FLAME only
            # promises a majority (its most stable HDBSCAN cluster, here about 20 of the 29 honest homes)
            assert not kept[:7].any() and kept[7:].float().mean() > {"krum": 0, "flame": 0.5}.get(name, 0.8), name
    # Warmup: honest changes are all zero, bad homes send noise. No rule may return NaN.
    warmup = torch.zeros(n, 100)
    warmup[:7] = torch.randn(7, 100)
    for name, rule in rules.items():
        assert rule(warmup)[0].isfinite().all(), name
    # Small groups merge into the nearest band
    assert merge_small(torch.tensor([0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2]), 5).tolist() == [0] * 6 + [2] * 5
    # The Normalized attack turns the mean around with only 7 of 36 homes.
    bad = torch.arange(7)
    attacked[bad] = normalized(changes, bad, lambda c: fedavg(c)[0])
    assert torch.cosine_similarity(fedavg(attacked)[0], fedavg(changes)[0], 0) < 0


if __name__ == "__main__":
    check()
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=["fedavg", "fedprox", "median", "trimmed", "krum", "fltrust", "flame", "tafdrl",
                                       "fedgrid"])
    ap.add_argument("days", type=int, nargs="?", default=60)
    ap.add_argument("seed", type=int, nargs="?", default=0, help="training randomness and bad homes, not the street")
    ap.add_argument("--attack", choices=[*ATTACKS, *DATA_ATTACKS], default="none")
    ap.add_argument("--share", type=float, default=0.3, help="share of bad homes, and the share robust rules expect")
    ap.add_argument("--last", action="store_true", help="keep the last policy, not the best on the held out weeks")
    # ponytail: mu = 0.01 is a common FedProx starting value; run_experiments.sh sweeps 0.001 and 0.1 too
    ap.add_argument("--mu", type=float, default=0.01, help="FedProx penalty weight")
    ap.add_argument("--without", choices=["clip", "groups", "trust"], help="FedGrid ablation: one step switched off")
    ap.add_argument("--lost", type=float, default=0.0, help="chance an honest home's change is lost in a round")
    ap.add_argument("--delayed", type=float, default=0.0, help="chance it arrives a round late instead")
    ap.add_argument("--feeder", default="L", help="CSIRO feeder, L (36 homes) or O (83 homes)")
    args = ap.parse_args()

    env = BatteryStreet(feeder=args.feeder)
    bad = np.array([], int)
    if args.attack != "none":
        bad = np.sort(np.random.default_rng(args.seed).choice(env.n, round(args.share * env.n), replace=False))
    reference = None
    if args.method == "fltrust" or (args.method == "fedgrid" and args.without != "trust"):
        # The server's own data: a street of other Ausgrid homes on the same feeder, outside the federation (the
        # research plan's public reference buildings). One central agent learns on it, from the global model each round.
        torch.manual_seed(args.seed + 1)
        reference = Learner(BatteryStreet(feeder=args.feeder, exclude=env.street.homes), "central", args.days,
                            args.seed + 1)
        assert not set(reference.env.street.homes) & set(env.street.homes), "a home is in both streets"
    # ponytail: 3 solar size bands, thirds of the street, as a home might state at sign up. Try 2 or 4 bands.
    pv = env.street.pv_kwp
    bands = torch.as_tensor(np.digitize(pv, np.quantile(pv, [1 / 3, 2 / 3])))
    server = Server(env.n, mu=args.mu if args.method == "fedprox" else 0.0, attack=ATTACKS.get(args.attack), bad=bad,
                    reference=reference, lost=args.lost, delayed=args.delayed, seed=args.seed)
    server.aggregate = aggregator(args.method, args.share, server, bands, args.without)
    poison = (bad, args.attack) if args.attack in DATA_ATTACKS else None
    start = time.time()
    agent = train(env, "local", args.days, seed=args.seed, server=server, keep_best=not args.last, poison=poison)
    size = server.global_model.numel()
    extras = [f"mu {args.mu}" if args.method == "fedprox" else "", f"without {args.without}" if args.without else "",
              f"lost {args.lost:.0%} delayed {args.delayed:.0%}" if args.lost or args.delayed else "",
              f"feeder {args.feeder}" if args.feeder != "L" else "", "last policy" if args.last else ""]
    print(f"{args.method} SAC, seed {args.seed}, attack {args.attack} from homes {bad.tolist()}"
          f"{''.join(', ' + e for e in extras if e)}: trained on {args.days} days in {time.time() - start:.0f} s, "
          f"{server.rounds} rounds, {size} shared values per home per round ({size * 4 / 1024:.0f} KB)")
    if server.received.any():
        rate = (server.rejected / server.received.clamp_min(1)).cpu().numpy()
        honest = np.setdiff1d(np.arange(env.n), bad)
        late = server.late.cpu().numpy()
        print(f"rejected: honest homes {rate[honest].mean():.1%} of their changes (most rejected home "
              f"{rate[honest].max():.1%}), bad homes {rate[bad].mean() if len(bad) else np.nan:.1%}, delayed honest "
              f"changes {late[1] / max(late[0], 1):.1%} of {late[0]:.0f}")
        print("rejected share by home (solar kWp): " + ", ".join(f"{r:.2f} ({p:.1f})" for r, p in zip(rate, pv)))

    # Check: every home holds the same shared layers (the kept policy may be from an earlier round than the critics).
    assert (server.flat(agent) == server.flat(agent)[0]).all()
    honest = np.setdiff1d(np.arange(env.n), bad)
    print(pd.Series(evaluate(env, policy(agent, env, "local"), *TEST, honest=honest)).round(2).to_string())

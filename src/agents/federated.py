"""Federated SAC: each home trains its own agent, and only the shared layers are combined across homes.

Each round (one day by default), every home sends the change in its shared layers since the last global model.
The server combines the changes with an aggregation rule, adds the result to the global model, and sends the new
shared layers back to every home. Each home's local layer, replay buffer and meter data never leave it.

  FedAvg   aggregation rule = mean of the changes
  FedProx  same, plus a penalty in local training that keeps the shared layers near the global model
  median, trimmed mean, Krum   standard robust aggregation rules, for the attack study

Attacks: a fixed set of bad homes train honestly but send a corrupted change. They know only their own change.

Run: .venv/bin/python src/agents/federated.py [fedavg|fedprox|median|trimmed|krum] [training days] [seed]
                                              [none|flip|scale|noise] [share of bad homes]
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.sac import policy, train
from env.battery_env import TEST, BatteryStreet, evaluate


# Aggregation rules. changes: (homes, parameters). Each returns the change to apply to the global model.
# ponytail: the robust rules assume up to 20% bad homes, the share used in the attack study.
def fedavg(changes):
    return changes.mean(0)


def median(changes):
    return changes.median(0).values


def trimmed_mean(changes, trim=0.2):
    k = int(trim * len(changes))
    return changes.sort(0).values[k:len(changes) - k].mean(0)


def krum(changes, assumed_bad=0.2):
    """The one change closest to its n - f - 2 nearest neighbours (Blanchard et al., 2017)."""
    n = len(changes)
    f = int(assumed_bad * n)
    distance = torch.cdist(changes, changes) ** 2
    return changes[distance.sort(1).values[:, 1:n - f - 1].sum(1).argmin()]  # column 0 is the distance to itself


# Attacks. own: the bad homes' honest changes, one row per bad home.
# ponytail: factor 10 is a starting value. Sweep it, and add the crafted attack of Fang et al. (2025) next.
def sign_flip(own):
    return -own


def scaling(own, factor=10.0):
    return factor * own


def noise(own, factor=10.0):
    return factor * own.std() * torch.randn_like(own)


AGGREGATE = {"fedavg": fedavg, "fedprox": fedavg, "median": median, "trimmed": trimmed_mean, "krum": krum}
ATTACK = {"none": None, "flip": sign_flip, "scale": scaling, "noise": noise}


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
            changes[self.bad] = self.attack(changes[self.bad])
        self.global_model = self.global_model + self.aggregate(changes)
        self.send(agent)
        self.rounds += 1


def check_aggregators():
    """One made up round: 29 honest homes near +1, 7 bad homes sending -10."""
    torch.manual_seed(0)
    changes = 1 + 0.1 * torch.randn(36, 100)
    changes[:7] = -10
    assert fedavg(changes).mean() < 0  # the mean is pulled the wrong way
    for rule in (median, trimmed_mean, krum):
        assert (rule(changes) - 1).abs().mean() < 0.15, rule.__name__


if __name__ == "__main__":
    check_aggregators()
    method = sys.argv[1] if len(sys.argv) > 1 else "fedavg"
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0  # training randomness and bad homes; the street stays the same
    attack = sys.argv[4] if len(sys.argv) > 4 else "none"
    share = float(sys.argv[5]) if len(sys.argv) > 5 else 0.2
    env = BatteryStreet()
    bad = np.sort(np.random.default_rng(seed).choice(env.n, round(share * env.n), replace=False)) if ATTACK[attack] else np.array([], int)
    # ponytail: mu = 0.01 is a common FedProx starting value. Tune it with a small sweep.
    server = Server(AGGREGATE[method], mu=0.01 if method == "fedprox" else 0.0, attack=ATTACK[attack], bad=bad)
    start = time.time()
    agent = train(env, "local", days, seed=seed, server=server)
    size = server.global_model.numel()
    print(f"{method} SAC, seed {seed}, attack {attack} from homes {bad.tolist()}: trained on {days} days in "
          f"{time.time() - start:.0f} s, {server.rounds} rounds, {size} shared values per home per round "
          f"({size * 4 / 1024:.0f} KB)")

    # Check: after the last round every home holds exactly the global shared layers.
    assert (server.flat(agent) == server.global_model).all()
    print(pd.Series(evaluate(env, policy(agent, env, "local"), *TEST)).round(2).to_string())

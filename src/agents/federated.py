"""Federated SAC: each home trains its own agent, and only the shared layers are combined across homes.

Each round (one day by default), every home sends the change in its shared layers since the last global model.
The server combines the changes with an aggregation rule, adds the result to the global model, and sends the new
shared layers back to every home. Each home's local layer, replay buffer and meter data never leave it.

  FedAvg   aggregation rule = mean of the changes
  FedProx  same, plus a penalty in local training that keeps the shared layers near the global model

The aggregation rule is a plain function of the changes (one row per home), so the robust aggregator can replace it.

Run: .venv/bin/python src/agents/federated.py [fedavg|fedprox] [training days]
"""
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.sac import policy, train
from env.battery_env import TEST, BatteryStreet, evaluate


def fedavg(changes):
    """changes: (homes, parameters). Returns the change to apply to the global model."""
    return changes.mean(0)


class Server:
    def __init__(self, aggregate=fedavg, round_days=1, mu=0.0):
        self.aggregate, self.round_days, self.mu = aggregate, round_days, mu
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
        self.global_model = self.global_model + self.aggregate(changes)
        self.send(agent)
        self.rounds += 1


if __name__ == "__main__":
    method = sys.argv[1] if len(sys.argv) > 1 else "fedavg"
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    # ponytail: mu = 0.01 is a common FedProx starting value. Tune it with a small sweep.
    server = Server(mu=0.01 if method == "fedprox" else 0.0)
    env = BatteryStreet()
    start = time.time()
    agent = train(env, "local", days, server=server)
    size = server.global_model.numel()
    print(f"{method} SAC: trained on {days} days in {time.time() - start:.0f} s, {server.rounds} rounds, "
          f"{size} shared values per home per round ({size * 4 / 1024:.0f} KB)")

    # Check: after the last round every home holds exactly the global shared layers.
    assert (server.flat(agent) == server.global_model).all()
    print(pd.Series(evaluate(env, policy(agent, env, "local"), *TEST)).round(2).to_string())

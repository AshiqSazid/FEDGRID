"""Device metrics of the research plan: decision time, training step time and memory for one home's agent.

Run it on the device itself, such as a Raspberry Pi (needs only torch and numpy):
  python src/device_benchmark.py
"""
import resource
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agents.sac import SAC, Buffer

OBS = 11  # values per home from BatteryStreet.observe()
torch.set_num_threads(1)
torch.manual_seed(0)
agent, buf = SAC(1, OBS), Buffer(1, 48 * 500, OBS)  # one home, 500 days of transitions
for _ in range(48 * 10):
    buf.add(torch.randn(1, 1, OBS), torch.rand(1, 1, 1) * 2 - 1, torch.randn(1, 1, 1), torch.randn(1, 1, OBS),
            torch.zeros(1, 1, 1))

o = torch.randn(1, 1, OBS)
start = time.perf_counter()
for _ in range(1000):
    agent.act(o, deterministic=True)
decide_ms = (time.perf_counter() - start)  # 1000 decisions, so seconds = ms per decision

start = time.perf_counter()
for _ in range(100):
    agent.update(*buf.sample(agent.batch))
update_ms = (time.perf_counter() - start) * 10

params = sum(p.numel() for net in (agent.actor, *agent.q) for p in net.parameters())
print(f"one decision {decide_ms:.2f} ms, one training step {update_ms:.1f} ms "
      f"(a day of training is 48 steps: {48 * update_ms / 1000:.1f} s)")
print(f"{params} network values ({params * 4 / 1024:.0f} KB), peak memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")

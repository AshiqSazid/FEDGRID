"""Central SAC on the training years with one fix, then the test year. Args: seed shaping(0/1) warmup_days"""
import sys
sys.path.insert(0, "src")
import pandas as pd
from agents.sac import policy, train
from env.battery_env import TEST, BatteryStreet, evaluate
seed, shaping, warmup = int(sys.argv[1]), bool(int(sys.argv[2])), int(sys.argv[3])
env = BatteryStreet()
agent = train(env, "central", 500, warmup_days=warmup, seed=seed, shaping=shaping)
r = evaluate(env, policy(agent, env, "central"), *TEST)
print(f"seed {seed} shaping {shaping} warmup {warmup}: test bill {r['bill $/home/yr']:.2f}, "
      f"curtailed {r['curtailed kWh/home/yr']:.1f}, cycles {r['battery cycles/yr']:.1f}")

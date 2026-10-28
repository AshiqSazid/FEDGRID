"""Central SAC with a given starting entropy weight, then the test year. Args: seed alpha"""
import sys
sys.path.insert(0, "src")
import agents.sac as S
from env.battery_env import TEST, BatteryStreet, evaluate
seed, alpha = int(sys.argv[1]), float(sys.argv[2])
Base = S.SAC
S.SAC = lambda *a, **k: Base(*a, alpha=alpha, **k)
env = BatteryStreet()
agent = S.train(env, "central", 500, seed=seed)
r = evaluate(env, S.policy(agent, env, "central"), *S.TEST)
print(f"seed {seed} alpha start {alpha}: test bill {r['bill $/home/yr']:.2f}, "
      f"curtailed {r['curtailed kWh/home/yr']:.1f}, cycles {r['battery cycles/yr']:.1f}")

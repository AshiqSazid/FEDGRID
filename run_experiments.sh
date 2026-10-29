#!/usr/bin/env bash
# Every training run behind the results, JOBS at a time with one CPU thread each. A run whose log already exists in
# results/ is skipped, so rerun this after an interruption. Groups run in the order given, most important first.
# Usage: [JOBS=4] ./run_experiments.sh [seeds] [attacks] [ablation] [faults] [longer] [scaling]
#   (all six when left out). PYTHON picks another interpreter (default .venv/bin/python).
# Log name: <method>_sac_<days>_seed<seed>[_<attack>_share<percent bad>][_<variant>]. The seed changes training
# randomness (and which homes are bad), never the street.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
METHODS="fedavg median trimmed krum fltrust flame tafdrl fedgrid"
ATTACKS="none normalized meter reward flip scale noise"

seeds() {  # clean runs, keeping the best policy on the held out weeks
  for seed in 0 1 2; do
    for mode in central local; do echo "${mode}_sac_500_seed$seed src/agents/sac.py $mode 500 $seed"; done
    for m in fedavg fedprox; do echo "${m}_sac_500_seed$seed src/agents/federated.py $m 500 $seed"; done
  done
}

attacks() {  # the last policy is kept, so damage late in training shows
  # Main setting: 30% of the homes bad (11 of 36), as in Fang et al. (2025). Seed 0 for everything first.
  for seed in 0 1 2; do
    for m in $METHODS; do
      for attack in $ATTACKS; do attack_run $m $attack 30 $seed; done
    done
  done
  # 10%, 20% and 40% bad homes, Normalized attack. Trimmed mean and Krum also change with the share they expect.
  for share in 10 20 40; do
    for m in $METHODS; do attack_run $m normalized $share; done
    for m in trimmed krum; do attack_run $m none $share; done
  done
}

ablation() {  # FedGrid with each of its three steps switched off in turn
  for seed in 0 1 2; do
    for without in clip groups trust; do
      for attack in none normalized; do attack_run fedgrid $attack 30 $seed _without_$without --without $without; done
    done
  done
}

faults() {  # honest but unreliable homes: 10% of their changes lost and 10% a round late
  for m in $METHODS; do
    for attack in none normalized; do attack_run $m $attack 30 0 _lost10_delayed10 --lost 0.1 --delayed 0.1; done
  done
}

longer() {  # 1000 training days, as every federated run was still improving at day 475 of 500; and FedProx's mu
  for seed in 0 1 2; do
    for mode in central local; do echo "${mode}_sac_1000_seed$seed src/agents/sac.py $mode 1000 $seed"; done
    for m in fedavg fedprox; do echo "${m}_sac_1000_seed$seed src/agents/federated.py $m 1000 $seed"; done
    for mu in 0.001 0.1; do echo "fedprox_sac_500_seed${seed}_mu$mu src/agents/federated.py fedprox 500 $seed --mu $mu"; done
  done
}

scaling() {  # a second, larger street: feeder O, 83 homes
  for m in fedavg fltrust flame fedgrid; do
    for attack in none normalized; do attack_run $m $attack 30 0 _feederO --feeder O; done
  done
}

attack_run() {  # method, attack, percent of homes bad, seed (0 if left out), then a log name suffix and extra flags
  local m=$1 attack=$2 share=$3 seed=${4:-0} suffix=${5:-}
  shift $(($# < 5 ? $# : 5))
  # no trailing space: xargs -L would join the next line on
  echo "${m}_sac_500_seed${seed}_${attack}_share$share$suffix src/agents/federated.py $m 500 $seed --attack $attack --share 0.$share --last${*:+ $*}"
}

for group in ${*:-seeds attacks ablation faults longer scaling}; do $group; done | xargs -P "${JOBS:-4}" -L 1 sh -c 'log=results/$0.log; [ -s "$log" ] ||
  { "${PYTHON:-.venv/bin/python}" "$@" > "$log.tmp" 2>&1 && mv "$log.tmp" "$log" && echo "done $0" || echo "FAILED $0, see $log.tmp"; }'; exit  # exit before bash reads on, so editing this file mid run is safe

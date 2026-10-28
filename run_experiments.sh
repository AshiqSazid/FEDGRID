#!/usr/bin/env bash
# Every training run behind the results, 4 at a time with one CPU thread each (about 80 minutes per 36 home run on a
# 4 core laptop). A run whose log already exists in results/ is skipped, so rerun this after an interruption.
# Usage: ./run_experiments.sh [seeds|attacks]   (both when left out)
# Log name: <method>_sac_<days>_seed<seed>[_<attack>_share<percent bad>]. The seed changes training randomness (and
# which homes are bad), never the street.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

seeds() {
  for seed in 0 1 2; do
    for mode in central local; do echo "${mode}_sac_500_seed$seed src/agents/sac.py $mode 500 $seed"; done
    for m in fedavg fedprox; do echo "${m}_sac_500_seed$seed src/agents/federated.py $m 500 $seed"; done
  done
}

attacks() {  # the last policy is kept, so damage late in training shows
  # Main setting: 30% of the homes bad (11 of 36), as in Fang et al. (2025)
  for m in fedavg median trimmed krum; do
    for attack in none flip scale noise normalized; do attack_run $m $attack 30; done
  done
  # Fewer and more bad homes, for the Normalized attack. Trimmed mean and Krum also change with the share they expect.
  for share in 20 40; do
    for m in fedavg median trimmed krum; do attack_run $m normalized $share; done
    for m in trimmed krum; do attack_run $m none $share; done
  done
}

attack_run() {  # method, attack, percent of homes bad
  echo "${1}_sac_500_seed0_${2}_share$3 src/agents/federated.py $1 500 0 --attack $2 --share 0.$3 --last"
}

{ [ "${1:-all}" = attacks ] || seeds; [ "${1:-all}" = seeds ] || attacks; } | xargs -P 4 -L 1 sh -c 'log=results/$0.log; [ -s "$log" ] ||
  { .venv/bin/python "$@" > "$log.tmp" 2>&1 && mv "$log.tmp" "$log" && echo "done $0" || echo "FAILED $0, see $log.tmp"; }'

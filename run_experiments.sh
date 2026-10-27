#!/usr/bin/env bash
# Every training run behind the results, 4 at a time with one CPU thread each (about 35 minutes per run on a 4 core
# laptop). A run whose log already exists in results/ is skipped, so rerun this after an interruption.
# Log name: <method>_sac_<days>_seed<seed>[_<attack>]. The seed changes training randomness (and which homes are bad),
# never the street.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

runs() {
  for seed in 0 1 2; do
    for mode in central local; do echo "${mode}_sac_500_seed$seed src/agents/sac.py $mode 500 $seed"; done
    for m in fedavg fedprox; do echo "${m}_sac_500_seed$seed src/agents/federated.py $m 500 $seed"; done
  done
  # Attack study: 7 of the 36 homes (20%) send bad updates
  for m in fedavg median trimmed krum; do
    for attack in none flip scale noise; do
      name=${m}_sac_500_seed0; [ "$attack" = none ] || name+=_$attack
      echo "$name src/agents/federated.py $m 500 0 $attack 0.2"
    done
  done
}

runs | xargs -P 4 -L 1 sh -c 'log=results/$0.log; [ -s "$log" ] ||
  { .venv/bin/python "$@" > "$log.tmp" 2>&1 && mv "$log.tmp" "$log" && echo "done $0" || echo "FAILED $0, see $log.tmp"; }'

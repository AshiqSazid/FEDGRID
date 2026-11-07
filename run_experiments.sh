#!/usr/bin/env bash
# Every training run behind the results, JOBS at a time with one CPU thread each. A run whose log already exists in
# results/ is skipped, so rerun this after an interruption. Groups run in the order given, most important first.
# Usage: [JOBS=4] ./run_experiments.sh [seeds] [attacks] [reference] [ablation] [faults] [longer] [scaling]
#   (all seven when left out), or these, only when named and in this order:
#     candidates tafdrl_settings    FedGrid versions and TAFDRL settings, read on validation weeks only
#     then src/choose_final.py candidates
#     settings unseen act           bands and z limit on the best FedGrid version; seeds 3 and 4, and ACT NextGen,
#                                   for every original defence
#     then src/choose_final.py settings
#     final unseen_final act_final stress    the final FedGrid and TAFDRL on seeds 0-2, seeds 3 and 4 and ACT
#                                   NextGen, and the 1.02 pu stress test
#   PYTHON picks another interpreter (default .venv/bin/python).
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

reference() {  # no attack again, as the first no-attack logs report no bill for the 25 homes the data attacks leave honest
  for seed in 0 1 2; do
    for m in $METHODS; do attack_run $m none 30 $seed _honest; done
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

candidates() {  # FedGrid versions, read on the validation weeks only by src/select_candidate.py. --no-test: no test-year bill
  { candidate _filter_first --filter-first
    candidate _filter_first_clip2 --filter-first --clip-factor 2
    candidate _filter_first_guard --filter-first --guard
    candidate _filter_first_home_trust --filter-first --home-trust
    # Clipping off: no attack and Normalized came with the ablation
    for seed in 0 1 2; do
      for attack in noise flip scale; do attack_run fedgrid $attack 30 $seed _without_clip --without clip --no-test; done
      attack_run fedgrid normalized 40 $seed _without_clip --without clip --no-test
    done
    # The original at 40% bad, to compare fairness. Seed 0 ran with the shares
    for seed in 1 2; do attack_run fedgrid normalized 40 $seed "" --no-test; done
  } | sed 's|^|candidates/|'
}

candidate() {  # log name suffix, then flags: the update attacks at 30% bad, and Normalized at 40% for fairness
  local suffix=$1
  shift
  for seed in 0 1 2; do
    for attack in none normalized noise flip scale; do attack_run fedgrid $attack 30 $seed $suffix "$@" --no-test; done
    attack_run fedgrid normalized 40 $seed $suffix "$@" --no-test
  done
}

tafdrl_settings() {  # the paper's minimum trust (best 0.3 to 0.4), and the limit values it doesn't give, on validation weeks
  { for seed in 0 1 2; do
      for attack in none normalized flip; do
        for lam in 1 2; do
          attack_run tafdrl $attack 30 $seed _lam${lam}_level99 --lam $lam --tau 0.35 --no-test
          attack_run tafdrl $attack 30 $seed _lam${lam}_level999 --lam $lam --tau 0.35 --level 0.999 --no-test
        done
      done
    done
  } | sed 's|^|candidates/|'
}

chosen() {  # flags of a version chosen on validation weeks by src/choose_final.py: its file holds a name, then flags
  local name flags
  read -r name flags < "results/candidates/$1.flags" || { echo "no $1 yet: run src/choose_final.py first" >&2; return 1; }
  echo "$flags"
}

settings() {  # 2 or 4 solar size bands, or z limit 2.0 or 3.0, one at a time on the best FedGrid version, like a candidate
  local name flags
  read -r name flags < results/candidates/best_fedgrid.flags || { echo "run src/choose_final.py candidates" >&2; exit 1; }
  [ "$name" = original ] && name="" || name="_$name"
  { candidate ${name}_bands2 $flags --bands 2
    candidate ${name}_bands4 $flags --bands 4
    candidate ${name}_z2 $flags --z 2.0
    candidate ${name}_z3 $flags --z 3.0
  } | sed 's|^|candidates/|'
}

unseen() {  # seeds 3 and 4, which no choice has used, for every original defence. --transfer: also scored on ACT homes
  for seed in 3 4; do
    for m in $METHODS; do
      for attack in $ATTACKS; do attack_run $m $attack 30 $seed "" --transfer; done
    done
  done
}

act() {  # ACT NextGen households on the same street, 2018 split by weeks (src/env/battery_env.py), every original
  # defence. Seed 0 for everything first
  echo "baselines_act src/baselines/mpc.py 1.0 nextgen"
  for seed in 0 1 2; do
    for m in $METHODS; do
      for attack in $ATTACKS; do attack_run $m $attack 30 $seed _act --dataset nextgen; done
    done
  done
}

final() {  # the final FedGrid and TAFDRL on seeds 0-2, logs ending _final: every attack (also scored on ACT homes),
  # the shares, message faults, and feeder O for FedGrid, as the original ran in attacks, faults and scaling
  local fg tf
  fg=$(chosen final_fedgrid) && tf=$(chosen final_tafdrl) || exit 1
  for seed in 0 1 2; do
    for attack in $ATTACKS; do
      attack_run fedgrid $attack 30 $seed _final $fg --transfer
      attack_run tafdrl $attack 30 $seed _final $tf --transfer
    done
  done
  for share in 10 20 40; do
    attack_run fedgrid normalized $share 0 _final $fg
    attack_run tafdrl normalized $share 0 _final $tf
  done
  for attack in none normalized; do
    attack_run fedgrid $attack 30 0 _lost10_delayed10_final --lost 0.1 --delayed 0.1 $fg
    attack_run tafdrl $attack 30 0 _lost10_delayed10_final --lost 0.1 --delayed 0.1 $tf
    attack_run fedgrid $attack 30 0 _feederO_final --feeder O $fg
  done
}

unseen_final() {  # the final FedGrid and TAFDRL on seeds 3 and 4
  local fg tf
  fg=$(chosen final_fedgrid) && tf=$(chosen final_tafdrl) || exit 1
  for seed in 3 4; do
    for attack in $ATTACKS; do
      attack_run fedgrid $attack 30 $seed _final $fg --transfer
      attack_run tafdrl $attack 30 $seed _final $tf --transfer
    done
  done
}

act_final() {  # the final FedGrid and TAFDRL on ACT NextGen
  local fg tf
  fg=$(chosen final_fedgrid) && tf=$(chosen final_tafdrl) || exit 1
  for seed in 0 1 2; do
    for attack in $ATTACKS; do
      attack_run fedgrid $attack 30 $seed _act_final --dataset nextgen $fg
      attack_run tafdrl $attack 30 $seed _act_final --dataset nextgen $tf
    done
  done
}

stress() {  # supply at 1.02 pu, reported apart from the main 1.00 pu setting, with the final FedGrid
  local fg
  fg=$(chosen final_fedgrid) || exit 1
  echo "baselines_pu102 src/baselines/mpc.py 1.02"
  for m in fedavg fltrust flame; do
    for attack in none normalized; do attack_run $m $attack 30 0 _pu102 --supply 1.02; done
  done
  for attack in none normalized; do attack_run fedgrid $attack 30 0 _pu102_final --supply 1.02 $fg; done
}

attack_run() {  # method, attack, percent of homes bad, seed (0 if left out), then a log name suffix and extra flags
  local m=$1 attack=$2 share=$3 seed=${4:-0} suffix=${5:-}
  shift $(($# < 5 ? $# : 5))
  # no trailing space: xargs -L would join the next line on
  echo "${m}_sac_500_seed${seed}_${attack}_share$share$suffix src/agents/federated.py $m 500 $seed --attack $attack --share 0.$share --last${*:+ $*}"
}

mkdir -p results/candidates
for group in ${*:-seeds attacks reference ablation faults longer scaling}; do $group; done | xargs -P "${JOBS:-4}" -L 1 sh -c 'log=results/$0.log; [ -s "$log" ] ||
  { "${PYTHON:-.venv/bin/python}" "$@" > "$log.tmp" 2>&1 && mv "$log.tmp" "$log" && echo "done $0" || echo "FAILED $0, see $log.tmp"; }'; exit  # exit before bash reads on, so editing this file mid run is safe

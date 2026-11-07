"""Choose the final FedGrid and TAFDRL from the validation weeks only, by the measure fixed on 5 Nov before any candidate
ran: lowest worst-case mean damage over the update attacks at 30% bad homes, seeds 0-2 (src/select_candidate.py), with
fairness printed beside it. A version counts only if all its runs finished: every attack it ran with, on all three
seeds. Ties go to the version listed first below.

FedGrid must also keep its fairness, as the 2 Nov plan to Dr Hossain said ("more resistant while keeping its
fairness"): a FedGrid version counts only if it rejects at most FAIR times the original FedGrid's share of honest
changes. Added on 6 Nov at 21:30, after the candidate results, where the most robust version rejected 30.8% of honest
changes against the original's 2.0%, and before any settings-step result was read. TAFDRL, a baseline whose paper
doesn't aim at fairness, keeps its most robust setting, and its fairness is reported beside it.

  candidates  after `candidates` and `tafdrl_settings`. Writes best_fedgrid, the FedGrid version the settings step
              starts from, and final_tafdrl, among the four settings with the paper's minimum trust 0.35. Those ran no
              attack, Normalized and flip, so their worst case is over Normalized and flip.
  settings    after `settings`. Writes final_fedgrid: the best version as it is (3 bands, z 2.5) or with one setting
              changed (2 or 4 bands, z 2.0 or 3.0).

Each choice goes to results/candidates/<choice>.flags as the version's name, then its flags (run_experiments.sh reads
them), and the tables behind it to results/candidates/choice_<step>.txt.

Run: .venv/bin/python src/choose_final.py candidates|settings [results folder]
"""
import sys
from pathlib import Path

from select_candidate import tables

FEDGRID = {"original": "", "filter_first": "--filter-first", "filter_first_clip2": "--filter-first --clip-factor 2",
           "filter_first_guard": "--filter-first --guard", "filter_first_home_trust": "--filter-first --home-trust",
           "without_clip": "--without clip"}
TAFDRL = {f"lam{lam}_level{level}": f"--lam {lam} --tau 0.35" + (" --level 0.999" if level == "999" else "")
          for lam in (1, 2) for level in ("99", "999")}
SETTINGS = {"bands2": "--bands 2", "bands4": "--bands 4", "z2": "--z 2.0", "z3": "--z 3.0"}
FAIR = 2.0  # a FedGrid version may reject at most this many times the original FedGrid's share of honest changes


def pick(table, versions, attacks, report, max_honest=None):
    """The version with the lowest worst case, among those whose runs all finished (and, for FedGrid, that are fair)."""
    done = table[table.index.isin(versions) & (table.attacks == attacks) & (table.runs == 3 * attacks)
                 & table.worst.notna()]
    report += [f"left out, runs missing: {v}" for v in versions if v not in done.index]
    if max_honest is not None:
        unfair = done["honest rejected %"] > max_honest
        report += [f"left out, rejects {done.at[v, 'honest rejected %']:.1f}% of honest changes, above "
                   f"{max_honest:.1f}%: {v}" for v in done.index[unfair]]
        done = done[~unfair]
    if done.empty:
        raise SystemExit("no version has all its runs: nothing chosen")
    best = min(done.index, key=lambda v: (done.worst[v], versions.index(v)))
    report.append(f"chosen: {best}, worst-case damage {done.worst[best]:+.1f} $/home over the validation weeks")
    return best


def main(step, folder):
    report, out = [], folder / "candidates"

    def write(choice, name, flags):
        (out / f"{choice}.flags").write_text(f"{name} {flags}".rstrip() + "\n")
        report.append(f"{choice}: {name} {flags}".rstrip())

    def fair(table):  # the most a FedGrid version may reject, from the original FedGrid on the same runs
        return FAIR * table.at["original", "honest rejected %"]

    if step == "candidates":
        for method, versions, attacks, choice in [("fedgrid", FEDGRID, 4, "best_fedgrid"),
                                                  ("tafdrl", TAFDRL, 2, "final_tafdrl")]:
            table = tables(method, folder)
            report += [f"\n## {method}\n", table.to_string()]
            name = pick(table, list(versions), attacks, report, fair(table) if method == "fedgrid" else None)
            write(choice, name, versions[name])
    elif step == "settings":
        base = (out / "best_fedgrid.flags").read_text().split()[0]
        named = {base: FEDGRID[base]}
        for s, flags in SETTINGS.items():
            named[s if base == "original" else f"{base}_{s}"] = f"{FEDGRID[base]} {flags}".strip()
        table = tables("fedgrid", folder)
        report += ["\n## fedgrid settings\n", table.to_string()]
        name = pick(table, list(named), 4, report, fair(table))
        write("final_fedgrid", name, named[name])
    else:
        raise SystemExit(__doc__)
    text = "\n".join(report)
    (out / f"choice_{step}.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "", Path(sys.argv[2]) if len(sys.argv) > 2 else
         Path(__file__).resolve().parents[1] / "results")

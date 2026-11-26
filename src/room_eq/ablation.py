"""Compare temperature continuation with fixed-temperature Gauss–Newton."""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import yaml

from room_eq.data import ARCHIVE, SAMPLE_RATE, MeshRIR
from room_eq.initialization import filter_setup, seeded_starts
from room_eq.objectives import SeatObjective
from room_eq.preprocess import log_spectra
from room_eq.solvers import generalized_gauss_newton, projected_gradient_norm


def solve_schedule(objective, initial, lower, upper, temperatures, *, budget, intermediate, tolerance):
    """Warm-start each stage and reserve the unspent budget for the final stage."""
    if not temperatures or temperatures[-1] != objective.temperature:
        raise ValueError("Schedule must end at the comparison temperature")
    if intermediate < 1 or budget <= intermediate * (len(temperatures) - 1):
        raise ValueError("Stage allocation must leave a positive final-stage budget")
    used = 0
    x = initial
    stages = []
    start = perf_counter()
    for index, temperature in enumerate(temperatures):
        cap = budget - used if index == len(temperatures) - 1 else intermediate
        fit = generalized_gauss_newton(
            replace(objective, temperature=temperature), x, lower, upper,
            budget=cap, tolerance=tolerance,
        )
        x = fit.x
        stages.append({
            "temperature_db_squared": temperature, "start_evaluation": used,
            "budget": cap, "evaluations": fit.nfev,
            "end_objective_at_stage_temperature": float(fit.fun),
            "status": str(fit.message), "history": fit.history,
        })
        used += fit.nfev
    norm = projected_gradient_norm(x, fit.jac, lower, upper)
    return {
        "final_objective": float(fit.fun), "projected_gradient": norm,
        "stationary": norm <= tolerance, "evaluations": used,
        "seconds": perf_counter() - start, "parameters": x.tolist(), "stages": stages,
    }


def paired_spatial_scores(scores, alpha):
    spatial = scores[scores["study"] == "spatial"]
    keys = ["seed", "train_count"]
    metrics = [
        f"{split}_{metric}_rmse_db"
        for split in ("train", "test") for metric in ("pooled", "worst_seat")
    ]
    mean = spatial[spatial["method"] == "mean"][keys + metrics]
    robust = spatial[(spatial["method"] == "robust") & (spatial["alpha"] == alpha)][keys + metrics]
    pairs = mean.merge(robust, on=keys, suffixes=("_mean", "_robust"), validate="one_to_one")
    if len(pairs) != len(mean) or len(pairs) != len(robust) or pairs.empty:
        raise ValueError("Expected one mean and robust result for every spatial split")
    for metric in metrics:
        pairs[f"{metric}_delta"] = pairs[f"{metric}_robust"] - pairs[f"{metric}_mean"]
    return pairs.sort_values(keys)


def run(archive, main_results, config, output):
    main = json.loads((main_results / "manifest.json").read_text(encoding="utf-8"))
    settings = main["config"]
    matched = next(split for split in main["splits"] if split["study"] == "matched")
    ids = np.asarray(matched["train_ids"])
    dataset = MeshRIR(archive)
    frequencies = np.geomspace(*settings["band_hz"], settings["frequency_count"])
    spectra = log_spectra(
        dataset.load(ids.tolist()), SAMPLE_RATE, frequencies,
        settings["fft_length"], settings["magnitude_floor"],
    )
    alpha = settings["primary_robust_alpha"]
    final_temperature = settings["temperature_db_squared"]
    final_only = [final_temperature]
    lower, upper, _ = filter_setup(settings)
    budget = settings["evaluation_budget"]
    tolerance = settings["projected_gradient_tolerance"]

    def evaluate(selected, initial, schedule):
        objective = SeatObjective(
            selected - selected.mean(), frequencies, SAMPLE_RATE,
            settings["gain_regularization"], alpha, final_temperature,
        )
        return solve_schedule(
            objective, initial, lower, upper, schedule,
            budget=budget, intermediate=config["intermediate_budget"], tolerance=tolerance,
        )

    # Three pilot subsets, each omitting a different third of the nine
    # designated training seats. The five benchmark starts are held separate.
    omit = np.array_split(np.random.default_rng(config["pilot_fold_seed"]).permutation(len(ids)), 3)
    pilot_starts = list(seeded_starts({**settings, "start_seeds": config["pilot_start_seeds"]}))[1:]
    schedules = config["candidate_schedules"]
    pilots = []
    for fold, omitted in enumerate(omit):
        keep = np.delete(np.arange(len(ids)), omitted)
        for name, initial in pilot_starts:
            baseline = evaluate(spectra[keep], initial, final_only)
            for index, schedule in enumerate(schedules):
                fit = evaluate(spectra[keep], initial, schedule)
                pilots.append({
                    "fold": fold, "start": name, "schedule_index": index,
                    "receiver_ids": ids[keep].tolist(),
                    "direct_objective": baseline["final_objective"],
                    "continuation_objective": fit["final_objective"],
                    "objective_delta": fit["final_objective"] - baseline["final_objective"],
                    "direct_evaluations": baseline["evaluations"],
                    "continuation_evaluations": fit["evaluations"],
                    "direct_stationary": baseline["stationary"],
                    "continuation_stationary": fit["stationary"],
                })
    pilot_table = pd.DataFrame(pilots)
    ranking = pilot_table.groupby("schedule_index").agg(
        median_objective_delta=("objective_delta", "median"),
        median_evaluations=("continuation_evaluations", "median"),
    ).sort_values(["median_objective_delta", "median_evaluations"])
    chosen = int(ranking.index[0])

    records = []
    for name, initial in seeded_starts(settings):
        for method, schedule in (("direct", final_only), ("continuation", schedules[chosen])):
            fit = evaluate(spectra, initial, schedule)
            records.append({"start": name, "method": method, "initial": initial.tolist(), **fit})
    pairs = paired_spatial_scores(pd.read_csv(main_results / "scores.csv"), alpha)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "main_results": str(main_results), "archive": str(archive), "training_ids": ids.tolist(),
        "alpha": alpha, "evaluation_budget": budget, "tolerance": tolerance,
        "ablation_config": config, "chosen_schedule": schedules[chosen],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    pilot_table.to_csv(output / "schedule_pilot.csv", index=False)
    ranking.to_csv(output / "schedule_ranking.csv")
    for row in records:
        assert row["evaluations"] <= budget
    with (output / "continuation_runs.jsonl").open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    results = pd.DataFrame([
        {key: value for key, value in row.items() if key not in {"stages", "parameters", "initial"}}
        for row in records
    ])
    results.to_csv(output / "continuation_runs.csv", index=False)
    pairs.to_csv(output / "paired_spatial.csv", index=False)
    plot_pairs(pairs, output / "paired_worst_seat.png")
    print(f"Selected schedule {schedules[chosen]}. Saved {len(records)} fits and {len(pairs)} spatial pairs.")


def plot_pairs(pairs, path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(1, 2, sharey=True, figsize=(8, 3.4), layout="constrained")
    for axis, split, title in zip(axes, ("train", "test"), ("Training", "Spatial holdout")):
        metric = f"{split}_worst_seat_rmse_db_delta"
        for seed, group in pairs.groupby("seed"):
            axis.plot(group["train_count"], group[metric], "o-", label=f"Seed {seed}")
        axis.axhline(0, color="0.35", linewidth=1, linestyle="--")
        axis.set(title=title, xlabel="Training positions", xticks=sorted(pairs["train_count"].unique()))
    axes[0].set_ylabel("Worst-seat RMSE difference (dB)\nRobust minus mean")
    axes[1].legend()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--main-results", type=Path, default=Path("results/main"))
    parser.add_argument("--config", type=Path, default=Path("configs/ablation.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/ablation"))
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    run(args.archive, args.main_results, config, args.output)


if __name__ == "__main__":
    main()

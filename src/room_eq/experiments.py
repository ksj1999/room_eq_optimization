"""Run matched solver and spatial holdout experiments on measured MeshRIR data."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import yaml

from room_eq.biquad import response_db
from room_eq.data import ARCHIVE, MD5, RECORD, SAMPLE_RATE, MeshRIR
from room_eq.initialization import filter_setup, seeded_starts
from room_eq.objectives import SeatObjective
from room_eq.preprocess import log_spectra
from room_eq.smoke import metrics, middle_plane, select_receivers
from room_eq.solvers import generalized_gauss_newton, lbfgsb, projected_gradient_norm


def spatial_splits(receivers, config):
    """Reserve the same block for every repeat and use nested training sets."""
    plane, _ = middle_plane(receivers)
    x = receivers[plane, 0]
    cutoff = config["max_training_x_m"]
    test_start = config["min_test_x_m"]
    if cutoff >= test_start:
        raise ValueError("Training and held-out regions must have a gap")
    candidates = plane[x <= cutoff]
    test = plane[x >= test_start]
    counts = config["training_counts"]
    if min(counts) < 1 or max(counts) > len(candidates) or not len(test):
        raise ValueError("Invalid training counts or empty holdout")
    center = (receivers[plane, :2].min(axis=0) + receivers[plane, :2].max(axis=0)) / 2
    anchor = candidates[np.argmin(np.linalg.norm(receivers[candidates, :2] - center, axis=1))]
    remaining = candidates[candidates != anchor]
    for seed in config["spatial_seeds"]:
        nested = np.concatenate(([anchor], np.random.default_rng(seed).permutation(remaining)))
        for count in counts:
            yield seed, nested[:count], test


def compare_split(study, seed, train, test, reference_db, frequencies, config, starts, runs, scores):
    lower, upper, _ = filter_setup(config)
    dense_hz = np.geomspace(1, SAMPLE_RATE / 2, 8192)
    common = {
        "study": study, "seed": seed, "train_count": len(train), "test_count": len(test),
        "level_reference_db": reference_db,
    }

    def score(method, alpha, fit, solver=None):
        eq = 0 if fit is None else response_db(fit["parameters"], frequencies, SAMPLE_RATE)
        boost = 0 if fit is None else response_db(fit["parameters"], dense_hz, SAMPLE_RATE).max()
        row = {
            **common, "method": method, "alpha": alpha, "solver": solver,
            "best_start": None if fit is None else fit["start"],
            "stationary": None if fit is None else fit["stationary"],
            "max_combined_boost_db": boost,
            "training_pooled_rmse_floor_db": np.sqrt(train.var(axis=0).mean()),
        }
        for name, data in (("train", train), ("test", test)):
            row.update({f"{name}_{key}": value for key, value in metrics(data + eq).items()})
        scores.append(row)

    score("no_eq", None, None)
    if study == "matched":
        objectives = [("mean", 1.0, train), ("robust", config["primary_robust_alpha"], train)]
        solvers = [lbfgsb, generalized_gauss_newton]
    else:
        objectives = [("single_seat", 1.0, train[:1])] + [
            ("mean" if alpha == 1 else "robust", alpha, train)
            for alpha in config["alpha_values"]
        ]
        solvers = [lbfgsb]
    for method, alpha, responses in objectives:
        objective = SeatObjective(
            responses, frequencies, SAMPLE_RATE, config["gain_regularization"],
            alpha, config["temperature_db_squared"],
        )
        for solve in solvers:
            fits = []
            for start, initial in starts:
                begin = perf_counter()
                fit = solve(
                    objective, initial, lower, upper, budget=config["evaluation_budget"],
                    tolerance=config["projected_gradient_tolerance"],
                )
                elapsed = perf_counter() - begin
                norm = projected_gradient_norm(fit.x, fit.jac, lower, upper)
                record = {
                    **common, "method": method, "alpha": alpha, "solver": solve.__name__,
                    "start": start, "initial_parameters": initial.tolist(),
                    "parameters": fit.x.tolist(), "objective": float(fit.fun),
                    "projected_gradient": norm,
                    "stationary": norm <= config["projected_gradient_tolerance"],
                    "success": bool(fit.success), "message": str(fit.message),
                    "seconds": elapsed, "nfev": fit.nfev, "njev": fit.njev,
                    "nhev": fit.get("nhev", 0), "iterations": fit.nit,
                    "history": fit.history,
                }
                fits.append(record)
                runs.append(record)
            best = min(fits, key=lambda fit: fit["objective"])
            score(method, alpha, best, solve.__name__)


def run(archive, config, output):
    dataset = MeshRIR(archive)
    frequencies = np.geomspace(*config["band_hz"], config["frequency_count"])
    train, test, _ = select_receivers(
        dataset.receivers, config["train_count"], config["test_count"], config["seed"],
    )
    plane, height = middle_plane(dataset.receivers)
    plane_spectra = log_spectra(
        dataset.load(plane.tolist()), SAMPLE_RATE, frequencies,
        config["fft_length"], config["magnitude_floor"],
    )
    spectra = dict(zip(plane, plane_spectra))
    spatial = list(spatial_splits(dataset.receivers, config))
    evaluations = [("matched", config["seed"], train, test)] + [
        ("spatial", seed, train, test) for seed, train, test in spatial
    ]
    anchor = int(spatial[0][1][0])
    starts = list(seeded_starts(config))
    runs, scores, splits = [], [], []
    for study, seed, train, test in evaluations:
        train_db = np.stack([spectra[index] for index in train])
        test_db = np.stack([spectra[index] for index in test])
        reference = float(train_db.mean() if study == "matched" else spectra[anchor].mean())
        split_starts = starts if study == "matched" else starts[:config["spatial_starts"]]
        compare_split(
            study, seed, train_db - reference, test_db - reference, reference,
            frequencies, config, split_starts, runs, scores,
        )
        splits.append({
            "study": study, "seed": seed, "train_ids": train.tolist(), "test_ids": test.tolist(),
        })
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "archive": str(archive), "dataset_record": RECORD, "release_md5": MD5,
        "sample_rate_hz": SAMPLE_RATE, "source_xyz_m": dataset.sources[0].tolist(),
        "plane_z_m": float(height), "spatial_reference_receiver": anchor,
        "config": config, "splits": splits,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    with (output / "runs.jsonl").open("w", encoding="utf-8") as stream:
        for record in runs:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
    solver_runs = pd.DataFrame([
        {key: value for key, value in row.items()
         if key not in {"history", "parameters", "initial_parameters"}}
        for row in runs
    ])
    solver_runs.to_csv(output / "solver_runs.csv", index=False)
    matched = solver_runs[solver_runs["study"] == "matched"]
    matched.groupby(["method", "solver"]).agg(
        runs=("success", "size"), stationary_runs=("stationary", "sum"),
        median_objective=("objective", "median"), best_objective=("objective", "min"),
        median_evaluations=("nfev", "median"), median_seconds=("seconds", "median"),
    ).to_csv(output / "solver_summary.csv")
    scores = pd.DataFrame(scores)
    scores.to_csv(output / "scores.csv", index=False)
    make_figures(output, scores, runs, config["primary_robust_alpha"])
    _, train_map, test_map = max(
        (split for split in spatial if split[0] == config["spatial_seeds"][0]),
        key=lambda split: len(split[1]),
    )
    make_spatial_map(
        output, dataset.receivers, spectra, frequencies,
        train_map, test_map, runs, config["spatial_seeds"][0], config["primary_robust_alpha"],
    )
    print(f"Saved {len(runs)} solver runs and {len(scores)} metric rows to {output}")


def make_figures(output, scores, runs, robust_alpha):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, method in zip(axes, ("mean", "robust")):
        subset = [row for row in runs if row["study"] == "matched" and row["method"] == method]
        floor = min(row["objective"] for row in subset)
        for algorithm, color in (("lbfgsb", "#3d76a3"), ("generalized_gauss_newton", "#9b4b30")):
            for index, row in enumerate(item for item in subset if item["solver"] == algorithm):
                history = row["history"]
                ax.semilogy(
                    [record["evaluation"] for record in history],
                    [max(record["objective"] - floor, 1e-9) for record in history],
                    color=color, alpha=0.7, label=algorithm.replace("_", " ") if index == 0 else None,
                )
        ax.set(title=method.capitalize(), xlabel="Objective/gradient evaluations")
    axes[0].set_ylabel("Objective above best observed")
    axes[0].legend()
    figure.savefig(output / "convergence.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    spatial = scores[scores["study"] == "spatial"]
    for method, subset in spatial.groupby("method"):
        if method == "robust":
            subset = subset[subset["alpha"] == robust_alpha]
        for ax, metric in zip(axes, ("pooled", "worst_seat")):
            group = subset.groupby("train_count")[f"test_{metric}_rmse_db"]
            medians = group.median()
            ax.plot(medians.index, medians, "o-", label=method.replace("_", " "))
            ax.fill_between(medians.index, group.quantile(0.25), group.quantile(0.75), alpha=0.15)
            ax.set(xlabel="Number of training positions", ylabel=f"{metric.replace('_', ' ').capitalize()} RMSE (dB)")
            ax.set_xticks(medians.index)
    axes[0].legend()
    figure.savefig(output / "spatial_holdout.png", dpi=150)
    plt.close(figure)

    figure, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    sweep = spatial[spatial["method"].isin(["mean", "robust"])]
    for count, subset in sweep.groupby("train_count"):
        medians = subset.groupby("alpha")[["test_pooled_rmse_db", "test_worst_seat_rmse_db"]].median()
        line, = ax.plot(*medians.to_numpy().T, "o-", label=f"{count} training positions")
        for alpha, (x, y) in medians.iterrows():
            ax.annotate(f"{alpha:g}", (x, y), xytext=(4, 4), textcoords="offset points", color=line.get_color())
    ax.set(xlabel="Held-out pooled RMSE (dB)", ylabel="Held-out worst-seat RMSE (dB)",
           title="Mean and worst-seat tradeoff by alpha")
    ax.legend()
    figure.savefig(output / "tradeoff.png", dpi=150)
    plt.close(figure)


def make_spatial_map(output, receivers, spectra, frequencies, train, test, runs, seed, robust_alpha):
    from matplotlib import pyplot as plt

    corrected = np.stack([spectra[index] for index in test]) - spectra[train[0]].mean()
    maps = {"No EQ": np.sqrt(np.mean(corrected**2, axis=1))}
    for label, alpha in (("Mean EQ", 1), (f"Robust EQ (α={robust_alpha:g})", robust_alpha)):
        fit = min(
            (row for row in runs if row["study"] == "spatial" and row["seed"] == seed
             and row["train_count"] == len(train) and row["method"] != "single_seat"
             and row["alpha"] == alpha),
            key=lambda row: row["objective"],
        )
        eq = response_db(fit["parameters"], frequencies, SAMPLE_RATE)
        maps[label] = np.sqrt(np.mean((corrected + eq)**2, axis=1))
    vmin = min(values.min() for values in maps.values())
    vmax = max(values.max() for values in maps.values())
    figure, axes = plt.subplots(1, 3, figsize=(10, 4), layout="constrained", sharey=True)
    for ax, (label, rmse) in zip(axes, maps.items()):
        ax.scatter(*receivers[train, :2].T, marker="x", color="0.25", s=15, label="Training")
        image = ax.scatter(*receivers[test, :2].T, c=rmse, marker="s", s=20,
                           vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set(title=label, xlabel="x (m)", aspect="equal")
    axes[0].set_ylabel("y (m)")
    axes[0].legend(loc="upper left")
    figure.colorbar(image, ax=axes, label="Held-out seat RMSE (dB)", shrink=0.7)
    figure.savefig(output / "spatial_map.png", dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--config", type=Path, default=Path("configs/main.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/main"))
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as stream:
        experiment = yaml.safe_load(stream)
    with args.config.with_name(experiment["smoke_config"]).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    run(args.archive, {**config, **experiment}, args.output)


if __name__ == "__main__":
    main()

"""Compare peak-informed, random-center and near-flat starts on fixed-temperature robust EQ."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from room_eq.data import ARCHIVE, SAMPLE_RATE, MeshRIR
from room_eq.initialization import filter_setup, peak_informed_start
from room_eq.objectives import SeatObjective
from room_eq.preprocess import log_spectra
from room_eq.solvers import generalized_gauss_newton, projected_gradient_norm


RANDOM_SEEDS = (6101, 6102, 6103, 6104, 6105)


def candidate_starts(responses, frequencies, config):
    lower, upper, near_flat = filter_setup(config)
    peak, peak_count = peak_informed_start(responses, frequencies, config)
    yield "peak", None, peak, peak_count
    yield "near_flat", None, near_flat, None
    # Uniform log centers with a fixed minimum gap. Gains and Q values
    # match the peak-informed fit, so only center placement changes.
    count = config["filter_count"]
    gap = np.arange(count) * np.log(2) / 6
    left, right = lower[1], upper[1] - gap[-1]
    if left >= right:
        raise ValueError("Band is too narrow for the number of separated centers")
    order = np.argsort(peak[1::3])
    for seed in RANDOM_SEEDS:
        random = peak.reshape(-1, 3).copy()
        positions = np.sort(np.random.default_rng(seed).uniform(left, right, size=count)) + gap
        random[order, 1] = positions
        yield "random_centers", seed, random.ravel(), None


def run(archive, main_results, output):
    manifest = json.loads((main_results / "manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    lower, upper, _ = filter_setup(config)
    frequencies = np.geomspace(*config["band_hz"], config["frequency_count"])
    splits = manifest["splits"]
    ids = sorted({index for split in splits for index in split["train_ids"]})
    dataset = MeshRIR(archive)
    spectra = log_spectra(
        dataset.load(ids), SAMPLE_RATE, frequencies, config["fft_length"], config["magnitude_floor"],
    )
    by_id = dict(zip(ids, spectra))
    anchor = manifest["spatial_reference_receiver"]
    records = []
    for split in splits:
        training = np.stack([by_id[index] for index in split["train_ids"]])
        reference = training.mean() if split["study"] == "matched" else by_id[anchor].mean()
        responses = training - reference
        objective = SeatObjective(
            responses, frequencies, SAMPLE_RATE, config["gain_regularization"],
            config["primary_robust_alpha"], config["temperature_db_squared"],
        )
        for method, seed, initial, peak_count in candidate_starts(responses, frequencies, config):
            started = perf_counter()
            fit = generalized_gauss_newton(
                objective, initial, lower, upper, budget=config["evaluation_budget"],
                tolerance=config["projected_gradient_tolerance"],
            )
            elapsed = perf_counter() - started
            records.append({
                "study": split["study"], "split_seed": split["seed"],
                "train_count": len(training), "method": method, "start_seed": seed,
                "detected_peak_count": peak_count, "evaluations": fit.nfev, "seconds": elapsed,
                "initial_objective": fit.history[0]["objective"], "final_objective": float(fit.fun),
                "projected_gradient": projected_gradient_norm(fit.x, fit.jac, lower, upper),
                "stationary": bool(fit.success), "status": str(fit.message),
                "initial_parameters": initial.tolist(), "final_parameters": fit.x.tolist(),
                "history": fit.history,
            })
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps({
        "main_results": str(main_results), "archive": str(archive), "config": config,
        "random_seeds": RANDOM_SEEDS, "minimum_peak_spacing_octaves": 1 / 6,
        "splits": [{key: value for key, value in split.items() if key != "test_ids"} for split in splits],
    }, indent=2) + "\n", encoding="utf-8")
    with (output / "runs.jsonl").open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    results = pd.DataFrame([
        {key: value for key, value in row.items()
         if key not in {"initial_parameters", "final_parameters", "history"}}
        for row in records
    ])
    results.to_csv(output / "runs.csv", index=False)
    summary = results.groupby(["study", "split_seed", "train_count", "method"]).agg(
        starts=("evaluations", "size"), stationary=("stationary", "sum"),
        median_initial_objective=("initial_objective", "median"),
        median_final_objective=("final_objective", "median"),
        best_final_objective=("final_objective", "min"),
        median_evaluations=("evaluations", "median"),
    )
    summary.to_csv(output / "summary.csv")
    print(f"Saved {len(records)} initialization fits to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--main-results", type=Path, default=Path("results/main"))
    parser.add_argument("--output", type=Path, default=Path("results/initialization"))
    args = parser.parse_args()
    run(args.archive, args.main_results, args.output)


if __name__ == "__main__":
    main()

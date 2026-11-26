"""Fit a bounded mean-loss equalizer to a small measured MeshRIR split."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import least_squares

from room_eq.biquad import response_db
from room_eq.data import ARCHIVE, MD5, RECORD, SAMPLE_RATE, MeshRIR
from room_eq.initialization import filter_setup
from room_eq.preprocess import log_spectra


def middle_plane(receivers):
    """Return the middle plane's receiver indices and height."""
    height = np.unique(receivers[:, 2].round(5))
    plane_z = height[len(height) // 2]
    plane = np.flatnonzero(np.isclose(receivers[:, 2], plane_z, rtol=0, atol=1e-5))
    return plane, plane_z


def select_receivers(receivers, train_count, test_count, seed):
    """Sample a reproducible split from the middle horizontal plane."""
    plane, plane_z = middle_plane(receivers)
    if min(train_count, test_count) < 1 or train_count + test_count > len(plane):
        raise ValueError("Training and test counts must fit on the selected plane")
    order = np.random.default_rng(seed).permutation(plane)
    return order[:train_count], order[train_count:train_count + test_count], plane_z


def fit_mean_eq(training_db, frequencies, config):
    """Fit the mean seat objective via SciPy's bounded trust-region method."""
    lower, upper, initial = filter_setup(config)
    # For shared EQ the seat variance is constant in theta. Fitting the
    # mean spectrum gives the same minimizer with a smaller residual vector.
    mean_db = training_db.mean(axis=0)
    regularization_scale = np.sqrt(config["gain_regularization"])

    def residual(parameters):
        corrected = mean_db + response_db(parameters, frequencies, SAMPLE_RATE)
        return np.concatenate((
            corrected / np.sqrt(frequencies.size),
            regularization_scale * parameters[::3],
        ))

    return least_squares(
        residual, initial, bounds=(lower, upper), method="trf",
        max_nfev=config["max_nfev"],
    )


def metrics(responses):
    seat_rmse = np.sqrt(np.mean(responses**2, axis=1))
    return {
        "mean_seat_rmse_db": float(seat_rmse.mean()),
        "pooled_rmse_db": float(np.sqrt(np.mean(responses**2))),
        "worst_seat_rmse_db": float(seat_rmse.max()),
        "p95_seat_rmse_db": float(np.quantile(seat_rmse, 0.95)),
        "max_abs_deviation_db": float(np.abs(responses).max()),
    }


def run(archive, config, output):
    dataset = MeshRIR(archive)
    train, test, plane_z = select_receivers(
        dataset.receivers, config["train_count"], config["test_count"], config["seed"]
    )
    ids = np.concatenate((train, test))
    frequencies = np.geomspace(*config["band_hz"], config["frequency_count"])
    spectra = log_spectra(
        dataset.load(ids), SAMPLE_RATE, frequencies,
        config["fft_length"], config["magnitude_floor"],
    )
    # Common level reference yields a flat 0 dB target after centering.
    level_reference = float(spectra[:len(train)].mean())
    before = spectra - level_reference
    start = perf_counter()
    fit = fit_mean_eq(before[:len(train)], frequencies, config)
    elapsed = perf_counter() - start
    eq = response_db(fit.x, frequencies, SAMPLE_RATE)
    after = before + eq
    dense_hz = np.geomspace(1, SAMPLE_RATE / 2, 8192)
    dense_eq = response_db(fit.x, dense_hz, SAMPLE_RATE)
    train_before, test_before = np.split(before, [len(train)])
    train_after, test_after = np.split(after, [len(train)])
    report = {
        "dataset": {
            "record": RECORD, "archive": str(archive), "release_md5": MD5,
            "sample_rate_hz": SAMPLE_RATE, "source_xyz_m": dataset.sources[0].tolist(),
            "plane_z_m": float(plane_z),
        },
        "config": config,
        "preprocessing": {
            "window": "rectangular", "smoothing": "none",
            "interpolation": "linear in Hz on the dB spectrum",
            "level_reference_db": level_reference, "target_db": 0.0,
        },
        "train_ids": train.tolist(),
        "test_ids": test.tolist(),
        "training_pooled_rmse_floor_db": float(np.sqrt(train_before.var(axis=0).mean())),
        "max_eq_boost_db": float(dense_eq.max()),
        "solver": {
            "method": "SciPy least_squares trf", "success": bool(fit.success),
            "status": int(fit.status), "message": fit.message, "seconds": elapsed,
            "nfev": fit.nfev, "njev": fit.njev,
        },
        "training": {"no_eq": metrics(train_before), "mean_eq": metrics(train_after)},
        "test": {"no_eq": metrics(test_before), "mean_eq": metrics(test_after)},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    params = fit.x.reshape(-1, 3)
    pd.DataFrame({
        "gain_db": params[:, 0],
        "frequency_hz": np.exp(params[:, 1]),
        "q": np.exp(params[:, 2]),
    }).to_csv(output / "filters.csv", index=False)
    pd.DataFrame({
        "receiver_id": ids,
        "split": ["train"] * len(train) + ["test"] * len(test),
        "x_m": dataset.receivers[ids, 0],
        "y_m": dataset.receivers[ids, 1],
        "z_m": dataset.receivers[ids, 2],
        "no_eq_rmse_db": np.sqrt(np.mean(before**2, axis=1)),
        "mean_eq_rmse_db": np.sqrt(np.mean(after**2, axis=1)),
    }).to_csv(output / "seats.csv", index=False)
    np.savez_compressed(
        output / "spectra.npz", receiver_ids=ids, frequencies_hz=frequencies,
        before_db=before, after_db=after, eq_db=eq,
    )
    plot_responses(output / "responses.png", frequencies, test_before, test_after)
    return report


def plot_responses(path, frequencies, before, after):
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True, layout="constrained")
    for axis, title, responses in zip(axes, ("No EQ", "Mean EQ"), (before, after)):
        axis.semilogx(frequencies, responses.T, color="#3777a8", alpha=0.25)
        axis.semilogx(frequencies, responses.mean(axis=0), color="#17405c", linewidth=2)
        axis.axhline(0, color="0.35", linestyle="--", linewidth=1)
        axis.set(title=title, xlabel="Frequency (Hz)")
    axes[0].set_ylabel("Response relative to target (dB)")
    figure.suptitle("Held-out MeshRIR positions")
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/smoke"))
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    report = run(args.archive, config, args.output)
    before = report["test"]["no_eq"]["pooled_rmse_db"]
    after = report["test"]["mean_eq"]["pooled_rmse_db"]
    print(f"Held-out pooled RMSE: {before:.2f} dB before EQ, {after:.2f} dB after EQ")
    print(f"Saved results to {args.output}")
    if not report["solver"]["success"]:
        parser.exit(1, f"Solver stopped: {report['solver']['message']}\n")


if __name__ == "__main__":
    main()

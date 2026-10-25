#!/usr/bin/env python3
"""Aggregate completed packet-vorticity diagnostics without changing native data.

Every nonempty space/time/age bin must contain all seven matched seeds. The
driver supplies volume-weighted moments within each seed; this script gives
those seeds equal weight before forming norms or ratios. It never averages
ratios, and never pools different snapshot times or local packet ages.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import tempfile

SEEDS = (1, 2, 3, 4, 6, 7, 8)
PAIRED_MEMBERS = ("packet_turbulence", "turbulence_control")
REGIONS = {"full", "stokes_layer", "piv_window"}
AGES = {"all", "before", "leading", "core", "trailing", "after"}
META = ("age_low", "age_high", "volume_m3", "x_columns", "z_levels",
        "z_min_m", "z_max_m")
KEYS = ("member", "time_s", "stencil_step", "region", "age_bin")
REQUIRED_TERMS = {
    "native_ux", "native_vy", "native_wz", "native_divergence",
    "native_xz_divergence", "u_x", "w_z", "xz_divergence", "q", "R3D",
    "u_qx", "w_qz", "omegaEx_vx", "q_vy", "omegaEz_vz", "v_qy",
}


def number(value, label, *, infinite=False):
    try:
        result = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid {label}: {value!r}") from exc
    if math.isnan(result) or (not infinite and not math.isfinite(result)):
        raise ValueError(f"Nonfinite {label}: {value!r}")
    return result


def integer(value, label):
    result = number(value, label)
    if result != int(result):
        raise ValueError(f"Noninteger {label}: {value!r}")
    return int(result)


def same(a, b):
    return a == b or math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)


def load_regions(folder):
    """Return group -> seed -> term; exclude consistently empty geometric bins."""
    folder = Path(folder)
    files = [(seed, folder / f"seed-{seed}" / "regions.csv") for seed in SEEDS]
    if (folder / "seed-0" / "regions.csv").is_file():
        files.append((0, folder / "seed-0" / "regions.csv"))
    groups = {}
    metadata = {}
    empty_rows = 0
    for expected_seed, path in files:
        if not path.is_file():
            raise ValueError(f"Missing completed seed diagnostics: {path}")
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            required_columns = set(KEYS) | set(META) | {
                "seed", "term", "mean", "mean_square", "max_abs"}
            if not required_columns <= set(reader.fieldnames or ()):
                raise ValueError(f"Missing required columns in {path}")
            for line, raw in enumerate(reader, 2):
                where = f"{path.name}:{line}"
                seed = integer(raw["seed"], f"seed at {where}")
                member = raw["member"]
                if seed != expected_seed:
                    raise ValueError(f"Seed does not match its directory at {where}")
                if (seed == 0 and member != "packet_null") or (
                    seed != 0 and member not in PAIRED_MEMBERS
                ):
                    raise ValueError(f"Unexpected member {member!r} for seed {seed}")
                if raw["region"] not in REGIONS or raw["age_bin"] not in AGES:
                    raise ValueError(f"Unknown region/age bin at {where}")
                step = integer(raw["stencil_step"], "stencil_step")
                if step < 1:
                    raise ValueError("stencil_step must be positive")
                key = (member, number(raw["time_s"], "time_s"), step,
                       raw["region"], raw["age_bin"])
                meta = {name: number(raw[name], name, infinite=name in {"age_low", "age_high"})
                        for name in META}
                for name in ("x_columns", "z_levels"):
                    meta[name] = integer(raw[name], name)
                if meta["volume_m3"] < 0 or min(meta["x_columns"], meta["z_levels"]) < 0:
                    raise ValueError(f"Negative geometric measure at {where}")
                if meta["age_low"] >= meta["age_high"]:
                    raise ValueError(f"Invalid age bounds at {where}")
                if meta["volume_m3"] == 0:
                    empty_rows += 1
                    continue
                if min(meta["x_columns"], meta["z_levels"]) < 1:
                    raise ValueError(f"Nonempty volume with empty grid at {where}")
                if meta["z_min_m"] > meta["z_max_m"]:
                    raise ValueError(f"Reversed depth bounds at {where}")
                # Pair members, all terms and all seeds must sample the same cells.
                geometry_key = key[1:]
                if geometry_key in metadata:
                    if not all(same(meta[name], metadata[geometry_key][name]) for name in META):
                        raise ValueError(f"Inconsistent paired sampling geometry for {geometry_key}")
                else:
                    metadata[geometry_key] = meta
                row = {name: number(raw[name], name) for name in ("mean", "mean_square", "max_abs")}
                if row["mean_square"] < 0 or row["max_abs"] < 0:
                    raise ValueError(f"Negative squared/absolute moment at {where}")
                tolerance = 1e-10 * max(row["mean_square"], row["max_abs"]**2, 1e-30)
                if row["mean"]**2 > row["mean_square"] + tolerance:
                    raise ValueError(f"Mean exceeds RMS at {where}")
                if row["mean_square"] > row["max_abs"]**2 + tolerance:
                    raise ValueError(f"RMS exceeds maximum at {where}")
                terms = groups.setdefault(key, {}).setdefault(seed, {})
                if raw["term"] in terms:
                    raise ValueError(f"Duplicate term {raw['term']} for seed {seed}, {key}")
                terms[raw["term"]] = row
    if not groups:
        raise ValueError("No nonempty diagnostic bins")
    expected_groups = {key[1:] for key in groups if key[0] in PAIRED_MEMBERS}
    for geometry_key in expected_groups:
        for member in PAIRED_MEMBERS:
            key = (member, *geometry_key)
            if key not in groups:
                raise ValueError(f"Missing matched member group: {key}")
    reference_terms = None
    for key, seeds in groups.items():
        expected = {0} if key[0] == "packet_null" else set(SEEDS)
        if set(seeds) != expected:
            raise ValueError(f"Incomplete seed ensemble for {key}: got {sorted(seeds)}")
        for seed, terms in seeds.items():
            if not REQUIRED_TERMS <= set(terms):
                raise ValueError(f"Missing diagnostic terms for {key}, seed {seed}: "
                                 f"{sorted(REQUIRED_TERMS-set(terms))}")
            if reference_terms is None:
                reference_terms = set(terms)
            elif set(terms) != reference_terms:
                raise ValueError(f"Inconsistent term inventory for {key}, seed {seed}")
    return groups, metadata, empty_rows


def ensemble_rows(groups, metadata):
    output = []
    for key, seed_rows in sorted(groups.items()):
        common = dict(zip(KEYS, key)) | metadata[key[1:]] | {"realizations": len(seed_rows)}
        for term in sorted(next(iter(seed_rows.values()))):
            rows = [terms[term] for terms in seed_rows.values()]
            mean_square = math.fsum(row["mean_square"] for row in rows) / len(rows)
            output.append(common | {"term": term,
                "mean": math.fsum(row["mean"] for row in rows) / len(rows),
                "mean_square": mean_square, "rms": math.sqrt(mean_square),
                "max_abs": max(row["max_abs"] for row in rows)})
    return output


def metric_rows(ensemble, tau, q_floor=0.0, advection_floor=0.0, gradient_floor=0.0):
    by_group = {}
    for row in ensemble:
        by_group.setdefault(tuple(row[name] for name in KEYS), {})[row["term"]] = row
    result = []
    for key, terms in sorted(by_group.items()):
        common = {name: next(iter(terms.values()))[name] for name in (*KEYS, *META, "realizations")}
        norm = lambda name: terms[name]["rms"]
        quadrature = lambda names: math.sqrt(math.fsum(norm(name)**2 for name in names))
        definitions = (
            ("epsilon_vy_native", norm("native_vy"), quadrature(("native_ux", "native_wz")), gradient_floor),
            ("epsilon_div_native", norm("native_xz_divergence"), quadrature(("native_ux", "native_wz")), gradient_floor),
            ("epsilon_div_centered", norm("xz_divergence"), quadrature(("u_x", "w_z")), gradient_floor),
            ("native_full_divergence_relative", norm("native_divergence"), quadrature(("native_ux", "native_vy", "native_wz")), gradient_floor),
            ("r3d_to_uncancelled_advection", norm("R3D"), norm("u_qx") + norm("w_qz"), advection_floor),
            ("tau_r3d_to_q", tau * norm("R3D"), norm("q"), q_floor),
            ("r3d_cancellation", norm("R3D"), math.fsum(norm(name) for name in ("omegaEx_vx", "q_vy", "omegaEz_vz", "v_qy")), advection_floor),
        )
        for metric, numerator, denominator, floor in definitions:
            status = "denominator_zero" if denominator == 0 else (
                "below_specified_floor" if denominator <= floor else "defined")
            result.append(common | {"metric": metric, "numerator": numerator,
                "denominator": denominator, "denominator_floor": floor,
                "ratio": numerator / denominator if status == "defined" else None,
                "status": status})
    return result


def csv_text(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def make_plot(metrics, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "unavailable_matplotlib"
    selected = [row for row in metrics if row["region"] == "stokes_layer"
                and row["age_bin"] == "all" and row["stencil_step"] == 1]
    if not selected:
        return "no_stokes_layer_all_step1_bins"
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout="constrained")
    for axis, metric, ylabel in zip(axes,
        ("epsilon_vy_native", "tau_r3d_to_q", "r3d_to_uncancelled_advection"),
        (r"$\|v_y\|/(\|u_x\|^2+\|w_z\|^2)^{1/2}$",
         r"$\tau_0\|R_{3D}\|/\|q\|$",
         r"$\|R_{3D}\|/(\|u q_x\|+\|w q_z\|)$")):
        for member, label, color in (("packet_turbulence", "Packet + turbulence", "#2166ac"),
                                     ("turbulence_control", "Matched control", "#555555")):
            rows = sorted((row for row in selected if row["metric"] == metric and row["member"] == member),
                          key=lambda row: row["time_s"])
            axis.plot([row["time_s"] for row in rows],
                      [math.nan if row["ratio"] is None else row["ratio"] for row in rows],
                      "o-", ms=4, color=color, label=label)
        axis.set_xlabel("Time from initialization (s)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=.18)
    axes[0].legend(fontsize=8)
    fig.suptitle("Upper Stokes layer, full fields: four edge cells excluded; no closed budget", fontsize=11)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return "written"


def build_report(folder, output=None, *, tau=2.8, plot=True,
                 q_floor=0.0, advection_floor=0.0, gradient_floor=0.0):
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be positive and finite")
    for floor in (q_floor, advection_floor, gradient_floor):
        if not math.isfinite(floor) or floor < 0:
            raise ValueError("Denominator floors must be nonnegative and finite")
    folder = Path(folder)
    output = Path(output) if output is not None else folder
    groups, metadata, empty_rows = load_regions(folder)
    ensemble = ensemble_rows(groups, metadata)
    metrics = metric_rows(ensemble, tau, q_floor, advection_floor, gradient_floor)
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "complete": True, "seeds": list(SEEDS), "tau0_seconds": tau,
        "nonempty_member_time_region_age_stencil_groups": len(groups),
        "ensemble_rows": len(ensemble), "metric_rows": len(metrics),
        "ignored_zero_volume_rows": empty_rows,
        "snapshot_times_seconds": sorted({key[1] for key in groups if key[0] in PAIRED_MEMBERS}),
        "stencil_steps": sorted({key[2] for key in groups}),
        "undefined_ratios": sum(row["ratio"] is None for row in metrics),
        "aggregation": "Equal weights across seven matched seeds after volume-weighted spatial moments within each seed/bin; ensemble mean squares precede square roots and ratios. No pooling across snapshot times or ages.",
        "age": "Local age (t - t_peak(x))/tau0; control uses the same virtual packet coordinate. The all bin includes every sampled age.",
        "frame": "Current-following simulation frame; the advection-based ratio is frame dependent.",
        "fields": "Full uncentred fields, including mean and coherent motion. q is transverse Eulerian vorticity; velocity in advection/stretching is Lagrangian.",
        "sampling": "Every reported term uses 1<=x<=11 m and a common four-cell exclusion at both vertical boundaries for both stencil steps. Thus even the full region denotes the interior depth range; the top exclusion is about 8 mm on the M2 grid. Exact z limits and sample volumes are retained in each row. Native full-grid divergence maxima remain in each seed qc.csv, outside this regional aggregation.",
        "ratio_definitions": {
            "epsilon_vy_native": "||native_vy|| / sqrt(||native_ux||^2+||native_wz||^2)",
            "epsilon_div_native": "||native_ux+native_wz|| / sqrt(||native_ux||^2+||native_wz||^2)",
            "epsilon_div_centered": "||u_x+w_z|| / sqrt(||u_x||^2+||w_z||^2)",
            "native_full_divergence_relative": "||native_divergence|| / sqrt(||native_ux||^2+||native_vy||^2+||native_wz||^2)",
            "r3d_to_uncancelled_advection": "||R3D|| / (||u*qx||+||w*qz||)",
            "tau_r3d_to_q": "tau0*||R3D||/||q||",
            "r3d_cancellation": "||R3D||/(||omegaEx*vx||+||q*vy||+||omegaEz*vz||+||v*qy||)",
        },
        "units": {"q": "s^-1", "R3D": "s^-2", "q_floor": "s^-1", "advection_floor": "s^-2", "gradient_floor": "s^-1"},
        "denominator_floors": {"q": q_floor, "advection": advection_floor, "gradient": gradient_floor},
        "limits": [
            "No uncertainty or statistical-significance claim is attached to the seven-seed ratios.",
            "No threshold labels Taylor-Proudman balance, quasi-two-dimensional flow, or a transport barrier.",
            "Small R3D can reflect cancellation of substantial three-dimensional terms; retain individual norms.",
            "Undefined ratios are blank in CSV. A numerical noise floor is not inferred automatically.",
            "Sparse snapshots and implicit numerical dissipation prevent a closed temporal vorticity budget.",
            "Stencil/coarsening sensitivity is not simulation-resolution convergence.",
            "Profiles remain in their seed directories; this report aggregates regions.csv only.",
        ],
    }
    with tempfile.TemporaryDirectory(prefix=".report-", dir=output) as temporary:
        stage = Path(temporary)
        (stage / "ensemble.csv").write_text(csv_text(ensemble))
        (stage / "metrics.csv").write_text(csv_text(metrics))
        summary["plot"] = make_plot(metrics, stage / "diagnostic-ratios.png") if plot else "disabled"
        text = ("Packet vorticity audit: aggregation completed.\n"
                f"Matched seeds: {', '.join(map(str, SEEDS))}.\n"
                f"Snapshot times (s): {', '.join(f'{t:.6g}' for t in summary['snapshot_times_seconds'])}.\n"
                f"Stencil steps: {', '.join(map(str, summary['stencil_steps']))}.\n"
                f"Nonempty member/time/region/age/stencil groups: {len(groups)}.\n"
                f"Undefined ratios: {summary['undefined_ratios']}; no denominator clipping.\n\n"
                + summary["aggregation"] + "\n" + summary["fields"] + "\n" + summary["sampling"] + "\n\n"
                + "\n".join(summary["limits"]) + "\n")
        (stage / "briefsummary.txt").write_text(text)
        (stage / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        for path in stage.iterdir():
            if path.name != "summary.json":
                os.replace(path, output / path.name)
        os.replace(stage / "summary.json", output / "summary.json")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tau", type=float, default=2.8)
    parser.add_argument("--q-floor", type=float, default=0.0, help="Explicit RMS q denominator floor (s^-1)")
    parser.add_argument("--advection-floor", type=float, default=0.0, help="Explicit RMS tendency denominator floor (s^-2)")
    parser.add_argument("--gradient-floor", type=float, default=0.0, help="Explicit RMS velocity-gradient denominator floor (s^-1)")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    summary = build_report(args.folder, args.output, tau=args.tau, plot=not args.no_plot,
                           q_floor=args.q_floor, advection_floor=args.advection_floor,
                           gradient_floor=args.gradient_floor)
    print(json.dumps({key: summary[key] for key in ("complete", "ensemble_rows", "metric_rows", "undefined_ratios", "plot")}))


if __name__ == "__main__":
    main()

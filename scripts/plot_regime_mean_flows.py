#!/usr/bin/env python3
"""Render matched mean flows from verified, derived y-mean cache arrays.

Usage: python plot_mean_flows.py --data data --campaign campaign.json --output gallery
No simulation source or cache is modified. Inputs must already be receipt-verified.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

STYLES = {
    "2d": dict(color="#182b3a", lw=2.8, ls="-", label="2D"),
    "tiny": dict(color="#008f9c", lw=1.55, ls=(0, (5, 3)), label="Weakly 3D (η = 10⁻⁶)"),
    "finite": dict(color="#c84e21", lw=1.9, ls="-", label="Finite 3D (η = 1)"),
}
ORDER = ("2d", "tiny", "finite")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig, output, stem):
    fig.savefig(output / (stem + ".png"), dpi=190, facecolor="white")
    fig.savefig(output / (stem + ".pdf"), facecolor="white")
    plt.close(fig)
    return {"png": stem + ".png", "pdf": stem + ".pdf"}


def curves(ax, group, index, change=False, initial=False):
    for role in ORDER:
        d = group["data"][role]
        p = d["U"][index].mean(axis=0)
        if change:
            p = p - d["U"][0].mean(axis=0)
        ax.plot(p * 1000, d["z"], **STYLES[role])
    if initial:
        d = group["data"]["2d"]
        ax.plot(d["U"][0].mean(axis=0) * 1000, d["z"], color="#8a8a8a", lw=1, ls=":", label="Initial profile")
    ax.grid(alpha=.18)
    ax.set_ylabel("z (m)")
    ax.tick_params(labelsize=9)


def legend(fig, initial=False):
    handles = [Line2D([], [], **STYLES[r]) for r in ORDER]
    if initial:
        handles.append(Line2D([], [], color="#8a8a8a", lw=1, ls=":", label="Initial profile"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=10)


def group_title(g):
    c = g["config"]
    state = "waves" if c.get("waves", 1) else "no waves"
    return f'{g["name"]} · seed {g["seed"]} · α = {c["alpha"]:g} · eddy speed {1000*c["eddy_rms"]:g} mm/s · {state}'


def profiles(g, output):
    d = g["data"]["2d"]
    t = d["time"][-1]
    delta = 1 / (2 * g["config"]["k"])
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.1), layout="constrained")
    fig.get_layout_engine().set(rect=(0, .055, 1, .94))
    for i in range(2):
        for j in range(2):
            curves(axes[i, j], g, -1, change=(j == 1), initial=(j == 0))
            axes[i, j].set_ylim((d["z"][0], 0) if i == 0 else (max(d["z"][0], -2*delta), 0))
            axes[i, j].set_xlabel("Eulerian mean current (mm/s)" if j == 0 else "Change from initial mean current (mm/s)")
        axes[i, 0].set_title("Full depth" if i == 0 else "Upper two Stokes depths", loc="left", fontsize=11)
    axes[0, 1].set_title(f"Final − initial, t = {t:g} s", loc="left", fontsize=11)
    fig.suptitle(f"Horizontally averaged mean flow at {t:g} s\n{group_title(g)}", fontsize=12)
    legend(fig, initial=True)
    return save(fig, output, g["slug"] + "-profiles")


def fields(g, output):
    d = g["data"]["2d"]
    c = g["config"]
    x = (np.arange(d["U"].shape[1]) + .5) * c["lx"] / d["U"].shape[1]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.6), layout="constrained", sharex=True, sharey=True)
    for row in range(2):
        arrays = [(g["data"][r]["U"][-1] - (g["data"][r]["U"][0].mean(axis=0)[None, :] if row else 0)) * 1000 for r in ORDER]
        if row:
            limit = max(float(np.max(np.abs(a))) for a in arrays)
            vmin, vmax = -limit, limit
            cmap = "RdBu_r"
        else:
            vmin = min(float(a.min()) for a in arrays)
            vmax = max(float(a.max()) for a in arrays)
            cmap = "viridis"
        if vmin == vmax:
            vmin -= 1e-12
            vmax += 1e-12
        for col, (r, a) in enumerate(zip(ORDER, arrays)):
            m = axes[row, col].pcolormesh(x, d["z"], a.T, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
            if not row:
                axes[row, col].set_title(STYLES[r]["label"], fontsize=11)
            if row:
                axes[row, col].set_xlabel("x (m)")
        axes[row, 0].set_ylabel(("Mean U\n" if row == 0 else "U − initial mean\n") + "z (m)")
        colorbar = fig.colorbar(m, ax=list(axes[row]), shrink=.9, pad=.02)
        colorbar.set_label("mm/s")
    fig.suptitle(f'Transverse-averaged Eulerian current, t = {d["time"][-1]:g} s\n{group_title(g)}', fontsize=12)
    return save(fig, output, g["slug"] + "-fields")


def split_profiles(profile_path, groups, output):
    """Keep downloads below large-file limits without changing any values."""
    final_name = "mean-profiles-final.csv"
    with contextlib.ExitStack() as stack:
        reader = csv.DictReader(stack.enter_context(profile_path.open()))
        final = csv.DictWriter(stack.enter_context((output/final_name).open("w", newline="")), fieldnames=reader.fieldnames)
        final.writeheader()
        by_group = {}
        for g in groups:
            name = g["slug"] + "-profiles.csv"
            writer = csv.DictWriter(stack.enter_context((output/name).open("w", newline="")), fieldnames=reader.fieldnames)
            writer.writeheader()
            by_group[(g["name"], str(g["seed"]))] = (writer, g["selected_field_times_s"][-1])
            g["profiles_csv"] = name
        for row in reader:
            writer, final_time = by_group[(row["matched_group"], row["seed"])]
            writer.writerow(row)
            if float(row["time_s"]) == final_time:
                final.writerow(row)
    return final_name


def refresh_fields(data, output):
    manifest_path = output / "index-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for g in manifest["groups"]:
        g["data"] = {}
        c = g["config"]
        for role, rid in g["run_ids"].items():
            with np.load(data/rid/"pair-fields.npz", allow_pickle=False) as a:
                us = c["waves"]*c["S_reference"]*np.exp(2*c["k"]*a["z"])
                g["data"][role] = {"U": a["Uc"]-us[None, None, :], "time": a["time"], "z": a["z"], "dz": a["dz"]}
        g["field_plot"] = fields(g, output)
        del g["data"]
        print("Refreshed fields", g["slug"], flush=True)
    manifest["profiles_csv"] = split_profiles(output/"mean-profiles.csv", manifest["groups"], output)
    manifest["definitions"]["change"] = "The horizontally averaged current profile at t minus its own initial profile. Initial Eulerian current and q0 are not assumed zero."
    manifest["definitions"]["field_anomaly"] = "The lower field row shows U(x,z,t_final)-mean_x U(x,z,0): departure from the initial horizontally averaged profile, retaining the present in-plane eddies and jets. It is not a subtraction of the initial x-z eddy pattern."
    manifest["definitions"]["colour_scaling"] = "Within each matched group, all three total-U panels share the exact data minimum and maximum; all three departures from the initial mean profile share a symmetric range reaching their largest absolute value. No quantile clipping. Colour ranges differ across groups and rows."
    manifest["generator_sha256"] = sha(Path(__file__))
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")


def primary_overview(groups, output):
    primary = [g for g in groups if g["experiment"] == "primary"]
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 7.3), sharey=True, layout="constrained")
    fig.get_layout_engine().set(rect=(0, .06, 1, .95))
    for ax, g in zip(axes.flat, primary):
        curves(ax, g, -1, change=True)
        c = g["config"]
        ax.set_title(f'{1000*c["eddy_rms"]:g} mm/s; α = {c["alpha"]:g}', fontsize=11)
        ax.set_xlabel("Current change (mm/s)", fontsize=10)
        ax.set_ylim(g["data"]["2d"]["z"][0], 0)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 3), useMathText=True)
    for ax in list(axes.flat)[len(primary):]:
        ax.set_axis_off()
    fig.suptitle("Final mean-current change across the seven primary parameter cases\nHorizontally averaged Eulerian current: final − initial", fontsize=13)
    legend(fig)
    return save(fig, output, "primary-mean-current-changes")


def anchor_evolution(g, output):
    d = g["data"]["2d"]
    targets = [0, 120, 300, 600]
    indices = [int(np.argmin(abs(d["time"] - target))) for target in targets]
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 7.4), sharey=True, layout="constrained")
    fig.get_layout_engine().set(rect=(0, .06, 1, .94))
    for j, i in enumerate(indices):
        for row in range(2):
            curves(axes[row, j], g, i, change=bool(row))
            axes[row, j].set_xlabel("Mean current (mm/s)" if row == 0 else "Current change (mm/s)", fontsize=10)
            axes[row, j].set_ylim(d["z"][0], 0)
            axes[row, j].ticklabel_format(axis="x", style="sci", scilimits=(-2, 3), useMathText=True)
        axes[0, j].set_title(f't = {d["time"][i]:g} s', fontsize=11)
    fig.suptitle("Central case: evolution of the horizontally averaged Eulerian current\n" + group_title(g), fontsize=12)
    legend(fig)
    paths = save(fig, output, "anchor-profile-evolution")
    return {**paths, "requested_times_s": targets, "actual_times_s": [float(d["time"][i]) for i in indices]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fields-only", action="store_true", help="Refresh field panels and split profile downloads in an existing complete output")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    if args.fields_only:
        refresh_fields(args.data, args.output)
        return
    campaign = json.loads(args.campaign.read_text())
    collections = {}
    for entry in campaign["runs"]:
        if entry["group"] in ORDER:
            collections.setdefault((entry["matched_group"], entry["seed"]), {})[entry["group"]] = entry
    groups = []
    source_records = {}
    profile_path = args.output / "mean-profiles.csv"
    with profile_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["matched_group", "seed", "run_id", "role", "eta", "time_s", "z_m", "mean_u_m_s", "initial_mean_u_m_s", "change_mean_u_m_s"])
        writer.writeheader()
        for (name, seed), entries in collections.items():
            if set(entries) != set(ORDER):
                raise ValueError(f"Incomplete matched triplet: {name}, seed {seed}")
            data = {}
            summaries = {}
            for role in ORDER:
                entry = entries[role]
                folder = args.data / entry["id"]
                summary_path, array_path = folder / "summary.json", folder / "pair-fields.npz"
                summary = json.loads(summary_path.read_text())
                if summary["status"] != "complete":
                    raise ValueError(f"Run is not complete: {entry['id']}")
                summaries[role] = summary
                with np.load(array_path, allow_pickle=False) as arrays:
                    data[role] = {"U": arrays["Uc"].copy(), "time": arrays["time"].copy(), "z": arrays["z"].copy(), "dz": arrays["dz"].copy()}
                    if "profile_time" in arrays and "profile_Uc" in arrays:
                        data[role]["profile_time"] = arrays["profile_time"].copy()
                        data[role]["profile_Uc"] = arrays["profile_Uc"].copy()
                d = data[role]
                # Oceananigans' saved Uc is the Lagrangian prognostic current.
                # The time-independent actual Stokes drift must be removed
                # before labelling a total field or profile as Eulerian.
                c = summary["config"]
                if c.get("velocity_convention") != "Lagrangian":
                    raise ValueError(f"Unexpected velocity convention: {entry['id']}")
                us_reference = c["S_reference"] * np.exp(2*c["k"]*d["z"])
                us_actual = c["waves"] * us_reference
                if not np.isclose(c["S"], c["waves"]*c["S_reference"], atol=1e-15, rtol=1e-12):
                    raise ValueError(f"Inconsistent Stokes amplitudes: {entry['id']}")
                d["U"] -= us_actual[None, None, :]
                if "profile_Uc" in d:
                    d["profile_Uc"] -= us_actual[None, :]
                expected_initial = -c["alpha"]*us_reference + c.get("uoffset", 0)
                if not np.allclose(d["U"][0].mean(axis=0), expected_initial, atol=1e-12, rtol=1e-10):
                    raise ValueError(f"Initial Eulerian mean disagrees with prescribed q0 background: {entry['id']}")
                if d["U"].shape != (len(d["time"]), summary["config"]["nx"], len(d["z"])) or not all(np.isfinite(v).all() for v in d.values()):
                    raise ValueError(f"Invalid mean-field array: {entry['id']}")
                if not (np.all(np.diff(d["time"]) > 0) and np.all(np.diff(d["z"]) > 0) and np.all(d["dz"] > 0)):
                    raise ValueError(f"Invalid coordinates: {entry['id']}")
                if not np.isclose(d["time"][-1], summary["final_time_s"], atol=1e-8, rtol=0):
                    raise ValueError(f"Final time mismatch: {entry['id']}")
                source_records[entry["id"]] = {"summary_sha256": sha(summary_path), "pair_fields_sha256": sha(array_path), "native_receipt_sha256": summary["receipt_sha256"], "source_version": summary["source_version"]}
            base = data["2d"]
            config = summaries["2d"]["config"]
            initial_checks = {}
            for role in ORDER:
                d = data[role]
                for key in ("nx", "ny", "nz", "lx", "ly", "k", "alpha", "eddy_rms", "nu", "waves", "S_reference", "S", "uoffset", "velocity_convention"):
                    if summaries[role]["config"].get(key) != config.get(key):
                        raise ValueError(f"Physical/grid mismatch {name} {role}: {key}")
                if d["U"].shape != base["U"].shape or not np.array_equal(d["z"], base["z"]) or not np.array_equal(d["dz"], base["dz"]) or not np.allclose(d["time"], base["time"], atol=1e-8, rtol=0):
                    raise ValueError(f"Unmatched mean-field coordinates: {name} {role}")
                initial_error = float(np.max(abs(d["U"][0] - base["U"][0])))
                initial_checks[role] = initial_error
                if initial_error > max(1e-12, 1e-9*config["eddy_rms"]):
                    raise ValueError(f"Initial transverse mean differs: {name} {role}, {initial_error}")
                p = d.get("profile_Uc", d["U"].mean(axis=1))
                pt = d.get("profile_time", d["time"])
                if p.shape != (len(pt), len(d["z"])) or not np.all(np.diff(pt) > 0):
                    raise ValueError(f"Invalid profile series: {name} {role}")
                if not np.allclose(p[0], d["U"][0].mean(axis=0), atol=1e-13, rtol=1e-12) or not np.allclose(p[-1], d["U"][-1].mean(axis=0), atol=1e-13, rtol=1e-12):
                    raise ValueError(f"Profile/field mismatch: {name} {role}")
                for ti, t in enumerate(pt):
                    for zi, z in enumerate(d["z"]):
                        writer.writerow({"matched_group": name, "seed": seed, "run_id": entries[role]["id"], "role": role, "eta": summaries[role]["config"]["perturb_energy_ratio"], "time_s": float(t), "z_m": float(z), "mean_u_m_s": float(p[ti, zi]), "initial_mean_u_m_s": float(p[0, zi]), "change_mean_u_m_s": float(p[ti, zi]-p[0, zi])})
            g = {"name": name, "seed": seed, "slug": f"{name}-s{seed}", "experiment": entries["2d"]["experiment"], "config": config, "run_ids": {r: entries[r]["id"] for r in ORDER}, "selected_field_times_s": base["time"].tolist(), "z_cell_center_range_m": [float(base["z"][0]), float(base["z"][-1])], "initial_ymean_max_absolute_difference_m_s": initial_checks, "data": data}
            g["profile_plot"] = profiles(g, args.output)
            g["field_plot"] = fields(g, args.output)
            # Keep complete arrays for the central time evolution only. Seven
            # primary groups need their initial/final slices for the overview.
            if name != "anchor" or seed != 1:
                for d in data.values():
                    d["U"] = d["U"][[0, -1]]
                    d["time"] = d["time"][[0, -1]]
            groups.append(g)
            print(f"Rendered {g['slug']}", flush=True)
    overview = primary_overview(groups, args.output)
    anchor = next(g for g in groups if g["name"] == "anchor" and g["seed"] == 1)
    evolution = anchor_evolution(anchor, args.output)
    compressed_profiles = profile_path.with_suffix(".csv.gz")
    with profile_path.open("rb") as source, compressed_profiles.open("wb") as destination:
        with gzip.GzipFile(filename="", mode="wb", fileobj=destination, mtime=0) as compressed:
            shutil.copyfileobj(source, compressed)
    public_groups = [{key: value for key, value in g.items() if key != "data"} for g in groups]
    final_profiles = split_profiles(profile_path, public_groups, args.output)
    manifest = {
        "schema_version": 1,
        "title": "Matched 2D, weakly 3D and finite-3D mean flows",
        "campaign_sha256": sha(args.campaign), "generator_sha256": sha(Path(__file__)),
        "group_count": len(groups), "plotted_run_count": len(source_records),
        "definitions": {
            "U": "Cell-centered Eulerian velocity averaged in the transverse y direction, converted from the saved Lagrangian field: U(x,z,t)=mean_y u_E(x,y,z,t). No temporal, ensemble, or x average in the field panels.",
            "velocity_conversion": "The cached Uc field is Lagrangian. Plots and CSV subtract the actual Stokes drift waves*S_reference*exp(2*k*z) to obtain Eulerian velocity. The no-wave controls subtract zero. Time differences are identical in the two conventions because the Stokes drift is steady.",
            "profile": "Horizontal mean mean_x U(x,z,t); x cells are uniform, so the arithmetic x mean is the horizontal-area mean.",
            "change": "The horizontally averaged current profile at t minus its own initial profile. Initial Eulerian current and q0 are not assumed zero.",
            "field_anomaly": "The lower field row shows U(x,z,t_final)-mean_x U(x,z,0): departure from the initial horizontally averaged profile, retaining the present in-plane eddies and jets. It is not a subtraction of the initial x-z eddy pattern.",
            "roles": {"2d": "Exact 2D initial condition, eta=0", "tiny": "Weak 3D perturbation, initial transverse/2D eddy energy eta=1e-6", "finite": "Finite 3D perturbation, initial transverse/2D eddy energy eta=1"},
            "initial_energy": "Finite perturbations add initial energy. These plots do not compare equal-total-energy initial states.",
            "coordinates": "All saved cell centers are included, with no boundary exclusion, smoothing, or filtering. x=(i+1/2)Lx/Nx; z and dz are stored cache coordinates.",
            "colour_scaling": "Within each matched group, all three total-U panels share the exact data minimum and maximum; all three departures from the initial mean profile share a symmetric range reaching their largest absolute value. No quantile clipping. Colour ranges differ across groups and rows.",
            "units": "Plots use mm/s and m. CSV values use m/s, m and s.",
            "stokes_depth": "delta_S=1/(2k); upper views show z>=-2delta_S.",
            "scope": "Finite-time 600 s individual realizations. Mean fields are derived arrays, not native three-dimensional snapshots. The smaller tiny100 amplitude is not plotted again; it belongs to the separate twin-amplitude test.",
        },
        "overview": overview, "anchor_evolution": evolution, "profiles_csv": final_profiles,
        "groups": public_groups, "sources": source_records,
    }
    (args.output / "index-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(groups)} matched groups and {len(source_records)} run profiles", flush=True)


if __name__ == "__main__":
    main()

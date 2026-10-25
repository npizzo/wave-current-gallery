#!/usr/bin/env python3
"""Render compact public figures from export_member.jl's derived CSVs."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def render(directory):
    directory = Path(directory)
    meta = json.loads((directory / "metadata.json").read_text())
    fields = np.atleast_1d(np.genfromtxt(directory / "mean-fields.csv", delimiter=",", names=True))
    profiles = np.atleast_1d(np.genfromtxt(directory / "mean-profiles.csv", delimiter=",", names=True))
    stats = np.atleast_1d(np.genfromtxt(directory / "velocity-statistics.csv", delimiter=",", names=True))
    frames = np.unique(fields["frame"])
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white", "figure.facecolor": "white"})
    limit = max(float(np.max(np.abs(fields["u_E_m_s"]))) * 1000, 1e-6)
    fig, axes = plt.subplots(len(frames), 1, figsize=(9, 2.1 * len(frames)), squeeze=False,
                             layout="constrained")
    for ax, frame in zip(axes[:, 0], frames):
        a = fields[fields["frame"] == frame]
        x, z = np.unique(a["x_m"]), np.unique(a["z_m"])
        # Extraction order is vertical level then streamwise index.
        image = ax.pcolormesh(x, z, a["u_E_m_s"].reshape(len(z), len(x)) * 1000,
                             shading="nearest", cmap="RdBu_r", vmin=-limit, vmax=limit, rasterized=True)
        ax.set(xlabel="Along-wave position x (m)", ylabel="Depth z (m)",
               title=f"t = {a['time_s'][0]:.2f} s")
    fig.colorbar(image, ax=axes[:, 0], label="Eulerian y-mean velocity (mm/s)", shrink=.8)
    fig.savefig(directory / "eulerian-mean.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5), layout="constrained")
    for frame in frames:
        a = profiles[profiles["frame"] == frame]
        ax.plot(a["mean_u_E_m_s"] * 1000, a["z_m"], label=f"{a['time_s'][0]:.2f} s")
    ax.axvline(0, color="0.7", linewidth=.7)
    ax.set(xlabel="Interior mean Eulerian velocity (mm/s)", ylabel="Depth z (m)")
    ax.legend(title="Simulation time", frameon=False)
    fig.savefig(directory / "mean-profiles.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4), layout="constrained")
    for component, color in zip("uvw", ("#2166ac", "#b2182b", "#547344")):
        ax.plot(stats["time_s"], stats[f"{component}_L_rms_m_s"] * 1000,
                label=f"{component}$_L$", color=color)
    ax.set(xlabel="Simulation time (s)", ylabel="Volume-weighted RMS velocity (mm/s)")
    ax.legend(frameon=False, ncol=3)
    fig.savefig(directory / "lagrangian-rms.png", dpi=180)
    plt.close(fig)
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    render(parser.parse_args().directory)

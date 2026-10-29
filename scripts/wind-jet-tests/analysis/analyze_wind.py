#!/usr/bin/env python3
"""Publishable, reproducible assessment of the wind-forced jet tests.

Only a bundle with a verified completion receipt may supply scientific values.
Native fields stay private. Public exports contain scalar diagnostics, curves,
profiles, criteria and source fingerprints, never machine paths or credentials.
"""
from __future__ import annotations
import argparse, csv, datetime, hashlib, html, json, math, os, re, shutil, tempfile
from pathlib import Path
import numpy as np
from wind_science import (DEFAULT_CRITERIA, closure_window, modal_response,
                          profile_properties, rms_xz, profile_rms, resolve_criteria, mask_regions, current_organization)

PUBLIC_CONFIG = ("nx", "ny", "nz", "lx", "ly", "h", "k", "S", "S_reference", "alpha",
                 "eddy_rms", "injection_wavelength", "perturb_energy_ratio", "seed", "nu",
                 "max_dt", "wind_ustar", "ustar", "wind_balance", "maintain_background",
                 "velocity_convention", "closure", "advection", "waves", "start_time", "stop_time")
MAP_FIELDS = ("Uc", "Wc", "Q", "Qdot_3d_native")
PROFILE_FIELDS = ("UL", "UE", "adv_full", "adv_ky0", "adv_3d", "wave", "closure", "forcing", "pressure_mean",
                  "flux_total", "flux_ky0", "flux_3d", "continuum_flux_rhs")


def clean(v):
    if isinstance(v, dict): return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)): return [clean(x) for x in v]
    if isinstance(v, np.generic): return clean(v.item())
    if isinstance(v, float) and not math.isfinite(v): return None
    return v


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def digest(path):
    with Path(path).open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()


def safe_path(root, name):
    p = Path(name)
    if p.is_absolute() or ".." in p.parts or not p.parts: raise ValueError("Unsafe relative file")
    path = Path(root)/p
    if not path.resolve().is_relative_to(Path(root).resolve()): raise ValueError("Path escapes root")
    return path


def identifier(s):
    if not isinstance(s, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", s):
        raise ValueError("Invalid case ID")
    return s


def numeric_csv(path):
    with Path(path).open(newline="") as f:
        result = []
        for row in csv.DictReader(f):
            result.append({k: (float(v) if v else None) for k, v in row.items()})
    return result


def write_csv(path, rows):
    if not rows: return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader()
        writer.writerows({k: (format(float(v),'.8g') if isinstance(v,(float,np.floating)) and math.isfinite(v) else clean(v)) for k, v in row.items()} for row in rows)


class Bundle:
    def __init__(self, root, entry, receipt):
        self.folder = safe_path(root, entry.get("path", "data/runs/"+entry["id"]))
        self.receipt = receipt
        self.checked = {}

    def file(self, name):
        path = safe_path(self.folder, name)
        expected = self.receipt.get("files", {}).get(name)
        if not expected or not path.is_file(): raise ValueError("Missing verified diagnostic: "+name)
        if name not in self.checked:
            if digest(path) != expected: raise ValueError("Changed completed diagnostic: "+name)
            self.checked[name] = expected
        return path

    def json(self, name): return json.loads(self.file(name).read_text())

    def maps(self, branch):
        config = self.json(branch+"/configuration.json")
        grid = self.json(branch+"/grid.json")
        completed = self.json(branch+"/completed.json")
        manifests = sorted(n for n in self.receipt["files"]
            if re.fullmatch(re.escape(branch)+r"/mean_[0-9]+\.[0-9]+\.json", n))
        if len(manifests) < 3: raise ValueError("Not enough mean maps in "+branch)
        fields = {n: [] for n in MAP_FIELDS}; time = []
        for name in manifests:
            m = self.json(name); time.append(float(m["time"]))
            for field in MAP_FIELDS:
                entry = m["arrays"][field]
                relative = branch+"/"+entry["file"]
                if entry["sha256"] != self.receipt["files"].get(relative):
                    raise ValueError("Map and bundle fingerprints differ")
                if entry["dtype"] != "float64" or entry["order"] != "F": raise ValueError("Unsupported map encoding")
                a = np.fromfile(self.file(relative), dtype="<f8")
                if a.size != np.prod(entry["shape"]): raise ValueError("Map size mismatch")
                a = a.reshape(entry["shape"], order="F")
                if a.ndim != 2 or not np.all(np.isfinite(a)): raise ValueError("Invalid x-z map")
                fields[field].append(a)
        if np.any(np.diff(time) <= 0) or not np.isclose(time[-1], completed["time"], rtol=0, atol=1e-7):
            raise ValueError("Incomplete map timeline")
        result = {n: np.array(v) for n, v in fields.items()}
        result.update(time=np.array(time), z=np.array(grid["z_centers"]), dz=np.diff(grid["z_faces"]))
        result["C"] = result.pop("Qdot_3d_native")
        return config, result

    def profiles(self, branch):
        rows = numeric_csv(self.file(branch+"/jet-profiles.csv"))
        time = np.unique([r["time_s"] for r in rows]); z = np.unique([r["z_m"] for r in rows])
        if len(time) < 3 or len(z) < 8: raise ValueError("Incomplete current profiles")
        arrays = {n: np.empty((len(time), len(z))) for n in PROFILE_FIELDS}
        dz = np.empty(len(z)); seen = set()
        for r in rows:
            i, j = np.searchsorted(time, r["time_s"]), np.searchsorted(z, r["z_m"])
            if (i, j) in seen: raise ValueError("Duplicate profile row")
            seen.add((i, j)); dz[j] = r["dz_m"]
            for name in PROFILE_FIELDS: arrays[name][i, j] = r[name]
        if len(seen) != len(time)*len(z) or not all(np.all(np.isfinite(v)) for v in arrays.values()):
            raise ValueError("Ragged or nonfinite profiles")
        return dict(time=time, z=z, dz=dz, **arrays)


def current_case(root, entry, source, approved):
    status = root/"data/status"/identifier(entry["id"])
    receipt_path = status/"completed.json"
    if receipt_path.exists() and (approved is None or entry["id"] in approved):
        r = json.loads(receipt_path.read_text())
        if r.get("id") != entry["id"] or not r.get("completed") or r.get("source_version") != source:
            raise ValueError("Completion receipt does not match source")
        if r.get("persistent_backup_verified") is not True:
            raise ValueError("Scientific completion requires a verified backup")
        return "complete", r
    if (status/"failed.json").exists(): return "failed", None
    if (status/"scheduler.json").exists():
        state = json.loads((status/"scheduler.json").read_text()).get("state", "").upper()
        if state in ("RUNNING", "COMPLETING"): return "running", None
        if state in ("PENDING", "CONFIGURING", "SUSPENDED", "REQUEUED"): return "queued", None
    if (status/"started.json").exists(): return "running", None
    return ("queued" if (root/"submission.json").exists() else "waiting"), None


def assess_case(root, entry, receipt, criteria):
    bundle = Bundle(root, entry, receipt)
    config, parent = bundle.maps("parent"); profiles = bundle.profiles("parent")
    wind_budget = numeric_csv(bundle.file("parent/wind-budget.csv")) if "parent/wind-budget.csv" in receipt["files"] else []
    if wind_budget:
        parent["native_norms"] = dict(time=np.array([r["time_s"] for r in wind_budget]),
            **{prefix+region:np.array([r[prefix+region] for r in wind_budget])
               for prefix in ("C_native_rms_","q_eddy_rms_") for region in ("full","interior","stokes")})
    windows, closure_history, responses, response_history = [], [], [], []
    for start in entry.get("forecast_times", (600, 1000)):
        branch = "forecast-"+str(start)
        forecast_config, forecast = bundle.maps(branch)
        for name in ("nx", "nz", "lx", "h", "k", "S", "S_reference", "nu", "max_dt", "wind_ustar", "alpha", "waves", "maintain_background", "advection", "closure"):
            if config.get(name) != forecast_config.get(name): raise ValueError("Forecast physics/grid differs: "+name)
        if int(forecast_config["ny"]) != 1: raise ValueError("Forecast is not restricted to transverse invariance")
        if not np.isclose(forecast["time"][0],start,atol=1e-7) or not np.isclose(forecast["time"][-1],start+entry.get("branch_duration",200),atol=1e-7):
            raise ValueError("Forecast does not match the requested physical time window")
        result, history = closure_window(parent, forecast, config, criteria)
        windows.append(result); closure_history.extend(history)
    for mode in entry.get("probe_modes", (1, 2, 4)):
        plus_name, minus_name = f"probe-m{mode}-plus", f"probe-m{mode}-minus"
        pc, mc = bundle.json(plus_name+"/configuration.json"), bundle.json(minus_name+"/configuration.json")
        amp = float(pc["jet_amplitude"])
        if pc.get("jet_mode") != mode or mc.get("jet_mode") != mode or pc.get("jet_sign") != 1 or mc.get("jet_sign") != -1 or mc.get("jet_amplitude") != amp:
            raise ValueError("Invalid opposite-sign mode pair")
        if not pc.get("initial_snapshot_sha256") or pc.get("initial_snapshot_sha256") != mc.get("initial_snapshot_sha256"):
            raise ValueError("Probe pair did not start from the same state")
        if not np.isclose(amp,entry.get("probe_amplitude",.0002),rtol=1e-12,atol=0):raise ValueError("Probe amplitude differs from requested case")
        for bc in (pc,mc):
            for name in ("nx","ny","nz","lx","ly","h","k","S","S_reference","nu","max_dt","wind_ustar","alpha","waves","maintain_background","advection","closure"):
                if bc.get(name)!=config.get(name):raise ValueError("Probe physics/grid differs: "+name)
        pp,mp=bundle.profiles(plus_name),bundle.profiles(minus_name)
        start=entry.get("probe_start",600);stop=start+entry.get("branch_duration",200)
        if any(not np.isclose(p["time"][0],start,atol=1e-7) or not np.isclose(p["time"][-1],stop,atol=1e-7) for p in (pp,mp)):
            raise ValueError("Probe time window differs from requested case")
        response, history = modal_response(pp, mp, mode, amp, float(config["h"]), criteria)
        responses.append(response); response_history.extend(dict(mode=mode, **r) for r in history)
    budget = numeric_csv(bundle.file("parent/regime-budget.csv"))
    z, dz = profiles["z"], profiles["dz"]
    final_profile = profile_properties(profiles["UE"][-1], profiles["UE"][0], z, dz)
    organization, organization_history, extrema_history=current_organization(profiles,float(config["h"]),float(config["eddy_rms"]))
    # A local Rhines diagnostic, not a calibrated input-only prediction of jets.
    beta = np.gradient(np.gradient(profiles["UE"], z, axis=1, edge_order=2), z, axis=1, edge_order=2)
    mask = mask_regions(z,float(config["k"]),criteria["trim_cells"],criteria["active_stokes_efolding_depths"],criteria["reference_vertical_cells"])["active_interior"]
    beta_rms = np.array([profile_rms(a[mask], dz[mask]) for a in beta])
    bt = np.array([r["time_s"] for r in budget]); eddy = np.array([r["eddy_rms"] for r in budget])
    bt, eddy = bt[np.isfinite(eddy)], eddy[np.isfinite(eddy)]
    final_u = float(np.interp(profiles["time"][-1], bt, eddy))
    final_beta = float(beta_rms[-1])
    S = float(config["S"]); ustar = float(config.get("wind_ustar", config.get("ustar", entry.get("env", {}).get("WIND_USTAR", 0.))))
    result = dict(id=entry["id"], status="complete", source_version=receipt["source_version"],
        configuration={k: config[k] for k in PUBLIC_CONFIG if k in config},
        parameters=dict(u_star_m_s=ustar, surface_stokes_m_s=S,
            langmuir_number=math.sqrt(ustar/S) if S > 0 else None,
            alpha=float(config["alpha"]), seed=int(config["seed"]), nu_m2_s=float(config["nu"])),
        closure_windows=windows, mode_responses=responses, final_profile=final_profile,current_organization=organization,
        diagnosed_scale=dict(active_depth_beta_rms_s_inverse_m_inverse=final_beta,
            final_eddy_rms_m_s=final_u, local_rhines_length_m=math.sqrt(final_u/final_beta) if final_beta > 0 else None,
            scope="Uses the evolved current curvature and measured turbulence; no dispersion or universal spacing test is claimed"),
        geometry=dict(final_chi_3d=budget[-1].get("chi_3d"), maximum_chi_3d=max((r["chi_3d"] for r in budget if r.get("chi_3d") is not None), default=None)),
        checks={k: entry[k] for k in ("verify_of", "linearity_of", "group", "check_kind") if k in entry},
        validation=dict(max_native_divergence=max(r["native_divergence_max"] for r in budget),
                        max_pressure_curl=max(r["native_pressure_curl_max"] for r in budget)),
        scope="Conditional mean-closure test and finite-time weak-band response; small correction without jets is not failure")
    curves = dict(time=profiles["time"], z=z, dz=dz, UE=profiles["UE"], UL=profiles["UL"], beta=beta,
                  budget_time=np.array([r["time_s"] for r in budget]), chi=np.array([r.get("chi_3d", np.nan) for r in budget]),
                  eddy=np.array([r.get("eddy_rms", np.nan) for r in budget]))
    for name in ("flux_total","flux_ky0","flux_3d","continuum_flux_rhs","adv_full","adv_ky0","adv_3d","closure","forcing"):
        curves[name]=profiles[name]
    if wind_budget:
        curves["wind_time"] = np.array([r["time_s"] for r in wind_budget])
        curves["roll_energy"] = np.array([r.get("roll_energy_xmean", np.nan) for r in wind_budget])
    return result, closure_history, response_history, curves, bundle.checked, organization_history, extrema_history


def qualify_results(results, entries, criteria):
    """Numerical checks qualify an apparent failure, never invent a jet theorem."""
    by_id = {r["id"]: r for r in results if r["status"] == "complete"}
    by_entry = {e["id"]: e for e in entries}
    for r in by_id.values():
        verifiers = [x for x in by_id.values() if by_entry[x["id"]].get("verify_of") == r["id"]]
        fine = [x for x in verifiers if x["configuration"].get("nz", 0) > r["configuration"].get("nz", 0)]
        timestep = [x for x in verifiers if x["configuration"].get("nz") == r["configuration"].get("nz") and x["configuration"].get("max_dt", 1e99) < r["configuration"].get("max_dt", 0)]
        for window in r["closure_windows"]:
            for region in window["regions"]:
                if not region.get("apparent_failure"): continue
                def match(check):
                    w = next((w for w in check["closure_windows"] if w["start_s"] == window["start_s"]), None)
                    return next((q for q in w["regions"] if q["region"] == region["region"]), None) if w else None
                def confirmed(check):
                    q = match(check)
                    if not q or not q.get("apparent_failure"): return False
                    for key in ("epsilon_C", "max_q_forecast_error", "max_current_forecast_error"):
                        if abs(q[key]-region[key]) > criteria["refinement_metric_relative_tolerance"]*max(abs(region[key]), .1): return False
                    return True
                confirming_fine=[match(x) for x in fine if confirmed(x)]
                confirming_step=[match(x) for x in timestep if confirmed(x)]
                resolved_failure=False
                for f in confirming_fine:
                    for s in confirming_step:
                        epsilon_uncertainty=abs(f["epsilon_C"]-region["epsilon_C"])+abs(s["epsilon_C"]-region["epsilon_C"])
                        safely_weak=region["epsilon_C"]+epsilon_uncertainty<criteria["weak_correction_threshold"]
                        for key in ("max_q_forecast_error","max_current_forecast_error"):
                            uncertainty=abs(f[key]-region[key])+abs(s[key]-region[key])
                            if safely_weak and region[key]-criteria["forecast_error_threshold"]>uncertainty:
                                resolved_failure=True
                if resolved_failure:
                    region["assessment"] = "FAIL"
                    region["reason"] = "Small-correction forecast criterion fails on matched spatial and timestep checks"
            required = [x for x in window["regions"] if x["region"] in ("interior", "active_interior")]
            if any(x["assessment"] == "FAIL" for x in required):
                window["assessment"] = "FAIL"; window["reason"] = "A prescribed depth region has a numerically checked closure failure"
        half = [x for x in by_id.values() if by_entry[x["id"]].get("linearity_of") == r["id"]]
        # Independent seed: same physical controls, resolution, and probe amplitude.
        seed_checks = [x for x in by_id.values() if x["id"] != r["id"] and
            x["parameters"]["seed"] != r["parameters"]["seed"] and
            by_entry[x["id"]].get("probe_amplitude", .0002) == by_entry[r["id"]].get("probe_amplitude", .0002) and
            all(x["configuration"].get(k) == r["configuration"].get(k) for k in
                ("nx", "ny", "nz", "S", "alpha", "nu", "perturb_energy_ratio", "max_dt")) and
            x["parameters"]["u_star_m_s"] == r["parameters"]["u_star_m_s"]]
        for response in r["mode_responses"]:
            response["linearity_verified"] = False; response["independent_seed_positive"] = False
            response["spatial_check_positive"] = False; response["timestep_check_positive"] = False
            response["band_type"] = "broad monotonic shear response" if response["mode"] == 1 else "single interior band" if response["mode"] == 2 else "multiple interior bands"
            if response["assessment"] != "POSITIVE_RESPONSE_CANDIDATE": continue
            mode = response["mode"]
            def mode_of(check): return next(a for a in check["mode_responses"] if a["mode"] == mode)
            def positive(m): return m["assessment"] in ("POSITIVE_RESPONSE_CANDIDATE", "SUPPORTED_POSITIVE_RESPONSE")
            response["linearity_verified"] = any(
                positive(mode_of(x)) and
                abs(mode_of(x)["final_response"]-response["final_response"]) <= criteria["linearity_relative_tolerance"]*max(abs(response["final_response"]), 1.)
                for x in half)
            response["independent_seed_positive"] = any(positive(mode_of(x)) for x in seed_checks)
            def consistent_mode(x):
                m = mode_of(x)
                return positive(m) and abs(m["final_response"]-response["final_response"]) <= criteria["refinement_metric_relative_tolerance"]*max(abs(response["final_response"]), 1.)
            response["spatial_check_positive"] = any(consistent_mode(x) for x in fine)
            response["timestep_check_positive"] = any(consistent_mode(x) for x in timestep)
            if all(response[key] for key in ("linearity_verified", "independent_seed_positive", "spatial_check_positive", "timestep_check_positive")):
                response["assessment"] = "SUPPORTED_POSITIVE_RESPONSE"


def plot_case(output, result, closure_history, response_history, curves):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True);ax=axes.ravel()
    for t in (0, 600, 1000, 1200):
        idx = int(np.argmin(abs(curves["time"]-t)))
        ax[0].plot(curves["UE"][idx]*1000, curves["z"], label=f'{curves["time"][idx]:g} s')
        ax[1].plot((curves["UE"][idx]-curves["UE"][0])*1000,curves["z"],label=f'{curves["time"][idx]:g} s')
    ax[0].set(xlabel="Eulerian mean current (mm/s)", ylabel="Depth (m)", title="Mean current evolves freely")
    ax[0].legend(fontsize=8)
    ax[1].set(xlabel="Change in Eulerian mean current (mm/s)",ylabel="Depth (m)",title="Redistribution from the initial profile")
    ax[2].plot(curves["budget_time"], 100*curves["chi"], color="#a34f30")
    ax[2].set(xlabel="Time (s)", ylabel="3D eddy energy (%)", title="Geometry is a separate diagnostic")
    ax[3].plot(curves["budget_time"], 1000*curves["eddy"], color="#146b70", label="All eddy motion")
    if "roll_energy" in curves and np.all(np.isfinite(curves["roll_energy"])):
        ax[3].plot(curves["wind_time"], 1000*np.sqrt(2*curves["roll_energy"]), label="Along-wave roll component")
    ax[3].set(xlabel="Time (s)", ylabel="RMS speed (mm/s)", title="Available motion and roll organization")
    ax[3].legend(fontsize=8)
    for a in ax: a.grid(alpha=.16)
    fig.savefig(output/"current-and-geometry.png", dpi=130); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for j, start in enumerate((600, 1000)):
        rows = [r for r in closure_history if r["start_s"] == start and r["region"] == "active_interior"]
        for key, label, color in (("epsilon_C", "Accumulated correction", "#8b5e3b"),
                                  ("q_error_normalized", "Vorticity forecast error", "#146b70"),
                                  ("current_error_normalized", "Current forecast error", "#5466a8")):
            axes[0, j].plot([r["time_s"]-start for r in rows], [r[key] for r in rows], label=label, color=color)
        axes[0, j].axhline(.1, ls="--", color=".5", lw=1)
        axes[0, j].set(xlabel="Forecast time (s)", ylabel="Normalized magnitude", title=f"Restart from the actual mean at {start} s")
        axes[0, j].legend(fontsize=8)
    for mode in [r["mode"] for r in result["mode_responses"]]:
        rows = [r for r in response_history if r["mode"] == mode]
        axes[1, 0].plot([r["elapsed_s"] for r in rows], [r["response_amplitude"] for r in rows], label="Mode 1: broad shear" if mode == 1 else f"Mode {mode}: interior bands")
        response = next(r for r in result["mode_responses"] if r["mode"] == mode)
        terms = response["integrated_modal_energy_supplies"]
        x = np.array([0, 1, 2])+({1: -.22, 2: 0, 4: .22}[mode])
        axes[1, 1].bar(x, [terms["adv_full"], terms["closure"], terms["forcing"]+terms["wave"]], width=.2, label=f"Mode {mode}")
    axes[1, 0].axhline(1, ls="--", color=".5", lw=1)
    axes[1, 0].set(xlabel="Time after perturbation (s)", ylabel="Signed response / imposed amplitude", title="Opposite-sign weak-band response")
    axes[1, 1].set(xticks=[0, 1, 2], xticklabels=["Advection", "Viscosity", "Force + waves"], ylabel="Integrated response-energy supply", title="Does turbulent transport reinforce the band?")
    for a in axes.flat: a.grid(alpha=.16); a.legend(fontsize=8)
    if not result['mode_responses']:
        for a in axes[1]:
            a.clear();a.text(.5,.5,'No response probes requested\n(exact 2D null)',ha='center',va='center',transform=a.transAxes);a.set_axis_off()
    fig.savefig(output/"closure-and-response.png", dpi=150); plt.close(fig)


STYLE = """html{color-scheme:light}body{font:17px/1.6 system-ui,sans-serif;color:#203238;background:#f6f8f7;margin:0}main{max-width:1100px;margin:auto;padding:35px 24px 70px}h1{font-size:clamp(2rem,4vw,3rem);line-height:1.12}h2{font-size:1.5rem;margin-top:2rem}a{color:#146b70}p{max-width:88ch}.eyebrow{font-size:.8rem;letter-spacing:.12em;text-transform:uppercase;color:#617a7d}.panel{background:white;border:1px solid #d6e0df;border-radius:12px;padding:20px;margin:22px 0}figure{margin:25px 0}img{width:100%;height:auto;display:block;background:white}figcaption,.small{font-size:.88rem;color:#54666a}table{border-collapse:collapse;width:100%;font-size:.88rem}th,td{text-align:left;border-bottom:1px solid #dce5e3;padding:9px 10px}th{background:#e7efed}.table-scroll{overflow:auto}.PASS{color:#146341}.FAIL{color:#a32920}.INCONCLUSIVE{color:#795c22}.badge{font-size:.8rem;font-weight:700}code{font-size:.85em}nav{display:flex;gap:20px;flex-wrap:wrap}.metrics{display:flex;gap:15px;flex-wrap:wrap}.metric{background:#e6eeec;border-radius:8px;padding:12px 20px}.metric strong{display:block;font-size:1.4rem}details{border-top:1px solid #ddd;padding:12px 0}summary{cursor:pointer}footer{margin-top:35px;color:#607276;font-size:.8rem}"""


def fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}g}"


def badge(state):
    return '<span class="badge '+html.escape(state)+'">'+html.escape(state.replace("_", " "))+'</span>'


def page(title, body, prefix=""):
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+' — Wave-current gallery</title><link rel="stylesheet" href="'+prefix+'style.css"></head><body><main>'+body+'</main></body></html>\n'


def case_page(folder, result):
    c = result["configuration"]; p = result["parameters"]
    rows = []
    for w in result["closure_windows"]:
        for r in w["regions"]:
            rows.append(f'<tr><td>{w["start_s"]:g}–{w["stop_s"]:g} s</td><td>{r["region"]}</td><td>{fmt(r.get("epsilon_C"))}</td><td>{fmt(r.get("max_q_forecast_error"))}</td><td>{fmt(r.get("max_current_forecast_error"))}</td><td>{badge(r["assessment"])}</td></tr>')
    responses = ''.join(f'<tr><td>{r["mode"]}</td><td>{fmt(r["vertical_wavelength_m"])} m</td><td>{fmt(r["final_absolute_gain"])}</td><td>{"yes" if r["eddy_reinforcement"] else "no"}</td><td>{badge(r["assessment"])}</td></tr>' for r in result["mode_responses"])
    extrema=result.get('current_organization',{}).get('late_mean_extrema',[])
    extrema_rows=''.join(f'<tr><td>{e["kind"]}</td><td>{fmt(e["z_m"])}</td><td>{fmt(e["half_prominence_width_m"])}</td><td>{100*e["presence_fraction"]:.0f}%</td><td>{"yes" if e["persistent"] else "no"}</td></tr>' for e in extrema)
    if not extrema_rows:extrema_rows='<tr><td colspan="5">No resolved extrema in the late mean profile above the stated prominence floor.</td></tr>'
    body = f'''<nav><a href="../../">All wind-forced tests</a><a href="../../../../">Gallery</a></nav>
<p class="eyebrow">Verified simulation bundle</p><h1>{html.escape(result['id'])}</h1>
<p>Water-side friction velocity {p['u_star_m_s']*1000:g} mm/s; surface Stokes drift {p['surface_stokes_m_s']*1000:.3g} mm/s; α = {p['alpha']:g}; seed {p['seed']}; grid {c['nx']} × {c['ny']} × {c['nz']}.</p>
<p><a href="summary.json">All numeric results</a> · <a href="closure-history.csv">Closure histories</a> · <a href="response-history.csv">Band-response histories and mode mixing</a> · <a href="profiles.csv">Mean profiles and momentum fluxes</a> · <a href="current-organization.csv">Vertical current modes</a></p>
<figure><img src="current-and-geometry.png" alt="Mean-current profiles, transverse geometry and motion strength"><figcaption>A current change can be driven directly by wind. Neither its magnitude nor the final 3D energy fraction alone establishes spontaneous jets.</figcaption></figure>
<h2>Does the mean profile contain persistent interior extrema?</h2><div class="table-scroll"><table><thead><tr><th>Type</th><th>Depth (m)</th><th>Half-prominence width (m)</th><th>Presence in final 200 s</th><th>At least 80%?</th></tr></thead><tbody>{extrema_rows}</tbody></table></div>
<p class="small">The prominence floor is 1% of the initial base eddy speed. Extrema are matched within half their width or two cells, whichever is larger. These describe the profile; they do not independently identify a beta-driven jet. Full-cadence data determine the metrics; public profiles and vertical-mode histories are sampled every 40 s.</p>
<h2>Does the reduced mean equation predict the next 200 seconds?</h2>
<div class="table-scroll"><table><thead><tr><th>Window</th><th>Depth region</th><th>εC</th><th>Vorticity error</th><th>Current error</th><th>Assessment</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="small">Both interior regions must pass. εC ≤ 0.1 is the predeclared small-correction criterion; both maximum forecast errors must also be ≤ 0.1. Quiet windows and correction values outside the assumed regime are inconclusive. An apparent failure requires matching spatial and timestep verification.</p>
<figure><img src="closure-and-response.png" alt="Restart forecast errors and weak band response with momentum-budget reinforcement"><figcaption>Forecasts start from the actual contemporaneous transverse mean. Opposite-sign perturbations isolate a weak band’s response from the evolving wind-driven mean.</figcaption></figure>
<h2>Does turbulent transport reinforce a weak band?</h2><div class="table-scroll"><table><thead><tr><th>Mode</th><th>Wavelength</th><th>Final gain</th><th>Eddy reinforcement</th><th>Assessment</th></tr></thead><tbody>{responses}</tbody></table></div>
<p class="small">A positive candidate has gain ≥ 1.2, positive integrated advective reinforcement, and a closed response budget. Supporting a positive response also requires amplitude, independent-seed, spatial and timestep checks. Mode 1 is a broad monotonic shear perturbation; its growth alone is not jet formation. Modes 2 and 4 contain interior bands. This is a finite-time response, not a proven unstable eigenmode; absence of growth does not rule out finite-amplitude jets.</p>
<footer>Source {result['source_version']}. Native three-dimensional snapshots remain private.</footer>'''
    (folder/"index.html").write_text(page(result["id"], body, "../../"))


def summary_plots(output, results):
    completed = [r for r in results if r["status"] == "complete"]
    if not completed: return []
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    colors = {0.: "#737c83", .002: "#237b9b", .004: "#b26830", .008: "#70499a"}
    for r in completed:
        p = r["parameters"]; color = colors.get(p["u_star_m_s"], "#146b70")
        for w in r["closure_windows"]:
            region = next(q for q in w["regions"] if q["region"] == "active_interior")
            if not region.get("active"): continue
            x = region["epsilon_C"]
            for ax, key in zip(axes, ("max_q_forecast_error", "max_current_forecast_error")):
                ax.scatter(max(x, 1e-7), max(region[key], 1e-7), c=color,
                           marker="o" if w["start_s"] == 600 else "s", s=40, alpha=.8)
    for ax, title in zip(axes, ("Vorticity forecast", "Mean-current forecast")):
        ax.set(xscale="log", yscale="log", xlabel="Accumulated omitted correction εC", ylabel="Maximum normalized error", title=title)
        ax.axvline(.1, ls="--", c=".5"); ax.axhline(.1, ls="--", c=".5"); ax.grid(alpha=.16)
    fig.savefig(output/"closure-summary.png", dpi=170); fig.savefig(output/"closure-summary.pdf"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for r in completed:
        p = r["parameters"]
        if r.get("checks", {}).get("verify_of") or r.get("checks", {}).get("linearity_of"): continue
        for a in r["mode_responses"]:
            axes[0].scatter(p["u_star_m_s"]*1000, a["final_absolute_gain"],
                marker={1:"o",2:"s",4:"^"}[a["mode"]], facecolors="none" if not a["eddy_reinforcement"] else colors.get(p["u_star_m_s"],"#146b70"),
                edgecolors=colors.get(p["u_star_m_s"],"#146b70"), s=50)
        axes[1].scatter(p["u_star_m_s"]*1000, 1000*r["final_profile"]["current_change_rms_m_s"],
                        marker="o" if p["seed"]==1 else "s", color=colors.get(p["u_star_m_s"],"#146b70"), s=45)
    axes[0].axhline(1, ls="--", c=".5"); axes[0].axhline(1.2, ls=":", c=".5")
    axes[0].set(xlabel="Water-side friction velocity (mm/s)", ylabel="Final weak-band response gain", title="Broad shear / single band / multiple bands")
    axes[1].set(xlabel="Water-side friction velocity (mm/s)", ylabel="RMS mean-current change (mm/s)", title="Redistribution, including direct wind response")
    for ax in axes: ax.grid(alpha=.16)
    fig.savefig(output/"response-summary.png", dpi=170); fig.savefig(output/"response-summary.pdf"); plt.close(fig)
    return ["closure-summary.png", "response-summary.png"]


def main():
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--approved-runs", type=Path)
    args = p.parse_args(); root = args.root.resolve(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    campaign = json.loads((root/"campaign.json").read_text()); source = json.loads((root/"source-manifest.json").read_text())
    criteria = resolve_criteria(campaign.get("criteria", {}))
    approved = set(json.loads(args.approved_runs.read_text())) if args.approved_runs else None
    results, plotdata, verified = [], {}, {}
    cache = root/"data/analysis-cache/wind"; cache.mkdir(parents=True, exist_ok=True)
    code_hashes = {name: digest(Path(__file__).with_name(name)) for name in ("analyze_wind.py", "wind_science.py")}
    for entry in campaign["runs"]:
        state, receipt = current_case(root, entry, source["version"], approved)
        if state != "complete":
            results.append(dict(id=entry["id"], status=state)); continue
        key = hashlib.sha256(json.dumps([receipt, criteria, code_hashes], sort_keys=True).encode()).hexdigest()
        folder = cache/identifier(entry["id"]); marker = folder/"cache.json"
        valid = marker.exists() and json.loads(marker.read_text()).get("key") == key
        if valid:
            old = json.loads(marker.read_text()); valid = all((folder/n).is_file() and digest(folder/n)==sha for n,sha in old.get("files",{}).items())
        if valid:
            # The cache avoids recomputation, not scientific input validation.
            # A mutated source file must never inherit an old successful report.
            live = Bundle(root, entry, receipt)
            for name, expected in json.loads((folder/"input-hashes.json").read_text()).items():
                if receipt["files"].get(name) != expected: raise ValueError("Cached input inventory differs from receipt")
                live.file(name)
        if not valid:
            result, ch, rh, curves, hashes, org, extrema = assess_case(root, entry, receipt, criteria)
            tmp = Path(tempfile.mkdtemp(prefix=entry["id"]+"-", dir=cache))
            try:
                write_json(tmp/"summary.json", result); write_csv(tmp/"closure-history.csv", ch); write_csv(tmp/"response-history.csv", rh)
                if not rh:
                    (tmp/"response-history.csv").write_text("mode,time_s,elapsed_s,response_amplitude\n")
                write_json(tmp/"input-hashes.json", hashes); np.savez_compressed(tmp/"plot-data.npz", **curves)
                write_csv(tmp/"profiles.csv", [dict(time_s=float(t), z_m=float(z), U_E_m_s=float(curves["UE"][i,j]),
                    U_L_m_s=float(curves["UL"][i,j]), generated_U_E_m_s=float(curves["UE"][i,j]-curves["UE"][0,j]),beta_eff_s_inverse_m_inverse=float(curves["beta"][i,j]),
                    **{name:float(curves[name][i,j]) for name in ("flux_total","flux_ky0","flux_3d","continuum_flux_rhs","adv_full","adv_ky0","adv_3d","closure","forcing")})
                    for i,t in enumerate(curves["time"]) if np.isclose(t/40,round(t/40),atol=1e-8) or i==len(curves["time"])-1 for j,z in enumerate(curves["z"])])
                write_csv(tmp/"current-organization.csv",[r for r in org if np.isclose(r["time_s"]/40,round(r["time_s"]/40),atol=1e-8)])
                write_json(tmp/"extrema-history.json",[r for r in extrema if np.isclose(r["time_s"]/40,round(r["time_s"]/40),atol=1e-8)])
                plot_case(tmp, result, ch, rh, curves)
                files={x.name:digest(x) for x in tmp.iterdir() if x.is_file()}; write_json(tmp/"cache.json", dict(key=key, files=files))
                if folder.exists(): shutil.rmtree(folder)
                tmp.replace(folder)
            finally:
                if tmp.exists(): shutil.rmtree(tmp)
        result=json.loads((folder/"summary.json").read_text()); results.append(result)
        verified[entry["id"]]=json.loads((folder/"input-hashes.json").read_text())
    qualify_results(results, campaign["runs"], criteria)
    (out/"style.css").write_text(STYLE)
    for result in results:
        if result["status"] != "complete": continue
        folder=out/"runs"/result["id"]; folder.mkdir(parents=True, exist_ok=True)
        for name in ("closure-history.csv","response-history.csv","profiles.csv","current-organization.csv","extrema-history.json","current-and-geometry.png","closure-and-response.png"):
            shutil.copy2(cache/result["id"]/name, folder/name)
        write_json(folder/"summary.json", result); case_page(folder,result)
    figures=summary_plots(out,results)
    counts={state:sum(r["status"]==state for r in results) for state in ("complete","running","queued","waiting","failed")}
    windows=[w for r in results if r["status"]=="complete" for w in r["closure_windows"]]
    assessments={state:sum(w["assessment"]==state for w in windows) for state in ("PASS","FAIL","INCONCLUSIVE")}
    supported=[(r["id"], m["mode"]) for r in results if r["status"]=="complete" for m in r["mode_responses"] if m["assessment"]=="SUPPORTED_POSITIVE_RESPONSE" and m["mode"] in (2,4)]
    summary=dict(title="Wind forcing, mean closure and jet response", source_version=source["version"], counts=counts,
        criteria=criteria, closure_assessments=assessments, supported_positive_responses=supported, runs=results,
        mechanism_assessment="INCONCLUSIVE", mechanism_reason="This campaign tests conditional closure and weak-band response. It does not yet independently verify vortex-wave dispersion or an inverse energy-transfer mechanism.")
    write_json(out/"summary.json",summary); write_json(out/"criteria.json",criteria)
    write_json(out/"provenance.json",dict(source_version=source["version"],analysis_sources=code_hashes,verified_inputs=verified,
        campaign_sha256=digest(root/"campaign.json"),public_profile_export_interval_s=40,public_csv_significant_digits=8,
        science_profile_interval_s=5,scope="Only verified backed-up bundles. Full precision and full cadence used for analysis. Native snapshots remain private."))
    metrics=[]
    for r in results:
        if r["status"] != "complete": continue
        for w in r["closure_windows"]:
            for region in w["regions"]:
                metrics.append(dict(id=r["id"],**r["parameters"],**{k:v for k,v in region.items() if not isinstance(v,dict)}))
    write_csv(out/"closure-metrics.csv",metrics)
    response_metrics=[dict(id=r["id"],**r["parameters"],**{k:v for k,v in m.items() if not isinstance(v,(dict,list))}) for r in results if r["status"]=="complete" for m in r["mode_responses"]]
    write_csv(out/"response-metrics.csv",response_metrics)
    rows=''.join('<tr><td>'+('<a href="runs/'+r['id']+'/">'+html.escape(r['id'])+'</a>' if r['status']=='complete' else html.escape(r['id']))+'</td><td>'+r['status']+'</td><td>'+(' / '.join(w['assessment'] for w in r['closure_windows']) if r['status']=='complete' else 'Awaiting verified results')+'</td></tr>' for r in results)
    figs=''
    if figures:
        figs='''<figure><img src="closure-summary.png" alt="Accumulated 3D correction versus restarted 2D forecast error"><figcaption>Active interior region; circles start at 600 s and squares at 1000 s. Both axes are measured independently. Dashed lines mark the predeclared 0.1 thresholds. Quiet windows are excluded from this figure, and both depth regions enter the assessment. <a href="closure-summary.pdf">PDF</a></figcaption></figure>
<figure><img src="response-summary.png" alt="Wind forcing versus weak-band response and current redistribution"><figcaption>Filled symbols have positive integrated advective reinforcement. The chart includes background-current and wave controls; individual pages identify each configuration. Current redistribution alone is not a jet detector. <a href="response-summary.pdf">PDF</a></figcaption></figure>'''
    conclusion=(f'{assessments["PASS"]} forecast windows support the reduced mean dynamics; {assessments["FAIL"]} have a numerically checked failure; {assessments["INCONCLUSIVE"]} remain inconclusive or outside the small-correction regime.' if windows else 'The tests are defined and awaiting completed, verified simulation bundles.')
    body=f'''<nav><a href="../../">Gallery</a><a href="../vorticity-moments/">Vorticity moments</a><a href="#assessment">Assessment</a></nav>
<p class="eyebrow">Sustained wind forcing · preregistered tests</p><h1>When does the mean evolve autonomously—and when do bands grow?</h1>
<p>Surface stress continually supplies momentum while the Stokes profile affects flow organization. These experiments test the reduced mean-vorticity equation separately from the formation of depth-banded currents. Small three-dimensional feedback with no jets is consistent with a necessary, rather than sufficient, applicability argument.</p>
<div class="metrics">{''.join(f'<div class="metric"><strong>{n}</strong>{state}</div>' for state,n in counts.items())}</div>
<section class="panel"><h2>What counts as a pass?</h2><p><strong>Mean closure:</strong> integrate the RMS native omitted tendency over a 200 s window, normalized by the starting x-dependent vorticity. Restart a 2D forecast from the actual transverse mean. A pass requires εC ≤ 0.1 and maximum normalized vorticity and current errors ≤ 0.1 in both prescribed interior regions. The 0.05 and 0.2 alternatives are retained in the downloads.</p><p><strong>Band response:</strong> compare opposite-sign weak current perturbations at three vertical wavelengths. Positive candidates require amplification, reinforcing turbulent momentum transport and a closed tendency budget. Amplitude, independent-seed, spatial and timestep checks qualify a supported response. Mode 1 tests broad shear; only modes 2 and 4 test interior bands.</p><p><strong>Scope:</strong> wind forcing and dissipation remain in every reduced forecast. This is not a test of unforced global vorticity conservation, and a nonzero wind-driven current is not automatically a jet.</p><p><a href="PROTOCOL.txt">Full protocol</a> · <a href="criteria.json">Exact criteria</a> · <a href="campaign.json">Simulation matrix</a> · <a href="summary.json">All results</a> · <a href="provenance.json">Provenance</a></p></section>
{figs}<h2>Simulation bundles</h2><div class="table-scroll"><table><thead><tr><th>Case</th><th>Status</th><th>Closure at 600 / 1000 s</th></tr></thead><tbody>{rows}</tbody></table></div>
<section class="panel" id="assessment"><h2>What can we conclude?</h2><p>{conclusion}</p><p>{len(supported)} case/mode combinations have a positive response supported by the specified amplitude, independent-seed, spatial and timestep checks.</p><p><strong>The broader mechanism remains unassessed:</strong> these measurements do not independently establish vortex-wave dispersion or inverse energy transfer. A small correction without a growing band does not fail the necessary applicability argument. A failed numerical job is a computational failure, not a physical falsification.</p></section>
<footer>Simulation source {source['version']}. Results update after each bundle is completed, verified and backed up. No native snapshots or credentials are published.</footer>'''
    (out/"index.html").write_text(page("Wind forcing and jet response",body))
    for name in ("PROTOCOL.txt","campaign.json","source-manifest.json"):
        if (root/name).exists(): shutil.copy2(root/name,out/name)
    size=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    if size>30*1024**2: raise ValueError("Wind test page exceeds 30 MiB publication budget")
    print(json.dumps(dict(counts=counts,closure_assessments=assessments,public_bytes=size)))


if __name__=="__main__": main()

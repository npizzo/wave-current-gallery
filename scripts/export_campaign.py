#!/usr/bin/env python3
"""Publish derived products from verified finished packet runs, never live JLD2.

The caller supplies the gallery update lock. Scientific inputs are read-only;
all extraction and rendering occur in an external cache. The manifest is the
last file replaced, so an interrupted update leaves the previous page usable.
"""
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile

# Importing the campaign's guard must not create __pycache__ in its live tree.
sys.dont_write_bytecode = True

SEEDS = (1, 2, 3, 4, 6, 7, 8)
COMMIT = "9ff503ca9bf4672a30f32c5dcfcc336f0be6faed"
MEMBERS = ("turbulence_control", "packet_turbulence")
LABELS = {"turbulence_control": "No-wave control", "packet_turbulence": "Wave packet + turbulence",
          "packet_null": "Wave packet without turbulence", "quiescent_control": "Quiescent control"}
NATIVE = ("metadata.jld2", "run_summary.jld2", "y_averages.jld2", "statistics.jld2")
FAILED_STATES = {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "PREEMPTED"}


def scheduler_states(submission, runner=subprocess.run):
    """Optional read-only accounting. Unavailable accounting never invents state."""
    jobs = [str(submission.get(k) or "") for k in ("array_job", "analysis_job")]
    if not all(j.isdigit() for j in jobs):
        return {}
    try:
        result = runner(["sacct", "--array", "--noheader", "--parsable2", "--allocations",
                         "--jobs=" + ",".join(jobs), "--format=JobID%64,State%32"],
                        check=True, text=True, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return {}
    states = {}
    for line in result.stdout.splitlines():
        fields = line.strip().split("|")
        if len(fields) < 2:
            continue
        job = fields[0].strip()
        if job not in jobs and not re.fullmatch(re.escape(jobs[0]) + r"_[0-7]", job):
            continue
        state = fields[1].strip().split()[0].rstrip("+") if fields[1].strip() else "UNKNOWN"
        if re.fullmatch(r"[A-Z_]+", state):
            states[job] = state
    return states


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def within(root, relative):
    relative = PurePosixPath(str(relative))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts or "\\" in str(relative):
        raise RuntimeError("Unsafe provenance path")
    path = root.joinpath(*relative.parts).resolve()
    if not path.is_relative_to(root.resolve()):
        raise RuntimeError("Provenance path escaped campaign")
    return path


def load_guard(root):
    spec = importlib.util.spec_from_file_location("packet_campaign_guard", root / "scripts/guard.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_receipt(root, path, fingerprints, required=()):
    """A present but bad receipt is an error, not a pending member."""
    record = read_json(path)
    if record.get("completed") is not True or record.get("fingerprints") != fingerprints:
        raise RuntimeError("Member completion receipt failed source/completion checks")
    outputs = record.get("verified_outputs")
    if not isinstance(outputs, list) or not outputs:
        raise RuntimeError("Receipt has no verified outputs")
    seen = set()
    for item in outputs:
        relative = item["path"]
        if relative in seen:
            raise RuntimeError("Duplicate receipt output")
        seen.add(relative)
        source = within(root, relative)
        if not source.is_file() or digest(source) != item["sha256"]:
            raise RuntimeError("Completed member output was changed or removed")
    if not set(required).issubset(seen):
        raise RuntimeError("Completion receipt does not cover this member")
    return record


def member_directory(root, member, seed, timestep="0.020"):
    prefix = f"seed_{seed:04d}" if member in MEMBERS else "quiescent"
    return root / "data/campaign/case_1D/M2" / member / f"{prefix}_dt{timestep}_boundedx"


def finished_step(root, member, seed, timestep, fingerprints):
    path = root / f"data/receipts/{member}-seed{seed}-dt{timestep}-completed.json"
    if not path.exists():
        return None
    directory = member_directory(root, member, seed, timestep)
    required = [str((directory / name).relative_to(root)) for name in NATIVE[:2]]
    return verify_receipt(root, path, fingerprints, required), directory


def input_hashes(directory):
    # Raw time-series are not hashed by the original completion receipt. After
    # that receipt is verified, independently hash them before AND after reading.
    return {name: digest(directory / name) for name in NATIVE}


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".manifest-", delete=False) as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write("\n")
        temporary = Path(file.name)
    temporary.replace(path)


def ensure_cache(root, directory, record, cache, scripts, python, julia, runner=subprocess.run):
    before = input_hashes(directory)
    identity = {"receipt": record, "inputs": before,
                "exporter": {name: digest(scripts / name) for name in ("export_campaign.py", "export_member.jl", "render_member.py")}}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    destination = cache / key
    expected = ("metadata.json", "mean-profiles.csv", "velocity-statistics.csv",
                "eulerian-mean.png", "mean-profiles.png", "lagrangian-rms.png")
    seal = destination / "complete.json"
    if seal.exists():
        saved = read_json(seal)
        if saved.get("inputs") != before or set(saved.get("files", {})) != set(expected):
            raise RuntimeError("Invalid gallery cache seal")
        if any(digest(destination / name) != checksum for name, checksum in saved["files"].items()):
            raise RuntimeError("Gallery cache was changed")
        return destination, key
    cache.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="extract-", dir=cache))
    try:
        env = os.environ.copy()
        env.update(JULIA_DEPOT_PATH=str(root / "depot"), JULIA_PROJECT=str(root / "upstream"),
                   MPLBACKEND="Agg", MPLCONFIGDIR=str(cache / "matplotlib"),
                   JULIA_NUM_THREADS=env.get("SLURM_CPUS_PER_TASK", env.get("JULIA_NUM_THREADS", "2")))
        runner([julia, "--startup-file=no", f"--project={root / 'upstream'}",
                str(scripts / "export_member.jl"), str(root), str(directory), str(scratch)],
               cwd=root, env=env, check=True)
        runner([python, str(scripts / "render_member.py"), str(scratch)], cwd=root, env=env, check=True)
        if before != input_hashes(directory):
            raise RuntimeError("Finished input changed during extraction; refusing publication")
        atomic_json(scratch / "complete.json", {"inputs": before, "files": {name: digest(scratch / name) for name in expected}})
        scratch.rename(destination)
    except BaseException:
        shutil.rmtree(scratch, ignore_errors=True)
        raise
    return destination, key


def copy_public(source, gallery, relative):
    destination = within(gallery, relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name("." + destination.name + ".tmp")
    shutil.copyfile(source, temp)
    temp.replace(destination)
    return str(relative)


def public_step(cached, key, gallery, member, seed, timestep):
    meta = read_json(cached / "metadata.json")
    if meta.get("member") != member or meta.get("seed") != seed:
        raise RuntimeError("Derived member identity mismatch")
    title = LABELS[member] + (f" · Δt = {timestep} s" if member == "packet_null" else "")
    # Content-addressed URLs prevent stale image caching and make an update's
    # manifest refer only to its own complete derived assets.
    prefix = f"assets/members/{key[:20]}"
    captions = {
        "eulerian-mean.png": "Saved y-averaged Eulerian along-wave velocity: prescribed Stokes drift is subtracted at native x faces, then adjacent faces are averaged. Same colour scale within this figure. Display samples at most about 384 x positions; every depth level is retained. This is one realization, not an ensemble result.",
        "mean-profiles.png": "Eulerian y-mean velocity averaged over 1 ≤ x ≤ 11 m at selected times. Absolute profiles retain the mean offset. These snapshots use different windows from the final laboratory comparison.",
        "lagrangian-rms.png": "Native volume-weighted RMS of the full Lagrangian velocity, including its mean and prescribed wave contribution. This is not turbulent RMS, an Eulerian energy, or a mixing-rate diagnostic."
    }
    if member not in MEMBERS:
        captions["eulerian-mean.png"] = captions["eulerian-mean.png"].replace(
            "This is one realization, not an ensemble result.", "This is a deterministic control with no initial turbulence.")
    titles = {"eulerian-mean.png": "Eulerian current through the tank", "mean-profiles.png": "Current profiles", "lagrangian-rms.png": "Velocity history"}
    figures = [{"src": copy_public(cached / name, gallery, f"{prefix}/{name}"),
                "title": f"{title}: {titles[name]}", "caption": captions[name]} for name in captions]
    downloads = [{"href": copy_public(cached / name, gallery, f"data/members/{key[:20]}/{name}"),
                  "label": f"{title}: {label}"} for name, label in
                 (("mean-profiles.csv", "Eulerian profiles (CSV)"), ("velocity-statistics.csv", "Lagrangian statistics (CSV)"))]
    return figures, downloads, {"member": member, "final_time_s": float(meta["final_time_s"]),
                                "iterations": int(meta["iterations"]), "dt_s": float(meta["dt_s"])}


def verified_final(root, fingerprints, all_arrays_complete):
    path = root / "data/analysis/completed.json"
    if not path.exists():
        return None
    if not all_arrays_complete:
        raise RuntimeError("Analysis receipt exists before all array completions")
    record = read_json(path)
    if record.get("completed") is not True or record.get("fingerprints") != fingerprints:
        raise RuntimeError("Final analysis is incomplete or source-mismatched")
    output = within(root, record["output"])
    if not output.is_relative_to((root / "data/analysis").resolve()):
        raise RuntimeError("Analysis output outside analysis directory")
    files = record.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("Final analysis has no sealed files")
    verified = {}
    for item in files:
        source = within(root, item["path"])
        if not source.is_relative_to(output) or digest(source) != item["sha256"]:
            raise RuntimeError("Final analysis file failed integrity check")
        relative = str(source.relative_to(output))
        if relative in verified:
            raise RuntimeError("Duplicate final output")
        verified[relative] = source
    required = {"figure3-reproduction.png", "figure3-reproduction.csv", "reference-comparison.json",
                "added/dimensionality.png", "added/momentum-flux-partition.png", "added/ensemble-snapshots.csv"}
    if not required.issubset(verified):
        raise RuntimeError("Final analysis receipt does not contain required outputs")
    return record, verified


def final_products(final, gallery):
    record, files = final
    key = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()[:20]
    figures = []
    definitions = (
        ("figure3-reproduction.png", "Laboratory comparison", "Seven retained realizations of bounded case 1.D. The original analysis shows raw and paired/control-corrected Eulerian changes; centred-shape agreement is distinguished from the absolute offset."),
        ("added/dimensionality.png", "Transverse invariance of the eddies", "Ensemble-incoherent eddy energy in ky = 0 modes and vertical velocity variance. The repeatable packet response is removed. This diagnostic does not itself establish a Taylor–Proudman regime or transport barrier."),
        ("added/momentum-flux-partition.png", "Vertical momentum transport", "Eulerian vertical momentum-flux covariance partitioned between ky = 0 and ky ≠ 0 eddies after ensemble-response subtraction. Flux signs and changes should be assessed alongside the current; no tracer diffusivity is measured."),
    )
    for name, title, caption in definitions:
        figures.append({"src": copy_public(files[name], gallery, f"assets/results/{key}/{Path(name).name}"), "title": title, "caption": caption})
    downloads = []
    for name, label in (("figure3-reproduction.csv", "Laboratory-comparison profiles (CSV)"),
                        ("added/ensemble-snapshots.csv", "Ensemble dimensionality and flux (CSV)"),
                        ("added/null-timestep-check.csv", "Deterministic null timestep check (CSV)")):
        if name in files:
            downloads.append({"href": copy_public(files[name], gallery, f"data/results/{key}/{Path(name).name}"), "label": label})
    comparison = read_json(files["reference-comparison.json"])
    if comparison.get("source_commit") != COMMIT:
        raise RuntimeError("Reference comparison source commit mismatch")
    metrics = []
    for mode, label in (("raw", "Raw LES"), ("paired", "Paired/control-corrected LES")):
        data = comparison["comparison"][mode]
        for field, title, note in (
            ("lab_centered_shape_rmse_mm_s", "centred profile RMSE", "Each curve's native PIV-window arithmetic mean is removed; correlated digitized laboratory samples, not a goodness-of-fit test."),
            ("lab_minus_les_window_offset_mm_s", "laboratory − LES window offset", "Mean offset before centring; the absolute and shape comparisons answer different questions."),
            ("upstream_profile_rmse_in_piv_window_mm_s", "difference from saved upstream profile", "RMSE over −1.2 ≤ k₀z ≤ −0.1; turbulent realizations need not be hardware-identical.")):
            value = float(data[field])
            if not __import__("math").isfinite(value):
                raise RuntimeError("Nonfinite final metric")
            metrics.append({"label": f"{label}: {title}", "value": round(value, 5), "unit": "mm/s", "note": note})
    # Reconstruct a public JSON rather than exposing arbitrary source metadata.
    summary = {"source_commit": COMMIT, "metrics": metrics,
               "scope": "Completed seven-seed reproduction. These are measurements, not a convergence or significance claim."}
    relative = f"data/results/{key}/comparison-summary.json"
    atomic_json(within(gallery, relative), summary)
    downloads.append({"href": relative, "label": "Comparison summary (JSON)"})
    return figures, downloads, metrics


def validate_public_manifest(value):
    text = json.dumps(value, ensure_ascii=False, allow_nan=False)
    # Do not serialize source records, command lines, local paths or credentials.
    if re.search(r"(?:/work/|/Users/|/home/|/u/|/scratch/|private[_ -]?key|BEGIN .*PRIVATE KEY|@login)", text, re.I):
        raise RuntimeError("Private path or credential marker in public manifest")
    def walk(item):
        if isinstance(item, dict):
            for key, val in item.items():
                if key in ("src", "href"):
                    p = PurePosixPath(val)
                    if p.is_absolute() or ".." in p.parts or ":" in val or "\\" in val:
                        raise RuntimeError("Unsafe public asset URL")
                walk(val)
        elif isinstance(item, list):
            for child in item:
                walk(child)
    walk(value)


def export(root, gallery, cache, scripts, *, guard=None, extractor=ensure_cache):
    root, gallery, cache, scripts = (Path(p).resolve() for p in (root, gallery, cache, scripts))
    if cache.is_relative_to(root) or gallery.is_relative_to(root):
        raise RuntimeError("Gallery/cache must be outside the scientific campaign")
    guard = guard or load_guard(root)
    submission = guard.check_submission(root)
    scheduler = scheduler_states(submission)
    fingerprints = guard.fingerprints(root)
    if getattr(guard, "COMMIT", COMMIT) != COMMIT:
        raise RuntimeError("Unexpected upstream source")
    validation = read_json(root / "validation/passed.json")
    validation_match = re.fullmatch(r"report-(\d+)\.json", Path(validation.get("report", "")).name)
    validation_job = validation_match.group(1) if validation_match else None
    arrays = []
    for index in range(8):
        path = root / f"data/receipts/array-{index}-completed.json"
        if path.exists():
            record = verify_receipt(root, path, fingerprints)
            if record.get("index") != index:
                raise RuntimeError("Array identity mismatch")
            arrays.append(index)
    # Resolve a sealed final analysis before producing member assets.
    final = verified_final(root, fingerprints, len(arrays) == 8)
    python = str(root / "runtime/python-env/bin/python3")
    julia = (root / "runtime/julia-path.txt").read_text().strip()
    members = []
    complete_pairs = 0
    for index, seed in enumerate(SEEDS, 1):
        figures, downloads, summaries = [], [], []
        for member in MEMBERS:
            step = finished_step(root, member, seed, "0.020", fingerprints)
            if step:
                record, directory = step
                cached, key = extractor(root, directory, record, cache, scripts, python, julia)
                figs, data, summary = public_step(cached, key, gallery, member, seed, "0.020")
                figures.extend(figs); downloads.extend(data); summaries.append(summary)
        status = "complete" if len(summaries) == 2 else "partial" if summaries else "pending"
        complete_pairs += status == "complete"
        slurm_state = scheduler.get(f"{submission.get('array_job')}_{index}")
        if status != "complete" and slurm_state in FAILED_STATES:
            status = "failed"
        state_note = f" Scheduler reports {slurm_state}." if slurm_state else ""
        members.append({"id": f"seed-{seed}", "seed": seed, "status": status, "label": f"Realization {seed}",
                        "summary": f"{len(summaries)} of 2 matched runs verified complete.{state_note}",
                        "scheduler_state": slurm_state,
                        "runs": summaries, "figures": figures, "downloads": downloads})
    figures, downloads, summaries = [], [], []
    for member, timestep in (("quiescent_control", "0.020"), ("packet_null", "0.020"), ("packet_null", "0.010")):
        step = finished_step(root, member, 0, timestep, fingerprints)
        if step:
            record, directory = step
            cached, key = extractor(root, directory, record, cache, scripts, python, julia)
            figs, data, summary = public_step(cached, key, gallery, member, 0, timestep)
            figures.extend(figs); downloads.extend(data); summaries.append(summary)
    control_state = scheduler.get(f"{submission.get('array_job')}_0")
    control_status = "complete" if len(summaries) == 3 else "failed" if control_state in FAILED_STATES else "partial" if summaries else "pending"
    state_note = f" Scheduler reports {control_state}." if control_state else ""
    members.append({"id": "controls", "seed": None, "status": control_status,
                    "label": "Deterministic controls", "summary": f"{len(summaries)} of 3 controls verified complete.{state_note}",
                    "scheduler_state": control_state,
                    "runs": summaries, "figures": figures, "downloads": downloads})
    if final and (complete_pairs != 7 or len(summaries) != 3):
        raise RuntimeError("Final analysis requires every individual member completion receipt")
    results, result_downloads, metrics = final_products(final, gallery) if final else ([], [], [])
    status = "complete" if final else "analysis_pending" if len(arrays) == 8 else "running"
    if not final and any(s in FAILED_STATES for s in scheduler.values()):
        status = "failed"
    manifest = {"schema_version": 1, "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "title": "Surface waves & turbulence",
        "campaign": {"id": "packet-1d", "title": "Wave-packet laboratory reproduction", "status": status,
            "source_commit": COMMIT,
            "parameters": {"case": "1.D", "grid": "1536 × 128 × 128", "tank": "12 × 0.8 × 0.4 m", "wave_steepness": "ak = 0.22", "wave_number": "k = 9.3 rad/m", "packet_width": "τ₀ = 2.8 s", "time_step": "0.02 s; additional deterministic null at 0.01 s", "numerics": "WENO5, RK3, Float32", "seeds": "1, 2, 3, 4, 6, 7, 8"},
            "completed_pairs": complete_pairs, "total_pairs": 7,
            "validation": {"status": "passed", "job_id": validation_job},
            "jobs": {"array": str(submission.get("array_job", "")), "analysis": str(submission.get("analysis_job", ""))},
            "scheduler_states": scheduler},
        "members": members, "results": results, "downloads": result_downloads, "metrics": metrics,
        "notices": [
            "Only verified finished runs are shown. A pending label means no completion receipt has been verified; it does not distinguish queued, running, or failed jobs.",
            "Member figures are individual realizations. Final laboratory and ensemble diagnostics appear only after the complete analysis passes its source and file-integrity checks.",
            "The original seven retained seeds exclude upstream numerical failures 5 and 9. This gallery preserves that selection; it is not a population-unbiased or convergence study.",
            "No tracer was simulated. Vertical momentum flux and ky = 0 energy alone do not establish a transport barrier or a Taylor–Proudman regime.",
            "Initial-condition, source and validation checks do not substitute for physical validation against the laboratory data."
        ]}
    # Revalidate provenance before publishing the new manifest. Data hashes were
    # checked around each JLD2 read; final-analysis files are immutable by receipt.
    guard.check_submission(root)
    if guard.fingerprints(root) != fingerprints:
        raise RuntimeError("Scientific source changed during gallery export")
    if final:
        verified_final(root, fingerprints, len(arrays) == 8)
    validate_public_manifest(manifest)
    atomic_json(gallery / "data/campaign.json", manifest)
    print(f"Exported {complete_pairs}/7 complete pairs; campaign {status}.")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", "--campaign-root", dest="campaign_root", type=Path, required=True)
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    export(args.campaign_root, args.gallery, args.cache, Path(__file__).resolve().parent)


if __name__ == "__main__":
    main()

"""Receipt and publication tests. Synthetic bytes stay in temporary directories."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/export_campaign.py"
spec = importlib.util.spec_from_file_location("gallery_export", SCRIPT)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


class FakeGuard:
    COMMIT = exporter.COMMIT

    def fingerprints(self, root):
        return {"fixture": "fixed"}

    def check_submission(self, root):
        return {"array_job": "66075619", "analysis_job": "66075620"}


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / "campaign"
        self.gallery = self.base / "gallery"
        self.root.mkdir()
        self.guard = FakeGuard()
        self.fingerprints = self.guard.fingerprints(self.root)
        self.put_json("validation/passed.json", {"passed": True})
        p = self.root / "runtime/julia-path.txt"
        p.parent.mkdir(parents=True); p.write_text("/not/executed/julia")

    def tearDown(self):
        self.temp.cleanup()

    def put_json(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return path

    def member(self, member="turbulence_control", seed=1, timestep="0.020"):
        directory = exporter.member_directory(self.root, member, seed, timestep)
        directory.mkdir(parents=True, exist_ok=True)
        outputs = []
        for name in exporter.NATIVE:
            p = directory / name; p.write_bytes(b"synthetic test bytes, not scientific data")
            if name in exporter.NATIVE[:2]:
                outputs.append({"path": str(p.relative_to(self.root)), "sha256": exporter.digest(p)})
        record = {"completed": True, "fingerprints": self.fingerprints, "verified_outputs": outputs}
        receipt = self.put_json(f"data/receipts/{member}-seed{seed}-dt{timestep}-completed.json", record)
        return directory, receipt, record

    def extractor(self, root, directory, record, cache, scripts, python, julia):
        member = directory.parent.name
        seed = 0 if directory.name.startswith("quiescent") else int(directory.name[5:9])
        timestep = float(directory.name.split("_dt")[1].split("_")[0])
        target = self.base / "derived" / member / directory.name
        target.mkdir(parents=True, exist_ok=True)
        (target / "metadata.json").write_text(json.dumps({"member": member, "seed": seed, "dt_s": timestep,
                                                         "final_time_s": 45.78, "iterations": 2289}))
        for name in ("eulerian-mean.png", "mean-profiles.png", "lagrangian-rms.png", "mean-profiles.csv", "velocity-statistics.csv"):
            (target / name).write_bytes(b"synthetic derived test asset")
        return target, exporter.digest(target / "metadata.json")

    def run_export(self):
        return exporter.export(self.root, self.gallery, self.base / "cache", SCRIPT.parent,
                               guard=self.guard, extractor=self.extractor)

    def test_absent_receipt_never_opens_native_data(self):
        directory = exporter.member_directory(self.root, "packet_turbulence", 1)
        directory.mkdir(parents=True)
        (directory / "y_averages.jld2").write_bytes(b"live uncompleted data")
        self.assertIsNone(exporter.finished_step(self.root, "packet_turbulence", 1, "0.020", self.fingerprints))
        manifest = self.run_export()
        self.assertEqual(manifest["campaign"]["completed_pairs"], 0)
        self.assertFalse(any(m["figures"] for m in manifest["members"]))

    def test_scheduler_accounting_is_explicit_and_sanitized(self):
        def runner(argv, **kwargs):
            self.assertIn("--format=JobID%64,State%32", argv)
            return SimpleNamespace(stdout="66075619_0|COMPLETED\n66075619_1|OUT_OF_MEMORY\n66075620|CANCELLED by 12345\n666666_0|RUNNING\n66075619_2.batch|FAILED\n")
        actual = exporter.scheduler_states(self.guard.check_submission(self.root), runner=runner)
        self.assertEqual(actual, {"66075619_0": "COMPLETED", "66075619_1": "OUT_OF_MEMORY", "66075620": "CANCELLED"})
        def unavailable(*a, **kw):
            raise FileNotFoundError("no sacct")
        self.assertEqual(exporter.scheduler_states(self.guard.check_submission(self.root), runner=unavailable), {})

    def test_rejects_incomplete_receipt(self):
        directory, path, record = self.member()
        record["completed"] = False; path.write_text(json.dumps(record))
        with self.assertRaisesRegex(RuntimeError, "completion"):
            self.run_export()

    def test_rejects_tampered_receipt_output(self):
        directory, path, record = self.member()
        (directory / "metadata.jld2").write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "changed"):
            self.run_export()

    def test_rejects_source_mismatch(self):
        directory, path, record = self.member()
        record["fingerprints"] = {"fixture": "changed"}; path.write_text(json.dumps(record))
        with self.assertRaises(RuntimeError):
            self.run_export()

    def test_partial_pair_does_not_publish_ensemble(self):
        self.member()
        manifest = self.run_export()
        self.assertEqual(manifest["members"][0]["status"], "partial")
        self.assertEqual(len(manifest["members"][0]["figures"]), 3)
        self.assertEqual(manifest["campaign"]["completed_pairs"], 0)
        self.assertEqual(manifest["results"], [])
        self.member("packet_turbulence")
        manifest = self.run_export()
        self.assertEqual(manifest["campaign"]["completed_pairs"], 1)
        self.assertEqual(manifest["results"], [])

    def test_requires_all_arrays_before_final(self):
        self.put_json("data/analysis/completed.json", {"completed": True})
        with self.assertRaisesRegex(RuntimeError, "before all"):
            self.run_export()

    def test_rejects_unsealed_or_changed_final(self):
        output = "data/analysis/attempt-test"
        p = self.root / output / "figure3-reproduction.png"
        p.parent.mkdir(parents=True); p.write_bytes(b"plot")
        record = {"completed": True, "fingerprints": self.fingerprints, "output": output,
                  "files": [{"path": str(p.relative_to(self.root)), "sha256": exporter.digest(p)}]}
        self.put_json("data/analysis/completed.json", record)
        with self.assertRaisesRegex(RuntimeError, "required outputs"):
            exporter.verified_final(self.root, self.fingerprints, True)
        p.write_bytes(b"tampered")
        with self.assertRaisesRegex(RuntimeError, "integrity"):
            exporter.verified_final(self.root, self.fingerprints, True)

    def test_private_paths_and_unsafe_urls_rejected(self):
        for value in ({"summary": "/work/private/data"}, {"src": "../../private.key"},
                      {"href": "file:///Users/person/key"}, {"src": "https://example.test/x"}):
            with self.assertRaises(RuntimeError):
                exporter.validate_public_manifest(value)
        exporter.validate_public_manifest({"src": "assets/member.png", "href": "data/member.csv"})

    def test_receipt_cannot_escape_campaign(self):
        outside = self.base / "private.json"; outside.write_bytes(b"secret")
        p = self.put_json("data/receipts/bad.json", {"completed": True, "fingerprints": self.fingerprints,
             "verified_outputs": [{"path": "../private.json", "sha256": exporter.digest(outside)}]})
        with self.assertRaisesRegex(RuntimeError, "Unsafe"):
            exporter.verify_receipt(self.root, p, self.fingerprints)

    def test_changed_native_input_during_extraction_cannot_publish(self):
        directory, _, record = self.member()

        def runner(argv, **kwargs):
            if "render_member.py" in argv[1]:
                scratch = Path(argv[2])
                for name in ("metadata.json", "mean-profiles.csv", "velocity-statistics.csv",
                             "eulerian-mean.png", "mean-profiles.png", "lagrangian-rms.png"):
                    (scratch / name).write_bytes(b"derived test data")
                (directory / "statistics.jld2").write_bytes(b"changed during extraction")

        with self.assertRaisesRegex(RuntimeError, "changed during extraction"):
            exporter.ensure_cache(self.root, directory, record, self.base / "cache", SCRIPT.parent,
                                  "python", "julia", runner=runner)
        self.assertFalse(list((self.base / "cache").glob("*/complete.json")))

    def test_failed_export_preserves_previous_manifest(self):
        initial = self.run_export()
        directory, _, _ = self.member()
        (directory / "metadata.jld2").write_bytes(b"tamper")
        with self.assertRaises(RuntimeError):
            self.run_export()
        self.assertEqual(json.loads((self.gallery / "data/campaign.json").read_text()), initial)

    def test_complete_sealed_campaign_publishes_only_allowlisted_final_products(self):
        for seed in exporter.SEEDS:
            for member in exporter.MEMBERS:
                self.member(member, seed)
        for member, timestep in (("quiescent_control", "0.020"), ("packet_null", "0.020"), ("packet_null", "0.010")):
            self.member(member, 0, timestep)
        _, _, sample = self.member()
        for index in range(8):
            self.put_json(f"data/receipts/array-{index}-completed.json", {**sample, "index": index})
        output = self.root / "data/analysis/attempt-completed"
        names = ("figure3-reproduction.png", "figure3-reproduction.csv", "added/dimensionality.png",
                 "added/momentum-flux-partition.png", "added/ensemble-snapshots.csv", "private-debug.txt")
        for name in names:
            p = output / name; p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("synthetic test only")
        metric = {"lab_centered_shape_rmse_mm_s": 1.0, "lab_minus_les_window_offset_mm_s": -2.0,
                  "upstream_profile_rmse_in_piv_window_mm_s": .2}
        (output / "reference-comparison.json").write_text(json.dumps({"source_commit": exporter.COMMIT,
            "comparison": {"raw": metric, "paired": metric}}))
        record = {"completed": True, "fingerprints": self.fingerprints, "output": str(output.relative_to(self.root)),
                  "files": [{"path": str(p.relative_to(self.root)), "sha256": exporter.digest(p)}
                            for p in output.rglob("*") if p.is_file()]}
        self.put_json("data/analysis/completed.json", record)
        manifest = self.run_export()
        self.assertEqual(manifest["campaign"]["status"], "complete")
        self.assertEqual(manifest["campaign"]["completed_pairs"], 7)
        self.assertEqual(len(manifest["results"]), 3)
        self.assertEqual(len(manifest["metrics"]), 6)
        self.assertFalse(list(self.gallery.rglob("private-debug.txt")))
        exporter.validate_public_manifest(manifest)


if __name__ == "__main__":
    unittest.main()

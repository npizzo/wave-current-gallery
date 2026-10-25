import csv
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("audit_report", Path(__file__).with_name("report.py"))
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def fixture(folder, mutate=None, *, empty=False):
    for seed in report.SEEDS:
        rows = []
        for member in report.PAIRED_MEMBERS:
            for term in sorted(report.REQUIRED_TERMS):
                value = float(seed)
                row = dict(seed=seed, member=member, time_s=22.88, stencil_step=1,
                           region="stokes_layer", age_bin="all", age_low="-inf", age_high="inf",
                           volume_m3=.4, x_columns=20, z_levels=8, z_min_m=-.05, z_max_m=-.001,
                           term=term, mean=value, mean_square=value**2, max_abs=value)
                if mutate:
                    mutate(row)
                rows.append(row)
                if empty:
                    rows.append(row | dict(age_bin="before", age_low=-5, age_high=-3,
                                           volume_m3=0, x_columns=0, mean="NaN", mean_square="NaN", max_abs="NaN"))
        path = Path(folder) / f"seed-{seed}" / "regions.csv"
        path.parent.mkdir(parents=True)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)

    def test_ensemble_norms_precede_ratios(self):
        fixture(self.folder)
        report.build_report(self.folder, plot=False)
        rows = read_csv(self.folder / "ensemble.csv")
        q = next(r for r in rows if r["member"] == "packet_turbulence" and r["term"] == "q")
        self.assertAlmostEqual(float(q["rms"]), math.sqrt(sum(s*s for s in report.SEEDS)/7))
        metrics = read_csv(self.folder / "metrics.csv")
        values = {r["metric"]: float(r["ratio"]) for r in metrics if r["member"] == "packet_turbulence"}
        self.assertAlmostEqual(values["epsilon_div_native"], 1/math.sqrt(2))
        self.assertAlmostEqual(values["r3d_to_uncancelled_advection"], .5)
        self.assertAlmostEqual(values["tau_r3d_to_q"], 2.8)
        self.assertAlmostEqual(values["r3d_cancellation"], .25)

    def test_not_a_mean_of_per_seed_ratios(self):
        def change(row):
            if row["term"] == "R3D":
                value = float(row["seed"])**2
                row.update(mean=value, mean_square=value**2, max_abs=value)
        fixture(self.folder, change)
        report.build_report(self.folder, plot=False)
        metric = next(row for row in read_csv(self.folder / "metrics.csv")
                      if row["member"] == "packet_turbulence"
                      and row["metric"] == "r3d_to_uncancelled_advection")
        expected = math.sqrt(sum(s**4 for s in report.SEEDS)/7)/(2*math.sqrt(sum(s*s for s in report.SEEDS)/7))
        self.assertAlmostEqual(float(metric["ratio"]), expected)
        self.assertNotAlmostEqual(float(metric["ratio"]), sum(report.SEEDS)/14)

    def test_complete_seven_seed_check(self):
        fixture(self.folder)
        (self.folder / "seed-8" / "regions.csv").unlink()
        with self.assertRaisesRegex(ValueError, "Missing completed seed"):
            report.build_report(self.folder, plot=False)
        self.assertFalse((self.folder / "summary.json").exists())

    def test_member_times_must_match(self):
        fixture(self.folder, lambda r: r.update(time_s=23) if r["member"] == "turbulence_control" else None)
        with self.assertRaisesRegex(ValueError, "Missing matched member"):
            report.load_regions(self.folder)

    def test_partial_seed_bin_rejected(self):
        fixture(self.folder, lambda r: r.update(volume_m3=0, x_columns=0) if r["seed"] == 8 else None)
        with self.assertRaisesRegex(ValueError, "Incomplete seed ensemble"):
            report.load_regions(self.folder)

    def test_shared_sampling_geometry(self):
        fixture(self.folder, lambda r: r.update(volume_m3=.5) if r["seed"] == 8 else None)
        with self.assertRaisesRegex(ValueError, "Inconsistent paired sampling geometry"):
            report.load_regions(self.folder)

    def test_empty_bins_are_ignored(self):
        fixture(self.folder, empty=True)
        result = report.build_report(self.folder, plot=False)
        self.assertEqual(result["nonempty_member_time_region_age_stencil_groups"], 2)
        self.assertGreater(result["ignored_zero_volume_rows"], 0)

    def test_zero_denominators_are_blank_not_infinity(self):
        fixture(self.folder, lambda r: r.update(mean=0, mean_square=0, max_abs=0))
        result = report.build_report(self.folder, plot=False)
        rows = read_csv(self.folder / "metrics.csv")
        self.assertTrue(all(r["ratio"] == "" for r in rows))
        self.assertTrue(all(r["status"] == "denominator_zero" for r in rows))
        self.assertEqual(result["undefined_ratios"], len(rows))
        json.loads((self.folder / "summary.json").read_text())

    def test_explicit_floor_does_not_clip(self):
        fixture(self.folder)
        report.build_report(self.folder, plot=False, q_floor=100)
        rows = read_csv(self.folder / "metrics.csv")
        selected = [r for r in rows if r["metric"] == "tau_r3d_to_q"]
        self.assertTrue(all(r["status"] == "below_specified_floor" and r["ratio"] == "" for r in selected))
        self.assertTrue(all(float(r["denominator"]) > 0 for r in selected))

    def test_invalid_second_moment(self):
        fixture(self.folder, lambda r: r.update(mean_square=-1) if r["seed"] == 8 else None)
        with self.assertRaisesRegex(ValueError, "Negative squared"):
            report.load_regions(self.folder)

    def test_optional_null_member_stays_separate(self):
        fixture(self.folder)
        path = self.folder / "seed-0" / "regions.csv"
        path.parent.mkdir()
        source = (self.folder / "seed-1" / "regions.csv").read_text()
        lines = source.splitlines()
        rows = list(csv.DictReader(lines))
        rows = [r | {"seed": 0, "member": "packet_null"} for r in rows if r["member"] == "packet_turbulence"]
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        report.build_report(self.folder, plot=False)
        ensemble = read_csv(self.folder / "ensemble.csv")
        self.assertTrue(all(r["realizations"] == "1" for r in ensemble if r["member"] == "packet_null"))
        self.assertTrue(all(r["realizations"] == "7" for r in ensemble if r["member"] in report.PAIRED_MEMBERS))


if __name__ == "__main__":
    unittest.main()

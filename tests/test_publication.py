"""Publication guards and event attachment without contacting Unity or GitHub."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ATTACH = load("publication_attach_test", ROOT / "unity/enable-updates.py")
CHECK = load("publication_gallery_check_test", ROOT / "scripts/check_gallery.py")
PUBLISH = load("publication_push_test", ROOT / "scripts/publish_update.py")


class EventAttachment(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.root = self.base / "publisher"
        self.campaign = self.base / "scientific-campaign"
        (self.root / "unity").mkdir(parents=True)
        state = self.root / ".unity-publisher"
        state.mkdir()
        (state / "id_ed25519").write_text("fixture, not a key\n")
        (state / "known_hosts").write_text("fixture\n")
        (self.campaign / "data").mkdir(parents=True)
        (self.campaign / "data/submission.json").write_text(json.dumps({"array_job": "66075619", "analysis_job": "66075620"}))
        (self.campaign / "scientific-input.txt").write_text("unchanged input\n")
        self.states = {"66075619_" + str(i): "PENDING" for i in range(8)}
        self.states["66075620"] = "PENDING"
        self.commands = []
        self.submitted = []

    def tearDown(self):
        self.temporary.cleanup()

    def output(self, argv, **kwargs):
        self.commands.append(list(argv))
        if argv[0] == "sacct":
            return "".join(key + "|" + value + "\n" for key, value in self.states.items())
        if argv[0] == "sbatch":
            self.submitted.append(list(argv))
            return str(70000000 + len(self.submitted)) + ";test-cluster\n"
        raise AssertionError("Unexpected external command: " + repr(argv))

    def run_command(self, argv, **kwargs):
        self.commands.append(list(argv))
        stdout = ""
        if "--show-current" in argv:
            stdout = "main\n"
        elif "get-url" in argv:
            stdout = "git@github.com:npizzo/test-gallery.git\n"
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    def invoke(self, campaign=None):
        before = {str(p.relative_to(self.campaign)): p.read_bytes() for p in self.campaign.rglob("*") if p.is_file()}
        argv = ["enable-updates.py", "--campaign", str(campaign or self.campaign), "--repository", "npizzo/test-gallery"]
        with mock.patch.object(ATTACH, "__file__", str(self.root / "unity/enable-updates.py")), \
             mock.patch.object(sys, "argv", argv), \
             mock.patch.object(ATTACH.subprocess, "run", side_effect=self.run_command), \
             mock.patch.object(ATTACH.subprocess, "check_output", side_effect=self.output), \
             contextlib.redirect_stdout(io.StringIO()):
            ATTACH.main()
        after = {str(p.relative_to(self.campaign)): p.read_bytes() for p in self.campaign.rglob("*") if p.is_file()}
        self.assertEqual(before, after, "Attaching publishers must not modify scientific inputs or receipts")

    def test_one_immediate_and_nine_afterany_events(self):
        self.invoke()
        self.assertEqual(len(self.submitted), 10)
        dependencies = [arg for argv in self.submitted for arg in argv if arg.startswith("--dependency=")]
        self.assertEqual(set(dependencies), {"--dependency=afterany:" + key for key in self.states})
        self.assertFalse(any(arg.startswith("--dependency=") for arg in self.submitted[0]))
        self.assertTrue(all("publish.sbatch" in " ".join(argv) for argv in self.submitted))
        record = json.loads((self.root / ".unity-publisher/updates.json").read_text())
        self.assertEqual(record["state"], "attached")
        self.assertEqual(len(record["jobs"]), 10)

    def test_completed_failed_and_cancelled_jobs_are_covered_by_immediate_snapshot(self):
        self.states = {key: "COMPLETED" for key in self.states}
        self.states["66075619_2"] = "FAILED"
        self.states["66075619_4"] = "CANCELLED by 12345"
        self.states["66075619_5"] = "OUT_OF_MEMORY+"
        self.invoke()
        self.assertEqual(len(self.submitted), 1)

    def test_only_nonterminal_parents_receive_dependencies(self):
        self.states = {key: "COMPLETED" for key in self.states}
        self.states["66075619_7"] = "RUNNING"
        self.states["66075620"] = "PENDING"
        self.invoke()
        self.assertEqual(len(self.submitted), 3)

    def test_duplicate_attachment_refused_before_more_submissions(self):
        self.invoke()
        with self.assertRaises(RuntimeError):
            self.invoke()
        self.assertEqual(len(self.submitted), 10)

    def test_missing_scheduler_record_does_not_submit_anything(self):
        del self.states["66075619_4"]
        with self.assertRaises(RuntimeError):
            self.invoke()
        self.assertEqual(self.submitted, [])
        self.assertFalse((self.root / ".unity-publisher/updates.json").exists())

    def test_array_query_expands_pending_elements(self):
        self.invoke()
        queries = [argv for argv in self.commands if argv[0] == "sacct"]
        self.assertTrue(queries)
        self.assertTrue(all("--array" in argv for argv in queries), "Compressed pending-array IDs omit individual dependency parents")

    def test_publisher_inside_campaign_is_rejected(self):
        with self.assertRaises(ValueError):
            self.invoke(campaign=self.root)
        self.assertFalse(self.submitted)

    def test_ambiguous_submission_keeps_ledger_and_blocks_retry(self):
        original = self.output

        def ambiguous(argv, **kwargs):
            if argv[0] == "sbatch":
                self.submitted.append(list(argv))
                return "unrecognized scheduler response\n"
            return original(argv, **kwargs)

        with mock.patch.object(self, "output", side_effect=ambiguous):
            with self.assertRaises(RuntimeError):
                self.invoke()
        ledger = json.loads((self.root / ".unity-publisher/updates.json").read_text())
        self.assertEqual(ledger["state"], "submitting")
        with self.assertRaises(RuntimeError):
            self.invoke()
        self.assertEqual(len(self.submitted), 1)


class PublicAssetGuards(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "data").mkdir()
        (self.root / "index.html").write_text("<!doctype html><title>Test</title>\n")
        self.data = {"schema_version": 1, "campaign": {"total_pairs": 7}, "members": [], "results": [], "downloads": []}
        self.write()

    def tearDown(self):
        self.temporary.cleanup()

    def write(self):
        (self.root / "data/campaign.json").write_text(json.dumps(self.data))

    def test_valid_compact_data_download(self):
        (self.root / "data/results.csv").write_text("time,value\n0,1\n")
        self.data["downloads"] = [{"href": "data/results.csv", "label": "Results"}]
        self.write()
        CHECK.check(self.root)

    def test_top_level_download_missing_is_rejected(self):
        self.data["downloads"] = [{"href": "data/missing.csv", "label": "Results"}]
        self.write()
        with self.assertRaises(ValueError):
            CHECK.check(self.root)

    def test_missing_member_figure_is_rejected(self):
        self.data["members"] = [{"figures": [{"src": "missing.png"}]}]
        self.write()
        with self.assertRaises(ValueError):
            CHECK.check(self.root)

    def test_private_paths_and_raw_simulation_files_are_rejected(self):
        (self.root / "data/raw.jld2").write_bytes(b"raw state")
        with self.assertRaises(ValueError):
            CHECK.check(self.root)
        (self.root / "data/raw.jld2").unlink()
        (self.root / "data/private.txt").write_text("/work/researcher/raw-data\n")
        with self.assertRaises(ValueError):
            CHECK.check(self.root)

    def test_parent_path_and_external_urls_are_rejected(self):
        for path in ("../private.csv", "/absolute.csv", "https://example.org/result.csv", "data\\file.csv"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                CHECK.local_asset(self.root, path)


class PublicationRetries(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.root = self.base / "publisher"
        self.campaign = self.base / "scientific-campaign"
        (self.root / "scripts").mkdir(parents=True)
        self.campaign.mkdir()
        self.commands = []
        self.checkouts = []
        self.push_codes = [0]
        self.diff_code = 1
        self.export_fails = False

    def tearDown(self):
        self.temporary.cleanup()

    def checked(self, argv, root, **kwargs):
        self.commands.append((list(argv), Path(root)))
        if argv[:2] == ["git", "clone"]:
            checkout = Path(argv[-1])
            checkout.mkdir()
            (checkout / "gallery").mkdir()
            self.checkouts.append(checkout)
        elif len(argv) > 1 and argv[1].endswith("export_campaign.py"):
            gallery = Path(argv[argv.index("--gallery") + 1])
            (gallery / "snapshot.txt").write_text("derived result\n")
            if self.export_fails:
                raise subprocess.CalledProcessError(1, argv)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    def unchecked(self, argv, **kwargs):
        self.commands.append((list(argv), Path(kwargs["cwd"])))
        if argv[:2] == ["git", "diff"]:
            return subprocess.CompletedProcess(argv, self.diff_code)
        if argv[:2] == ["git", "push"]:
            return subprocess.CompletedProcess(argv, self.push_codes.pop(0))
        raise AssertionError("Unexpected unchecked command: " + repr(argv))

    def invoke(self):
        argv = ["publish_update.py", "--campaign", str(self.campaign),
                "--repository", "npizzo/test-gallery", "--key", str(self.root / "key"),
                "--known-hosts", str(self.root / "known_hosts")]
        with mock.patch.object(PUBLISH, "__file__", str(self.root / "scripts/publish_update.py")), \
             mock.patch.object(sys, "argv", argv), \
             mock.patch.object(PUBLISH, "run", side_effect=self.checked), \
             mock.patch.object(PUBLISH.subprocess, "run", side_effect=self.unchecked), \
             contextlib.redirect_stdout(io.StringIO()):
            PUBLISH.main()

    def test_rejected_push_rebuilds_fresh_without_force(self):
        self.push_codes = [1, 0]
        self.invoke()
        self.assertEqual(len(self.checkouts), 2)
        self.assertNotEqual(self.checkouts[0], self.checkouts[1])
        self.assertTrue(all(not path.exists() for path in self.checkouts))
        clones = [argv for argv, _ in self.commands if argv[:2] == ["git", "clone"]]
        pushes = [argv for argv, _ in self.commands if argv[:2] == ["git", "push"]]
        self.assertTrue(all("npizzo/test-gallery.git" in " ".join(argv) for argv in clones + pushes))
        self.assertFalse(any("--force" in arg or arg.startswith("+") for argv in pushes for arg in argv))
        adds = [argv for argv, _ in self.commands if argv[:2] == ["git", "add"]]
        self.assertEqual(adds, [["git", "add", "--", "gallery"]] * 2)
        self.assertFalse(list(self.campaign.iterdir()), "Publisher must leave campaign directory untouched")

    def test_push_failures_stop_after_three_attempts(self):
        self.push_codes = [1, 1, 1]
        with self.assertRaises(RuntimeError):
            self.invoke()
        self.assertEqual(len(self.checkouts), 3)
        self.assertTrue(all(not path.exists() for path in self.checkouts))

    def test_failed_export_leaves_no_dirty_checkout_and_can_retry(self):
        self.export_fails = True
        with self.assertRaises(subprocess.CalledProcessError):
            self.invoke()
        self.assertTrue(all(not path.exists() for path in self.checkouts))
        self.assertFalse(any(argv[:2] == ["git", "push"] for argv, _ in self.commands))
        self.export_fails = False
        self.invoke()
        self.assertEqual(len(self.checkouts), 2)
        self.assertTrue(all(not path.exists() for path in self.checkouts))

    def test_unchanged_snapshot_does_not_commit_or_push(self):
        self.diff_code = 0
        self.invoke()
        self.assertFalse(any("commit" in argv or argv[:2] == ["git", "push"] for argv, _ in self.commands))
        self.assertEqual(len(self.checkouts), 1)


if __name__ == "__main__":
    unittest.main()

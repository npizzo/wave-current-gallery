import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from guard import digest,write_json
from references import verify_references
from publish import confirmed_push


def reference_fixture(root):
    old=root/'old-campaign';run=old/'data/runs/old';run.mkdir(parents=True)
    (run/'completed.json').write_text('{"time":1800}')
    files={'completed.json':digest(run/'completed.json')}
    receipt=dict(id='old',completed=True,persistent_backup_verified=True,source_version='old-version',files=files)
    write_json(old/'data/status/old/completed.json',receipt)
    backup=root/'backup';saved=backup/'runs/old';saved.mkdir(parents=True)
    write_json(saved/'completion-receipt.json',receipt)
    write_json(saved/'COMPLETE.json',dict(complete=True,source_version='old-version',receipt_sha256=digest(saved/'completion-receipt.json')))
    (saved/'native-output.tar.gz').write_bytes(b'retained original archive')
    write_json(old/'runtime/paths.json',dict(backup=str(backup)))
    case=dict(id='ref',source_version='old-version',case_id='old')
    write_json(root/'campaign.json',dict(runs=[],references=[case]))
    write_json(root/'references-manifest.json',dict(references=dict(ref=dict(case_id='old',source_version='old-version',files=files,receipt_sha256=digest(old/'data/status/old/completed.json')))))
    write_json(root/'runtime/references.json',dict(references=dict(ref=dict(root=str(run),campaign_root=str(old),files=files))))
    return run,saved


class ShearReferenceTests(unittest.TestCase):
    def test_reference_completion_backup_and_exact_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);run,backup=reference_fixture(root)
            self.assertEqual(verify_references(root)['ref']['case_id'],'old')
            (run/'completed.json').write_text('{"time":1801}')
            with self.assertRaises(RuntimeError):verify_references(root)

    def test_private_paths_cannot_replace_frozen_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);reference_fixture(root)
            value=json.loads((root/'runtime/references.json').read_text())
            value['references']['ref']['files']={'completed.json':'0'*64}
            write_json(root/'runtime/references.json',value)
            with self.assertRaises(RuntimeError):verify_references(root)

    def test_reference_requires_persistent_backup_proof(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);run,backup=reference_fixture(root)
            (backup/'native-output.tar.gz').unlink()
            with self.assertRaises(RuntimeError):verify_references(root)

    def test_failed_transport_accepted_push_is_detected(self):
        results=[SimpleNamespace(returncode=x) for x in (1,0,0)]
        with patch('publish.subprocess.run',side_effect=results) as call:
            self.assertTrue(confirmed_push(Path('/fixture'),{}))
            self.assertEqual(call.call_args_list[-1].args[0],['git','merge-base','--is-ancestor','HEAD','FETCH_HEAD'])

    def test_failed_push_without_remote_acceptance_is_not_success(self):
        results=[SimpleNamespace(returncode=x) for x in (1,0,1)]
        with patch('publish.subprocess.run',side_effect=results):
            self.assertFalse(confirmed_push(Path('/fixture'),{}))

if __name__=='__main__':unittest.main()

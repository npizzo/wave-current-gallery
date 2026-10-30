import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from guard import inventory,version_for,verify,write_json,digest,safe_relative
from inputs import approved_input,branch_settings,preserve_inputs
from backup import tree_hashes,backup_run,archive_hashes
from publish import update_landing
from submit import submit_once,find_submission


def toml_text(value,path=()):
    lines=[]
    if path:lines.append('['+'.'.join(json.dumps(p) for p in path)+']')
    for key,item in value.items():
        if not isinstance(item,dict):lines.append(json.dumps(key)+' = '+json.dumps(item))
    for key,item in value.items():
        if isinstance(item,dict):lines.append(toml_text(item,(*path,key)))
    return '\n'.join(lines)+'\n'


def input_fixture(root):
    parent=root/'old-parent';parent.mkdir()
    arrays={}
    for name in ('u','v','w'):
        file=parent/('t01000.000000_'+name+'.bin');file.write_bytes((name*64).encode())
        arrays[name]=dict(file=file.name,sha256=digest(file),shape=[2,2,2],dtype='float64',order='F')
    manifest=parent/'t01000.000000.json';write_json(manifest,dict(time=1000.,arrays=arrays))
    metadata=dict(format='wind-native-snapshot-v1',time=1000.,json_sha256=digest(manifest),
                  parameters=dict(nx=2,ny=2,nz=2,ustar=.002),source_hashes=dict(driver='frozen'),arrays=arrays)
    companion=manifest.with_suffix('.toml');companion.write_text(toml_text(metadata))
    reference=parent/'mean_01010.000000_Uc.bin';reference.write_bytes(b'continuation-reference')
    record=dict(snapshot_json_sha256=digest(manifest),snapshot_metadata_sha256=digest(companion),
                snapshot_time=1000.,expected_parent_ustar=.002,allowed_branch_ustar=[0.,.002],
                parameters=metadata['parameters'],source_hashes=metadata['source_hashes'],arrays=arrays,
                validation_reference_files={reference.name:digest(reference)})
    approval=dict(format='wind-decay-parent-approval-v1',campaign='unity-wind-decay-tests',sources=dict(parent=record))
    (root/'source_provenance.toml').write_text(toml_text(approval))
    write_json(root/'runtime/input-paths.json',dict(sources=dict(parent=str(manifest))))
    case=dict(id='off',source_id='parent',path='data/runs/off',env=dict(START_TIME=1000,STOP_TIME=4000,WIND_USTAR=0))
    write_json(root/'campaign.json',dict(runs=[case]));(root/'PROTOCOL.txt').write_text('fixture')
    files=inventory(root);source=dict(files=files,version=version_for(files));write_json(root/'source-manifest.json',source)
    return manifest,case,source


class DecayOperationTests(unittest.TestCase):
    def test_approved_input_and_only_allowed_intervention(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,case,source=input_fixture(root)
            actual,record,sha=approved_input(root,'parent',source);self.assertEqual(actual,path)
            settings,proof=branch_settings(root,case,source)
            self.assertEqual(settings['INIT_SNAPSHOT'],str(path));self.assertEqual(settings['INIT_KIND'],'full')
            self.assertEqual(settings['INIT_APPROVAL_SHA256'],sha);self.assertEqual(proof['source_time'],1000)
            self.assertEqual(settings['WIND_USTAR'],0);self.assertEqual(settings['JET_MODE'],0)
            case['env']['WIND_USTAR']=.004
            with self.assertRaises(RuntimeError):branch_settings(root,case,source)

    def test_input_tamper_and_wrong_start_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,case,source=input_fixture(root)
            case['env']['START_TIME']=1200
            with self.assertRaises(RuntimeError):branch_settings(root,case,source)
            (path.parent/'t01000.000000_u.bin').write_bytes(b'changed')
            with self.assertRaises(RuntimeError):approved_input(root,'parent',source)

    def test_approval_is_part_of_source_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,_,source=input_fixture(root);verify(root)
            self.assertIn('source_provenance.toml',source['files'])
            with (root/'source_provenance.toml').open('a') as f:f.write('\n# changed\n')
            with self.assertRaises(RuntimeError):verify(root)
            with self.assertRaises(RuntimeError):approved_input(root,'parent',source)

    def test_continuation_reference_is_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,_,source=input_fixture(root)
            (path.parent/'mean_01010.000000_Uc.bin').write_bytes(b'different continuation')
            with self.assertRaises(RuntimeError):approved_input(root,'parent',source)

    def test_exact_persistent_parent_copy_idempotent_and_tamper_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path,case,source=input_fixture(root);backup=root/'backup'
            with patch('inputs.shutil.disk_usage',return_value=SimpleNamespace(free=10**12)):
                preserve_inputs(root,backup,source);preserve_inputs(root,backup,source)
                saved=backup/'initial-states/parent'/path.name
                self.assertEqual(saved.read_bytes(),path.read_bytes())
                saved.write_bytes(b'changed')
                with self.assertRaises(RuntimeError):preserve_inputs(root,backup,source)

    def test_submission_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);result=SimpleNamespace(returncode=0,stdout='1234\n',stderr='')
            with patch('submit.subprocess.run',return_value=result) as run:
                self.assertEqual(submit_once(root,'array','scripts/run.sbatch',['--array=0-1%2']),'1234')
                self.assertEqual(submit_once(root,'array','scripts/run.sbatch',['--array=0-1%2']),'1234')
                self.assertEqual(run.call_count,1)
            with self.assertRaises(RuntimeError):submit_once(root,'array','scripts/run.sbatch',['--array=0-3%2'])

    def test_ambiguous_submission_never_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);calls=[]
            def call(args,**kw):
                calls.append(args[0]);return SimpleNamespace(returncode=1 if args[0]=='sbatch' else 0,stdout='',stderr='connection ended')
            with patch('submit.subprocess.run',side_effect=call):
                with self.assertRaises(RuntimeError):submit_once(root,'array','scripts/run.sbatch',[])
                with self.assertRaises(RuntimeError):submit_once(root,'array','scripts/run.sbatch',[])
            self.assertEqual(calls.count('sbatch'),1);self.assertEqual(calls.count('squeue'),2);self.assertEqual(calls.count('sacct'),2)

    def test_ambiguous_submission_recovers_exact_token(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            write_json(root/'submission-journal/array.json',dict(script='scripts/run.sbatch',options=[],token='winddecay-fixture',
                       created_at='2026-10-30T00:00:00+00:00',state='attempting'))
            with patch('submit.subprocess.run',return_value=SimpleNamespace(returncode=0,stdout='1234_1|winddecay-fixture\n',stderr='')) as run:
                self.assertEqual(submit_once(root,'array','scripts/run.sbatch',[]),'1234')
                self.assertTrue(all(c.args[0][0]!='sbatch' for c in run.call_args_list))

    def test_multiple_token_matches_stop(self):
        with patch('submit.subprocess.run',return_value=SimpleNamespace(returncode=0,stdout='123|x\n456|x\n',stderr='')):
            with self.assertRaises(RuntimeError):find_submission('x','2026-10-30')

    def test_navigation_does_not_replace_existing_wind_campaign(self):
        original='<div class="nav-links"><a href="experiments/wind-jet-tests/">Wind and jets</a></div><main id="main" class="wrap"><section id="wind-jet-campaign">Existing</section></main>'
        changed=update_landing(original)
        self.assertIn('experiments/wind-decay/',changed);self.assertIn('<section id="wind-jet-campaign">Existing</section>',changed)
        self.assertEqual(update_landing(changed),changed)

    def test_persistent_branch_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'branch';folder.mkdir();(folder/'state.bin').write_bytes(bytes(range(256)))
            receipt=dict(files=tree_hashes(folder),source_version='fixture',completed=True)
            with patch('backup.shutil.disk_usage',return_value=SimpleNamespace(free=10**12)):
                saved=backup_run(folder,root/'backup',dict(id='off'),receipt)
                self.assertEqual(archive_hashes(saved/'native-output.tar.gz'),receipt['files'])
                backup_run(folder,root/'backup',dict(id='off'),receipt)

    def test_unsafe_relative_paths_rejected(self):
        for value in ('../x','/secret','a/../../b','.'):
            with self.assertRaises(ValueError):safe_relative(value)

if __name__=='__main__':unittest.main()

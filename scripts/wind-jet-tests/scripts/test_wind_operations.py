import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from guard import inventory,version_for,verify,write_json,safe_relative
from launch import stage_plan,snapshot_at
from backup import tree_hashes,backup_run,archive_hashes
from publish import update_landing
from submit import submit_once,find_submission


class WindOperationTests(unittest.TestCase):
    def case(self):
        return dict(id='test',path='data/runs/test',env=dict(STOP_TIME=1200,NX=128,NY=64,NZ=96,REGIME_INTERVAL=10))

    def test_parent_and_independent_branches(self):
        plan=stage_plan(self.case());self.assertEqual(len(plan),9)
        self.assertEqual(plan[0][0],'parent');self.assertIsNone(plan[0][2])
        self.assertEqual(plan[0][1]['SNAPSHOT_TIMES'],'0,600,1000,1200')
        lookup={name:(env,time) for name,env,time in plan}
        for start in (600,1000):
            env,source=lookup[f'forecast-{start}']
            self.assertEqual(source,start);self.assertEqual(env['NY'],1)
            self.assertEqual(env['STOP_TIME'],start+200);self.assertEqual(env['INIT_KIND'],'mean')
        for mode in (1,2,4):
            p,_=lookup[f'probe-m{mode}-plus'];m,_=lookup[f'probe-m{mode}-minus']
            self.assertEqual(p['START_TIME'],600);self.assertEqual(p['STOP_TIME'],800)
            self.assertEqual(p['JET_SIGN'],1);self.assertEqual(m['JET_SIGN'],-1)
            self.assertEqual({k:v for k,v in p.items() if k!='JET_SIGN'},
                             {k:v for k,v in m.items() if k!='JET_SIGN'})
            self.assertEqual(p['NY'],64)
        self.assertEqual(self.case()['env']['NY'],64)

    def test_probe_half_amplitude_and_no_probes(self):
        case=self.case();case.update(probe_amplitude=.0001,probe_modes=[2])
        plan=stage_plan(case);self.assertEqual(len(plan),5)
        self.assertEqual(plan[-1][1]['JET_AMPLITUDE'],.0001)
        case.update(probe_modes=[],forecasts=[])
        self.assertEqual(len(stage_plan(case)),1)

    def test_invalid_time_and_modes_rejected(self):
        case=self.case();case['forecast_times']=[1100]
        with self.assertRaises(ValueError):stage_plan(case)
        case=self.case();case['probe_modes']=[2,2]
        with self.assertRaises(ValueError):stage_plan(case)

    def test_submission_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            result=SimpleNamespace(returncode=0,stdout='1234\n',stderr='')
            with patch('submit.subprocess.run',return_value=result) as run:
                self.assertEqual(submit_once(root,'array','scripts/run.sbatch',['--array=0-2%2']),'1234')
                self.assertEqual(submit_once(root,'array','scripts/run.sbatch',['--array=0-2%2']),'1234')
                self.assertEqual(run.call_count,1)
            with self.assertRaises(RuntimeError):submit_once(root,'array','scripts/run.sbatch',['--array=0-3%2'])

    def test_ambiguous_submission_never_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);calls=[]
            def call(args,**kw):
                calls.append(args[0])
                return SimpleNamespace(returncode=1 if args[0]=='sbatch' else 0,stdout='',stderr='connection ended')
            with patch('submit.subprocess.run',side_effect=call):
                with self.assertRaises(RuntimeError):submit_once(root,'array','scripts/run.sbatch',[])
                with self.assertRaises(RuntimeError):submit_once(root,'array','scripts/run.sbatch',[])
            self.assertEqual(calls.count('sbatch'),1)
            self.assertEqual(calls.count('squeue'),2);self.assertEqual(calls.count('sacct'),2)

    def test_ambiguous_submission_recovers_from_slurm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            write_json(root/'submission-journal/array.json',dict(script='scripts/run.sbatch',options=[],
                 token='windjet-fixture',created_at='2026-10-29T00:00:00+00:00',state='attempting'))
            with patch('submit.subprocess.run',return_value=SimpleNamespace(returncode=0,stdout='1234_2|windjet-fixture\n',stderr='')) as run:
                self.assertEqual(submit_once(root,'array','scripts/run.sbatch',[]),'1234')
            self.assertTrue(all(c.args[0][0]!='sbatch' for c in run.call_args_list))

    def test_multiple_token_matches_stop(self):
        with patch('submit.subprocess.run',return_value=SimpleNamespace(returncode=0,stdout='123|x\n456|x\n',stderr='')):
            with self.assertRaises(RuntimeError):find_submission('x','2026-10-29')

    def test_navigation_is_additive_idempotent(self):
        original='<div class="nav-links"><a href="old/">Old</a></div><main id="main" class="wrap">old content</main>'
        changed=update_landing(original)
        self.assertIn('old content',changed);self.assertIn('href="old/"',changed)
        self.assertIn('experiments/wind-jet-tests/',changed)
        self.assertEqual(update_landing(changed),changed)

    def test_guard_and_unsafe_paths(self):
        for value in ('../x','/secret','a/../../b','.'):
            with self.assertRaises(ValueError):safe_relative(value)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'run').mkdir()
            for value in ('campaign.json','PROTOCOL.txt','run/physics.jl'):(root/value).write_text('old')
            files=inventory(root);write_json(root/'source-manifest.json',dict(files=files,version=version_for(files)))
            verify(root);(root/'run/physics.jl').write_text('new')
            with self.assertRaises(RuntimeError):verify(root)

    def test_persistent_bundle_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'case';(folder/'parent').mkdir(parents=True)
            (folder/'parent/state.bin').write_bytes(bytes(range(256)))
            receipt=dict(files=tree_hashes(folder),source_version='test',completed=True)
            with patch('backup.shutil.disk_usage',return_value=SimpleNamespace(free=10**12)):
                dest=backup_run(folder,root/'backup',dict(id='a'),receipt)
                self.assertEqual(archive_hashes(dest/'native-output.tar.gz'),receipt['files'])
                backup_run(folder,root/'backup',dict(id='a'),receipt)

if __name__=='__main__':unittest.main()

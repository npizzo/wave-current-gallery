import contextlib, importlib.util, io, json, tempfile, unittest
from pathlib import Path
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('audit_submit',Path(__file__).with_name('submit.py'))
submit=importlib.util.module_from_spec(spec);spec.loader.exec_module(submit)

class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name);self.gallery=self.base/'gallery';self.source=self.gallery/'unity/vorticity-audit'
        self.source.mkdir(parents=True);(self.gallery/'.git').mkdir()
        state=self.gallery/'.unity-publisher';state.mkdir();(state/'id_ed25519').write_text('synthetic-placeholder')
        for f in ('submit.py','audit.sbatch','report.sbatch'): (self.source/f).write_text('test source\n')
        self.campaign=self.base/'campaign';self.campaign.mkdir()
    def run_submit(self,answers):
        with patch.object(submit,'__file__',str(self.source/'submit.py')),patch.object(submit,'check_campaign',return_value={'x':'y'}),patch.object(submit.sys,'argv',['submit.py','--campaign',str(self.campaign)]),patch.object(submit.subprocess,'check_output',side_effect=answers) as call,contextlib.redirect_stdout(io.StringIO()):
            submit.main()
        return call
    def test_only_analysis_and_afterok_report(self):
        call=self.run_submit(['123\n','124\n'])
        self.assertEqual(call.call_count,2)
        self.assertIn('--dependency=afterok:123',call.call_args_list[1].args[0])
        self.assertTrue(any(x.endswith('audit.sbatch') for x in call.call_args_list[0].args[0]))
        self.assertEqual(self.run_submit([]).call_count,0)
    def test_partial_submission_preserved_no_duplicates(self):
        with self.assertRaises(RuntimeError):self.run_submit(['123\n',RuntimeError('scheduler unavailable')])
        path=next((self.gallery/'.unity-publisher').glob('vorticity-audit-*/submission.json'))
        record=json.loads(path.read_text());self.assertEqual(len(record['jobs']),1)
        self.assertEqual(record['status'],'submitting')
        self.assertEqual(self.run_submit([]).call_count,0)
    def test_unparseable_id_never_retried(self):
        with self.assertRaises(RuntimeError):self.run_submit(['unexpected scheduler reply'])
        self.assertEqual(self.run_submit([]).call_count,0)
if __name__=='__main__':unittest.main()

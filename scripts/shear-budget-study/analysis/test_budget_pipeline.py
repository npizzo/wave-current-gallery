"""Analytic fixtures test saved-map processing; no synthetic scientific results."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import onset_reference as old
from analyze_budget import ALL_FIELDS, map_budgets, check_scalar_maps


class SavedMapTests(unittest.TestCase):
    def fixture(self,root,ny):
        folder=root/'data/runs/test';folder.mkdir(parents=True);files={}
        cfg=dict(nx=8,ny=ny,nz=24,h=.5,k=6.1,start_time=600.,stop_time=604.)
        shape=np.sin(2*np.pi*np.arange(8)/8)[:,None]*np.ones((1,24))
        source=-.1*shape if ny>1 else np.zeros_like(shape)
        for time in (600.,602.,604.):
            values={name:np.zeros_like(shape) for name in ALL_FIELDS}
            values.update(Q=shape+(time-600)*source,Qdot_advection_full=source,Qdot_3d_native=source,Qdot_total=source)
            arrays={}
            for name,array in values.items():
                fname=f'mean_{time:012.6f}_{name}.bin';array.ravel(order='F').astype('<f8').tofile(folder/fname)
                sha=hashlib.sha256((folder/fname).read_bytes()).hexdigest();files[fname]=sha
                arrays[name]=dict(dtype='float64',order='F',sha256=sha,file=fname,shape=list(array.shape))
            fname=f'mean_{time:012.6f}.json';(folder/fname).write_text(json.dumps(dict(time=time,arrays=arrays)))
            files[fname]=hashlib.sha256((folder/fname).read_bytes()).hexdigest()
        return old.Run(root,dict(id='test'),dict(files=files)),cfg

    def test_signed_native_map_pipeline(self):
        with tempfile.TemporaryDirectory() as temp:
            run,cfg=self.fixture(Path(temp),2);result=map_budgets(run,cfg)
            self.assertEqual(result['algebraic_relative_error'],0.)
            self.assertLess(result['field_closure']['interior']['relative_to_change'],1e-14)
            self.assertLess(result['moments']['interior']['eddy']['integrated_signed_terms_s_minus2']['C'],0.)
            self.assertLess(result['moments']['interior']['eddy']['relative_to_change'],1e-14)
            self.assertEqual(check_scalar_maps(copy.deepcopy(result['rows']),result['rows']),0.)
            bad=copy.deepcopy(result['rows']);bad[-1]['interior_P_eddy_C']*=2
            with self.assertRaises(ValueError):check_scalar_maps(bad,result['rows'])

    def test_exactly_2d_native_map_pipeline(self):
        with tempfile.TemporaryDirectory() as temp:
            run,cfg=self.fixture(Path(temp),1);result=map_budgets(run,cfg)
            self.assertTrue(result['direct_3d_zero_check_2d'])
            self.assertEqual(result['max_abs_C_s_minus2'],0.)


if __name__=='__main__':unittest.main()

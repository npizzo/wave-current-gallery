import unittest,copy
from check_moments import check_scalar_maps
def fixture():
 r={'time_s':0.}
 for region in ('full','interior','stokes'):
  for part in ('Q','eddy','profile'):
   r[f'{region}_Z_{part}']=.5
   for term in ('A0','C','W','D','F'):r[f'{region}_P_{part}_{term}']=.1
  for term in ('A0','C','W','D','F'):r[f'{region}_source_rms_{term}']=1.
 return r
class Tests(unittest.TestCase):
 def test_cancellation(self):
  a=fixture();b=copy.deepcopy(a);a['full_P_profile_W']=1e-21;b['full_P_profile_W']=-2e-21
  self.assertLess(check_scalar_maps([a],[b]),1e-14)
 def test_nonzero_error(self):
  a=fixture();b=copy.deepcopy(a);b['full_P_Q_C']+=1e-7
  with self.assertRaises(ValueError):check_scalar_maps([a],[b])
 def test_energy_error(self):
  a=fixture();b=copy.deepcopy(a);b['full_Z_Q']+=1e-6
  with self.assertRaises(ValueError):check_scalar_maps([a],[b])
 def test_missing_time(self):
  a=fixture();b=copy.deepcopy(a);b['time_s']=1
  with self.assertRaises(ValueError):check_scalar_maps([a],[b])
if __name__=='__main__':unittest.main()

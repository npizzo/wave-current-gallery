import unittest
from analyze_shear import intervention_difference

class InterventionTests(unittest.TestCase):
    def curves(self,values):
        keys=('3d-u0-reference','3d-u002-reference','2d-u0','2d-u002')
        return {k:{'histories':[{'elapsed_s':t,'jet_prominence_fraction':v} for t in (200,400,600,800,1000,1200)]} for k,v in zip(keys,values)}
    def test_same_wind_effect_has_no_additional_3d_effect(self):
        # Different control trajectories must cancel in the paired contrast.
        r=intervention_difference(self.curves((.8,.4,.6,.2)))
        self.assertTrue(all(abs(x['difference_of_wind_effects'])<1e-14 for x in r['rows']))
    def test_additional_3d_erosion_is_negative(self):
        r=intervention_difference(self.curves((.8,.2,.7,.5)))
        self.assertAlmostEqual(r['rows'][0]['difference_of_wind_effects'],-.4)
    def test_existing_pair_alone_cannot_attribute_extra_erosion(self):
        c=self.curves((.8,.2,.7,.5));del c['2d-u002']
        self.assertEqual(intervention_difference(c)['status'],'AWAITING_MATCHED_2D_CONTROLS')

if __name__=='__main__':unittest.main()

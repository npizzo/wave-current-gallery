import unittest
from budget_comparisons import difference_in_differences,assess_interventions


class InterventionTests(unittest.TestCase):
    def test_3d_additional_erosion_sign(self):
        self.assertAlmostEqual(difference_in_differences(
            {'3d_control':.4,'3d_wind':.2,'2d_control':.4,'2d_wind':.5}),-.3)

    def test_matched_2d_response_cancels(self):
        self.assertAlmostEqual(difference_in_differences(
            {'3d_control':.4,'3d_wind':.2,'2d_control':.6,'2d_wind':.4}),0.)

    def test_uncomputed_control_is_pending(self):
        groups=[dict(id='test',**{'3d_control':'a','3d_wind':'b','2d_control':'c','2d_wind':'d'})]
        self.assertEqual(assess_interventions(groups,[],{},{} )[0]['status'],'pending')


if __name__=='__main__':unittest.main()

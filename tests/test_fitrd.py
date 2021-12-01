import unittest
import os
import pandas as pd
import numpy as np
import fitrd.fitrd as fitrd

class Test_fitrd(unittest.TestCase):
    def test_prepare_system(self):         
        test_dir = os.path.dirname(os.path.abspath(__file__))
        traj_filepath = os.path.join(test_dir,"a0.5_thi137_thf079.9_p.traj")
        wave_filepath = os.path.join(test_dir,"hm1_a0.5_thi137_thf079.9_p.dat")

        a = 0.5
        m = 1
        mmax = 4

        traj_data = pd.read_csv(traj_filepath, delim_whitespace=True,usecols=[0,1],names=['t','r'],skiprows=240000)
        tlrcross = np.interp(fitrd.lrradius(a), traj_data['r'].loc[::-1],traj_data['t'].loc[::-1])

        alphasystem = fitrd.preparesystem(m,a,mmax,cachedir=os.getcwd(),overwrite=True)
        spherical_modes, time, spheroidal_coefs = fitrd.solve_system(m,mmax,tlrcross,[wave_filepath],alphasystem)
        c_amps, c_phases = fitrd.postprocess(time,spheroidal_coefs,tlrcross+10,50,tlrcross+210)

        x0 = np.array([1.69175971, 1.04324328, 0.48921908, 0.23496657, 0.18614327, 0.06827545, 0.08147308, 0.02036668])
        self.assertAlmostEqual(np.linalg.norm(c_amps - x0),0.0,7)

if __name__ == '__main__':
    unittest.main()        
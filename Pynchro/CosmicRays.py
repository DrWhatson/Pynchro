import numpy as np
import healpy as hp
import Profiler as prof


class CRE():
    def __init__(self,grid):
        """Class to build and store distribution of Cosmic Rays
        """
        self.grid = grid
        n = grid.get_n_coords()
        self.cre_density = np.zeros(n,dtype='float')

    def parse(self,dict):
        """Take dictionary of parameter and make distributions
        """

        grid = self.grid
        cre_density = self.cre_density

        for key,value in dict.iteritems():
            if prof.has_profile(key):
                prof.apply_profile(key,grid,cre_density,value)

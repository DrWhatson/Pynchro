import numpy as np
import healpy as hp
import CR_Profiler as prof


class CRE():
    def __init__(self,grid):
        """Class to build and store distribution of Cosmic Rays
        """
        self.grid = grid
        n = grid.get_n_coords()
        self.Ndensity = np.zeros(n,dtype='float')
        self.ref_energy = 20.6 # GeV
        self.index = -3.0


    def parse(self,dict):
        """Take dictionary of parameter and make distributions
        """

        grid = self.grid
        self.zero()
        Ndensity = self.Ndensity

        for key,value in dict.iteritems():
            if prof.has_profile(key):
                prof.apply_profile(key,grid,Ndensity,value)

    def zero(self):
        self.Ndensity *= 0.0

    def regrid_xy(self,xy,z=0.0,Bval='mag'):
        from scipy.interpolate import griddata
        flg = np.where(self.grid.XYZ_gal[2]==z)[0]
        points = np.array([self.grid.XYZ_gal[0][flg],self.grid.XYZ_gal[1][flg]])
        points = np.transpose(points)
        values = self.Ndensity[flg]
        xy = np.transpose(xy)
#        print points.shape,values.shape, xy.shape
        grid_xy = griddata(points, values, xy, method='linear')
        return grid_xy

    def regrid_xz(self,xz,y=0.0):
        from scipy.interpolate import griddata
        flg = np.where(self.grid.XYZ_sun[1]==y)[0]
        points = np.array([self.grid.XYZ_sun[0][flg],self.grid.XYZ_sun[2][flg]])
        points = np.transpose(points)
        values = self.cre_density[flg]
        xz = np.transpose(xz)
#        print points.shape,values.shape, xz.shape
        grid_xz = griddata(points, values, xz, method='linear')
        return grid_xz

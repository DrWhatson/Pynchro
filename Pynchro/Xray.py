import numpy as np
import healpy as hp
from scipy.interpolate import RegularGridInterpolator
import Xray_Profiler as prof


class Xray():
    def __init__(self,grid):
        """Class to build and store local 3D dust extinction 
        """
        self.grid = grid
        n = grid.get_n_coords()
        self.xray = np.zeros(n,dtype='double')

    def zero(self):
        self.xray *= 0.0


    def parse(self,dict):
        """Take dictionary of parameter and make distributions
        """

        grid = self.grid
        self.zero()
        xray = self.xray

        for key,value in dict.iteritems():
            if prof.has_profile(key):
                prof.apply_profile(key,grid,xray,value)


    def regrid_xy(self,xy,z=0.0,Bval='mag'):
        from scipy.interpolate import griddata

        flg = np.where(np.abs(self.grid.XYZ_gal[2]-z)<0.01)[0]
        points = np.array([self.grid.XYZ_gal[0][flg],self.grid.XYZ_gal[1][flg]])
        points = np.transpose(points)
        values = self.xray[flg]
        xy = np.transpose(xy)
#        print points.shape,values.shape, xy.shape
        grid_xy = griddata(points, values, xy, method='linear')
        return grid_xy


    def regrid_xz(self,xz,y=0.0,lon=0.0,Bval='mag'):
        from scipy.interpolate import griddata

#        flg = np.where(self.grid.XYZ_sun[1]==y)[0]

        lon *= np.pi/180

        rot = self.grid.XYZ_gal[1]*np.cos(-lon)+(self.grid.XYZ_gal[0]+8.5)*np.sin(-lon)
#        flg = np.where(abs(self.grid.XYZ_gal[1]-y)<0.01)[0]
        flg = np.where(np.abs(rot-y)<0.01)[0]

        points = np.array([self.grid.XYZ_gal[0][flg],self.grid.XYZ_gal[2][flg]])
        points = np.transpose(points)
        values = self.xray[flg]
        xz = np.transpose(xz)
#        print points.shape,values.shape, xz.shape
        grid_xz = griddata(points, values, xz, method='linear')
        return grid_xz

import numpy as np
import healpy as hp
from scipy.interpolate import RegularGridInterpolator

class GAS():
    def __init__(self,grid):
        """Class to build and store local 3D Gas
        """
        self.grid = grid
        n = grid.get_n_coords()
        self.GAS = np.zeros(n,dtype='double')

    def read_file(self,filen,shape,ranges):

        gas = np.fromfile(filen)
        gas.shape = shape

        print gas.shape

        gas = np.transpose(gas)
#        gas = np.transpose(gas,(1,2,0))

        x = np.linspace(ranges[0][0],ranges[0][1],shape[0])
        y = np.linspace(ranges[1][0],ranges[1][1],shape[1])
        z = np.linspace(ranges[2][0],ranges[2][1],shape[2])

        intplr = RegularGridInterpolator((x, y, z), gas)

        grid = self.grid
        n = grid.get_n_coords()

        flg = np.where(grid.XYZ_gal[0]<ranges[0][0],0,1)
        flg *= np.where(grid.XYZ_gal[0]>ranges[0][1],0,1)
        flg *= np.where(grid.XYZ_gal[1]<ranges[1][0],0,1)
        flg *= np.where(grid.XYZ_gal[1]>ranges[1][1],0,1)
        flg *= np.where(grid.XYZ_gal[2]<ranges[2][0],0,1)
        flg *= np.where(grid.XYZ_gal[2]>ranges[2][1],0,1)

        flg = np.compress(flg,np.arange(n))

        pts = np.transpose(grid.XYZ_gal[:,flg])

        self.GAS[flg] = intplr(pts)


    def regrid_xy(self,xy,z=0.0,Bval='mag'):
        from scipy.interpolate import griddata

        flg = np.where(np.abs(self.grid.XYZ_gal[2]-z)<0.01)[0]
        points = np.array([self.grid.XYZ_gal[0][flg],self.grid.XYZ_gal[1][flg]])
        points = np.transpose(points)
        values = self.GAS[flg]
        xy = np.transpose(xy)
#        print points.shape,values.shape, xy.shape
        grid_xy = griddata(points, values, xy, method='linear')
        return grid_xy


    def regrid_xz(self,xz,y=0.0,lon=0.0,Bval='mag'):
        from scipy.interpolate import griddata

        sun = self.grid.sun
#        flg = np.where(self.grid.XYZ_sun[1]==y)[0]

        lon *= np.pi/180

        rot = self.grid.XYZ_gal[1]*np.cos(-lon)+(self.grid.XYZ_gal[0]-sun.x)*np.sin(-lon)
#        flg = np.where(abs(self.grid.XYZ_gal[1]-y)<0.01)[0]
        flg = np.where(np.abs(rot-y)<0.01)[0]

        points = np.array([self.grid.XYZ_gal[0][flg],self.grid.XYZ_gal[2][flg]])
        points = np.transpose(points)
        values = self.GAS[flg]
        xz = np.transpose(xz)
#        print points.shape,values.shape, xz.shape
        grid_xz = griddata(points, values, xz, method='linear')
        return grid_xz

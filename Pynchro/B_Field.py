import numpy as np
import healpy as hp
import B_Profiler as prof
from scipy.interpolate import RegularGridInterpolator
#from scipy.io import FortranFile

class GMF():
    def __init__(self,grid):
        """Class to build and store the Galactic magnetic field
        """

        self.grid = grid
        n = grid.get_n_coords()
        self.B_rho = np.zeros(n,dtype='float')
        self.B_phi = np.zeros(n,dtype='float')
        self.B_z = np.zeros(n,dtype='float')

    def zero(self):
        self.B_rho *= 0.0
        self.B_phi *= 0.0
        self.B_z *= 0.0


    def parse(self,dict):
        """Take dictionary of parameter and make distributions
        """

        self.zero()

        grid = self.grid
        B_rho = self.B_rho
        B_phi = self.B_phi
        B_z = self.B_z

        for key,value in dict.iteritems():
            if prof.has_profile(key):
                prof.apply_profile(key,grid,B_rho,B_phi,B_z,value)

    def read_file(self,filex,filey,filez,shape,ranges):

        tmp = np.fromfile(filex,dtype='float32')
        tmp = tmp[7:]*1e-6
        tmp.shape = shape

        Bx = np.zeros(shape,dtype='float32')
        Bx = np.transpose(tmp)

        tmp = np.fromfile(filex,dtype='float32')
        tmp = tmp[7:]*1e-6
        tmp.shape = shape

        By = np.zeros(shape,dtype='float32')
        By = np.transpose(tmp)

        tmp = np.fromfile(filex,dtype='float32')
        tmp = tmp[7:]*1e-6
        tmp.shape = shape

        Bz = np.zeros(shape,dtype='float32')
        Bz = np.transpose(tmp)

        x = np.linspace(ranges[0][0],ranges[0][1],shape[0])
        y = np.linspace(ranges[1][0],ranges[1][1],shape[1])
        z = np.linspace(ranges[2][0],ranges[2][1],shape[2])

        intplrx = RegularGridInterpolator((x, y, z), Bx)
        intplry = RegularGridInterpolator((x, y, z), By)
        intplrz = RegularGridInterpolator((x, y, z), Bz)

        grid = self.grid
        n = grid.get_n_coords()

        flg = np.where(grid.XYZ_gal[0]<ranges[0][0],0,1)
        flg *= np.where(grid.XYZ_gal[0]>ranges[0][1],0,1)
        flg *= np.where(grid.XYZ_gal[1]<ranges[1][0],0,1)
        flg *= np.where(grid.XYZ_gal[1]>ranges[1][1],0,1)
        flg *= np.where(grid.XYZ_gal[2]<ranges[2][0],0,1)
        flg *= np.where(grid.XYZ_gal[2]>ranges[2][1],0,1)

        flg = np.compress(flg,np.arange(n))

        ip = np.array([2,1,0])
        pts = np.transpose(grid.XYZ_gal[:,flg])

        B_x = intplrx(pts)
        del Bx

        B_y = intplry(pts)
        del By

        self.B_z[flg] = intplrz(pts)
        del Bz

        # Global
        phi = np.arctan2(self.grid.XYZ_gal[1][flg],self.grid.XYZ_gal[0][flg])

        # Finally rotate Bx,By to Brho Bphi

        B_rho = B_x #B_x*np.cos(-phi) + B_y*np.sin(-phi)
        B_phi = B_y #-B_x*np.sin(-phi) + B_y*np.cos(-phi)

        self.B_rho[flg] = B_rho
        self.B_phi[flg] = B_phi

        del B_rho, B_phi

    def regrid_xy(self,xy,z=0.0,Bval='Bmag'):
        from scipy.interpolate import griddata
        flg = np.where(np.abs(self.grid.XYZ_gal[2]-z)<0.01)[0]
        points = np.array([self.grid.XYZ_gal[0][flg],self.grid.XYZ_gal[1][flg]])
        points = np.transpose(points)

        if Bval=='Bmag':
            values = np.sqrt(self.B_rho[flg]**2 + self.B_phi[flg]**2 + self.B_z[flg]**2)
        elif Bval=='B_rho':
            values = self.B_rho[flg]
        elif Bval=='B_phi':
            values = self.B_phi[flg]
        elif Bval=='B_z':
            values = self.B_z[flg]
        elif Bval=='B_x':
            B_x,B_y,B_z = self.get_Bxyz()
            values = B_x[flg]
        elif Bval=='B_y':
            B_x,B_y,B_z = self.get_Bxyz()
            values = B_y[flg]
        else:
            print "Unrecognized val option %s" % Bval
            return flg*0.0

        xy = np.transpose(xy)
#        print points.shape,values.shape, xy.shape
        grid_xy = griddata(points, values, xy, method='linear')
        return grid_xy


    def get_AlphaGammaBtrans2_fromGal(self):
        """
        FUNCTION
            ===  GET_ALPHAGAMMA_FROMGAL : get position and inclination angles ===

        From a Galactic magnetic vector field given in  Galactocentric
        coordinate system, this function evaluates the relevant angles for
        polarizations inquieries in the oberser reference frame

        Created on Oct 25 2016
        Based on get_AlphaGamma.py written on Jul 6 2016
        @author: V.Pelgrims

        """

        #bfield components in GalactoCentric reference frame
        B_rho = self.B_rho
        B_z = self.B_z
        B_phi = self.B_phi


        #conversion of the vector field in GC ref. frame to the observer spherical
        #coordinate system
        dots = self.grid.dotproducts
        B_r = B_rho * dots[0] + B_z * dots[3] + B_phi * dots[6]
        B_t = B_rho * dots[1] + B_z * dots[4] + B_phi * dots[7]
        B_p = B_rho * dots[2] + B_z * dots[5] + B_phi * dots[8]

        #NOTE: computing this convresion internally turns out to be (much) faster
        #that calling __gal2sun_vector() internally or externally.

        #inclination angle
        Alpha = np.arccos( (B_r*B_r / ( B_r*B_r + B_t*B_t + B_p*B_p ))**.5)

        #polarization position angle
        Gamma = .5 * np.arctan2( - 2 * B_t*B_p , B_p*B_p - B_t*B_t )
        #written this way to be in HEALPix convention.
        #Gamma is the polarization position angle expected to be
        #perfectly perpendicular to the projection on the sky of
        #the magnetic field vector

        #Gamma = 0 means the polarization points to the South
        #      = pi/2                                   East


        Btransverse = np.sqrt(B_t*B_t + B_p*B_p)

        return Alpha,Gamma,Btransverse

    def get_Bxyz(self):

        # Global phi
        phi = np.arctan2(self.grid.XYZ_gal[1],self.grid.XYZ_gal[0])

        # Finally rotate Brho,Bphi to Bx and By

        Bx = self.B_rho*np.cos(phi) - self.B_phi*np.sin(phi)
        By = self.B_rho*np.sin(phi) + self.B_phi*np.cos(phi)

        return Bx, By, self.B_z

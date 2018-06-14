"""
Pynchro is a package intended to model synchrotron emission
from the galaxy.
"""

import sys
import os
from warnings import warn

#Setup will go here

import numpy as np
import healpy as hp

pi = np.pi

class Sun():
    def __init__(self,x=-8.5,y=0.0,z=0.0):
        self.x = x
        self.y = y
        self.z = z

class Grid():
    def __init__(self,NSIDE=64,radial_step=0.2,radial_max=20.0,sun=Sun()):
        """
        Build a grid of xyz points
        """

        self.NSIDE = NSIDE
        self.radial_step = radial_step
        self.radial_max = radial_max
        self.sun = sun

        NPIX = hp.nside2npix(NSIDE)
        r = np.arange(radial_step,radial_max+radial_step,radial_step)
        rSize = r.size

        self.Npix = NPIX
        self.rSize = rSize


        #unit pointing vectors
        xyz_n = np.asarray(hp.pix2vec(NSIDE,np.arange(NPIX)))

        r = r[np.newaxis,np.newaxis,:]
        xyz_n = xyz_n[:,:,np.newaxis]

        #pointing vectors to all wanted positions
        self.XYZ_sun = xyz_n*r
        shape = self.XYZ_sun.shape
        self.XYZ_sun.shape = [shape[0],shape[1]*shape[2]]

        self.bondary = xyz_n * (radial_max+radial_step)

        self.XYZ_gal = (self.XYZ_sun +
               np.tile(np.array([sun.x,sun.y,sun.z]),(rSize*NPIX,1)).T)

        self.__ifBfield()


    def get_n_coords(self):
        return self.XYZ_gal.shape[1]

    def get_cartesian(self):
        return self.XYZ_gal

    def __ifBfield(self):
        """
        FUNCTION
            That comptutes the dotproducts between the basis vectors of the
        cylindrical coordinate system centred on the Galactic centre and those
        of the spherical coordinate system centred on the Sun.
        """

        #compute the cylindrical-basis vectors in the GC reference frame
        u_rhoG,u_zG,u_phiG = self.__u_cyl(self.XYZ_gal)

        #compute the spherical-basis vectors in the heliocentric reference frame
        u_rS,u_tS,u_pS = self.__u_sph(self.XYZ_sun)

        #compute the required scalar products between basis vectors
        d_rGrS = np.sum(u_rhoG*u_rS, axis=0)    # u_rhoG . u_rS
        d_rGtS = np.sum(u_rhoG*u_tS, axis=0)    # u_rhoG . u_thetaS
        d_rGpS = np.sum(u_rhoG*u_pS, axis=0)    # u_rhoG . u_phiS

        d_zGrS = np.sum(u_zG*u_rS, axis=0)      # u_zG . u_rS
        d_zGtS = np.sum(u_zG*u_tS, axis=0)      # u_zG . u_thetaS
        d_zGpS = np.sum(u_zG*u_pS, axis=0)      # u_zG . u_phiS

        d_pGrS = np.sum(u_phiG*u_rS, axis=0)    # u_phiG . u_rS
        d_pGtS = np.sum(u_phiG*u_tS, axis=0)    # u_phiG . u_thetaS
        d_pGpS = np.sum(u_phiG*u_pS, axis=0)    # u_phiG . u_phiS

        self.dotproducts = np.array([d_rGrS,d_rGtS,d_rGpS,
                            d_zGrS,d_zGtS,d_zGpS,
                            d_pGrS,d_pGtS,d_pGpS])


    def __u_cyl(self,coord):
        """
        For input cartesian coordinates, returns the 3-basis vectors of the
        cylindrical coordinate system.
        """

        phi = np.arctan2(coord[1],coord[0])

        #compute the stuff
        u_rho1 = np.cos(phi)
        u_rho2 = np.sin(phi)
        u_rho3 = np.zeros(phi.size)

        u_rho = np.array([u_rho1, u_rho2, u_rho3])

        u_phi = np.array([-u_rho2, u_rho1, u_rho3])

        u_z = np.array([u_rho3, u_rho3, u_rho3 + 1.])

        return u_rho, u_z, u_phi


    def __u_sph(self,coord):
        """
        For input cartesian coordinates, returns the 3-basis vectors of the
        spherical coordinate system.
        """

        theta = np.arctan2((coord[0]**2 + coord[1]**2)**.5,coord[2])
        phi = np.arctan2(coord[1],coord[0])

        #compute the stuff
        Cphi = np.cos(phi)
        Sphi = np.sin(phi)
        Ctheta = np.cos(theta)
        Stheta = np.sin(theta)

        u_r = np.array([Cphi*Stheta, Sphi*Stheta, Ctheta])
        u_theta = np.array([Cphi*Ctheta, Sphi*Ctheta, -Stheta])
        u_phi = np.array([-Sphi, Cphi, np.zeros(Cphi.size)])

        return u_r, u_theta, u_phi


class Grid_cone(Grid):
    def __init__(self,NSIDE=64,radial_step=0.2,radial_max=20.0,sun=Sun(),
                 radial_min=0,centre=[0.0,0.0],ang_radius=5.0):
        """
        Build a grid of xyz points inside a cone
        """

        self.NSIDE = NSIDE
        self.radial_step = radial_step
        self.radial_max = radial_max
        self.radial_min = radial_min
        self.sun = sun
        self.ang_radius = ang_radius*np.pi/180.
        self.centre = centre

        r = np.arange(radial_min,radial_max+radial_step,radial_step)
        rSize = r.size
        self.rSize = rSize

        if len(centre) == 3: # Is it a xyz offset?
            mag = np.sqrt(centre[0]**2 + centre[1]**2 + centre[2]**2)
            vec = np.array(centre)/mag
        elif len(centre) ==2: # Or glon & glat
            vec = hp.ang2vec((90-centre[1])*np.pi/180,centre[0]*np.pi/180)
        else:
            raise ValueError("Incorrect number of centre coordinates %i" % len(centre))

        #unit pointing vectors
        self.ipix = hp.query_disc(NSIDE,vec,self.ang_radius)
        xyz_n = np.asarray(hp.pix2vec(NSIDE,self.ipix))

        NPIX = len(self.ipix)
        self.Npix = NPIX

        r = r[np.newaxis,np.newaxis,:]
        xyz_n = xyz_n[:,:,np.newaxis]

        #pointing vectors to all wanted positions
        self.XYZ_sun = xyz_n*r
        shape = self.XYZ_sun.shape
        self.XYZ_sun.shape = [shape[0],shape[1]*shape[2]]

        self.XYZ_gal = (self.XYZ_sun +
               np.tile(np.array([sun.x,sun.y,sun.z]),(rSize*NPIX,1)).T)

        self._Grid__ifBfield()

class Grid_strip(Grid):
    def __init__(self,NSIDE=64,radial_step=0.2,radial_max=20.0,sun=Sun(),
                 radial_min=0,lonra=[-45.0,45.0],latra=[-45.0,45.0]):
        """
        Build a grid of xyz points inside a range of galactic longitude and latitude
        """

        self.NSIDE = NSIDE
        self.radial_step = radial_step
        self.radial_max = radial_max
        self.radial_min = radial_min
        self.sun = sun
        self.lonra = np.array(lonra)
        self.latra = np.array(latra)

        r = np.arange(radial_min,radial_max+radial_step,radial_step)
        rSize = r.size

        #unit pointing vectors
        the_lo = np.pi/2-latra[1]*np.pi/180
        the_hi = np.pi/2-latra[0]*np.pi/180
        ipix = hp.query_strip(NSIDE,the_lo,the_hi)
        the, lon = hp.pix2ang(NSIDE,ipix)
        if lonra[0]<0:
            lon = np.where(lon>np.pi,lon-2*np.pi,lon)
        flg = np.where(lon>lonra[0]*np.pi/180,1,0)
        flg *= np.where(lon<lonra[1]*np.pi/180,1,0)
        self.ipix = np.compress(flg,ipix)
        xyz_n = np.asarray(hp.pix2vec(NSIDE,self.ipix))

        NPIX = len(self.ipix)

        r = r[np.newaxis,np.newaxis,:]
        xyz_n = xyz_n[:,:,np.newaxis]

#        print "NPIX =",NPIX
#        print xyz_n.shape
#        print r.shape

        #pointing vectors to all wanted positions
        self.XYZ_sun = xyz_n*r
        shape = self.XYZ_sun.shape
        self.XYZ_sun.shape = [shape[0],shape[1]*shape[2]]

#        print self.XYZ_sun.shape

        self.XYZ_gal = (self.XYZ_sun +
               np.tile(np.array([sun.x,sun.y,sun.z]),(rSize*NPIX,1)).T)

        self._Grid__ifBfield()

class Grid_list(Grid):
    def __init__(self,NSIDE=64,radial_step=0.2,radial_max=20.0,sun=Sun(),
                 radial_min=0,ipix=np.arange(12*64*64)):
        """
        Build a grid of xyz points from list of healpix pixel ids
        """

        self.NSIDE = NSIDE
        self.radial_step = radial_step
        self.radial_max = radial_max
        self.radial_min = radial_min
        self.sun = sun

        r = np.arange(radial_min,radial_max+radial_step,radial_step)
        rSize = r.size

        #unit pointing vectors
        self.ipix = ipix
        xyz_n = np.asarray(hp.pix2vec(NSIDE,self.ipix))

        NPIX = len(self.ipix)

        #pointing vectors to all wanted positions
        self.XYZ_sun = (np.tile(xyz_n,(1,rSize)) *
               np.tile(np.reshape(np.tile(r.T,(NPIX,1)),
               NPIX*rSize,1).T,(3,1)))

        self.XYZ_gal = (self.XYZ_sun +
               np.tile(np.array([sun.x,sun.y,sun.z]),(rSize*NPIX,1)).T)

        self._Grid__ifBfield()


def plot_galactic_xy_slice(val,xrange=(-20,20),yrange=(-20,20),step=0.05,Bval='Bmag'):

    import matplotlib.pyplot as plt
    import matplotlib

    x = np.arange(xrange[0]-step/2.,xrange[1]+step/2.,step)
    y = np.arange(yrange[0]-step/2.,yrange[1]+step/2.,step)
    X,Y = np.meshgrid(x,y)
    XY = np.asarray([X.ravel(),Y.ravel()])

    vals_regrid = val.regrid_xy(XY,Bval=Bval)

#    print XY.shape, vals_regrid.shape

    #plotting stuff
    limites = [x.min(),x.max(),y.min(),y.max()]
    plt.figure()
    plt.imshow(np.reshape(vals_regrid,[len(y),len(x)]),
               extent=limites,
               origin='lower')
#               norm=matplotlib.colors.LogNorm(vmin=5.0e-3,
#                                              vmax=1.1))
    plt.axis(limites)
    plt.colorbar()
    plt.plot(0,0,'ok')
#    plt.plot(x_sun,y_sun,'*k')
    plt.xlabel('$X$ [kpc]',fontsize=18)
    plt.ylabel('$Y$ [kpc]',fontsize=18)
    plt.show()

def plot_galactic_xz_slice(val,xrange=(-20,20),zrange=(-5,5),step=0.05,Bval='Bmag'):

    import matplotlib.pyplot as plt
    import matplotlib

    x = np.arange(xrange[0]-step/2.,xrange[1]+step/2.,step)
    z = np.arange(zrange[0]-step/2.,zrange[1]+step/2.,step)
    X,Z = np.meshgrid(x,z)
    XZ = np.asarray([X.ravel(),Z.ravel()])

    vals_regrid = val.regrid_xz(XZ,Bval=Bval)

#    print XZ.shape, vals_regrid.shape

    #plotting stuff
    limites = [x.min(),x.max(),z.min(),z.max()]
    plt.figure()
    plt.imshow(np.reshape(vals_regrid,[len(z),len(x)]),
               extent=limites,
               origin='lower')
    #           norm=matplotlib.colors.LogNorm(vmin=5.0e-3,vmax=1.1))

    plt.axis(limites)
    plt.colorbar()
    plt.plot(0,0,'ok')
#    plt.plot(x_sun,y_sun,'*k')
    plt.xlabel('$X$ [kpc]',fontsize=18)
    plt.ylabel('$Z$ [kpc]',fontsize=18)
    plt.show()



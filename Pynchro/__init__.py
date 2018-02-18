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

        #unit pointing vectors
        xyz_n = np.asarray(hp.pix2vec(NSIDE,np.arange(NPIX)))

        #pointing vectors to all wanted positions
        self.XYZ_sun = (np.tile(xyz_n,(1,rSize)) *
               np.tile(np.reshape(np.tile(r.T,(NPIX,1)),
               NPIX*rSize,1).T,(3,1)))

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

        #compute the requiered scalar products between basis vectors
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

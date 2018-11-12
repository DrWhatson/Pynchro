import numpy as np
import healpy as hp
import scipy.special as sf

class CGS():
    C_light = 2.99792458e10  #cm/sec
    qe = 4.80320425e-10 #
    eV = 1.60217733e-12 #ergs
    GeV = 1.e9 * eV
    MEC2 = 0.51099907e-3 * GeV
    ME = MEC2 /(C_light*C_light)
    MEC = MEC2/C_light
    E3MC2 = qe**3 / MEC2
    kpc = 3.086e+21 # cm
    kB =1.380622e-16


###########################  SYNCHROTRON EMISION  ##########################


def synchrotron_emissivity_tot_pol(Ngam, nu, Bper, index):
  
    # coefficients which do not attend integration
    norm = Ngam*np.sqrt(3)/2.*CGS.E3MC2*np.abs(Bper)

    # synchrotron integration
    A = 4.*np.pi*CGS.MEC*nu/(3.*CGS.qe*np.abs(Bper))
    mu = -0.5 * (3.+index)

    emiss_tot = norm * A**(0.5*(index+1)) * 2**(mu+1) * sf.gamma(0.5*mu+7./3.)*sf.gamma(0.5*mu+2./3.)/(mu+2)/(4*np.pi)

    emiss_pol = norm * A**(0.5*(index+1)) * 2**mu * sf.gamma(0.5*mu+4./3.)*sf.gamma(0.5*mu+2./3.)/(4*np.pi)

    return emiss_tot, emiss_pol


def dIQU_sync(bmag,cre,nu):
 
    alp_Bf, gam_Bf, Bper = bmag.get_AlphaGammaBtrans2_fromGal()

    emiss_tot,emiss_pol = synchrotron_emissivity_tot_pol(cre.Ndensity, nu, Bper, cre.index)

    #intensity
    dI = emiss_tot

    #Complex polarization
    dP = emiss_pol * np.exp(-2j*gam_Bf)
    
    return dI,dP



def IQU_sync(bmag, cre, nu):

    Npix = bmag.grid.Npix
    stepR = bmag.grid.radial_step
    rSize = bmag.grid.rSize
    fud = 1.15 # 15% normalisation to get to agree with hamx    


    I2bt = CGS.C_light**2/(2.*CGS.kB*nu**2)
    print nu, I2bt, stepR, CGS.kpc

    dI,dP = dIQU_sync(bmag, cre, nu)

    # If need to do Faraday rotation, do it here before sum

    I = np.sum(np.reshape(dI,[Npix,rSize]),axis=1)*stepR*CGS.kpc*I2bt*2*np.pi/fud
    P = np.sum(np.reshape(dP,[Npix,rSize]),axis=1)*stepR*CGS.kpc*I2bt*2*np.pi/fud

    Q = P.real
    U = P.imag

    I = expand_partial_healpix(I,bmag.grid)
    Q = expand_partial_healpix(Q,bmag.grid)
    U = expand_partial_healpix(U,bmag.grid)

    return I,Q,U


#def Extinction(vis, ext):

#    Npix = bmag.grid.Npix
#    stepR = bmag.grid.radial_step
#    rSize = bmag.grid.rSize

    # Go from outer shell inwards

#    for i in np.arange(rSize):
#        I = vis.


def expand_partial_healpix(ppix, grid):
    NPIX = hp.nside2npix(grid.NSIDE)
    pix = np.zeros(NPIX,dtype='double')
    pix[grid.ipix] = ppix
    return pix
    
    



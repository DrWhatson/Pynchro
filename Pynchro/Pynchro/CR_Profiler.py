#from numba import jit
import numpy as np
import healpy as hp
from Pynchro.Integrator import CGS


def has_profile(type):
    types = profiles.keys()

#    print 'profiles=',profiles

    if type in types:
        return True
    else:
        print "Unrecognized profile type %s" % type
        return False


def apply_profile(type,grid,Nden,parms):
    profiles[type](grid,Nden,parms)

# analytical CRE flux normalization factor at E0
# analytical CRE spectral integrations use N(\gamma)
def flux_norm(j0,E0,flux_idx):
    # j0 is in [GeV m^2 s sr]^-1 units
    gamma0 = E0*CGS.GeV/CGS.MEC2+1
    beta0 = np.sqrt(1.-1./gamma0**2)
    # from PHI(E) to N(\gamma) convertion
    unit = (4.*np.pi*CGS.MEC)/(CGS.GeV*100.*100.*beta0)
    norm = j0*gamma0**(-flux_idx)
   
#    print "Unit=",unit,"  norm=",norm," Gamma0=",gamma0," E0=",E0

    return norm*unit


def Loop(grid,Nden,parms={},
    r_in=113,
    r_out=120,
    b_0=[0.0,1.0,0.0],
    loopCentre=[-8.0,-0.045,0.07],
    J_0=0.0217,
    E_0=20.6,
    alpha=3):

    parm_dict = {'r_inner':r_in,
                 'r_outer':r_out,
                 'B_0':b_0,
                 'loopCentre':loopCentre,
                 'J_0':J_0,
                 'E_0':E_0,
                 'alpha':alpha}


    parm_keys = parm_dict.keys()

    for loop in parms:
        for key_p,val_p in loop.iteritems():
            if key_p in parm_keys:
                parm_dict[key_p] = val_p
            else:
                print "Unrecognized Loop parmeter %s" % key_p

    loop_profile(grid,Nden,parm_dict)

#@jit
def loop_profile(grid,Nden,parm):

#    print parm

    r_in = parm['r_inner']
    r_out = parm['r_outer']
    loopCentre = parm['loopCentre']
    J_0 = parm['J_0']
    E_0 = parm['E_0']
    alpha = parm['alpha']

    N_0 = flux_norm(J_0,E_0,-alpha)

    coord = grid.get_cartesian()
    lc = np.array(loopCentre)
    lc =lc[:,np.newaxis]

    xyz = coord - np.tile(lc,(1,len(coord[0])))
    r = np.sqrt(np.sum(xyz*xyz,axis=0))
    nb = Nden.shape[0]

#    print "N_0",N_0
#    print "Coord shape",coord.shape
#    print "r_in",r_in
#    print "r_out",r_out
#    print "LoopCentre",lc

    # Set inside of loop to zero
    flg = np.where(r<r_in)[0]
    Nden[flg] = 0.0

#    print np.min(r),np.max(r)
    flg = np.where(r>r_in,1,0)
    flg *= np.where(r<r_out,1,0)
    flg = np.compress(flg,np.arange(nb))

#    the = np.arcsin(coord[1][flg]/r[flg])
#    phi = np.arctan2(coord[0][flg],coord[2][flg])

    Nden[flg] += N_0
#    print "Nflag",flg.shape

#    bthe = np.arcsin(b_0[2]/Bmag)
#    bphi = np.arctan2(b_0[1],b_0[0])

    return


def Rexp_Zsech2(grid,Nden,parms={},
    r_scale=5, # Radius exp scale
    z_scale=1, # 
    J_0=0.0217,
    E_0=20.6,
    alpha=3):

    parm_dict = {'r_scale':r_scale,
                 'z_scale':z_scale,
                 'J_0':J_0,
                 'E_0':E_0,
                 'alpha':alpha}

    parm_keys = parm_dict.keys()

    for parm in parms:
        for key_p,val_p in parm.iteritems():
            if key_p in parm_keys:
                parm_dict[key_p] = val_p
            else:
                print "Unrecognized Rexp_Zsech2 parmeter %s" % key_p

    xyz = grid.get_cartesian()
    r = np.sqrt(np.sum(xyz*xyz,axis=0))

    r_scale = parm_dict['r_scale']
    z_scale = parm_dict['z_scale']
    J_0 = parm_dict['J_0']
    E_0 = parm_dict['E_0']
    alpha = parm_dict['alpha']

    N_0 = flux_norm(J_0,E_0,-alpha)

#    print "N_0 = ",N_0

    Nden += N_0 * np.exp(-r/r_scale)/np.cosh(xyz[2]/z_scale)/np.cosh(xyz[2]/z_scale)

def RING(grid, Nden, parms={},
    N_0 = 3.2e-4,
    r_in = 1.,
    r_ex = 5.):

    parm_dict = {'N_0':N_0,
                 'r_in':r_in,
                 'r_ex':r_ex}

    parm_keys = parm_dict.keys()

    for loop in parms:
        for key_p,val_p in loop.iteritems():
            if key_p in parm_keys:
                parm_dict[key_p] = val_p
            else:
                print "Unrecognized RING parmeter %s" % key_p

    N_0 = parm_dict['N_0']
    r_in = parm_dict['r_in']
    r_ex = parm_dict['r_ex']

    xyz = grid.get_cartesian()
    r = np.sqrt(np.sum(xyz[0:1,:]*xyz[0:1,:],axis=0))
    nb = len(r)

    flg = np.where(r>r_in,1,0)
    flg *= np.where(r<r_ex,1,0)
    flg = np.compress(flg,np.arange(nb))

    Nden[flg] += N_0



def OneParamFunc(var,param,form):

    if (form == 'Gaussian' or form == 'Gauss' or form == 'G'):
        f_of_var = np.exp(-(var*var) / (2 * param*param))
    elif (form == 'OffsetGaussian' or form == 'OffGauss' or form == 'OG'):
        f_of_var = np.exp(-((var-param)*(var-param)) / (param*param))
    elif (form == "Decreasing_Exp" or form == 'd_exp' or
          form == 'exp' or form == 'DE'):
        f_of_var = np.exp( - np.abs(var) / param )
    elif (form == 'HyperbolicSecant' or form == 'H' or
          form == 'HS' or form == 'cosh'):
        f_of_var = 1 / (np.cosh(- var / param) * np.cosh(- var / param))
    elif (form == 'PowerLaw' or form == 'PL' or form == 'power'):
        f_of_var = np.abs(var) ** param
    elif (form == 'Cosine' or form == 'C' or form == 'cos'):
        f_of_var = np.maximum(np.cos(pi*var / (2*param)), 0.0)
    else:
        print('The one-parameter functional referenced by',
              form,
              'is not recognized. \n Wrong entry or not',
              'yet implemented. Try again!')
        return

    return f_of_var



profiles = {'Rexp_Zsech2':Rexp_Zsech2, 'RING':RING, 'Loop':Loop}

from numba import jit
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

def Loop(grid,Nden,parms={},
    r_in=113,
    r_out=120,
    loopCentre=[-8.0,-0.045,0.07],
    J_0=1.0,
    dN=0.0,
    gthe = 0.,
    gphi = 90.):

    parm_dict = {'r_inner':r_in,
                 'r_outer':r_out,
                 'loopCentre':loopCentre,
                 'J_0':J_0,
                 'dN':dN,
                 'gthe':gthe,
                 'gphi':gphi}


    parm_keys = parm_dict.keys()

    for loop in parms:
        for key_p,val_p in loop.iteritems():
            if key_p in parm_keys:
                parm_dict[key_p] = val_p
            else:
                print "Unrecognized Loop parmeter %s" % key_p

    loop_profile(grid,Nden,parm_dict)



def loop_profile(grid,Nden,parm):

#    print parm

    r_in = parm['r_inner']
    r_out = parm['r_outer']
    loopCentre = parm['loopCentre']
    J_0 = parm['J_0']
 
    dN = np.array(parm['dN']) # Magnitude of density gradient
    gthe = parm['gthe']*np.pi/180. # Direction of gradient in density
    gphi = parm['gphi']*np.pi/180.

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

    flg = np.where(r>r_in,1,0)
    flg *= np.where(r<r_out,1,0)
#    flg *= np.where(xyz[0]<0,0,1)
#    flg *= np.where(xyz[1]<0,0,1)
    flg = np.compress(flg,np.arange(nb))

    x = xyz[0][flg]
    y = xyz[1][flg]
    z = xyz[2][flg]

    # Gradient in magnet field strenght 
    Nden[flg] = J_0 + dN*(np.sin(gthe)*np.cos(gphi)*x
           + np.sin(gthe)*np.sin(gphi)*y
           + np.cos(gthe)*z)/r_out


#    Nden[flg] += J_0

    return

def Sphere(grid,Nden,parms={},
    r_out=120,
    loopCentre=[-8.0,-0.045,0.07],
    J_0=1.0):

    parm_dict = {'r_outer':r_out,
                 'loopCentre':loopCentre,
                 'J_0':J_0}


    parm_keys = parm_dict.keys()

    for loop in parms:
        for key_p,val_p in loop.iteritems():
            if key_p in parm_keys:
                parm_dict[key_p] = val_p
            else:
                print "Unrecognized Loop parmeter %s" % key_p

    sphere_profile(grid,Nden,parm_dict)


def sphere_profile(grid,Nden,parm):

#    print parm

    r_out = parm['r_outer']
    loopCentre = parm['loopCentre']
    J_0 = parm['J_0']
 
    print "r_out=",r_out
    print "Centre=",loopCentre

    coord = grid.get_cartesian()
    lc = np.array(loopCentre)
    lc =lc[:,np.newaxis]

    xyz = coord - np.tile(lc,(1,len(coord[0])))
    r = np.sqrt(np.sum(xyz*xyz,axis=0))
    nb = Nden.shape[0]
  
    flg = np.where(r>r_out,0,1)
#    flg *= np.where(xyz[0]<0,0,1)
#    flg *= np.where(xyz[1]<0,0,1)

    flg = np.compress(flg,np.arange(nb))

    Nden[flg] += J_0

    return


profiles = {'Loop':Loop, 'Sphere':Sphere}


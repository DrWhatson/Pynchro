#from numba import jit
import numpy as np
import healpy as hp


def has_profile(type):
    types = profiles.keys()

    if type in types:
        return True
    else:
        print "Unrecognized profile type %s" % type
        return False


def apply_profile(type,grid,B_rho,B_phi,B_z,parms):
    profiles[type](grid,B_rho,B_phi,B_z,parms)


def Fixed(grid,B_rho,B_phi,B_z,parms={},
    b0=10,      # Galactic elevation
    l0=70.0,    # Galactic longitude
    B_0=2.0):   # B_field strenght

    parm_dict = {'b0':b0,
                 'l0':l0,
                 'B_0':B_0}

    parm_keys = parm_dict.keys()

    for loop in parms:
        for key_p,val_p in loop.iteritems():
            if key_p in parm_keys:
                parm_dict[key_p] = val_p
            else:
                print "Unrecognized Fixed parmeter %s" % key_p

    fixed_profile(grid,B_rho,B_phi,B_z,parm_dict)


def fixed_profile(grid,B_rho,B_phi,B_z,parm):

    B_0 = parm['B_0']
    b0 = parm['b0']
    l0 = parm['l0']

    xyz = grid.get_cartesian()
    r = np.sqrt(np.sum(xyz*xyz,axis=0))
    the = np.arcsin(xyz[2]/r)
    phi = np.arctan2(xyz[1],xyz[0])

    print "Coord shape",xyz.shape
    print "BField shape",B_rho.shape
    print "b0",b0
    print "l0",l0
    print "B_0", B_0

    B_z += B_0 * np.sin(b0*np.pi/180)
    B_phi += B_0 * np.cos(b0*np.pi/180) * np.cos(phi-l0*np.pi/180)
    B_rho += B_0 * np.cos(b0*np.pi/180) * np.sin(phi-l0*np.pi/180)

    return


def ASS(grid, B_rho, B_phi, B_z, parms={},
    B_0 = 2.0,
    B_amp_param = 8.0,
    B_amp_type = 'cyl',
    pitch = 11.5,
    rho_0 = 8.5,
    Xi_0 = 25.0,
    z_0 = 1.0):

    parm_dict = {'B_0':B_0,
                 'B_amp_parm':B_amp_param,
                 'B_amp_type':B_amp_type,
                 'pitch':pitch,
                 'rho_0':rho_0,
                 'Xi_0':Xi_0,
                 'z_0':z_0}

    parm_keys = parm_dict.keys()

    for loop in parms:
        for key_p,val_p in loop.iteritems():
            if key_p in parm_keys:
                parm_dict[key_p] = val_p
            else:
                print "Unrecognized ASS parmeter %s" % key_p

    B_0 = parm_dict['B_0']
    B_amp_parm = parm_dict['B_amp_parm']
    B_amp_type = parm_dict['B_amp_type']
    pitch = parm_dict['pitch']*np.pi/180.
    rho_0 = parm_dict['rho_0']
    Xi_0 = parm_dict['Xi_0']*np.pi/180.
    z_0 = parm_dict['z_0']
    
    xyz = grid.get_cartesian()
    r = np.sqrt(np.sum(xyz*xyz,axis=0))
    the = np.arcsin(xyz[2]/r)
    phi = np.arctan2(xyz[1],xyz[0])

    if B_amp_type == 'cst':
        B_amp = B_0
    elif B_amp_type == 'cyl':
        B_0 = B_0 * (1. + rho_0/B_amp_param)
        B_amp = B_0 * 1./(1. + r/B_amp_param)
    elif B_amp_type == 'sph':
        B_amp = (B_0 * np.exp(-((r**2 + xyz[2]**2)**.5 - rho_0)/B_amp_param))
    else:
        raise ValueError('''
        Bad entry for optional argument 'B_amp_type'.
        Key must be one either: 'cst', 'sph' or 'cyl' ''')

    ### Tilt angle
    Xi_z = Xi_0 * np.tanh(xyz[2] / z_0)


    #cylindrical component of the magnetic vector field
    B_rho += B_amp * np.sin(pitch) * np.cos(Xi_z)
    B_phi += B_amp * np.cos(pitch) * np.cos(Xi_z)
    B_z += B_amp * np.sin(Xi_z)


def RING(grid, B_rho, B_phi, B_z, parms={},
    B_0 = 2.0,
    rho_in = 2.,
    rho_ex = 5.):

    parm_dict = {'B_0':B_0,
                 'rho_in':rho_in,
                 'rho_ex':rho_ex}

    parm_keys = parm_dict.keys()

    for loop in parms:
        for key_p,val_p in loop.iteritems():
            if key_p in parm_keys:
                parm_dict[key_p] = val_p
            else:
                print "Unrecognized ASS parmeter %s" % key_p

    B_0 = parm_dict['B_0']*1e-6 # uG
    rho_in = parm_dict['rho_in']
    rho_ex = parm_dict['rho_ex']

    xyz = grid.get_cartesian()
    r = np.sqrt(np.sum(xyz*xyz,axis=0))
    nb = len(r)

    flg = np.where(r>rho_in,1,0)
    flg *= np.where(r<rho_ex,1,0)
    flg = np.compress(flg,np.arange(nb))

    B_phi[flg] += B_0



def Loop(grid,B_rho,B_phi,B_z,parms={},
    r_in=0.113,
    r_out=0.120,
    B_0=10.0,
    dB = 0.0,
    the = 0.,
    phi = 90.,
    gthe = 0.,
    gphi = 90.,
    loopCentre=[-8.0,-0.0,0.0]):

    parm_dict = {'r_inner':r_in,
                 'r_outer':r_out,
                 'B_0':B_0,
                 'loopCentre':loopCentre,
                 'dB':dB,
                 'gthe':gthe,
                 'gphi':gphi, 
                 'the':the,
                 'phi':phi}


    parm_keys = parm_dict.keys()

    for loop in parms:
        for key_p,val_p in loop.iteritems():
            if key_p in parm_keys:
                parm_dict[key_p] = val_p
            else:
                print "Unrecognized Loop parmeter %s" % key_p

    loop_profile(grid,B_rho,B_phi,B_z,parm_dict)

#@jit
def loop_profile(grid,B_rho,B_phi,B_z,parm):

    print parm

    b_0 = np.array(parm['B_0'])*1e-6

    r_in = parm['r_inner']
    r_out = parm['r_outer']
    loopCentre = parm['loopCentre']

    the0 = np.pi/2-parm['the']*np.pi/180.  # Direction of magnetic field
    phi0 = np.pi/2+parm['phi']*np.pi/180.

    dB = np.array(parm['dB'])*1e-6 # Magnitude of mag field gradient
    gthe = parm['gthe']*np.pi/180. # Direction of gradient in magnetic field
    gphi = parm['gphi']*np.pi/180.

    coord = grid.get_cartesian()
    lc = np.array(loopCentre)
    lc =lc[:,np.newaxis]

    xyz = coord - np.tile(lc,(1,len(coord[0])))
    r = np.sqrt(np.sum(xyz*xyz,axis=0))
    nb = B_z.shape[0]

#    print "r_in",r_in
#    print "r_out",r_out
#    print "the0",the0
#    print "phi0",phi0
   

    # Set inside of loop to zero
    flg = np.where(r<r_in)[0]
    B_rho[flg] = 0.0
    B_phi[flg] = 0.0
    B_z[flg] = 0.0

    flg = np.where(r>r_in,1,0)
    flg *= np.where(r<r_out,1,0)
    flg = np.compress(flg,np.arange(nb))

    x = xyz[0][flg]
    y = xyz[1][flg]
    z = xyz[2][flg]
    r = r[flg]

    # Gradient in magnet field strenght 
    Bmag = b_0 + dB*(np.sin(gthe)*np.cos(gphi)*x
           + np.sin(gthe)*np.sin(gphi)*y
           + np.cos(gthe)*z)/r_out

    print "the0=",the0*180/np.pi," phi0=",phi0*180/np.pi

    # Rotate x,y,z 
    x1 = x*np.cos(phi0) - y*np.sin(phi0)
    y1 = x*np.sin(phi0) + y*np.cos(phi0)

    y2 = y1*np.cos(the0) - z*np.sin(the0)
    z1 = y1*np.sin(the0) + z*np.cos(the0)

    # local the phi

    the = np.arcsin(z1/r)
    phi = np.arctan2(y2,x1)

    # Local Bx, By, Bz
    bx = -Bmag*np.sin(the)*np.sin(phi)
    by = -Bmag*np.sin(the)*np.cos(phi)
    bz =  Bmag*np.cos(the)

    # Rotate back
    bx1  = bx*np.cos(-the0) - bz*np.sin(-the0)
    Bz  = bx*np.sin(-the0) + bz*np.cos(-the0)

    Bx = (bx1*np.cos(phi0) - by*np.sin(phi0))
    By = bx1*np.sin(phi0) + by*np.cos(phi0)


    # Global phi
    phi = -np.arctan2(coord[1][flg],coord[0][flg])

    print "max min phi ",np.max(phi),np.min(phi)

    # Finally rotate Bx,By to Brho and Bphi

    B_rho[flg] += By*np.cos(phi) - Bx*np.sin(phi)
    B_phi[flg] += By*np.sin(phi) + Bx*np.cos(phi)
    B_z[flg] += Bz

    return


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

def __B0_of_r(rho,z,**kwargs):
    """
    Internal function
        Intended to modulate the magnetic field amplitude by a function
        of the radial (cyl or sph) coordinate

    If constant:     B_amp = B_0     for all rho,z
    If cylindrical:  B_amp = B_0 * 1/(1 + rho/B_amp_param)
    If spherical:    B_amp = B_0 * exp(-(r - rho_0)/B_amp_param)
    B_amp = B_0    if constant
            B_0 * 1/(1 + rho/rho_0)    if cylindrical
            B_0 * exp(-(r - rho_0)/B_amp_param) if spherical

    B_amp is automatically normalized such that B_amp(at sun) = B_sun meant
    to be given by the 'B_0' param in kwargs.

    INPUT:
    ------
      - rho : cylindircal radial coordinate
      - z : height coordinates

      **kwargs : containing the information to build the wanted function
         - B_0 : an overall amplitude
         - B_amp_param : an additional parameter for the radial function
         - B_amp_type : string to specify the fonctional form. Should be
                        'cst','cyl' or 'sph'
         - rho_0 : is supposed to contained the dist. btw the Sun and the GC


    OUTPUT:
    ------
      - B_amp : the field amplitude at each location specified by rho,z


    Creation date : Jul 5 2017
    @author: V.Pelgrims
    """

    param = {'B_0':2.1,
             'B_amp_type':'cyl',
             'B_amp_param':8.0,
             'rho_0':8.0
             }

    #set parameter value to entries, if given
    for key,value in kwargs.iteritems():
        param[key] = value
    
    ###
    if param['B_amp_type'] == 'cst':
        B_amp = param['B_0']
    elif param['B_amp_type'] == 'cyl':
        B_0 = param['B_0'] * (1. + param['rho_0']/param['B_amp_param'])
        B_amp = B_0 * 1./(1. + rho/param['B_amp_param'])
    elif param['B_amp_type'] == 'sph':
        B_amp = (param['B_0']
                 * np.exp(-((rho**2 + z**2)**.5 - param['rho_0'])
                          /param['B_amp_param']))
    else:
        raise ValueError('''
        Bad entry for optional argument 'B_amp_type'.
        Key must be one either: 'cst', 'sph' or 'cyl' ''')
    
    return B_amp



profiles = {'Loop':Loop,'Fixed':Fixed, 'ASS':ASS, 'RING':RING}

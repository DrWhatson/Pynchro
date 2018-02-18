import numpy as np
import healpy as hp



def has_profile(type):
    types = profiles.keys()

    if type in types:
        return True
    else:
        print "Unrecognized profile type %s" % type
        return False


def apply_profile(type,grid,cre_density,parms):
    profiles[type](grid,cre_density,parms)



def Loop(grid,cre_density,parms={},
    amp=1.0,
    radial_form='Gaussian',
    radial_scale=0.03,
    shell_radius=0.14,
    loopCentre=[-8.0,-0.045,0.07]):

    parm_dict = {'amplitude':amp,
                 'radial_form':radial_form,
                 'radial_scale':radial_scale,
                 'shell_radius':shell_radius,
                 'loopCentre':loopCentre}

    parm_keys = parm_dict.keys()

    for loop in parms:
        for key_p,val_p in loop.iteritems():
            if key_p in parm_keys:
                parm_dict[key_p] = val_p
            else:
                print "Unrecognized Loop parmeter %s" % key_p

        loop_profile(grid, cre_density,
                     amp,
                     radial_form,
                     radial_scale,
                     shell_radius,
                     loopCentre)


def loop_profile(grid, cre_density,
                  amp,
                  radial_form,
                  radial_scale,
                  shell_radius,
                  loopCentre):

    coord = grid.get_cartesian()
    lc = np.array(loopCentre)
    lc =lc[:,np.newaxis]

    print coord.shape, lc.shape

    xyz = coord - np.tile(lc,(1,len(coord[0])))


    r = np.sum(xyz*xyz,axis=0)**.5
    #shift the vales of r for shell-like density
    r_shifted = r-shell_radius

    #build the density profile using the HaloProfile function
    loop = amp * OneParamFunc(r_shifted,radial_scale,radial_form)
    cre_density += loop

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



profiles = {'Loop':Loop}

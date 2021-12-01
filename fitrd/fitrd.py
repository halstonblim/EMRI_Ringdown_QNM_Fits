"""
This module contains code to calculate QNM amplitude and phases given a set of
spherical waveform modes of a given azimuthal index m.

e.g. given h22, h32, h42 spherical modes, these routines will compute the
m = 2, l = 2,3,4 QNMs and m = -2, l = 2,3,4 mirror QNMS
"""
import numpy as np
import pandas as pd
import qnm
from os.path import exists

def mulmlpnp(m,l,lprime,n,a):
    """
    Returns spherical-spheroidal overlap mu_{m,l,l',n'}(a)
    defined in Eq. (5) of Berti and Klein (2014)
    """
    omega_s, a_val, _ = qnm.modes_cache(s=-2,l=lprime,m=m,n=n)(a=a)
    i = np.where(qnm.angular.ells(-2,m,abs(m)+100) == l)[0][0]
    overlap_s = qnm.angular.C_and_sep_const_closest(a_val,-2,a*omega_s,m,abs(m) + 100)[1][i]
    return overlap_s.real + 1j*overlap_s.imag

def loadqnm(lprime,m,n,a):
    """Read in QNM frequencies using qnm package."""
    omega_s,_,_ = qnm.modes_cache(s=-2,l=lprime,m=m,n=n)(a=a)
    return omega_s

def lrradius(a):
    """Calculate light ring radius for equatorial prograde orbit."""
    return 2*(1+np.cos((2./3.)*np.arccos(-np.abs(a))))
    

def get_lrange(m,mmax,lmin=2):
    """
    Return range of angular indices l_min <= l <= l_max
    where l_min = max(2,|m|) and l_max = |m| + mmax
    """
    return np.arange(np.amax([lmin,np.abs(m)]),mmax + np.abs(m) + 1)

def preparesystem(m,a,mmax,cachedir=None,overwrite=False):
    """
    Calculate matrix elements containing spherical-spheroidal
    overlaps and qnm frequencies needed for Eq. (3.10) in
    Lim, Khanna, Apte, and Hughes (2019)

    Inputs:
    - m: azimuthal index
    - a: spin parameter
    - mmax: number of mixed modes to include beyond l == |m| [lmin,..,...,|m| + mmax]
    - cachedir: directory where computations are saved or loaded

    Output:
    - alphasystem[lp][l], matrix of coefficients
    - l: spherical index
    - lp: spheroidal index
    """

    if cachedir is not None and overwrite == False and exists(f'{cachedir}/prepare_system_{m}_{a:.4f}_{mmax}.npy'):
        alphasystem = np.load(f'{cachedir}/prepare_system_{m}_{a:.4f}_{mmax}.npy')        
    else:

        #  Setup the system matrix to solve
        larray  = [[l,is_derivative]
                   for is_derivative in [False,True]
                   for l in get_lrange(m,mmax)]

        lparray = [[l,mprime]
                   for l in get_lrange(m,mmax)
                   for mprime in [m,-m]]

        # For m = 0, take into account degenerate mirror modes
        # Label the pair modes as (|l|,0) and (-|l|,0)
        if m == 0:
            lparray = [[lprime*(-1)**i,mprime] for i,(lprime,mprime) in enumerate(lparray)]

        alphasystem = np.zeros((len(larray),len(larray),2),dtype=complex)

        for lindex, (l, is_derivative) in enumerate(larray):
            for lpindex, (lprime,mprime) in enumerate(lparray):

                overlap = mulmlpnp(mprime,l,np.abs(lprime),0,a)
                omega_j = -1j * loadqnm(np.abs(lprime),mprime,0,a)

                # Check if mirror mode
                # m == 0, use sign(lp) to denote mirror modes
                # m != 0, use sign(mp*m) to denote mirror modes
                if (m != 0 and mprime*m < 0) or (m == 0 and lprime < 0):
                    overlap = np.conjugate(overlap) * (-1)**l 
                    omega_j = np.conjugate(omega_j)
                # If derivative, multiply mu by 1j*omega
                if is_derivative is True:
                    overlap *= omega_j
                alphasystem[lindex][lpindex] = [overlap,omega_j]
        if cachedir is not None:
            np.save(f"{cachedir}/prepare_system_{m}_{a:.4f}_{mmax}.npy",alphasystem)
    return alphasystem

def get_linearsystem(alphasystem,t,t_0):
    """
    Calculate LHS of matrix equation.
    Consists of spherical-spheroidal overlaps and QNMs,
    according to Eq. (3.10) in Lim,Khanna,Apte,Hughes (2019)
    """
    musystem = alphasystem[:,:,0]
    omegasystem = alphasystem[:,:,1]
    return musystem * np.exp(omegasystem * (t - t_0))

def load_waveforms(wavefiles, spherical_modes, mmax, tstart=0):
    """
    Load and cut waveform data

    Assumes set of wavefiles have naming conventions:
    - hm{m}_*.dat for m >= 0
    - hmm{m}_*.dat for m < 0

    Inputs:
    - wavefiles: list of filepaths for each wavefile of index m
    - spherical_modes: list of modes corresponding to wavefiles [[2,2],[3,2],...]
    - mmax: number of mixed modes to model beyond l == |m|

    Ouputs:
    - wavedatas (N)
        contains N rows for each mode in spherical_modes
        each row has wavedatas[i] = hlm_+ - 1j * hlm_x
    """
    # Check consistency of inputs
    for (l,m) in spherical_modes:
        assert l in get_lrange(m,mmax), f"Mode ({l},{m}) out of range"

    # Find which modes are contained in each wavefile
    filemodelist = []
    for wavefile in wavefiles:
        wavefile = wavefile.split("/")[-1]

        # sign(m), check by examining fourth character in filename
        # m > 0, e.g. hm1*.dat
        if wavefile[3] == "_":
            m = int(wavefile[2])
            filemodelist.append([[l,m] for l in get_lrange(m,mmax,lmin=m)])

        # m < 0, e.g. hmm1*.dat
        elif wavefile[3] != "_":
            m = -int(wavefile[3])
            filemodelist.append([[l,m] for l in get_lrange(m,mmax,lmin=-m)])

    # Read in waveforms in order of modelist
    wavedatas = []
    for l,m in spherical_modes:
        # Find file containing desired mode
        for j,wavefile in enumerate(wavefiles):
            if [l,m] in filemodelist[j]:
                data = pd.read_csv(wavefile,delim_whitespace=True,header=None,
                                   usecols=[0,1+2*(l-np.abs(m)),2+2*(l-np.abs(m))]).to_numpy()
                time,h_plus,h_cross = data[data[:,0] >= tstart].T
                wavedatas.append(h_plus - 1j * h_cross)
    return time, np.array(wavedatas)

def get_spheroidalmodes(spherical_modes):
    """
    Return list of pairs of modeled spheroidal modes given
    list of inputted spherical modes
    For N spherical modes, will model N spheroidal mode (QNM) pairs
    """
    spheroidal_modes = []
    for l,m in spherical_modes:
        spheroidal_modes.append([l,m,0])
        spheroidal_modes.append([l,-m,1])
    return spheroidal_modes

def get_sphericalcoefs(time,wavedatas):
    """
    Calculate RHS of matrix equation.
    Consists of spherical modes and their derivatives,
    according to Eq. (3.10) in Lim,Khanna,Apte,Hughes (2019)
    """
    delta_t = time[1] - time[0]
    wavedatas_deriv = np.gradient(wavedatas,delta_t,axis=1)
    return np.vstack((wavedatas,wavedatas_deriv)).T

def solve_system(m,mmax,tlrcross,wavefilepaths,alphasystem):
    """
    Solve for QNM amplitudes at each point in time
    Eq. (3.10) in Lim, Khanna, Apte, Hughes (2019)

    Inputs:
    - m: spherical index describing input waveform files
    - a: spin parameter
    - thinc: spin-orbit misalignment parameter describing input trajectory
    - thf: plunge parameter describing input trajectory
    - mmax: number of mixed QNMs to model, lmax = mmax + max(2,|m|)
    - tlrcross: time of lightring crossing

    Outputs:
    - spherical modes: list of spherical modes used to find QNMs
    - time: array of times at which spheroidal modes are calculated
    - spheroidal coefs: solved spheroidal modes at each time
    """
    spherical_modes = [[l,m] for l in get_lrange(m,mmax)]

    time, wavedatas = load_waveforms(wavefilepaths,spherical_modes,
                                     mmax,tstart=tlrcross-50)
    spherical_coefs = get_sphericalcoefs(time,wavedatas)
    spheroidal_coefs = np.zeros(spherical_coefs.shape,dtype=complex)
    for t,time_i in enumerate(time):
        alplm = get_linearsystem(alphasystem,time_i,tlrcross)
        spheroidal_coefs[t] = np.linalg.solve(alplm,spherical_coefs[t])

    return spherical_modes, time, spheroidal_coefs

def mlabel(mval):
    """
    Outputs m-index as string
    if m >= 0, maps to f"m{m}"
    if m < 0, maps to f"mm{m}"
    """
    if mval < 0:
        return f"mm{np.abs(mval)}"
    return f"m{mval}"

def postprocess(time,spheroidal_coefs,tstart,timewindow,tend):
    """
    Take average of spheroidal coefs over many fitting times
    to extract QNM. Procedure described in Lim,Khanna,Apte,Hughes (2019)
    """
    c_amps = np.abs(spheroidal_coefs)
    c_phases = np.unwrap(np.angle(spheroidal_coefs))

    stdmin = 1e99

    tmax = np.min([time[-1] - timewindow,tend])
    i = np.argmin(np.abs(time - tstart))

    while time[i] + timewindow <= tmax:
        mask = (time >= time[i]) & (time <= time[i] + timewindow)
        c_amps_mean = np.mean(c_amps[mask,:],axis=0)
        stdtotal = np.sum(np.std(c_amps[mask,:],axis=0) / c_amps_mean)
        if stdtotal < stdmin:
            stdmin = stdtotal
            c_amps_mean_best = c_amps_mean
            c_phases_mean_best = np.mod(np.mean(c_phases[mask,:],axis=0),2*np.pi)
        i += 1
    return c_amps_mean_best, c_phases_mean_best

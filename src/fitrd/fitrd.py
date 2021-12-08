"""
This module contains code to calculate QNM amplitude and phases given a set of
spherical waveform modes of a given azimuthal index m.

e.g. given h22, h32, h42 spherical modes, these routines will compute the
m = 2, l = 2,3,4 QNMs and m = -2, l = 2,3,4 mirror QNMS
"""
import os
import numpy as np
import pandas as pd
import qnm

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

def get_lrange(m,k_ell,lmin=2):
    """
    Return range of angular indices l_min <= l <= l_max
    where l_min = max(2,|m|) and l_max = |m| + k_ell
    """
    return np.arange(np.amax([lmin,np.abs(m)]),k_ell + np.abs(m) + 1)

def calculate_matrix_components(m, a, larray, lparray):
    """
    Helper function to calculate matrix elements containing spherical-spheroidal
    overlaps and qnm frequencies needed for Eq. (3.10) in
    Lim, Khanna, Apte, and Hughes (2019)

    Inputs:
    - larray: list [[l,is_derivative],...] describing each
              spherical mode and whether derivative was taken
    - lparray: list of spheroidal modes
    """
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
    return alphasystem

def preparesystem(m,a,k_ell,cachedir=None,overwrite=False):
    """
    Calculate matrix elements containing spherical-spheroidal
    overlaps and qnm frequencies needed for Eq. (3.10) in
    Lim, Khanna, Apte, and Hughes (2019)

    Inputs:
    - m: azimuthal index
    - a: spin parameter
    - k_ell: number of mixed modes to include beyond l == |m| [lmin,..,...,|m| + k_ell]
    - cachedir: directory where computations are saved or loaded

    Output:
    - alphasystem[lp][l], matrix of coefficients
    """
    if (cachedir is not None and overwrite is False and
        os.path.exists(f'{cachedir}/prepare_system_{m}_{a:.4f}_{k_ell}.npy')):

        alphasystem = np.load(f'{cachedir}/prepare_system_{m}_{a:.4f}_{k_ell}.npy')

    else:

        #  Setup the system matrix to solve
        larray  = [[l,is_derivative]
                   for is_derivative in [False,True]
                   for l in get_lrange(m,k_ell)]

        lparray = [[l,mprime]
                   for l in get_lrange(m,k_ell)
                   for mprime in [m,-m]]

        # For m = 0, take into account degenerate mirror modes
        # Label the pair modes as (|l|,0) and (-|l|,0)
        if m == 0:
            lparray = [[lprime*(-1)**lpindex,mprime] for
                       lpindex,(lprime,mprime) in enumerate(lparray)]

        alphasystem = calculate_matrix_components(m, a, larray, lparray)

        if cachedir is not None:
            if os.path.exists(cachedir) == False:
                os.makedirs(cachedir)
            np.save(f"{cachedir}/prepare_system_{m}_{a:.4f}_{k_ell}.npy",alphasystem)
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

def load_waveforms(wavefiles, spherical_modes, k_ell, t_cut=0):
    """
    Load and cut waveform data

    Assumes set of wavefiles have naming conventions:
    - hm{m}_*.dat for m >= 0
    - hmm{m}_*.dat for m < 0

    Inputs:
    - wavefiles: list of filepaths for each wavefile of index m
    - spherical_modes: list of modes corresponding to wavefiles [[2,2],[3,2],...]
    - k_ell: number of mixed modes to model beyond l == |m|

    Ouputs:
    - wavedatas (N)
        contains N rows for each mode in spherical_modes
        each row has wavedatas[i] = hlm_+ - 1j * hlm_x
    """
    # Check consistency of inputs
    for (l,m) in spherical_modes:
        assert l in get_lrange(m,k_ell), f"Mode ({l},{m}) out of range"

    # Find which modes are contained in each wavefile
    filemodelist = []
    for wavefile in wavefiles:
        wavefile = wavefile.split("/")[-1]

        # sign(m), check by examining fourth character in filename
        # m > 0, e.g. hm1*.dat
        if wavefile[3] == "_":
            m = int(wavefile[2])
            filemodelist.append([[l,m] for l in get_lrange(m,k_ell,lmin=m)])

        # m < 0, e.g. hmm1*.dat
        elif wavefile[3] != "_":
            m = -int(wavefile[3])
            filemodelist.append([[l,m] for l in get_lrange(m,k_ell,lmin=-m)])

    # Read in waveforms in order of modelist
    wavedatas = []
    for l,m in spherical_modes:
        # Find file containing desired mode
        for j,wavefile in enumerate(wavefiles):
            if [l,m] in filemodelist[j]:
                data = pd.read_csv(wavefile,delim_whitespace=True,header=None,engine='python',
                                   usecols=[0,1+2*(l-np.abs(m)),2+2*(l-np.abs(m))]).to_numpy()
                time,h_plus,h_cross = data[data[:,0] >= t_cut].T
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

def solve_system(m,k_ell,t_fiducial,wavefilepaths,alphasystem):
    """
    Solve for QNM amplitudes at each point in time
    Eq. (3.10) in Lim, Khanna, Apte, Hughes (2019)

    Inputs:
    - m: spherical index describing input waveform files
    - a: spin parameter
    - thinc: spin-orbit misalignment parameter describing input trajectory
    - thf: plunge parameter describing input trajectory
    - k_ell: number of mixed QNMs to model, lmax = k_ell + max(2,|m|)
    - t_fiducial: fiducial time

    Outputs:
    - spherical modes: list of spherical modes used to find QNMs
    - time: array of times at which spheroidal modes are calculated
    - spheroidal coefs: solved spheroidal modes at each time
    """
    spherical_modes = [[l,m] for l in get_lrange(m,k_ell)]

    time, wavedatas = load_waveforms(wavefilepaths,spherical_modes,
                                     k_ell,t_cut=t_fiducial-50)
    spherical_coefs = get_sphericalcoefs(time,wavedatas)
    spheroidal_coefs = np.zeros(spherical_coefs.shape,dtype=complex)
    for t,time_i in enumerate(time):
        alplm = get_linearsystem(alphasystem,time_i,t_fiducial)
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

def postprocess(time,spheroidal_coefs,t_start,t_end,t_window):
    """
    Take average of spheroidal coefs over many fitting times
    to extract QNM. Procedure described in Lim,Khanna,Apte,Hughes (2019)
    """
    assert t_end - t_start >= t_window, "Need at least one averaging period"
    c_amps = np.abs(spheroidal_coefs)
    c_phases = np.unwrap(np.angle(spheroidal_coefs))

    stdmin = 1e99

    t_max = np.min([time[-1] - t_window,t_end])
    i = np.argmin(np.abs(time - t_start))

    c_amps_mean_best = None

    while time[i] + t_window <= t_max:
        mask = (time >= time[i]) & (time <= time[i] + t_window)
        c_amps_mean = np.mean(c_amps[mask,:],axis=0)
        stdtotal = np.sum(np.std(c_amps[mask,:],axis=0) / c_amps_mean)
        if stdtotal < stdmin:
            stdmin = stdtotal
            c_amps_mean_best = c_amps_mean
            c_phases_mean_best = np.mod(np.mean(c_phases[mask,:],axis=0),2*np.pi)
        i += 1
    assert c_amps_mean_best is not None, "Could not find QNM amplitude, terminating"
    return c_amps_mean_best, c_phases_mean_best

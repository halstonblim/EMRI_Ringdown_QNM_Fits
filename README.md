fitrd
=====

This module implements the fitting algorithm described in Lim, Khanna,
Apte, and Hughes (2019) to extract quasi-normal mode amplitudes from
ringdown waveforms \[1\]. We provide some sample data and notebooks to
demonstrate how to use this code

Installation
------------

Dependencies include [numpy](https://numpy.org/), [pandas](https://pandas.pydata.org/), and [qnm](https://github.com/duetosymmetry/qnm)

To install, run `pip install fitrd` in the package directory. To test installation, run `python tests/test_fitrd.py`

During testing, we have found several potential installation issues

-   qnm requires numba, but numba requies numpy\<1.21,\>=1.17. To fix
    this, install a compatible version of numpy. We suggest creating a
    different python environment with a compatible numpy version
-   pip fails to install numba (numba is required by qnm). numba has a lot of dependencies that can lead to issues when not using conda (e.g. [llvmlite](https://llvmlite.readthedocs.io/en/latest/admin-guide/install.html#using-pip)) We suggest following instructions on the [qnm github](https://github.com/duetosymmetry/qnm) and installing the qnm package either with conda: `conda install -c conda-forge qnm` or installing from the source by cloning the repository. After qnm is installed, `pip install fitrd` should work
-   python 3.10 is not supported by several of the dependencies. If you have python 3.10 installed, you will have to downgrade to (or create new environment with) python >=3.7, <3.10 

## Formatting Waveform Files

The current version of the code expects the waveform data to be formatted in
a particular way. The outgoing radiation must be decomposed into a set
of -2-spin-weighted spherical harmonic modes,
`h(l,m) = h+(l,m) - 1j * hx(l,m)`. Each waveform file must contain all
spherical modes with index `m`. The first column should contain the
time, the subsequent columns should contain both the + and x components
of each spherical mode, starting with `l - abs(m) = 0` mode.

For example, a valid waveform file for `m = 0` may have 7 columns:
`t, h+(0,0), hx(0,0),  h+(1,0), hx(1,0), h+(2,0), hx(2,0)`. It may also
have 11 columns:
`t, h+(0,0), hx(0,0),  h+(1,0), hx(1,0), h+(2,0), hx(2,0), h+(3,0), hx(3,0), h+(4,0), hx(4,0)`.
Even though modes with `l < 2` are zero for gravitational radiation,
they should still be included as columns in the waveform file.

The name of the waveform file should indicate which mode it contains.
The naming convention is

-   `f"hm{m}*.dat"` for `m >= 0`
-   `f"hmm{abs(m)}*.dat"` for `m < 0`

User Inputs
-----------

There are several user-inputted parameters which we describe here.

-   `m` - The azimuthal index desribing the spherical modes and the QNM
    mode pairs to be extracted.
-   `a` - The spin parameter describing the spacetime.
-   `t_fiducial` - Fiducial time of the ringdown model. For
    quasi-circular EMRI plunges, we advocate for using the retarded time
    at which the small body crosses the equivalent equatorial light
    ring. However, this can be set to any value, such as the time when
    `h(2,2)` is maximized.
-   `k_ell` - The number of spheroidal modes pairs to include beyond
    `max(2,abs(m))`, as defined in Eq. (3.10) in \[1\].
-   `t_start,t_end,t_window` - Used in the `fitrd.fitrd.postprocess`
    function, which describe the range of times over which the
    spheroidal modes should be calculated, and how large the sliding
    window should be when averaging the spheroidal modes

References
----------

1.  [H. Lim, G. Khanna, A. Apte, and S. A. Hughes, Phys. Rev. D 100,
    084032 (2019).](https://doi.org/10.1103/PhysRevD.100.084032)
2.  [S. A. Hughes, A. Apte, G. Khanna, and H. Lim, Phys. Rev. Lett. 123,
    161101 (2019).](https://doi.org/10.1103/PhysRevLett.123.161101)
3.  [A. Apte and S. A. Hughes, Phys. Rev. D 100, 084031
    (2019).](https://doi.org/10.1103/PhysRevD.100.084031)

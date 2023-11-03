import numpy as np
import scipy.linalg as linalg
import scipy.optimize as opt
import cmath
from dptb.utils.constants import Boltzmann, eV2J


def pole_maker(Emin, ChemPot, kT, reltol):
    """This is an alternate pole summation method implemented by Areshkin-Nikolic [CITE].
    Similar to the continued-fraction representation of Ozaki, this method allows for
    efficient computation of the density matrix by complex pole summation. Both methods
    approximate the Fermi-Dirac distribution function so that it is more easily manipulable.
    One of the exciting uses of this method is for finite differences, and the computation
    of quasi-equilibrium solutions for use in efficient non-equilibrium density matrices
    as the user can reuse certain poles but with different weights for little extra cost.
    What A-N does is replaces f(E,mu) with one that approximates it on the real line:
    f(E,mu,kT) -> f(E,1j*mu_im,1j*kT_im)*( f(E,mu,kT) - f(Emin, mu_re,kT_re) ) but has
    drastic changes in the upper complex plane. By judicious choice of mu_im to be at
    least p*kT away from the real line and mu_re to be at least p*kT away from the
    minimum eigenvalue Emin, this gives a controlled relative error of e^-p.
    This function automatically switches from first order to second order when
    appropriate. This is when the interval divided by the temperature exceeds 10^3.
    To-do: add third order method.
     Parameters
    ----------
    Emin    : scalar (dtype=numpy.float)
           Minimum occupied state, e.g. a band edge.
    ChemPot : scalar (dtype=numpy.float)
           The chemical potential
    kT      : scalar (dtype=numpy.float)
           The temperature (in units of energy)
    reltol : scalar (dtype=numpy.float)
           The desired relative tolerance. p = -np.log(reltol)
    """
    p = -np.log(reltol)  # Compute the exponent for the relative tolerance desired.

    # When energy exceeds 10^3, switch to second order poles.
    z = (ChemPot - Emin) / kT

    if z < 10 ** 3:
        poles, residues = pole_order_one(Emin, ChemPot, kT, p)
    else:
        poles, residues = pole_order_two(Emin, ChemPot, kT, p)

    return poles, residues


def pole_order_one(Emin, ChemPot, kT, p):
    kTRe, kTIm, muRe, muIm = pole_minimizer_one(Emin, ChemPot, kT, p)

    # Three pole branches
    #   :  :
    # - - - - C
    #   :  :
    #   :  :
    #   A  B

    dummyindex = np.arange(-10000, 10000 + 1)  # Integers from -500 to 500 inclusive,.
    # Need to make a more reasonable guess for valid integers, real line is easy, imag is hard

    # Residues for Fermi-Dirac are simply -kT, residue theorem multiplies
    # by 2*pi*i, then the windowing function multiplies in its own factor:

    # Branch A poles:
    poleA = muRe + 1j * np.pi * kTRe * (2 * dummyindex + 1)  # Analytical pole location from definition of fermi-fun
    # Compute their residues
    resA = (2j * np.pi * kTRe) * fermi_fun(poleA, 1j * muIm, 1j * kTIm)
    # Determine which to trim, they must have magnitude over tol and be in upper complex plane
    zA = (np.abs(resA) >= np.exp(-p) * kT) & (np.imag(poleA) > 0)
    poleA = poleA[zA]
    resA = resA[zA]

    # Branch B poles:
    poleB = ChemPot + 1j * np.pi * kT * (2 * dummyindex + 1)  # Analytical pole location from definition of fermi-fun
    # Compute their residues
    resB = -(2j * np.pi * kT) * fermi_fun(poleB, 1j * muIm, 1j * kTIm)
    # Determine which to trim, they must have magnitude over tol and be in upper complex plane
    zB = (np.abs(resB) >= np.exp(-p) * kT) & (np.imag(poleB) > 0)
    poleB = poleB[zB]
    resB = resB[zB]

    # Branch C poles:
    poleC = 1j * muIm - np.pi * kTIm * (2 * dummyindex + 1)  # Analytical pole location from definition of fermi-fun
    # Residues
    resC = (2 * np.pi * kTIm) * (fermi_fun(poleC, ChemPot, kT) - fermi_fun(poleC, muRe, kTRe))

    zC = (np.abs(resC) >= np.exp(-p) * kT) & (np.imag(poleC) > 0)
    poleC = poleC[zC]
    resC = resC[zC]

    poles = np.concatenate((poleA, poleB, poleC))
    residues = np.concatenate((resA, resB, resC))

    return poles, residues


def pole_minimizer_one(Emin, ChemPot, kT, p):
    """
    This function is minimizing the problem:
    N = N_AB + N_BC + N_CD, where:
    N_AB = (muIm + p*kTIm)/(2*pi*kT)
    N_BC = ( (mu + p*kT) - (muRe - p*kTre) )/(2*pi*kTIm)
    N_CD = (muIm + p*kTIm)/(2*pi*kTRe)
    though this has 4 free parameters (excluding p), we
    set certain parameters so that we do not violate
    our relative tolerance, these are:
    muRe = Emin - p*kTre
    muIm = p*kTim
    which then reduces this to a two variable problem:
    N = 2*p*kTIm*(1/kT + 1/kTRe) +
        ( (ChemPot - Emin) + p*(kT + 2*kTRe) )/( 2*kTIm )
    The function then optimizes the position of the
    pole branches not to overlap then outputs the temp-
    eratures and chemical potentials
    """

    # There's a lot going on here, first we desig-
    # nate the cost function for the whole problem
    def pole_cost_one_A(z):
        kTReA = z[0]  # Added the A to stop it from shadowing
        kTImA = z[1]  # the variable later on in the function.
        return 2 * p * kTImA * (1 / kT + 1 / kTReA) + (ChemPot - Emin + p * (kT + 2 * kTReA)) / (2 * kTImA)

    z0 = np.array([kT, kT])
    zAout = opt.minimize(pole_cost_one_A, z0, bounds=((kT, None), (kT, None)), tol=1e-8)

    # Now we modify it so that the imaginary chemical potential is
    # directly in-between the poles generated from the real line
    l_opt = np.ceil(p * zAout.x[1] / (2 * np.pi * kT))
    muIm = np.pi * kT * (2 * l_opt)
    kTIm = muIm / p  # Because muIm = p kTIm, kTIm = muIm/p

    # Now that we have fixed the imaginary part, we re-optimize
    # kTRe, because some marginal improvement may be made after
    # applying the previous constraints

    def pole_cost_one_B(z):
        kTReB = z[0]  # Added the zeros to stop it from shadowing
        return 2 * p * kTIm * (1 / kT + 1 / kTReB) + (ChemPot - Emin + p * (kT + 2 * kTReB)) / (2 * kTIm)

    z1 = np.array(zAout.x[0])
    bndsB = [(kT, None)]
    zBout = opt.minimize(pole_cost_one_B, z1, bounds=bndsB, tol=1e-8)

    # After it is optimized we once again correct the ans-
    # wer to be between the poles of the nearest branch,
    # thereby preventing a second order pole situation.

    m_opt = np.ceil(np.abs(ChemPot - Emin + p * zBout.x[0]) / (2 * np.pi * kTIm))
    muRe = ChemPot - np.pi * kTIm * (2 * m_opt)
    kTRe = (Emin - muRe) / p

    # The extra poles introduced from this
    # positional fixing process is negligible
    # and is typically no greater than a small
    # handful of poles.

    return kTRe, kTIm, muRe, muIm


def pole_order_two(Emin, ChemPot, kT, p):
    kTRe1, kTIm1, muRe1, muIm1, kTRe2, kTIm2, muRe2, muIm2 = pole_minimizer_two(Emin, ChemPot, kT, p)

    poles = 1
    residues = 1

    return poles, residues


def pole_minimizer_two(Emin, ChemPot, kT, p):
    """
    This function is minimizing the problem:
    N = N_AB + N_BC + N_CD + N_DE + N_EF, where:
    N_AB =
    N_BC =
    N_CD =
    N_DE =
    N_EF =
    though this has x free parameters (excluding p), we
    set certain parameters so that we do not violate
    our relative tolerance, these are:
    muRe1 = x
    muIm1 = y
    muRe2 = x
    muIm2 = y
    which then reduces this to a x variable problem:
    N =
    """

    # Run code, get output, real answer will be different than these initial filler values.
    kTIm1 = kT
    kTIm2 = kT
    kTRe1 = kT
    kTRe2 = kT
    muRe2 = Emin - p * kTRe2
    muRe1 = 0.5 * (muRe2 + ChemPot)  # Initial guess is halfway between
    muIm1 = p * kTIm1
    muIm2 = p * kTIm2

    return kTRe1, kTIm1, muRe1, muIm1, kTRe2, kTIm2, muRe2, muIm2


def pole_finite_difference(muL, muR, kT, reltol):
    """
    For computing the finite difference derivative using pole summation, see Vaitkus thesis.
    By using 2x one-sided differences, we can get the forwards, backwards, and centred
    first derivatives and the centred second derivative. The poles will be written as a vector
    containing all points, and the residues will be two vectors containing the poles for the
    backwards and forward differences respectively as (f_mid-f_min) and (f_max-f_mid).
    This choice is so that the user need compute the poles only once and
    can carry out whichever residues you like using the following recipes:
    h = np.abs(muR - muC)
    Forward  1st = ( (f_max-f_mid)                 ) / (   h)
    Centred  1st = ( (f_max-f_mid) + (f_mid-f_min) ) / ( 2*h)
    Backward 1st = ( (f_mid-f_min)                 ) / (   h)
    Centred  2nd = ( (f_max-f_mid) - (f_mid-f_min) ) / (h**2)
    Parameters
    ----------
    muL    : scalar (dtype=numpy.float)
           Left chemical potential
    muR    : scalar (dtype=numpy.float)
           Right chemical potential
    kT     : scalar (dtype=numpy.float)
           The temperature (in units of energy)
    reltol : scalar (dtype=numpy.float)
           The desired relative tolerance. p = -np.log(reltol)
    """

    p = -np.log(reltol)  # Value of p that gives desired relative tolerance.

    muMin = np.min([muL, muR])
    muMax = np.max([muL, muR])
    muMid = 0.5 * (muMin + muMax)

    kTIm = np.sqrt((muMax - muMin + 2 * p * kT) / (6 * p / kT))  # Analytical solution for minimum pole number
    # Correcting temperature so that p*kTIm will be directly between the poles from the other funcs.
    kTIm = (2 * np.pi * kT / p) * np.ceil((p * kTIm) / (2 * np.pi * kT))
    muIm = p * kTIm  # Where muIm should be to get e^-p error

    # Four pole branches
    #   :  :  :
    # - - - - - - D
    #   :  :  :
    #   :  :  :
    #   A  B  C

    dummyindex = np.arange(-10000, 10000 + 1)  # Integers from -500 to 500 inclusive,.
    # Need to make a more reasonable guess for valid integers, real line is easy, imag is hard

    # Residues for Fermi-Dirac are simply -kT, residue theorem multiplies
    # by 2*pi*i, then the windowing function multiplies in its own factor:

    # Branch A poles:
    poleA = muMin + 1j * np.pi * kT * (2 * dummyindex + 1)  # Analytical pole location from definition of fermi-fun
    # Residues for left
    resA_L = (2j * np.pi * kT) * fermi_fun(poleA, 1j * muIm, 1j * kTIm)
    # Residues for right
    resA_R = 0 * poleA
    # Determine which to trim, they must have magnitude over tol and be in upper complex plane
    zA = (np.abs(resA_L) >= np.exp(-p) * kT) & (np.imag(poleA) > 0)
    poleA = poleA[zA]
    resA_L = resA_L[zA]
    resA_R = resA_R[zA]

    # Branch B poles:
    poleB = muMid + 1j * np.pi * kT * (2 * dummyindex + 1)  # Analytical pole location from definition of fermi-fun
    # Residues for left
    resB_L = -(2j * np.pi * kT) * fermi_fun(poleB, 1j * muIm, 1j * kTIm)
    # Residues for right are exactly the negatives of the left by design
    resB_R = -resB_L
    # Determine which to trim
    zB = (np.abs(resB_L) >= np.exp(-p) * kT) & (np.imag(poleB) > 0)
    poleB = poleB[zB]
    resB_L = resB_L[zB]
    resB_R = resB_R[zB]

    # Branch C poles:
    poleC = muMax + 1j * np.pi * kT * (2 * dummyindex + 1)  # Analytical pole location from definition of fermi-fun
    # Residues for left
    resC_L = 0 * poleC
    # Residues for right
    resC_R = -(2j * np.pi * kT) * fermi_fun(poleC, 1j * muIm, 1j * kTIm)
    # Determine which to trim
    zC = (np.abs(resC_R) >= np.exp(-p) * kT) & (np.imag(poleC) > 0)
    poleC = poleC[zC]
    resC_L = resC_L[zC]
    resC_R = resC_R[zC]

    # Branch D poles:
    poleD = 1j * muIm - np.pi * kTIm * (2 * dummyindex + 1)  # Analytical pole location from definition of fermi-fun
    # Residues for left
    resD_L = (2 * np.pi * kTIm) * (fermi_fun(poleD, muMid, kT) - fermi_fun(poleD, muMin, kT))
    # Residues for right
    resD_R = (2 * np.pi * kTIm) * (fermi_fun(poleD, muMax, kT) - fermi_fun(poleD, muMid, kT))
    # Determine which to trim

    zD = np.maximum(np.abs(resD_L), np.abs(resD_R)) > np.exp(-p) * kT
    poleD = poleD[zD]
    resD_L = resD_L[zD]
    resD_R = resD_R[zD]

    poles = np.concatenate((poleA, poleB, poleC, poleD))
    residuesL = np.concatenate((resA_L, resB_L, resC_L, resD_L))
    residuesR = np.concatenate((resA_R, resB_R, resC_R, resD_R))

    return poles, residuesL, residuesR


def fermi_fun(E, mu, kT):
    """
    Computes the Fermi-Dirac distribution function. Auxiliary function just to tidy the workspace.
    Parameters
    ----------
    E    : scalar (dtype=numpy.float)
         Energy being evaluated
    mu   : scalar (dtype=numpy.float)
         Chemical potential
    kT   : scalar (dtype=numpy.float)
         Temperature (in units of energy)
    """
    # Usually the good way
    #    x = (E - mu) / (2 * kT)
    #    return 0.5 * (1 - np.tanh(x))
    # Usual way that most do
    #    x = (E - mu)/kT
    #    return 1/(np.exp(x) + 1)
    # However both cmath/numpy throw errors for either
    # so we fix it using the following:

    x = (E - mu) / kT
    # the function is periodic modulo 2pi, so we subtract
    # off a large integer imaginary part because the math
    # packages numpy/cmath do not like large imaginary parts.
    # Right now it's commented out because of debugging.
    # x = x - 2j * np.pi * np.round(0.5 * np.imag(x) / np.pi)
    out = []

    # stack exchange advice, to avoid overflow issues, make
    # the real part of numbers always smaller than 1 by using
    # alternative definitions for positive and negative Re(x)

    if x.size < 2:
        if np.real(x) <= 0:
            out = 1 / (cmath.exp(x) + 1)
        else:
            out = cmath.exp(-x) / (1 + cmath.exp(-x))
    else:
        for xi in x:
            if np.real(xi) <= 0:
                out.append(1 / (cmath.exp(xi) + 1))
            else:
                out.append(cmath.exp(-xi) / (1 + cmath.exp(-xi)))

    return np.array(out)


def fermi_deriv(E, mu, kT):
    """
    Computes the Fermi-Dirac distribution function derivative. Auxiliary function just to tidy the workspace.
    Parameters
    ----------
    E    : scalar (dtype=numpy.float)
         Energy being evaluated
    mu   : scalar (dtype=numpy.float)
         Chemical potential
    kT   : scalar (dtype=numpy.float)
         Temperature (in units of energy)
    """
    # Tanh should be more numerically stable than 1/(exp(x) + 1), however
    # numpy has issues with large complex values in both exp and tanh
    # x = (E - mu)/kT
    # return 1/(np.exp(x) + 1)

    x = (E - mu) / (2 * kT)
    return (np.cosh(x) ** -2) / (4 * kT)


def fermi_deriv2(E, mu, kT):
    """
    Computes the Fermi-Dirac distribution function second derivative. Auxiliary function just to tidy the workspace.
    Parameters
    ----------
    E    : scalar (dtype=numpy.float)
         Energy being evaluated
    mu   : scalar (dtype=numpy.float)
         Chemical potential
    kT   : scalar (dtype=numpy.float)
         Temperature (in units of energy)
    """
    # Tanh should be more numerically stable than 1/(exp(x) + 1), however
    # numpy has issues with large complex values in both exp and tanh
    # x = (E - mu)/kT
    # return 1/(np.exp(x) + 1)

    x = (E - mu) / (2 * kT)
    return np.tanh(x) * (np.cosh(x) ** -2) / (4 * kT ** 2)

# if __name__ == '__main__':
#     pole, residue = pole_maker(Emin=-27, ChemPot=-4, kT=Boltzmann*T/eV2J, reltol=1e-30)
#     print(residue.shape)
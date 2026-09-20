from filters import filters
import numpy as np
import math
from collections import defaultdict

def calculate_photon_count(A, dt, T_atm, QE, dlambda, F0, m):
    """Calculates the number of photons for the given observing setup."""
    N_photons = A * dt * T_atm * QE * dlambda * F0 * 10**(-0.4 * m)
    return N_photons






def calculate_SNR_poisson(N_photons):
    """Calculate the SNR assuming Poisson statistics."""
    SNR =  math.sqrt(N_photons)
    return SNR




def calculate_Tatm(lam):
    """Calculate the atmospheric transmission at a wavelength of lambda."""
    lambdas, coeffs = [], []
    with open("data/atmospheric_transmission.txt", "r", encoding='utf-8') as file:
        for line in file:
            if line[0] != "#":
                cur = line.split()
                lambdas.append(int(float(cur[0])))
                coeffs.append(float(cur[1]))


    #Tatm = (lam/5 - lam//5) * float(lambda_to_coeff[5*np.round(lam/5)]) + (1-lam/5 + lam//5) * float(lambda_to_coeff[5*np.round(lam/5)+5])

    Tatm = np.interp(lam, lambdas, coeffs)
    print(lam, Tatm)
    return Tatm



def calculate_QE(lam):
    """Calculate the quantum efficiency at a wavelength of lambda."""
    QE = 1

    return QE



def calculate_F0(lam):
    """Calculate the spectral photon flux for an m=0 source at lambda."""
    lambdas, coeffs = [], []
    with open("data/photon_fluxes.txt", "r", encoding='utf-8') as file:
        for line in file:
            if line[0] != "#":
                cur = line.split()
                lambdas.append(int(float(cur[0])))
                coeffs.append(float(cur[1]))


    #Tatm = (lam/5 - lam//5) * float(lambda_to_coeff[5*np.round(lam/5)]) + (1-lam/5 + lam//5) * float(lambda_to_coeff[5*np.round(lam/5)+5])

    F0 = np.interp(lam, lambdas, coeffs)
    return F0



def calculate_exposure_time(A, SNR, T_atm, QE, dlambda, F0, m):
    """Calculate the exposure time needed to achieve the required SNR."""

    dt = SNR **2 /(A * T_atm * QE * dlambda * F0 * 10 ** (-0.4 * m))

    return dt


## Calculating SNR for specific examples:

# An m_B = 14 star, observed in the B filter with a shed telescope for 30 minutes.
lam = filters["B"]["center_nm"]
N_photons = calculate_photon_count(
    math.pi * (0.3556 / 2) ** 2, 30 * 60, calculate_Tatm(lam),
    calculate_QE(lam), filters["B"]["width_nm"], calculate_F0(lam), 14,
)
print(f"Shed telescope, B=14 star: SNR = {calculate_SNR_poisson(N_photons):.2f}")




# The same source and integration time, but with the Elliot telescope.
N_photons = calculate_photon_count(
    math.pi * (0.6096 / 2) ** 2, 30 * 60, calculate_Tatm(lam),
    calculate_QE(lam), filters["B"]["width_nm"], calculate_F0(lam), 14,
)
print(f"Elliot telescope, B=14 star: SNR = {calculate_SNR_poisson(N_photons):.2f}")




# An m_r' = 6.2 asteroid, observed in the r' filter with a shed telescope for 3 seconds.
lam = filters["r'"]["center_nm"]
N_photons = calculate_photon_count(
    math.pi * (0.3556 / 2) ** 2, 3, calculate_Tatm(lam),
    calculate_QE(lam), filters["r'"]["width_nm"], calculate_F0(lam), 6.2,
)
print(f"Shed telescope, r'=6.2 asteroid: SNR = {calculate_SNR_poisson(N_photons):.2f}")

#!/usr/bin/env python3
"""
A module for the bow-tie (https://www.utupub.fi/handle/10024/152846 and references therein)
analysis of a response function of a particle instrument.
"""
__author__ = "Philipp Oleynik"
__credits__ = ["Philipp Oleynik"]

import math
from scipy import interpolate
from scipy import optimize
from statistics import geometric_mean
import numpy as np


def make_energy_grid(*, channels_per_decade = 256, min_energy = 0.01, max_energy = 1.0E5):
    """
    Calculates a standard logarithmic energy grid.
    :param channels_per_decade: Number of channels between 1 and 10, excluding exact value of 10.
    :param min_energy: Starting energy. Can be anything, not necessarily powers of 10
    :param max_energy: Upper limit of energy. Can be anything, not necessarily powers of 10
    :return: An integer and three float arrays: number_of_energy_steps, energy_midpoint, energy_cut, energy_bin_width
            The cuts are the upper limits.
    """
    emin_start = (np.floor(np.log10(min_energy) * channels_per_decade) / channels_per_decade)
    emax_stop = (np.floor(np.log10(max_energy) * channels_per_decade) / channels_per_decade)
    number_of_energy_steps = int((emax_stop - emin_start) * channels_per_decade + 1)
    log_step = 1.0 / channels_per_decade
    
    energy_midpoint = np.zeros(shape = (number_of_energy_steps,), dtype = float)
    energy_cut = np.zeros(shape = (number_of_energy_steps,), dtype = float)
    energy_low = np.zeros(shape = (number_of_energy_steps,), dtype = float)
    energy_bin_width = np.zeros(shape = (number_of_energy_steps,), dtype = float)
    
    for i in range(0, number_of_energy_steps, 1):
        midpoint = np.power(10, emin_start) * np.power(10, log_step * (i + 0.5))
        energy_bin_low = np.power(10, emin_start) * np.power(10, log_step * i)
        energy_bin_high = np.power(10, emin_start) * np.power(10, log_step * (i + 1))
        
        energy_cut[i] = energy_bin_high
        energy_midpoint[i] = midpoint
        energy_bin_width[i] = energy_bin_high - energy_bin_low
        energy_low[i] = energy_bin_low
    
    return { 'nstep': number_of_energy_steps,
             'midpt': energy_midpoint,
             'ehigh': energy_cut,
             'enlow': energy_low,
             'binwd': energy_bin_width }


def generate_pwlaw_spectra(energy_grid_dict,
                           gamma_pow_min = -3.5, gamma_pow_max = -1.5,
                           num_steps = 100, use_integral_bowtie = False):
    """
    The function generates a power-law spectra in a given range of indices.
    :param energy_grid_dict: A dictionary, must have 'midpt' (a numpy array)
    :type energy_grid_dict: the energy_grid_data (dictionary, see make_energy_grid),
    :param gamma_pow_min: The lower limit for power-law index
    :type gamma_pow_min: float
    :param gamma_pow_max: The upper limit for power-law index
    :type gamma_pow_max: float
    :param num_steps: The number of power-law spectra to generate
    :type num_steps: int
    :param use_integral_bowtie: True if integral spectrum is requested
    :type use_integral_bowtie: bool
    :return: A list of dictionaries containing the power-law spectra.
    :rtype: a list of dictionaries with float 'gamma' field and an array with a spectrum as 'spect' field.
    """
    model_spectra = []  # generate power-law spectra for folding
    if use_integral_bowtie:
        for power_law_gamma in np.linspace(gamma_pow_min, gamma_pow_max, num = num_steps, endpoint = True):
            model_spectra.append({
                'gamma': power_law_gamma,
                'spect': generate_powerlaw_np(energy_grid = energy_grid_dict, power_index = power_law_gamma),
                'intsp': generate_integral_powerlaw_np(energy_grid = energy_grid_dict,
                                                       power_index = power_law_gamma)
            })
    else:
        for power_law_gamma in np.linspace(gamma_pow_min, gamma_pow_max, num = num_steps, endpoint = True):
            model_spectra.append({
                'gamma': power_law_gamma,
                'spect': generate_powerlaw_np(energy_grid = energy_grid_dict, power_index = power_law_gamma)
            })
    return model_spectra


def generate_exppowlaw_spectra(energy_grid_dict,
                               gamma_pow_min = -3.5, gamma_pow_max = -1.5,
                               num_steps = 100, use_integral_bowtie = False,
                               cutoff_energy = 1.0):
    """
    The function generates exponentially cut off power-law spectra in a given range of indices.
    The exponential cutoff is applied by the formula dJ/dE = E^(gamma) * exp( - E0 / (E - E0)); E > E0
    :param energy_grid_dict: A dictionary, must have 'midpt' (a numpy array)
    :type energy_grid_dict: the energy_grid_data (dictionary, see make_energy_grid),
    :param gamma_pow_min: The lower limit for power-law index
    :type gamma_pow_min: float
    :param gamma_pow_max: The upper limit for power-law index
    :type gamma_pow_max: float
    :param num_steps: The number of power-law spectra to generate
    :type num_steps: int
    :param use_integral_bowtie: True if integral spectrum is requested
    :type use_integral_bowtie: bool
    :param cutoff_energy: The lower cutoff energy; E0 in the formula
    :type cutoff_energy: float
    :return: A list of dictionaries containing the exponentially cut off power-law spectra.
    :rtype: a list of dictionaries with float 'gamma' field and an array with a spectrum as 'spect' field.
    """
    model_spectra = []
    if use_integral_bowtie:
        print("Not implemented!")
        return None
    
    for power_law_gamma in np.linspace(gamma_pow_min, gamma_pow_max, num = num_steps, endpoint = True):
        spectrum = 1.0 * np.power(energy_grid_dict['midpt'], power_law_gamma) * \
                   np.exp(-cutoff_energy / (energy_grid_dict['midpt'] - cutoff_energy))
        
        index_cutoff = np.searchsorted(energy_grid_dict['midpt'], cutoff_energy)
        np.put(spectrum, range(0, index_cutoff + 1), 1.0E-30)
        model_spectra.append({
            'gamma': power_law_gamma,
            'spect': spectrum
        })
    return model_spectra


def generate_integral_powerlaw_np(*, energy_grid = None,
                                  power_index = -3.5, sp_norm = 1.0):
    """
    The function generates a single power-law integral spectrum.
    :param energy_grid_dict: A dictionary, must have 'midpt' (a numpy array)
    :type energy_grid_dict: the energy_grid_data (dictionary, see make_energy_grid),
    :param power_index: power-law index.
    :param sp_norm: norm factor.
    :return: spectrum as numpy array.
    """
    if energy_grid is not None:
        spectrum = - sp_norm * np.power(energy_grid['enlow'], power_index + 1) / (power_index + 1)
        return spectrum
    return None


def generate_powerlaw_np(*, energy_grid = None, power_index = -2, sp_norm = 1.0):
    """
    The function generates a single power-law differential spectrum.
    :param energy_grid_dict: A dictionary, must have 'midpt' (a numpy array)
    :type energy_grid_dict: the energy_grid_data (dictionary, see make_energy_grid),
    :param power_index: power-law index.
    :param sp_norm: norm factor.
    :return: spectrum as numpy array.
    """
    spectrum = sp_norm * np.power(energy_grid['midpt'], power_index)
    return spectrum


def fold_spectrum_np(*, grid = None, spectrum = None, response = None):
    """
    Folds incident spectrum with an instrument response. Int( spectrum * response * dE)
    :param grid: energy grid, midpoints of each energy bin
    :param spectrum: intensities defined at the midpoint of each energy bin
    :param response: geometric factor curve defined at the midpoint of each energy bin
    :return: countrate in the channel described by the response.
    """
    if grid is None:
        return math.nan
    if spectrum is None or response is None:
        return 0
    if (len(spectrum) == len(response)) and (len(spectrum) == len(grid['midpt'])):
        result = np.trapz(np.multiply(spectrum, response), grid['midpt'])
        return result
    return 0


def calculate_bowtie_gf(response_data,
                        model_spectra,
                        emin = 0.01, emax = 1000,
                        gamma_index_steps = 100,
                        use_integral_bowtie = False,
                        sigma = 3,
                        return_gf_stddev = False):
    """
    Calculates the bowtie geometric factor for a single channel
    :param return_gf_stddev: True if the margin of the channel geometric factor is requested.
    :type return_gf_stddev: bool
    :param response_data: The response data for the channel.
    :type response_data: A dictionary, must have 'grid', the energy_grid_data (dictionary, see make_energy_grid),
                         and 'resp', the channel response (an array of a length of energy_grid_data['nstep'])
    :param model_spectra: The model spectra for the analysis.
    :type model_spectra: A dictionary (see generate_pwlaw_spectra)
    :param emin: the minimal energy to consider
    :type emin: float
    :param emax: the maximum energy to consider
    :type emax: float
    :param gamma_index_steps:
    :type gamma_index_steps:
    :param use_integral_bowtie:
    :type use_integral_bowtie:
    :param sigma: Cutoff sigma value for the energy margin.
    :type sigma: float
    :return: (The geometric factor, [the standard dev of GF], the effective energy, lower margin for the effective energy, upper margin for the effective energy)
    :rtype: list
    """
    energy_grid_local = response_data['grid']['midpt']
    
    index_emin = np.searchsorted(energy_grid_local, emin)  # search for an index corresponding to start energy
    index_emax = np.searchsorted(energy_grid_local, emax)
    multi_geometric_factors = np.zeros((gamma_index_steps, response_data['grid']['nstep']), dtype = float)
    # for each model spectrum do the folding.
    
    for model_spectrum_idx, model_spectrum in enumerate(model_spectra):
        
        spectral_folding_int = fold_spectrum_np(grid = response_data['grid'],
                                                spectrum = model_spectrum['spect'],
                                                response = response_data['resp'])
        if use_integral_bowtie:
            spectrum_data = model_spectrum['intsp']
        else:
            spectrum_data = model_spectrum['spect']
        
        multi_geometric_factors[model_spectrum_idx, index_emin:index_emax] = spectral_folding_int / spectrum_data[index_emin:index_emax]
    
    # Create a discrete standard deviation vector for each energy in the grid.
    # This standard deviation is normalized to the local mean, so that a measure of spreading of points is obtained.
    # Mathematically, this implies normalization of the random variable to its mean.
    non_zero_gf = np.mean(multi_geometric_factors, axis = 0) > 0
    multi_geometric_factors_usable = multi_geometric_factors[:, non_zero_gf]
    means = np.exp(np.mean(np.log(multi_geometric_factors_usable), axis = 0))  # logarithmic mean
    gf_stddev = np.std(multi_geometric_factors_usable, axis = 0) / means
    gf_stddev_norm = gf_stddev / np.min(gf_stddev)
    bowtie_cross_index = np.argmin(gf_stddev_norm)  # The minimal standard deviation point - bowtie crossing point.
    
    # Interpolate the discrete standard deviation so that it could be used in a equation solver.
    # The discrete standard deviation is normalized to 1 in the minimum, so that 1.0 must be subtracted
    # before sigma level to make a discrete "equation" for the the interpolator because
    # the optimize.bisect looks for zeroes of a function, which is the interpolator.
    stddev_interpolator = interpolate.interp1d(energy_grid_local[non_zero_gf], gf_stddev_norm - 1.0 - sigma)
    try:
        (channel_energy_low) = optimize.bisect(stddev_interpolator,
                                               energy_grid_local[non_zero_gf][0],
                                               energy_grid_local[non_zero_gf][bowtie_cross_index])  # to the left of bowtie_cross_index
    except ValueError:
        channel_energy_low = 0
    
    try:
        (channel_energy_high) = optimize.bisect(stddev_interpolator,
                                                energy_grid_local[non_zero_gf][bowtie_cross_index],
                                                energy_grid_local[non_zero_gf][-1])  # to the right of bowtie_cross_index
    except ValueError:
        channel_energy_high = 0
    
    # gf = np.mean(multi_geometric_factors_usable, axis = 0)  # Average geometric factor for all model spectra
    # gf_cross = gf[bowtie_cross_index]  # The mean geometric factor for the bowtie crossing point
    
    gf_cross = geometric_mean(multi_geometric_factors_usable[:, bowtie_cross_index])
    energy_cross = energy_grid_local[bowtie_cross_index]
    if return_gf_stddev:
        return gf_cross, gf_stddev[bowtie_cross_index], energy_cross, channel_energy_low, channel_energy_high
    
    return gf_cross, energy_cross, channel_energy_low, channel_energy_high


if __name__ == "__main__":
    pass

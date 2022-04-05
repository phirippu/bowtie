#!/usr/bin/env python3
import numpy as np
import cProfile

from matplotlib import pyplot as plt

import bowtie


def gf_test(use_integral_bowtie = False, gamma_min = -3.5, gamma_max = -1.5,
            gamma_steps = 100, channels_per_decade = 256, powerlaw = True,
            plotspectrum = False, cutoff_energy = 1.0):
    """
    A test function for the bowtie module.
    :param use_integral_bowtie: True if the bowtie analysis is applied to an integral channel.
    :type use_integral_bowtie: Boolean
    :param gamma_min: Lower value for the model power law index.
    :type gamma_min: float
    :param gamma_max: Upper value for the model power law index.
    :type gamma_max: float
    :param gamma_steps: The number of model spectra to be used.
    :type gamma_steps: integer
    :param channels_per_decade: The number of energy steps per decade in the energy grid.
    :type channels_per_decade: integer
    """
    response_matrix = []
    energy_grid_data = bowtie.make_energy_grid(channels_per_decade = channels_per_decade, max_energy = 1000)
    if powerlaw:
        exppower_law_spectra = bowtie.generate_pwlaw_spectra(energy_grid_data,
                                                             gamma_pow_min = gamma_min,
                                                             gamma_pow_max = gamma_max,
                                                             num_steps = gamma_steps,
                                                             use_integral_bowtie = use_integral_bowtie)
    else:
        print("Experimental spectrum is used.")
        exppower_law_spectra = bowtie.generate_exppowlaw_spectra(energy_grid_data,
                                                                 gamma_pow_min = gamma_min,
                                                                 gamma_pow_max = gamma_max,
                                                                 num_steps = gamma_steps,
                                                                 use_integral_bowtie = use_integral_bowtie,
                                                                 cutoff_energy = cutoff_energy)
    
    if plotspectrum:
        fig, ax = plt.subplots(1)
        ax.scatter(energy_grid_data['midpt'], exppower_law_spectra[0]['spect'], s = 0.1, label = 'Side')
        ax.scatter(energy_grid_data['midpt'], exppower_law_spectra[gamma_steps // 2]['spect'], s = 0.1, label = 'Side')
        ax.scatter(energy_grid_data['midpt'], exppower_law_spectra[gamma_steps - 1]['spect'], s = 0.1, label = 'Side')
        ax.set_yscale('log')
        ax.set_xscale('log')
        plt.show()
    
    response_single_ch = np.zeros_like(energy_grid_data['midpt'])
    emin = 1.5  # test boxcar energies
    emax = 2.0
    index_emin = np.searchsorted(energy_grid_data['midpt'], emin)  # search for an index corresponding to start energy
    index_emax = np.searchsorted(energy_grid_data['midpt'], emax)
    real_emin = energy_grid_data['enlow'][index_emin]
    real_emax = energy_grid_data['ehigh'][index_emax]
    delta_e = real_emax - real_emin
    if use_integral_bowtie:
        np.put(response_single_ch, range(index_emin, energy_grid_data['nstep']), 1.0)  # a geometric factor of 1 from emin
    else:
        np.put(response_single_ch, range(index_emin, index_emax + 1), 1.0 / delta_e)  # a geometric factor of 1 from emin to emax
    
    response_matrix.append({  # in the real application, the response_matrix is a collection of responses for multiple channels
        'name': 'Test boxcar',  # channel name to print
        'grid': energy_grid_data,  # the energy grid dictionary
        'resp': response_single_ch  # channel response
    })
    
    for channel, response in enumerate(response_matrix):
        (gf_to_print, eff_energy_to_print, _, _) = bowtie.calculate_bowtie_gf(response,
                                                                              exppower_law_spectra,
                                                                              gamma_index_steps = gamma_steps,
                                                                              use_integral_bowtie = use_integral_bowtie)
        print(f"Channel {response_matrix[channel]['name']}: G ={gf_to_print:6.3g}; E ={eff_energy_to_print:6.3g}.")


if __name__ == "__main__":
    gf_test(use_integral_bowtie = False,
            powerlaw = True,
            plotspectrum = True,
            cutoff_energy = 1.5,
            gamma_max = -1.5,
            gamma_min = -3.5,
            channels_per_decade = 256)
    # cProfile.run('gf_test(use_integral_bowtie = True)', sort = 'time')

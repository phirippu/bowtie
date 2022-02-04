#!/usr/bin/env python3

import numpy as np
import sys

import bowtie


def main(use_integral_bowtie = False,
         particle = 'e'):
    base_path = "..."  # adjust to your path
    channels_per_decade = 256
    gamma_min = -3.5
    gamma_max = -1.5
    gamma_steps = 100
    global_emin = 0.01
    global_emax = 1000
    
    if particle == 'e':
        instrument_channels = 7
        channel_start = 1
        channel_stop = 8
        # For contamination studies
        # instrument_channels = 3
        # channel_start = 8
        # channel_stop = 8 + 3
    else:
        instrument_channels = 9
        channel_start = 8
        channel_stop = 8 + 9
        # For contamination studies
        # instrument_channels = 7
        # channel_start = 1
        # channel_stop = 8
    
    #        y = particles_response[:, chdraw, 2] / (particles_shot / radiation_area + 1E-24) * const.pi
    data_file_name = base_path + \
                     '/Results/SIXS/array_vault_{1:s}_{0:d}.npz'.format(channels_per_decade, particle)
    print("Using response file:", data_file_name)
    
    npzfile = np.load(data_file_name)
    particles_shot = npzfile['particles_Shot']  # The number of particles shot in a simulation of all energy bins
    particles_response = npzfile['particles_Respo']  # The number of particles detected per particle channel in all energy bins
    other_params = npzfile['other_params']
    nstep = int(other_params[0])  # The total number of energy bins
    radiation_area = other_params[2]  # The radiation area (isotropically radiating sphere) around the Geant4 instrument model in cm2
    energy_midpoint = npzfile['energy_Mid']  # midpoints of the energy bins in MeV
    energy_toppoint = npzfile['energy_Cut']  # high cuts of the energy bins in MeV
    energy_channel_width = npzfile['energy_Width']  # the energy bin widths in MeV
    # energy grid in the format compatible with the output of a function in the bowtie package
    energy_grid = { 'nstep': nstep, 'midpt': energy_midpoint,
                    'ehigh': energy_toppoint, 'enlow': energy_toppoint - energy_channel_width,
                    'binwd': energy_channel_width }
    
    channel_names = ["O", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]
    side = 0
    
    response_matrix = []
    normalize_to_area = 1.0 / ((particles_shot + 1) / radiation_area) * np.pi
    for i in range(channel_start, channel_stop):
        resp_cache = particles_response[:, i, side] * normalize_to_area
        
        response_matrix.append({
            'name': channel_names[i],  # last added name
            'grid': energy_grid,
            'resp': resp_cache,  # channel response
        })
    
    power_law_spectra = bowtie.generate_pwlaw_spectra(energy_grid, gamma_min, gamma_max, gamma_steps)
    gf_to_print = np.zeros(len(response_matrix))
    eff_energies_to_print = np.zeros(len(response_matrix))
    
    for channel, response in enumerate(response_matrix):
        (gf_to_print[channel], eff_energies_to_print[channel], _, _) = bowtie.calculate_bowtie_gf(response,
                                                                                                  power_law_spectra,
                                                                                                  emin = global_emin,
                                                                                                  emax = global_emax,
                                                                                                  gamma_index_steps = gamma_steps,
                                                                                                  use_integral_bowtie = use_integral_bowtie,
                                                                                                  sigma = 3)
        print(f"Channel {response['name']}: G = {gf_to_print[channel]:.3g}, cm2srMeV; E = {eff_energies_to_print[channel]:.2g}, MeV")


if __name__ == "__main__":
    particle = 'e'  # default particle
    if len(sys.argv) > 1:
        particle = 'e' if sys.argv[1] == '-e' else 'p'
    
    main(particle = particle)

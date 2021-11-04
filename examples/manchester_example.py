#!/usr/bin/env python

# -----------------------------------------------------------------------------
#  manchester_example.py
#
#  Example for reading from ROOT file
#   * Author: Everybody is an author!
#   * Creation date: 3 November 2021
# -----------------------------------------------------------------------------

import argparse
import numpy as np
import awkward as ak
import uproot

# parse arguments from command
parser = argparse.ArgumentParser(description="example program")
parser.add_argument("file", type=str, default=None,
                    help="path to ROOT file")

# parse arguments
args = parser.parse_args()
file_path = str(args.file)

# initialize lists
event_list = []
neutrino_energy_list = []
vertex_x_list = []
vertex_y_list = []
vertex_z_list = []
pdg_code_list = []

# open the ROOT file
with uproot.open(file_path) as f:

    #--------------------------------------------------------------------------
    # get metadata from ROOT file
    #--------------------------------------------------------------------------

    metadata = f['metadata']

    # get detector dimensions
    detector_length_x = metadata['detector_length_x'].array()[0]  # cm
    detector_length_y = metadata['detector_length_y'].array()[0]  # cm
    detector_length_z = metadata['detector_length_z'].array()[0]  # cm

    # get parameters used in Q_PIX_RTD
    drift_velocity = metadata['drift_velocity'].array()[0]  # cm/s
    longitudinal_diffusion = metadata['longitudinal_diffusion'].array()[0]  # cm^2/s
    transverse_diffusion = metadata['transverse_diffusion'].array()[0]  # cm^2/s
    electron_lifetime = metadata['electron_lifetime'].array()[0]  # s
    readout_dimensions = metadata['readout_dimensions'].array()[0]  # cm
    pixel_size = metadata['pixel_size'].array()[0]  # cm
    reset_threshold = metadata['reset_threshold'].array()[0]  # electrons
    sample_time = metadata['sample_time'].array()[0]  # s
    buffer_window = metadata['buffer_window'].array()[0]  # s
    dead_time = metadata['dead_time'].array()[0]  # s
    charge_loss = metadata['charge_loss'].array()[0]  # 0 is off, 1 is on

    #--------------------------------------------------------------------------
    # get event tree from ROOT file
    #--------------------------------------------------------------------------

    tree = f['event_tree']

    # list of branches that we want to access
    branches = [

        # event number
        'event',

        # generator particle information [Q_PIX_GEANT4]
        'generator_initial_number_particles',
        'generator_initial_particle_pdg_code',
        'generator_initial_particle_x',
        'generator_initial_particle_y',
        'generator_initial_particle_z',
        'generator_initial_particle_t',
        'generator_initial_particle_px',
        'generator_initial_particle_py',
        'generator_initial_particle_pz',
        'generator_initial_particle_energy',
        'generator_initial_particle_mass',
        'generator_final_number_particles',
        'generator_final_particle_pdg_code',
        'generator_final_particle_x',
        'generator_final_particle_y',
        'generator_final_particle_z',
        'generator_final_particle_t',
        'generator_final_particle_px',
        'generator_final_particle_py',
        'generator_final_particle_pz',
        'generator_final_particle_energy',
        'generator_final_particle_mass',

        # MC particle information [Q_PIX_GEANT4]
        'number_particles', 'particle_track_id', 'particle_parent_track_id',
        'particle_pdg_code', 'particle_mass',
        'particle_initial_x', 'particle_initial_y', 'particle_initial_z',
        'particle_initial_t',
        'particle_initial_px', 'particle_initial_py', 'particle_initial_pz',
        'particle_initial_energy',

        # MC hit information [Q_PIX_GEANT4]
        'hit_energy_deposit', 'hit_track_id', 'hit_process_key',
        'hit_start_x', 'hit_start_y', 'hit_start_z', 'hit_start_t',
        'hit_end_x', 'hit_end_y', 'hit_end_z', 'hit_end_t',

        # MC energy deposited [Q_PIX_GEANT4]
        'energy_deposit',

        # pixel information [Q_PIX_RTD]
        'pixel_x', 'pixel_y', 'pixel_reset', 'pixel_tslr',
        'pixel_reset_truth_track_id', 'pixel_reset_truth_weight',

    ]

    #--------------------------------------------------------------------------
    # iterate through the event tree
    #--------------------------------------------------------------------------

    for arrays in tree.iterate(filter_name=branches):

        # get event number array
        event_array = arrays['event']

        # generator particle arrays
        generator_initial_number_particles_array = arrays['generator_initial_number_particles']
        generator_initial_particle_pdg_code_array = arrays['generator_initial_particle_pdg_code']
        generator_initial_particle_x_array = arrays['generator_initial_particle_x']
        generator_initial_particle_y_array = arrays['generator_initial_particle_y']
        generator_initial_particle_z_array = arrays['generator_initial_particle_z']
        generator_initial_particle_t_array = arrays['generator_initial_particle_t']
        generator_initial_particle_px_array = arrays['generator_initial_particle_px']
        generator_initial_particle_py_array = arrays['generator_initial_particle_py']
        generator_initial_particle_pz_array = arrays['generator_initial_particle_pz']
        generator_initial_particle_energy_array = arrays['generator_initial_particle_energy']
        generator_initial_particle_mass_array = arrays['generator_initial_particle_mass']
        generator_final_number_particles_array = arrays['generator_final_number_particles']
        generator_final_particle_pdg_code_array = arrays['generator_final_particle_pdg_code']
        generator_final_particle_x_array = arrays['generator_final_particle_x']
        generator_final_particle_y_array = arrays['generator_final_particle_y']
        generator_final_particle_z_array = arrays['generator_final_particle_z']
        generator_final_particle_t_array = arrays['generator_final_particle_t']
        generator_final_particle_px_array = arrays['generator_final_particle_px']
        generator_final_particle_py_array = arrays['generator_final_particle_py']
        generator_final_particle_pz_array = arrays['generator_final_particle_pz']
        generator_final_particle_energy_array = arrays['generator_final_particle_energy']
        generator_final_particle_mass_array = arrays['generator_final_particle_mass']

        # get MC particle arrays
        number_particles_array = arrays['number_particles']
        particle_track_id_array = arrays['particle_track_id']
        particle_parent_track_id_array = arrays['particle_parent_track_id']
        particle_pdg_code_array = arrays['particle_pdg_code']
        particle_mass_array = arrays['particle_mass']
        particle_initial_x_array = arrays['particle_initial_x']
        particle_initial_y_array = arrays['particle_initial_y']
        particle_initial_z_array = arrays['particle_initial_z']
        particle_initial_t_array = arrays['particle_initial_t']
        particle_initial_px_array = arrays['particle_initial_px']
        particle_initial_py_array = arrays['particle_initial_py']
        particle_initial_pz_array = arrays['particle_initial_pz']
        particle_initial_energy_array = arrays['particle_initial_energy']

        # get MC energy deposited arrays
        energy_deposit_array = arrays['energy_deposit']

        # get MC hit arrays
        hit_track_id_array = arrays['hit_track_id']
        hit_end_x_array = arrays['hit_end_x']
        hit_end_y_array = arrays['hit_end_y']
        hit_end_z_array = arrays['hit_end_z']
        hit_end_t_array = arrays['hit_end_t']
        hit_energy_deposit_array = arrays['hit_energy_deposit']
        hit_process_key_array = arrays['hit_process_key']

        # get pixel arrays
        pixel_x_array = arrays['pixel_x']
        pixel_y_array = arrays['pixel_y']
        pixel_reset_array = arrays['pixel_reset']
        pixel_tslr_array = arrays['pixel_tslr']

        # get pixel MC truth arrays
        pixel_reset_track_id_array = arrays['pixel_reset_truth_track_id']
        pixel_reset_weight_array = arrays['pixel_reset_truth_weight']

        # get number of events
        number_events = len(event_array)

        # loop over events
        for idx in range(number_events):

            # get event number
            event = event_array[idx]

            print('Event:', event)

            # get generator particle information for event
            generator_initial_number_particles = generator_initial_number_particles_array[idx]
            generator_initial_particle_pdg_code = generator_initial_particle_pdg_code_array[idx]
            generator_initial_particle_x = generator_initial_particle_x_array[idx]
            generator_initial_particle_y = generator_initial_particle_y_array[idx]
            generator_initial_particle_z = generator_initial_particle_z_array[idx]
            generator_initial_particle_t = generator_initial_particle_t_array[idx]
            generator_initial_particle_px = generator_initial_particle_px_array[idx]
            generator_initial_particle_py = generator_initial_particle_py_array[idx]
            generator_initial_particle_pz = generator_initial_particle_pz_array[idx]
            generator_initial_particle_energy = generator_initial_particle_energy_array[idx]
            generator_initial_particle_mass = generator_initial_particle_mass_array[idx]
            generator_final_number_particles = generator_final_number_particles_array[idx]
            generator_final_particle_pdg_code = generator_final_particle_pdg_code_array[idx]
            generator_final_particle_x = generator_final_particle_x_array[idx]
            generator_final_particle_y = generator_final_particle_y_array[idx]
            generator_final_particle_z = generator_final_particle_z_array[idx]
            generator_final_particle_t = generator_final_particle_t_array[idx]
            generator_final_particle_px = generator_final_particle_px_array[idx]
            generator_final_particle_py = generator_final_particle_py_array[idx]
            generator_final_particle_pz = generator_final_particle_pz_array[idx]
            generator_final_particle_energy = generator_final_particle_energy_array[idx]
            generator_final_particle_mass = generator_final_particle_mass_array[idx]

            # get MC particle information for event
            number_particles = number_particles_array[idx]
            particle_track_id = particle_track_id_array[idx]
            particle_parent_track_id = particle_parent_track_id_array[idx]
            particle_pdg_code = particle_pdg_code_array[idx]
            particle_mass = particle_mass_array[idx]
            particle_initial_x = particle_initial_x_array[idx]
            particle_initial_y = particle_initial_y_array[idx]
            particle_initial_z = particle_initial_z_array[idx]
            particle_initial_t = particle_initial_t_array[idx]
            particle_initial_px = particle_initial_px_array[idx]
            particle_initial_py = particle_initial_py_array[idx]
            particle_initial_pz = particle_initial_pz_array[idx]
            particle_initial_energy = particle_initial_energy_array[idx]

            # get MC energy deposited information for event
            # this is the total energy deposited in this event
            energy_deposit = energy_deposit_array[idx]

            neutrino_energy = None
            vertex_x = None
            vertex_y = None
            vertex_z = None
            pdg_codes = []

            #------------------------------------------------------------------
            # Loop over initial generator particles
            #------------------------------------------------------------------
            #
            # These are the initial particles from the event generator.
            #
            # Assuming MARLEY is used as the event generator:
            #
            #   (1) For CC events, the initial particles are the neutrino and
            #       an argon nucleus.
            #
            #   (2) For ES events, the initial particles are the neutrino and
            #       an electron.
            #
            #------------------------------------------------------------------

            for p_idx in range(generator_initial_number_particles):

                # skip if this is not an electron neutrino
                if (generator_initial_particle_pdg_code[p_idx] != 12):
                    continue

                # get energy of the electron neutrino
                neutrino_energy = generator_initial_particle_energy[p_idx]

                # get vertex position
                vertex_x = generator_initial_particle_x[p_idx]
                vertex_y = generator_initial_particle_y[p_idx]
                vertex_z = generator_initial_particle_z[p_idx]

            #------------------------------------------------------------------
            # Loop over final generator particles
            #------------------------------------------------------------------
            #
            # These are the final particles from the event generator.  These
            # particles are used as input to the GEANT4 simulation in which
            # each input particle is propagated through a volume of liquid
            # argon.
            #
            # This is one of two ways of accessing the final generator
            # particles.
            #
            #------------------------------------------------------------------

            for p_idx in range(generator_final_number_particles):

                # get energy of the particle
                energy = generator_final_particle_energy[p_idx]

                # get PDG code of the particle
                pdg_code = generator_final_particle_pdg_code[p_idx]

            #------------------------------------------------------------------
            # Loop over MC particles
            #------------------------------------------------------------------
            #
            # These are the MC particles from the GEANT4 simulation.  The final
            # generator particles all have a parent track ID of 0; we will use
            # this to access the final generator particles.
            #
            # This is the other way of accessing the final generator particles.
            # The advantage of this method is that we get to access the track
            # ID of the MC particle and other useful information that the
            # GEANT4 simulation provides.
            #
            #------------------------------------------------------------------

            for p_idx in range(number_particles):

                # get MC particle parent track ID
                parent_track_id = particle_parent_track_id[p_idx]

                # skip if this is not a final generator particle
                if parent_track_id != 0:
                    continue

                # get MC particle track ID
                track_id = particle_track_id[p_idx]

                # get MC particle PDG code
                pdg_code = particle_pdg_code[p_idx]
                pdg_codes.append(pdg_code)  # append to pdg_codes list

                # get initial 4-position and 4-momentum of MC particle
                position = [
                    particle_initial_x[p_idx],
                    particle_initial_y[p_idx],
                    particle_initial_z[p_idx],
                    particle_initial_t[p_idx],
                ]
                momentum = [
                    particle_initial_px[p_idx],
                    particle_initial_py[p_idx],
                    particle_initial_pz[p_idx],
                    particle_initial_energy[p_idx],
                ]

            #------------------------------------------------------------------

            # check that all the relevant information have been collected
            if neutrino_energy and vertex_x and vertex_y and vertex_z and pdg_codes:

                # append them to their respective lists
                neutrino_energy_list.append(neutrino_energy)
                vertex_x_list.append(vertex_x)
                vertex_y_list.append(vertex_y)
                vertex_z_list.append(vertex_z)
                pdg_code_list.append(pdg_codes)

                print("Neutrino energy [MeV]:", neutrino_energy)
                print("Energy deposited [MeV]:", energy_deposit)
                print("Vertex [cm]: (%s, %s, %s)" % (vertex_x, vertex_y, vertex_z))
                print("PDG codes of outgoing particles:", pdg_codes)

            print("-------------------------------------------------------------------------------")

            #------------------------------------------------------------------
            # let's skip the rest of the code in this event loop for now
            #------------------------------------------------------------------
            continue
            #------------------------------------------------------------------

            # get pixel information for event
            pixel_x = pixel_x_array[idx]
            pixel_y = pixel_y_array[idx]
            pixel_reset = pixel_reset_array[idx]
            pixel_tslr = pixel_tslr_array[idx]
            pixel_reset_track_id = pixel_reset_track_id_array[idx]
            pixel_reset_weight = pixel_reset_weight_array[idx]

            # get number of pixels in event
            number_pixels = len(pixel_x)

            # loop over pixels
            for px in range(number_pixels):

                # print the pixel coordinate
                print('  pixel (x, y): ({}, {})'.format(pixel_x[px], pixel_y[px]))

                # print list of resets associated with this pixel
                print('  resets:', pixel_reset[px])

                # print list of resets associated with this pixel
                print('  time since last reset:', pixel_tslr[px])

                # get number of resets associated with this pixel
                number_resets = len(pixel_reset[px])

                # loop over resets
                for rst in range(number_resets):

                    # get reset time
                    reset = pixel_reset[px][rst]

                    # get time since last reset
                    tslr = pixel_tslr[px][rst]

                    # get track IDs of MC particles responsible for this reset
                    mc_track_ids = pixel_reset_track_id[px][rst]
                    mc_weights = pixel_reset_weight[px][rst]

                    # get numbers of MC particles responsible for this reset
                    number_mc_particles = len(mc_track_ids)

                    # print reset, tslr, and z
                    print('    reset time [ns]:', reset * 1e9,
                          '; time since last reset [ns]:', tslr * 1e9,
                          '; z [cm]:', reset * drift_velocity)

                    # print list of track IDs and weights
                    print('      MC particle track IDs:', mc_track_ids)
                    print('      MC weights:', mc_weights)

                    # # loop over MC particles
                    # for mcp in range(number_mc_particles):
                    #     mc_track_id = mc_track_ids[mcp]
                    #     mc_weight = mc_weights[mcp]

            #------------------------------------------------------------------
            # extract spatial information from pixels and resets
            #------------------------------------------------------------------

            pixel_multiplicity = ak.count(pixel_reset, axis=1)
            pix_t = ak.flatten(pixel_reset).to_numpy()
            pix_x = np.repeat(pixel_x.to_numpy(), pixel_multiplicity)
            pix_y = np.repeat(pixel_y.to_numpy(), pixel_multiplicity)
            pix_tslr = ak.flatten(pixel_tslr).to_numpy()

            # convert to physical units
            pix_x = pix_x * pixel_size  # cm
            pix_y = pix_y * pixel_size  # cm
            pix_z = pix_t * drift_velocity  # cm
            pix_tslr = pix_tslr * 1e9  # ns

            #------------------------------------------------------------------
            # do things with pixels here
            #------------------------------------------------------------------

            # blah

            #------------------------------------------------------------------

#------------------------------------------------------------------------------
# convert to numpy arrays
#------------------------------------------------------------------------------
event_array = np.array(event_list)
neutrino_energy_array = np.array(neutrino_energy_list)
vertex_x_array = np.array(vertex_x_list)
vertex_y_array = np.array(vertex_y_list)
vertex_z_array = np.array(vertex_z_list)

#------------------------------------------------------------------------------
# convert to awkward array (jagged array)
#------------------------------------------------------------------------------
pdg_code_array = ak.Array(pdg_code_list)

#------------------------------------------------------------------------------
# print arrays
#------------------------------------------------------------------------------
print('event_array:', event_array)
print('neutrino_energy_array:', neutrino_energy_array)
print('vertex_x_array:', vertex_x_array)
print('vertex_y_array:', vertex_y_array)
print('vertex_z_array:', vertex_z_array)
print('pdg_code_array:', pdg_code_array)

#------------------------------------------------------------------------------
# make plots with matplotlib here
#------------------------------------------------------------------------------
# import matplotlib.pyplot as plt


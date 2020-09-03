#!/usr/bin/env python

# -----------------------------------------------------------------------------
#  read_root.py
#
#  Example for reading from ROOT file
#   * Author: Everybody is an author!
#   * Creation date: 1 September 2020
# -----------------------------------------------------------------------------

import argparse
import numpy as np
import uproot

# parse arguments from command
parser = argparse.ArgumentParser(description="pixel event display")
parser.add_argument("file", type=str, default=None,
                    help="path to ROOT file")

args = parser.parse_args()
file_path = str(args.file)

with uproot.open(file_path) as f:

    #--------------------------------------------------------------------------
    # get metadata from ROOT file
    #--------------------------------------------------------------------------

    metadata = f['metadata']

    # get detector dimensions
    detector_length_x = metadata.array('detector_length_x')[0]  # cm
    detector_length_y = metadata.array('detector_length_y')[0]  # cm
    detector_length_z = metadata.array('detector_length_z')[0]  # cm

    # get parameters used in Q_PIX_RTD
    drift_velocity = metadata.array('drift_velocity')[0]  # cm/ns
    longitudinal_diffusion = metadata.array('longitudinal_diffusion')[0]  # cm^2/ns
    transverse_diffusion = metadata.array('transverse_diffusion')[0]  # cm^2/ns
    electron_lifetime = metadata.array('electron_lifetime')[0]  # ns
    readout_dimensions = metadata.array('readout_dimensions')[0]  # cm
    pixel_size = metadata.array('pixel_size')[0]  # cm
    reset_threshold = metadata.array('reset_threshold')[0]  # electrons
    sample_time = metadata.array('sample_time')[0]  # ns
    buffer_window = metadata.array('buffer_window')[0]  # ns
    dead_time = metadata.array('dead_time')[0]  # ns
    charge_loss = metadata.array('charge_loss')[0]  # 0 is off, 1 is on

    #--------------------------------------------------------------------------
    # get event tree from ROOT file
    #--------------------------------------------------------------------------

    tree = f['event_tree']

    # list of branches that we want to access
    branches = [

        # event number
        'event',

        # MC particle information [Q_PIX_GEANT4]
        'particle_track_id', 'particle_pdg_code',
        'particle_mass', 'particle_initial_energy',

        # MC hit information [Q_PIX_GEANT4]
        'hit_energy_deposit', 'hit_track_id', 'hit_process_key',
        'hit_start_x', 'hit_start_y', 'hit_start_z', 'hit_start_t',
        'hit_end_x', 'hit_end_y', 'hit_end_z', 'hit_end_t',

        # pixel information [Q_PIX_RTD]
        'pixel_x', 'pixel_y', 'pixel_reset', 'pixel_tslr',

    ]

    #--------------------------------------------------------------------------
    # iterate through the event tree
    #--------------------------------------------------------------------------

    for arrays in tree.iterate(branches=branches, namedecode='utf-8'):

        # get event number array
        event_array = arrays['event']

        # get MC particle arrays
        particle_track_id_array = arrays['particle_track_id']
        particle_pdg_code_array = arrays['particle_pdg_code']

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

        # get number of events
        number_events = len(event_array)

        # loop over events
        for idx in range(number_events):

            # get event number
            event = event_array[idx]

            # get pixel information for event
            pixel_x = pixel_x_array[idx]
            pixel_y = pixel_y_array[idx]
            pixel_reset = pixel_reset_array[idx]
            pixel_tslr = pixel_tslr_array[idx]

            # get number of pixels in event
            number_pixels = len(pixel_x)

            print('Event:', event)

            # loop over pixels
            for px in range(number_pixels):

                # print the pixel coordinate
                print('  (x, y): ({}, {})'.format(pixel_x[px], pixel_y[px]))

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

            #------------------------------------------------------------------
            # extract spatial information from pixels and resets
            #------------------------------------------------------------------

            # pixel array
            pix_array = list(np.empty((1, 4)))

            # iterate through pixels and fill pixel array
            for px in range(number_pixels):

                # get number of resets for this pixel
                number_resets = len(pixel_reset[px])

                # iterate through resets
                for rst in range(number_resets):

                    # get reset time
                    reset = pixel_reset[px][rst]

                    # get time since last reset
                    tslr = pixel_tslr[px][rst]

                    # append to pixel array
                    pix_array.append([pixel_x[px], pixel_y[px], reset, tslr])

            # convert list to array
            if len(pix_array) > 1:
                # ignore dummy element
                pix_array = np.array(pix_array[1:], dtype=int)
            else:
                pix_array = np.array(pix_array, dtype=int)

            # convert to physical units
            pix_x = pix_array[:, 0] * pixel_size  # cm
            pix_y = pix_array[:, 1] * pixel_size  # cm
            pix_z = pix_array[:, 2] * drift_velocity  # cm
            pix_tslr = pix_array[:, 3]  # ns

            #------------------------------------------------------------------

            # do things with pixels here


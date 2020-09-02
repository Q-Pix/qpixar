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

    # get metadata from ROOT file
    metadata = f['metadata']
    detector_length_x = metadata.array('detector_length_x')[0]
    detector_length_y = metadata.array('detector_length_y')[0]
    detector_length_z = metadata.array('detector_length_z')[0]

    # get event tree from ROOT file
    tree = f['event_tree']

    # list of branches that we want to access
    branches = [
        # event number
        'event',

        # MC particle information
        'particle_track_id', 'particle_pdg_code',
        'particle_mass', 'particle_initial_energy',

        # MC hit information
        'hit_energy_deposit', 'hit_track_id', 'hit_process_key',
        'hit_start_x', 'hit_start_y', 'hit_start_z', 'hit_start_t',
        'hit_end_x', 'hit_end_y', 'hit_end_z', 'hit_end_t',

        # pixel information
        'pixel_x', 'pixel_y', 'pixel_reset',
    ]

    # loop over branches in the event tree
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

            # get number of pixels in event
            number_pixels = len(pixel_x)

            print('Event:', event)

            # loop over pixels
            for px in range(number_pixels):

                # print the pixel coordinate
                print('  (x, y): ({}, {})'.format(pixel_x[px], pixel_y[px]))

                # print list of resets associated with this pixel
                print('  resets:', pixel_reset[px])

                # loop over resets
                for reset in pixel_reset[px]:
                    # do something with `reset'
                    pass


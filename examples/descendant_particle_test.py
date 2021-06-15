#!/usr/bin/env python

# -----------------------------------------------------------------------------
#  descendant_particle_test.py
#
#  Example for reading from ROOT file
#   * Author: Everybody is an author!
#   * Creation date: 15 June 2021
# -----------------------------------------------------------------------------

import argparse
import numpy as np
import uproot
import awkward as ak

import qpixtools as qp

import pprint
import textwrap
import json

# parse arguments from command
parser = argparse.ArgumentParser(description="TEST")
parser.add_argument("file", type=str, default=None,
                    help="path to ROOT file")

args = parser.parse_args()
file_path = str(args.file)

with uproot.open(file_path) as f:

    #--------------------------------------------------------------------------
    # get event tree from ROOT file
    #--------------------------------------------------------------------------

    tree = f['event_tree']

    # list of branches that we want to access
    branches = [

        # event number
        'event',

        # MC particle information [Q_PIX_GEANT4]
        'number_particles',
        'particle_track_id', 'particle_pdg_code', 'particle_parent_track_id',
        'particle_mass', 'particle_initial_energy',
        'particle_number_daughters', 'particle_daughter_track_id',

    ]

    #--------------------------------------------------------------------------
    # iterate through the event tree
    #--------------------------------------------------------------------------

    for arrays in tree.iterate(filter_name=branches):

        # get event number array
        event_array = arrays['event']

        # get MC particle arrays
        number_particles_array = arrays['number_particles']
        particle_track_id_array = arrays['particle_track_id']
        particle_pdg_code_array = arrays['particle_pdg_code']
        particle_parent_track_id_array = arrays['particle_parent_track_id']
        particle_number_daughters_array = arrays['particle_number_daughters']
        particle_daughter_track_id_array = arrays['particle_daughter_track_id']

        # get number of events
        number_events = len(event_array)

        # prompt message
        msg = """
You may...

  (1) type in "exit", "quit", or "q" and press enter to terminate the program; or
  (2) type in anything else or press enter to go to the next event.
"""

        print(msg)

        # loop over events
        for idx in range(number_events):

            # wait for user input
            user_input = input()
            try:
                value = user_input.strip().lower()
                if value == "exit" or value == "quit" or value == "q":
                    print("\nExiting...\n")
                    break
            except:
                continue

            # get event number
            event = event_array[idx]

            print('Event:', event)

            # get MC particle information
            number_particles = number_particles_array[idx]
            particle_track_id = particle_track_id_array[idx]
            particle_pdg_code = particle_pdg_code_array[idx]
            particle_parent_track_id = particle_parent_track_id_array[idx]
            particle_number_daughters = particle_number_daughters_array[idx]
            particle_daughter_track_id = particle_daughter_track_id_array[idx]

            # fill MC particle map
            particle_map = qp.fill_particle_map(particle_track_id)

            # map of descendant particle track ID to ancestor particle track ID
            # the descendant track ID is the key and the ancestor track ID is
            # the value
            ancestor_track_id = {}

            # loop over MC particles
            # for p_idx in range(number_particles):
            for track_id, p_idx in particle_map.items():

                # track_id = particle_track_id[p_idx]
                pdg_code = particle_pdg_code[p_idx]
                parent_track_id = particle_parent_track_id[p_idx]
                number_daughters = particle_number_daughters[p_idx]
                daughter_track_id = particle_daughter_track_id[p_idx]

                # # check particle map
                # if p_idx != particle_map[track_id]:
                #     raise ValueError('Mismatch of particle index to track ID!')

                # print info only for the first generation of MC particles
                if parent_track_id > 1:
                    continue

                # create and fill MC shower particle map
                shower_particle_map = {}
                qp.fill_shower_particle_map(track_id,
                                            particle_daughter_track_id,
                                            particle_map, shower_particle_map)

                # get array of track IDs for the entire lineage
                lineage_track_id_array = ak.Array(list(shower_particle_map.keys()))

                # print info for MC particle
                print('\n  Track ID:', track_id)
                print('    Index:', particle_map[track_id])
                print('    PDG code:', pdg_code)
                print('    Parent track ID:', parent_track_id)
                print('    Number of daughters:', number_daughters)
                print('    Daughter track IDs:', daughter_track_id)

                s = json.dumps(shower_particle_map, sort_keys=True, indent=2)
                s = s.replace('"', '')
                print('    Shower particle map:', s[:1])
                print(textwrap.indent(s[2:], ' ' * 25))

                print('    All track IDs in lineage:', lineage_track_id_array)

                # fill ancestor track ID map
                for descendant_track_id in lineage_track_id_array:
                    ancestor_track_id[descendant_track_id] = track_id

            print('\nEnd of event', event)
            print(msg)


#!/usr/bin/env python

# -----------------------------------------------------------------------------
#  pyroot_interface_example.py
#
#  Example for reading from ROOT file using the pyroot interface
#   * Author: Everybody is an author!
#   * Creation date: 1 March 2021
# -----------------------------------------------------------------------------

import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
import pyroot

# file_path = './examples/root/muon_rtd_2020-09-08.root'
file_path = '/Volumes/seagate/work/qpix/supernova_test/rtd/nu_e/Nu_e-rtd-39.root'

file_handler = pyroot.FileHandler(file_path)
event_handler = pyroot.EventHandler(file_handler)

number_entries = event_handler.number_entries

for idx in range(number_entries):

    event_handler.get_entry(idx)
    event_handler.load_mc_generator_particles()
    event_handler.load_mc_particles()
    event_handler.load_mc_hits()
    event_handler.match_mc_hits_to_mc_particles()

    print(idx, event_handler.run, event_handler.event)


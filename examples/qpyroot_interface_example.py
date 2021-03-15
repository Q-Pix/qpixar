#!/usr/bin/env python

# -----------------------------------------------------------------------------
#  qpyroot_interface_example.py
#
#  Example for reading from ROOT file using the qpyroot interface
#   * Author: Everybody is an author!
#   * Creation date: 1 March 2021
# -----------------------------------------------------------------------------

import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
import qpyroot

# file_path = './examples/root/muon_rtd_2020-09-08.root'
file_path = '/Volumes/seagate/work/qpix/supernova_test/rtd/nu_e/Nu_e-rtd-39.root'

# intialize the file handler
file_handler = qpyroot.FileHandler(file_path)

# initialize the event handler
event_handler = qpyroot.EventHandler(file_handler)

# get number of entries
number_entries = event_handler.number_entries()

# prompt message
msg = """
You may...

  (1) type in "exit", "quit", or "q" and press enter to terminate the program; or
  (2) type in anything else or press enter to go to the next event.
"""

print(msg)

for idx in range(number_entries):

    # get entry
    event_handler.get_entry(idx)

    # load the MC generator particles
    event_handler.load_mc_generator_particles()

    # load the MC particles
    event_handler.load_mc_particles()

    # load the MC hits
    event_handler.load_mc_hits()

    # match the MC hits to the MC particles
    event_handler.match_mc_hits_to_mc_particles()

    # load pixels
    event_handler.load_pixels()

    # print some basic information about the current event
    info = """
-------------------------------------------------------------------------------
  Index %s, run %s, event %s
-------------------------------------------------------------------------------
""" % (idx, event_handler.run(), event_handler.event())
    print(info)

    # fetch MC generator particles
    mc_generator_initial_particles = event_handler.mc_generator_initial_particles()
    mc_generator_final_particles = event_handler.mc_generator_final_particles()

    # loop over MC generator initial particles
    for g_idx in range(event_handler.number_mc_generator_initial_particles()):
        mc_generator_particle = mc_generator_initial_particles[g_idx]
        mc_generator_particle_pdg_code = mc_generator_particle.pdg_code()
        mc_generator_particle_mass = mc_generator_particle.mass()
        mc_generator_particle_charge = mc_generator_particle.charge()
        mc_generator_particle_x = mc_generator_particle.x()
        mc_generator_particle_y = mc_generator_particle.y()
        mc_generator_particle_z = mc_generator_particle.z()
        mc_generator_particle_t = mc_generator_particle.t()
        mc_generator_particle_px = mc_generator_particle.px()
        mc_generator_particle_py = mc_generator_particle.py()
        mc_generator_particle_pz = mc_generator_particle.pz()
        mc_generator_particle_energy = mc_generator_particle.energy()

    # loop over MC generator final particles
    for g_idx in range(event_handler.number_mc_generator_final_particles()):
        mc_generator_particle = mc_generator_final_particles[g_idx]
        mc_generator_particle_pdg_code = mc_generator_particle.pdg_code()
        mc_generator_particle_mass = mc_generator_particle.mass()
        mc_generator_particle_charge = mc_generator_particle.charge()
        mc_generator_particle_x = mc_generator_particle.x()
        mc_generator_particle_y = mc_generator_particle.y()
        mc_generator_particle_z = mc_generator_particle.z()
        mc_generator_particle_t = mc_generator_particle.t()
        mc_generator_particle_px = mc_generator_particle.px()
        mc_generator_particle_py = mc_generator_particle.py()
        mc_generator_particle_pz = mc_generator_particle.pz()
        mc_generator_particle_energy = mc_generator_particle.energy()

    # fetch MC particles
    mc_particles = event_handler.mc_particles()

    # loop over MC particles
    for p_idx in range(event_handler.number_mc_particles()):
        mc_particle = mc_particles[p_idx]
        mc_particle_track_id = mc_particle.track_id()
        mc_particle_parent_track_id = mc_particle.parent_track_id()
        mc_particle_pdg_code = mc_particle.pdg_code()
        mc_particle_mass = mc_particle.mass()
        mc_particle_charge = mc_particle.charge()
        mc_particle_process_key = mc_particle.process_key()
        mc_particle_total_occupancy = mc_particle.total_occupancy()
        mc_particle_initial_x = mc_particle.initial_x()
        mc_particle_initial_y = mc_particle.initial_y()
        mc_particle_initial_z = mc_particle.initial_z()
        mc_particle_initial_t = mc_particle.initial_t()
        mc_particle_initial_px = mc_particle.initial_px()
        mc_particle_initial_py = mc_particle.initial_py()
        mc_particle_initial_pz = mc_particle.initial_pz()
        mc_particle_initial_energy = mc_particle.initial_energy()

    # fetch MC hits
    mc_hits = event_handler.mc_hits()

    # loop over MC hits
    for h_idx in range(event_handler.number_mc_hits()):
        mc_hit = mc_hits[h_idx]
        mc_hit_track_id = mc_hit.track_id()
        mc_hit_x = mc_hit.x()
        mc_hit_y = mc_hit.y()
        mc_hit_z = mc_hit.z()
        mc_hit_t = mc_hit.t()
        mc_hit_energy_deposit = mc_hit.energy_deposit()
        mc_hit_process_key = mc_hit.process_key()

    # fetch pixels
    pixels = event_handler.pixels()

    # loop over pixels
    for pix_idx in range(event_handler.number_pixels()):
        pixel = pixels[pix_idx]
        pixel_x = pixel.x()
        pixel_y = pixel.y()
        number_resets = pixel.number_resets()
        reset_array = pixel.reset_array()
        tslr_array = pixel.tslr_array()
        for reset_idx in range(number_resets):
            reset = reset_array[reset_idx]
            tslr = tslr_array[reset_idx]

    # wait for user input
    user_input = input()
    try:
        value = user_input.strip().lower()
        if value == "exit" or value == "quit" or value == "q":
            print("\nExiting...\n")
            break
    except:
        continue


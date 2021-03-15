#!/usr/bin/env python

# -----------------------------------------------------------------------------
#  EventHandler.py
#
#  EventHandler class for QPixAR
#   * Author: Everybody is an author!
#   * Creation date: 1 March 2021
# -----------------------------------------------------------------------------

import ROOT

from .FileHandler import FileHandler
from .MCGeneratorParticle import MCGeneratorParticle
from .MCParticle import MCParticle
from .MCHit import MCHit
from .Pixel import Pixel

class EventHandler:

    def __init__(self, file_handler):

        self.file_handler_ = file_handler
        self.run_ = 0
        self.entry_ = 0

        #----------------------------------------------------------------------
        # get event tree from ROOT file
        #----------------------------------------------------------------------

        self.tree_ = file_handler.file.Get('event_tree')
        self.number_entries_ = self.tree_.GetEntries()

        # initialize lists
        self.mc_generator_initial_particles_ = []
        self.mc_generator_final_particles_ = []
        self.mc_particles_ = []
        self.mc_hits_ = []
        self.pixels_ = []

        # initialize counters
        self.number_mc_generator_initial_particles_ = 0
        self.number_mc_generator_final_particles_ = 0
        self.number_mc_particles_ = 0
        self.number_mc_hits_ = 0
        self.number_pixels_ = 0

        # # get event number
        # self.event = self.array['event', self.entry]

        # # list of branches that we want to access
        # self.branches = [

        #     # event number
        #     'event',

        #     # MC particle information [Q_PIX_GEANT4]
        #     'particle_track_id', 'particle_pdg_code',
        #     'particle_mass', 'particle_initial_energy',

        #     # MC hit information [Q_PIX_GEANT4]
        #     'hit_energy_deposit', 'hit_track_id', 'hit_process_key',
        #     'hit_start_x', 'hit_start_y', 'hit_start_z', 'hit_start_t',
        #     'hit_end_x', 'hit_end_y', 'hit_end_z', 'hit_end_t',

        #     # pixel information [Q_PIX_RTD]
        #     'pixel_x', 'pixel_y', 'pixel_reset', 'pixel_tslr',

        # ]

    #--------------------------------------------------------------------------
    # set and get the entry of the event tree
    #--------------------------------------------------------------------------

    def get_entry(self, entry):

        # set entry index
        self.entry_ = entry

        # get entry
        self.tree_.GetEntry(entry)

        # get run number
        self.event_ = self.tree_.run

        # get event number
        self.event_ = self.tree_.event

        # reset lists
        self.mc_generator_initial_particles_ = []
        self.mc_generator_final_particles_ = []
        self.mc_particles_ = []
        self.mc_hits_ = []
        self.pixels_ = []

        # reset counters
        self.number_mc_generator_initial_particles_ = 0
        self.number_mc_generator_final_particles_ = 0
        self.number_mc_particles_ = 0
        self.number_mc_hits_ = 0
        self.number_pixels_ = 0

    #--------------------------------------------------------------------------
    # fetch MC generator particles from event tree
    #--------------------------------------------------------------------------

    def load_mc_generator_particles(self):

        #----------------------------------------------------------------------
        # fetch MC generator initial particles
        #----------------------------------------------------------------------

        self.mc_generator_initial_particles_ = []
        self.number_mc_generator_initial_particles_ = self.tree_.generator_initial_number_particles

        for idx in range(self.number_mc_generator_initial_particles_):

            mc_generator_particle = MCGeneratorParticle(
                state=0,
                pdg_code=self.tree_.generator_initial_particle_pdg_code[idx],
                mass=self.tree_.generator_initial_particle_mass[idx],
                charge=self.tree_.generator_initial_particle_charge[idx],
                x=self.tree_.generator_initial_particle_x[idx],
                y=self.tree_.generator_initial_particle_y[idx],
                z=self.tree_.generator_initial_particle_z[idx],
                t=self.tree_.generator_initial_particle_t[idx],
                px=self.tree_.generator_initial_particle_px[idx],
                py=self.tree_.generator_initial_particle_py[idx],
                pz=self.tree_.generator_initial_particle_pz[idx],
                energy=self.tree_.generator_initial_particle_energy[idx])

            self.mc_generator_initial_particles_.append(mc_generator_particle)

        #----------------------------------------------------------------------
        # fetch MC generator final particles
        #----------------------------------------------------------------------

        self.mc_generator_final_particles_ = []
        self.number_mc_generator_final_particles_ = self.tree_.generator_final_number_particles

        for idx in range(self.number_mc_generator_final_particles_):

            mc_generator_particle = MCGeneratorParticle(
                state=1,
                pdg_code=self.tree_.generator_final_particle_pdg_code[idx],
                mass=self.tree_.generator_final_particle_mass[idx],
                charge=self.tree_.generator_final_particle_charge[idx],
                x=self.tree_.generator_final_particle_x[idx],
                y=self.tree_.generator_final_particle_y[idx],
                z=self.tree_.generator_final_particle_z[idx],
                t=self.tree_.generator_final_particle_t[idx],
                px=self.tree_.generator_final_particle_px[idx],
                py=self.tree_.generator_final_particle_py[idx],
                pz=self.tree_.generator_final_particle_pz[idx],
                energy=self.tree_.generator_final_particle_energy[idx])

            self.mc_generator_final_particles_.append(mc_generator_particle)

    #--------------------------------------------------------------------------
    # fetch MC particles from event tree
    #--------------------------------------------------------------------------

    def load_mc_particles(self):

        self.mc_particles_ = []
        self.number_mc_particles_ = self.tree_.number_particles

        for idx in range(self.number_mc_particles_):

            mc_particle = MCParticle(
                track_id=self.tree_.particle_track_id[idx],
                parent_track_id=self.tree_.particle_parent_track_id[idx],
                pdg_code=self.tree_.particle_pdg_code[idx],
                mass=self.tree_.particle_mass[idx],
                charge=self.tree_.particle_charge[idx],
                process_key=self.tree_.particle_process_key[idx],
                total_occupancy=self.tree_.particle_total_occupancy[idx],
                initial_x=self.tree_.particle_initial_x[idx],
                initial_y=self.tree_.particle_initial_y[idx],
                initial_z=self.tree_.particle_initial_z[idx],
                initial_t=self.tree_.particle_initial_t[idx],
                initial_px=self.tree_.particle_initial_px[idx],
                initial_py=self.tree_.particle_initial_py[idx],
                initial_pz=self.tree_.particle_initial_pz[idx],
                initial_energy=self.tree_.particle_initial_energy[idx])

            self.mc_particles_.append(mc_particle)

    #--------------------------------------------------------------------------
    # fetch MC hits from event tree
    #--------------------------------------------------------------------------

    def load_mc_hits(self):

        self.mc_hits_ = []
        self.number_mc_hits_ = self.tree_.number_hits

        for idx in range(self.number_mc_hits_):

            mc_hit = MCHit(
                track_id=self.tree_.hit_track_id[idx],
                start_x=self.tree_.hit_start_x[idx],
                start_y=self.tree_.hit_start_y[idx],
                start_z=self.tree_.hit_start_z[idx],
                start_t=self.tree_.hit_start_t[idx],
                end_x=self.tree_.hit_end_x[idx],
                end_y=self.tree_.hit_end_y[idx],
                end_z=self.tree_.hit_end_z[idx],
                end_t=self.tree_.hit_end_t[idx],
                energy_deposit=self.tree_.hit_energy_deposit[idx],
                length=self.tree_.hit_length[idx],
                process_key=self.tree_.hit_process_key[idx])

            self.mc_hits_.append(mc_hit)

    #--------------------------------------------------------------------------
    # match MC hits to MC particles
    #--------------------------------------------------------------------------

    def match_mc_hits_to_mc_particles(self):

        for idx in range(self.number_mc_particles_):
            self.mc_particles_[idx].match_mc_hits(self.mc_hits_)

    #--------------------------------------------------------------------------
    # fetch pixels from event tree
    #--------------------------------------------------------------------------

    def load_pixels(self):

        self.pixels_ = []
        self.number_pixels_ = len(self.tree_.pixel_x)

        for idx in range(self.number_pixels_):

            pixel = Pixel(self.tree_.pixel_x[idx], self.tree_.pixel_y,
                          self.tree_.pixel_reset, self.tree_.pixel_tslr)

            self.pixels_.append(pixel)

    #--------------------------------------------------------------------------
    # return number of entries
    #--------------------------------------------------------------------------

    def number_entries(self):
        return self.number_entries_

    #--------------------------------------------------------------------------
    # return run number
    #--------------------------------------------------------------------------

    def run(self):
        return self.run_

    #--------------------------------------------------------------------------
    # return event number
    #--------------------------------------------------------------------------

    def event(self):
        return self.event_

    #--------------------------------------------------------------------------
    # return number of MC generator initial particles
    #--------------------------------------------------------------------------

    def number_mc_generator_initial_particles(self):
        return self.number_mc_generator_initial_particles_

    #--------------------------------------------------------------------------
    # return number of MC generator final particles
    #--------------------------------------------------------------------------

    def number_mc_generator_final_particles(self):
        return self.number_mc_generator_final_particles_

    #--------------------------------------------------------------------------
    # return number of MC particles
    #--------------------------------------------------------------------------

    def number_mc_particles(self):
        return self.number_mc_particles_

    #--------------------------------------------------------------------------
    # return number of MC hits
    #--------------------------------------------------------------------------

    def number_mc_hits(self):
        return self.number_mc_hits_

    #--------------------------------------------------------------------------
    # return number of pixels
    #--------------------------------------------------------------------------

    def number_pixels(self):
        return self.number_pixels_

    #--------------------------------------------------------------------------
    # return list of MC generator initial particles
    #--------------------------------------------------------------------------

    def mc_generator_initial_particles(self):
        return self.mc_generator_initial_particles_

    #--------------------------------------------------------------------------
    # return list of MC generator final particles
    #--------------------------------------------------------------------------

    def mc_generator_final_particles(self):
        return self.mc_generator_final_particles_

    #--------------------------------------------------------------------------
    # return list of MC particles
    #--------------------------------------------------------------------------

    def mc_particles(self):
        return self.mc_particles_

    #--------------------------------------------------------------------------
    # return list of MC hits
    #--------------------------------------------------------------------------

    def mc_hits(self):
        return self.mc_hits_

    #--------------------------------------------------------------------------
    # return list of pixels
    #--------------------------------------------------------------------------

    def pixels(self):
        return self.pixels_


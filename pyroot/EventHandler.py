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

class EventHandler:

    def __init__(self, file_handler):

        self.file_handler = file_handler
        self.run = 0
        self.entry = 0

        #----------------------------------------------------------------------
        # get event tree from ROOT file
        #----------------------------------------------------------------------

        self.tree = file_handler.file.Get('event_tree')
        self.number_entries = self.tree.GetEntries()

        # initialize lists
        self.mc_generator_initial_particles = []
        self.mc_generator_final_particles = []
        self.mc_particles = []
        self.mc_hits = []

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
        self.entry = entry

        # get entry
        self.tree.GetEntry(entry)

        # get run number
        self.event = self.tree.run

        # get event number
        self.event = self.tree.event

        # reset lists
        self.mc_generator_initial_particles = []
        self.mc_generator_final_particles = []
        self.mc_particles = []
        self.mc_hits = []

    #--------------------------------------------------------------------------
    # fetch MC generator particles from event tree
    #--------------------------------------------------------------------------

    def load_mc_generator_particles(self):

        #----------------------------------------------------------------------
        # fetch MC generator initial particles
        #----------------------------------------------------------------------

        self.mc_generator_initial_particles = []
        self.number_mc_generator_initial_particles = self.tree.generator_initial_number_particles

        for idx in range(self.number_mc_generator_initial_particles):

            mc_generator_particle = MCGeneratorParticle(
                state=0,
                pdg_code=self.tree.generator_initial_particle_pdg_code[idx],
                mass=self.tree.generator_initial_particle_mass[idx],
                charge=self.tree.generator_initial_particle_charge[idx],
                x=self.tree.generator_initial_particle_x[idx],
                y=self.tree.generator_initial_particle_y[idx],
                z=self.tree.generator_initial_particle_z[idx],
                t=self.tree.generator_initial_particle_t[idx],
                px=self.tree.generator_initial_particle_px[idx],
                py=self.tree.generator_initial_particle_py[idx],
                pz=self.tree.generator_initial_particle_pz[idx],
                energy=self.tree.generator_initial_particle_energy[idx])

            self.mc_generator_initial_particles.append(mc_generator_particle)

        #----------------------------------------------------------------------
        # fetch MC generator final particles
        #----------------------------------------------------------------------

        self.mc_generator_final_particles = []
        self.number_mc_generator_final_particles = self.tree.generator_final_number_particles

        for idx in range(self.number_mc_generator_final_particles):

            mc_generator_particle = MCGeneratorParticle(
                state=1,
                pdg_code=self.tree.generator_final_particle_pdg_code[idx],
                mass=self.tree.generator_final_particle_mass[idx],
                charge=self.tree.generator_final_particle_charge[idx],
                x=self.tree.generator_final_particle_x[idx],
                y=self.tree.generator_final_particle_y[idx],
                z=self.tree.generator_final_particle_z[idx],
                t=self.tree.generator_final_particle_t[idx],
                px=self.tree.generator_final_particle_px[idx],
                py=self.tree.generator_final_particle_py[idx],
                pz=self.tree.generator_final_particle_pz[idx],
                energy=self.tree.generator_final_particle_energy[idx])

            self.mc_generator_final_particles.append(mc_generator_particle)

    #--------------------------------------------------------------------------
    # fetch MC particles from event tree
    #--------------------------------------------------------------------------

    def load_mc_particles(self):

        self.mc_particles = []
        self.number_mc_particles = self.tree.number_particles

        for idx in range(self.number_mc_particles):

            mc_particle = MCParticle(
                track_id=self.tree.particle_track_id[idx],
                parent_track_id=self.tree.particle_parent_track_id[idx],
                pdg_code=self.tree.particle_pdg_code[idx],
                mass=self.tree.particle_mass[idx],
                charge=self.tree.particle_charge[idx],
                process_key=self.tree.particle_process_key[idx],
                total_occupancy=self.tree.particle_total_occupancy[idx],
                initial_x=self.tree.particle_initial_x[idx],
                initial_y=self.tree.particle_initial_y[idx],
                initial_z=self.tree.particle_initial_z[idx],
                initial_t=self.tree.particle_initial_t[idx],
                initial_px=self.tree.particle_initial_px[idx],
                initial_py=self.tree.particle_initial_py[idx],
                initial_pz=self.tree.particle_initial_pz[idx],
                initial_energy=self.tree.particle_initial_energy[idx])

            self.mc_particles.append(mc_particle)

    #--------------------------------------------------------------------------
    # fetch MC hits from event tree
    #--------------------------------------------------------------------------

    def load_mc_hits(self):

        self.mc_hits = []
        self.number_mc_hits = self.tree.number_hits

        for idx in range(self.number_mc_hits):

            mc_hit = MCHit(
                track_id=self.tree.hit_track_id[idx],
                start_x=self.tree.hit_start_x[idx],
                start_y=self.tree.hit_start_y[idx],
                start_z=self.tree.hit_start_z[idx],
                start_t=self.tree.hit_start_t[idx],
                end_x=self.tree.hit_end_x[idx],
                end_y=self.tree.hit_end_y[idx],
                end_z=self.tree.hit_end_z[idx],
                end_t=self.tree.hit_end_t[idx],
                energy_deposit=self.tree.hit_energy_deposit[idx],
                length=self.tree.hit_length[idx],
                process_key=self.tree.hit_process_key[idx])

            self.mc_hits.append(mc_hit)

    #--------------------------------------------------------------------------
    # match MC hits to MC particles
    #--------------------------------------------------------------------------

    def match_mc_hits_to_mc_particles(self):

        for idx in range(self.number_mc_particles):
            self.mc_particles[idx].match_mc_hits(self.mc_hits)


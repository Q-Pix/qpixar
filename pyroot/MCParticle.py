# -----------------------------------------------------------------------------
#  MCParticle.py
#
#  MCParticle class for QPixAR
#   * Author: Everybody is an author!
#   * Creation date: 1 March 2021
# -----------------------------------------------------------------------------

class MCParticle:

    def __init__(
        self, track_id, parent_track_id, pdg_code, mass, charge, process_key,
        total_occupancy, initial_x, initial_y, initial_z, initial_t,
        initial_px, initial_py, initial_pz, initial_energy):

        self.track_id = track_id
        self.parent_track_id = parent_track_id
        self.pdg_code = pdg_code
        self.mass = mass
        self.charge = charge
        self.process_key = process_key
        self.total_occupancy = total_occupancy
        self.initial_x = initial_x
        self.initial_y = initial_y
        self.initial_z = initial_z
        self.initial_t = initial_t
        self.initial_px = initial_px
        self.initial_py = initial_py
        self.initial_pz = initial_pz
        self.initial_energy = initial_energy

        self.mc_hit_indices = []

    def add_mc_hit_index(self, index):

        self.mc_hit_indices.append(index)

    def match_mc_hits(self, mc_hits):

        self.mc_hit_indices = []

        for idx in range(len(mc_hits)):
            if mc_hits[idx].track_id == self.track_id:
                self.mc_hit_indices.append(idx)


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

        self.track_id_ = track_id
        self.parent_track_id_ = parent_track_id
        self.pdg_code_ = pdg_code
        self.mass_ = mass
        self.charge_ = charge
        self.process_key_ = process_key
        self.total_occupancy_ = total_occupancy
        self.initial_x_ = initial_x
        self.initial_y_ = initial_y
        self.initial_z_ = initial_z
        self.initial_t_ = initial_t
        self.initial_px_ = initial_px
        self.initial_py_ = initial_py
        self.initial_pz_ = initial_pz
        self.initial_energy_ = initial_energy

        self.mc_hit_indices_ = []

    def add_mc_hit_index(self, index):

        self.mc_hit_indices_.append(index)

    def match_mc_hits(self, mc_hits):

        self.mc_hit_indices_ = []

        for idx in range(len(mc_hits)):
            if mc_hits[idx].track_id == self.track_id_:
                self.mc_hit_indices_.append(idx)

    def track_id(self):
        return self.track_id_

    def parent_track_id(self):
        return self.parent_track_id_

    def pdg_code(self):
        return self.pdg_code_

    def mass(self):
        return self.mass_

    def charge(self):
        return self.charge_

    def process_key(self):
        return self.process_key_

    def total_occupancy(self):
        return self.total_occupancy_

    def initial_x(self):
        return self.initial_x_

    def initial_y(self):
        return self.initial_y_

    def initial_z(self):
        return self.initial_z_

    def initial_t(self):
        return self.initial_t_

    def initial_px(self):
        return self.initial_px_

    def initial_py(self):
        return self.initial_py_

    def initial_pz(self):
        return self.initial_pz_

    def initial_energy(self):
        return self.initial_energy_

    def mc_hit_indices(self):
        return self.mc_hit_indices_


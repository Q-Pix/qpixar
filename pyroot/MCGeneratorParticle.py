# -----------------------------------------------------------------------------
#  MCGeneratorParticle.py
#
#  MCGeneratorParticle class for QPixAR
#   * Author: Everybody is an author!
#   * Creation date: 2 March 2021
# -----------------------------------------------------------------------------

class MCGeneratorParticle:

    def __init__(
        self, state, pdg_code, mass, charge, x, y, z, t, px, py, pz, energy):

        self.state = state
        self.pdg_code = pdg_code
        self.mass = mass
        self.charge = charge
        self.x = x
        self.y = y
        self.z = z
        self.t = t
        self.px = px
        self.py = py
        self.pz = pz
        self.energy = energy

        self.mc_hit_indices = []


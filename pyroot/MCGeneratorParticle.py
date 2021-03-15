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

        self.state_ = state
        self.pdg_code_ = pdg_code
        self.mass_ = mass
        self.charge_ = charge
        self.x_ = x
        self.y_ = y
        self.z_ = z
        self.t_ = t
        self.px_ = px
        self.py_ = py
        self.pz_ = pz
        self.energy_ = energy

    def state(self):
        return self.state_

    def pdg_code(self):
        return self.pdg_code_

    def mass(self):
        return self.mass_

    def charge(self):
        return self.charge_

    def x(self):
        return self.x_

    def y(self):
        return self.y_

    def z(self):
        return self.z_

    def t(self):
        return self.t_

    def px(self):
        return self.px_

    def py(self):
        return self.py_

    def pz(self):
        return self.pz_

    def energy(self):
        return self.energy_


# -----------------------------------------------------------------------------
#  MCHit.py
#
#  MCHit class for QPixAR
#   * Author: Everybody is an author!
#   * Creation date: 1 March 2021
# -----------------------------------------------------------------------------

class MCHit:

    def __init__(
        self, track_id, start_x, start_y, start_z, start_t, end_x, end_y,
        end_z, end_t, energy_deposit, length, process_key):

        self.track_id_ = track_id
        self.start_x_ = start_x
        self.start_y_ = start_y
        self.start_z_ = start_z
        self.start_t_ = start_t
        self.end_x_ = end_x
        self.end_y_ = end_y
        self.end_z_ = end_z
        self.end_t_ = end_t
        self.energy_deposit_ = energy_deposit
        self.process_key_ = process_key

    def track_id(self):
        return self.track_id_

    def start_x(self):
        return self.start_x_

    def start_y(self):
        return self.start_y_

    def start_z(self):
        return self.start_z_

    def start_t(self):
        return self.start_t_

    def end_x(self):
        return self.end_x_

    def end_y(self):
        return self.end_y_

    def end_z(self):
        return self.end_z_

    def end_t(self):
        return self.end_t_

    def energy_deposit(self):
        return self.energy_deposit_

    def process_key(self):
        return self.process_key_

    def x(self):
        return self.end_x_

    def y(self):
        return self.end_y_

    def z(self):
        return self.end_z_

    def t(self):
        return self.end_t_


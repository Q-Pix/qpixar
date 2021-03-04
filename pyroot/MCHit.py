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

        self.track_id = track_id
        self.start_x = start_x
        self.start_y = start_y
        self.start_z = start_z
        self.start_t = start_t
        self.end_x = end_x
        self.end_y = end_y
        self.end_z = end_z
        self.end_t = end_t
        self.energy_deposit = energy_deposit
        self.process_key = process_key


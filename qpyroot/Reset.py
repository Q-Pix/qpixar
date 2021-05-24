# -----------------------------------------------------------------------------
#  Reset.py
#
#  Reset class for QPixAR
#   * Author: Everybody is an author!
#   * Creation date: 22 May 2021
# -----------------------------------------------------------------------------

class Reset:

    def __init__(self, time, tslr,
                 mc_track_id_array=None, mc_weight_array=None):

        self.time_ = time
        self.tslr_ = tslr

        self.mc_track_ids_ = mc_track_id_array
        self.mc_weights_ = mc_weight_array

    def time(self):
        return self.time_

    def tslr(self):
        return self.tslr_

    def mc_track_ids(self):
        return self.mc_track_ids_

    def mc_weights(self):
        return self.mc_weights_


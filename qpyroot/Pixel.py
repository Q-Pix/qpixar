# -----------------------------------------------------------------------------
#  Pixel.py
#
#  Pixel class for QPixAR
#   * Author: Everybody is an author!
#   * Creation date: 3 March 2021
# -----------------------------------------------------------------------------

from .Reset import Reset

class Pixel:

    def __init__(self, pixel_x, pixel_y, pixel_reset, pixel_tslr,
                 pixel_reset_mc_track_id=None, pixel_reset_mc_weight=None):

        self.x_ = pixel_x
        self.y_ = pixel_y
        self.number_resets_ = len(pixel_reset)
        self.reset_array_ = pixel_reset
        self.tslr_array_ = pixel_tslr
        self.mc_track_id_array_ = pixel_reset_mc_track_id
        self.mc_weight_array_ = pixel_reset_mc_weight
        self.resets_ = []

    def x(self):
        return self.x_

    def y(self):
        return self.y_

    def number_resets(self):
        return self.number_resets_

    def reset_array(self):
        return self.reset_array_

    def tslr_array(self):
        return self.tslr_array_

    def load_resets(self):

        self.resets_ = []

        for idx in range(self.number_resets_):

            try:
                iter(self.mc_track_id_array_)
                iter(self.mc_weight_array_)

            except TypeError:
                reset = Reset(self.reset_array_[idx], self.tslr_array_[idx])

            else:
                reset = Reset(self.reset_array_[idx], self.tslr_array_[idx],
                              self.mc_track_id_array_[idx],
                              self.mc_weight_array_[idx])

                # print("else")

            # if (self.mc_track_id_array_ is None and
            #     self.mc_weight_array_ is None):

            #     reset = Reset(self.reset_array_[idx], self.tslr_array_[idx])

            # else:

            #     reset = Reset(self.reset_array_[idx], self.tslr_array_[idx],
            #                   self.mc_track_id_array_[idx],
            #                   self.mc_weight_array_[idx])

            self.resets_.append(reset)

    def resets(self):
        return self.resets_


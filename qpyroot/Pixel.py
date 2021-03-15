# -----------------------------------------------------------------------------
#  Pixel.py
#
#  Pixel class for QPixAR
#   * Author: Everybody is an author!
#   * Creation date: 3 March 2021
# -----------------------------------------------------------------------------

class Pixel:

    def __init__(self, pixel_x, pixel_y, pixel_reset, pixel_tslr):

        self.x_ = pixel_x
        self.y_ = pixel_y
        self.number_resets_ = len(pixel_reset)
        self.reset_array_ = pixel_reset
        self.tslr_array_ = pixel_tslr

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


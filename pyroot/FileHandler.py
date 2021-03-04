#!/usr/bin/env python

# -----------------------------------------------------------------------------
#  FileHandler.py
#
#  FileHandler class for QPixAR
#   * Author: Everybody is an author!
#   * Creation date: 28 February 2021
# -----------------------------------------------------------------------------

import ROOT

class FileHandler:

    def __init__(self, file_path):

        self.file_path = file_path
        self.file = ROOT.TFile.Open(self.file_path, 'read')

        #------------------------------------------------------------------
        # get metadata from ROOT file
        #------------------------------------------------------------------

        self.metadata = self.file.Get('metadata')

        # there should only be one entry per file
        for idx in range(self.metadata.GetEntries()):

            self.metadata.GetEntry(idx)

            self.detector_length_x = self.metadata.detector_length_x  # cm
            self.detector_length_y = self.metadata.detector_length_y  # cm
            self.detector_length_z = self.metadata.detector_length_z  # cm

            self.drift_velocity = self.metadata.drift_velocity  # cm/ns
            self.longitudinal_diffusion = self.metadata.longitudinal_diffusion  # cm^2/ns
            self.transverse_diffusion = self.metadata.transverse_diffusion  # cm^2/ns
            self.electron_lifetime = self.metadata.electron_lifetime  # ns
            self.readout_dimensions = self.metadata.readout_dimensions  # cm
            self.pixel_size = self.metadata.pixel_size  # cm
            self.reset_threshold = self.metadata.reset_threshold  # electrons
            self.sample_time = self.metadata.sample_time  # ns
            self.buffer_window = self.metadata.buffer_window  # ns
            self.dead_time = self.metadata.dead_time  # ns
            self.charge_loss = self.metadata.charge_loss  # 0 is off, 1 is on


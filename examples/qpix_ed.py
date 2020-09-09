#!/usr/bin/env python

# -----------------------------------------------------------------------------
#  qpix_ed.py
#
#  Example: A simple event display for Q-Pix
#   * Author: Everybody is an author!
#   * Creation date: 1 September 2020
# -----------------------------------------------------------------------------

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
import uproot

matplotlib.rcParams.update({'font.size': 8})

# parse arguments from command
parser = argparse.ArgumentParser(description="pixel event display")
parser.add_argument("file", type=str, default=None,
                    help="path to ROOT file")
parser.add_argument('--3d', dest='pix3d', default=False, action='store_true',
                    help="start with 3D Q-Pix view")

args = parser.parse_args()
file_path = str(args.file)
pix3d = args.pix3d

with uproot.open(file_path) as f:

    #--------------------------------------------------------------------------
    # get metadata from ROOT file
    #--------------------------------------------------------------------------

    metadata = f['metadata']

    # get detector dimensions
    detector_length_x = metadata.array('detector_length_x')[0]  # cm
    detector_length_y = metadata.array('detector_length_y')[0]  # cm
    detector_length_z = metadata.array('detector_length_z')[0]  # cm

    # get parameters used in Q_PIX_RTD
    drift_velocity = metadata.array('drift_velocity')[0]  # cm/ns
    longitudinal_diffusion = metadata.array('longitudinal_diffusion')[0]  # cm^2/ns
    transverse_diffusion = metadata.array('transverse_diffusion')[0]  # cm^2/ns
    electron_lifetime = metadata.array('electron_lifetime')[0]  # ns
    readout_dimensions = metadata.array('readout_dimensions')[0]  # cm
    pixel_size = metadata.array('pixel_size')[0]  # cm
    reset_threshold = metadata.array('reset_threshold')[0]  # electrons
    sample_time = metadata.array('sample_time')[0]  # ns
    buffer_window = metadata.array('buffer_window')[0]  # ns
    dead_time = metadata.array('dead_time')[0]  # ns
    charge_loss = metadata.array('charge_loss')[0]  # 0 is off, 1 is on

    #--------------------------------------------------------------------------
    # get event tree from ROOT file
    #--------------------------------------------------------------------------

    tree = f['event_tree']

    # list of branches to access
    branches = [

        # event number
        'event',

        # generator particle information [Q_PIX_GEANT4]
        'generator_initial_number_particles',
        'generator_initial_particle_pdg_code',
        'generator_initial_particle_energy',
        'generator_initial_particle_mass',
        'generator_final_number_particles',
        'generator_final_particle_pdg_code',
        'generator_final_particle_energy',
        'generator_final_particle_mass',

        # MC particle information [Q_PIX_GEANT4]
        'particle_track_id', 'particle_pdg_code',
        'particle_mass', 'particle_initial_energy',

        # MC hit information [Q_PIX_GEANT4]
        'hit_energy_deposit', 'hit_track_id', 'hit_process_key',
        'hit_start_x', 'hit_start_y', 'hit_start_z', 'hit_start_t',
        'hit_end_x', 'hit_end_y', 'hit_end_z', 'hit_end_t',

        # pixel information [Q_PIX_RTD]
        'pixel_x', 'pixel_y', 'pixel_reset', 'pixel_tslr',

    ]

    #--------------------------------------------------------------------------
    # prompt message
    #--------------------------------------------------------------------------

    msg = """
The GUI window can be closed by pressing the q key. This terminal window will
lose focus when the GUI window appears.

Press enter to continue... """

    input(msg)

    #--------------------------------------------------------------------------
    # iterate through the event tree
    #--------------------------------------------------------------------------

    for arrays in tree.iterate(branches=branches, namedecode='utf-8'):

        # get event number array
        event_array = arrays['event']

        # get generator particle arrays
        gen_ip_number_array   = arrays['generator_initial_number_particles']
        gen_ip_pdg_code_array = arrays['generator_initial_particle_pdg_code']
        gen_ip_energy_array   = arrays['generator_initial_particle_energy']
        gen_ip_mass_array     = arrays['generator_initial_particle_mass']
        gen_fp_number_array   = arrays['generator_final_number_particles']
        gen_fp_pdg_code_array = arrays['generator_final_particle_pdg_code']
        gen_fp_energy_array   = arrays['generator_final_particle_energy']
        gen_fp_mass_array     = arrays['generator_final_particle_mass']

        # get MC hit arrays
        hit_end_x_array = arrays['hit_end_x']
        hit_end_y_array = arrays['hit_end_y']
        hit_end_z_array = arrays['hit_end_z']
        hit_end_t_array = arrays['hit_end_t']
        hit_energy_deposit_array = arrays['hit_energy_deposit']

        # get pixel arrays
        pixel_x_array = arrays['pixel_x']
        pixel_y_array = arrays['pixel_y']
        pixel_reset_array = arrays['pixel_reset']
        pixel_tslr_array = arrays['pixel_tslr']

        # get number of events
        number_events = len(event_array)

        #----------------------------------------------------------------------
        # iterate over events
        #----------------------------------------------------------------------

        idx = 0
        while idx < number_events:

            #------------------------------------------------------------------
            # fetch event information
            #------------------------------------------------------------------

            # get event number
            event = event_array[idx]

            # get pixel information for event
            pixel_x = pixel_x_array[idx]
            pixel_y = pixel_y_array[idx]
            pixel_reset = pixel_reset_array[idx]
            pixel_tslr = pixel_tslr_array[idx]

            # get number of pixels in event
            number_pixels = len(pixel_x)

            # get MC hit information for event
            hit_end_x = hit_end_x_array[idx]
            hit_end_y = hit_end_y_array[idx]
            hit_end_z = hit_end_z_array[idx]
            hit_end_t = hit_end_t_array[idx]
            hit_energy_deposit = hit_energy_deposit_array[idx]

            # get number of hits in event
            number_hits = len(hit_end_x)

            # get generator particles in event
            gen_ip_number   = gen_ip_number_array[idx]
            gen_ip_pdg_code = gen_ip_pdg_code_array[idx]
            gen_ip_energy   = gen_ip_energy_array[idx]
            gen_ip_mass     = gen_ip_mass_array[idx]
            gen_fp_number   = gen_fp_number_array[idx]
            gen_fp_pdg_code = gen_fp_pdg_code_array[idx]
            gen_fp_energy   = gen_fp_energy_array[idx]
            gen_fp_mass     = gen_fp_mass_array[idx]

            #------------------------------------------------------------------
            # print event number
            #------------------------------------------------------------------

            print("\nEvent: {} / {}\n".format(idx, number_events))

            #------------------------------------------------------------------
            # print information of generator particles
            #------------------------------------------------------------------

            print("----------------------------------------------------------"
                  "\nNumber of initial generator particles: {}\n"
                  "----------------------------------------------------------"
                  "\n"
                  .format(gen_ip_number))

            for p in range(gen_ip_number):
                msg = """Initial particle #{}
  PDG code:       {}
  Energy:         {} MeV
  Kinetic energy: {} MeV
""".format(p+1, gen_ip_pdg_code[p], gen_ip_energy[p], gen_ip_energy[p]-gen_ip_mass[p])

                print(msg)

            print("----------------------------------------------------------"
                  "\nNumber of final generator particles: {}\n"
                  "----------------------------------------------------------"
                  "\n"
                  .format(gen_fp_number))

            for p in range(gen_fp_number):
                msg = """Final particle #{}
  PDG code:       {}
  Energy:         {} MeV
  Kinetic energy: {} MeV
""".format(p+1, gen_fp_pdg_code[p], gen_fp_energy[p], gen_fp_energy[p]-gen_fp_mass[p])

                print(msg)

            print("----------------------------------------------------------")

            #------------------------------------------------------------------
            # print event number
            #------------------------------------------------------------------

            print("\nEvent: {} / {}\n".format(idx, number_events))

            #------------------------------------------------------------------
            # initialize figure and gridspec
            #------------------------------------------------------------------

            fig = plt.figure(figsize=(12, 7))
            gs = fig.add_gridspec(nrows=3, ncols=4)

            #------------------------------------------------------------------

            # xyz plot of MC hits
            ax_hit_xyz = fig.add_subplot(gs[0, 0], projection='3d')

            #------------------------------------------------------------------

            # xy plot of MC hits
            gs_hit_xy = gridspec.GridSpecFromSubplotSpec(
                1, 3, subplot_spec=gs[0, 1], wspace=0.1, hspace=0.1)
            ax_hit_xy = fig.add_subplot(gs_hit_xy[1:])

            #------------------------------------------------------------------

            # xz and yz plot of MC hits
            gs_hit_xz_yz = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=gs[0, 2:], wspace=0.1, hspace=0.1)
            ax_hit_xz = fig.add_subplot(gs_hit_xz_yz[0])
            ax_hit_yz = fig.add_subplot(gs_hit_xz_yz[1])

            #------------------------------------------------------------------

            # aliases for MC hits
            hit_x, hit_y, hit_z = hit_end_x, hit_end_y, hit_end_z

            # plot MC hits
            ax_hit_xyz.scatter(hit_z, hit_x, hit_y, alpha=0.75, s=1)
            ax_hit_xy.scatter(hit_x, hit_y, alpha=0.75, s=1)
            ax_hit_xz.scatter(hit_z, hit_x, alpha=0.75, s=1)
            ax_hit_yz.scatter(hit_z, hit_y, alpha=0.75, s=1)

            #------------------------------------------------------------------

            # labels, axis limits, and aspect ratios of plots of MC hits

            ax_hit_xyz.set_title('g4')

            ax_hit_xy.text(0.95, 1.05, 'g4',
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax_hit_xy.transAxes)

            ax_hit_xz.text(0.975, 1.1, 'g4',
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax_hit_xz.transAxes)

            ax_hit_xy.set_xlabel('x [cm]')
            ax_hit_xy.set_ylabel('y [cm]')
            ax_hit_xy.set_xlim(0, detector_length_x)
            ax_hit_xy.set_ylim(0, detector_length_y)
            ax_hit_xy.set_aspect('equal')

            ax_hit_xyz.set_xlabel('z [cm]')
            ax_hit_xyz.set_ylabel('x [cm]')
            ax_hit_xyz.set_zlabel('y [cm]')

            ax_hit_xyz.set_xlim(0, detector_length_z)
            ax_hit_xyz.set_ylim(0, detector_length_x)
            ax_hit_xyz.set_zlim(0, detector_length_y)

            ax_hit_xz.set_xticklabels([])
            ax_hit_xz.set_ylabel('x [cm]')
            ax_hit_xz.set_xlim(0, detector_length_z)
            ax_hit_xz.set_ylim(0, detector_length_x)
            ax_hit_xz.set_aspect('equal')

            ax_hit_yz.set_xlabel('z [cm]')
            ax_hit_yz.set_ylabel('y [cm]')
            ax_hit_yz.set_xlim(0, detector_length_z)
            ax_hit_yz.set_ylim(0, detector_length_y)
            ax_hit_yz.set_aspect('equal')

            #------------------------------------------------------------------

            # link axes
            ax_hit_xz.get_shared_x_axes().join(ax_hit_xz, ax_hit_yz)

            #------------------------------------------------------------------

            # pixel dimensions
            nx_pix = int(detector_length_x/pixel_size)
            ny_pix = int(detector_length_y/pixel_size)

            # dummy arrays/lists
            a = np.zeros((nx_pix, ny_pix))
            b = list(np.empty((1, 4)))

            # iterate through pixels and fill dummy arrays/lists
            for px in range(number_pixels):
                number_resets = len(pixel_reset[px])
                a[pixel_y[px], pixel_x[px]] += number_resets
                for rst in range(number_resets):
                    reset = pixel_reset[px][rst]
                    tslr = pixel_tslr[px][rst]
                    b.append([pixel_x[px], pixel_y[px], reset, tslr])
                    # a[pixel_y[px], pixel_x[px]] += 1/tslr

            # convert list to array
            if len(b) > 1:
                # ignore dummy element
                b = np.array(b[1:], dtype=int)
            else:
                b = np.array(b, dtype=int)

            # convert to physical units
            pix_x = b[:, 0] * pixel_size  # cm
            pix_y = b[:, 1] * pixel_size  # cm
            pix_z = b[:, 2] * drift_velocity  # cm
            pix_tslr = b[:, 3]  # ns

            #------------------------------------------------------------------

            if not pix3d:

                #--------------------------------------------------------------

                # xy plot of pixels
                ax_pix_xy = fig.add_subplot(gs[1:3, 0:2])

                # xz and yz plots of pixels
                gs_pix_xz_yz = gridspec.GridSpecFromSubplotSpec(
                    2, 1, subplot_spec=gs[1:, 2:], wspace=0.1, hspace=0.1)
                ax_pix_xz = fig.add_subplot(gs_pix_xz_yz[0])
                ax_pix_yz = fig.add_subplot(gs_pix_xz_yz[1])

                #--------------------------------------------------------------

                # pixel bins
                pix_x_bins = np.linspace(0, detector_length_x, nx_pix+1)
                pix_y_bins = np.linspace(0, detector_length_y, ny_pix+1)

                # for display purposes only
                nz_pix = int(detector_length_z/pixel_size)
                pix_z_bins = np.linspace(0, detector_length_z, nz_pix+1)

                # plot pixels
                im_pix_xy = ax_pix_xy.hist2d(pix_x, pix_y, bins=(pix_x_bins, pix_y_bins))[3]
                im_pix_xz = ax_pix_xz.hist2d(pix_z, pix_x, bins=(pix_z_bins, pix_x_bins))[3]
                im_pix_yz = ax_pix_yz.hist2d(pix_z, pix_y, bins=(pix_z_bins, pix_y_bins))[3]

                # link axes
                ax_pix_xy.get_shared_x_axes().join(ax_pix_xy, ax_hit_xy)
                ax_pix_xy.get_shared_y_axes().join(ax_pix_xy, ax_hit_xy)

                ax_pix_xz.get_shared_x_axes().join(ax_pix_xz, ax_hit_xz)
                ax_pix_xz.get_shared_y_axes().join(ax_pix_xz, ax_hit_xz)
                ax_pix_xz.get_shared_x_axes().join(ax_pix_xz, ax_pix_yz)

                ax_pix_yz.get_shared_x_axes().join(ax_pix_yz, ax_hit_yz)
                ax_pix_yz.get_shared_y_axes().join(ax_pix_yz, ax_hit_yz)

                #--------------------------------------------------------------

                # color bars

                cb_pix_xy = fig.colorbar(im_pix_xy, ax=ax_pix_xy, shrink=0.8)
                cb_pix_xy.set_label('number of resets')

                cb_pix_xz = fig.colorbar(im_pix_xz, ax=ax_pix_xz, shrink=0.8)
                cb_pix_xz.set_label('number of resets')

                cb_pix_yz = fig.colorbar(im_pix_yz, ax=ax_pix_yz, shrink=0.8)
                cb_pix_yz.set_label('number of resets')

                #--------------------------------------------------------------

                # labels and aspect ratios of pixel plots

                ax_pix_xy.set_title('qpix')

                # ax_pix_xy.text(0.965, 1.025, 'qpix',
                #     horizontalalignment='center',
                #     verticalalignment='center',
                #     transform=ax_pix_xy.transAxes)

                ax_pix_xz.text(0.96, 1.05, 'qpix',
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=ax_pix_xz.transAxes)

                ax_pix_xy.set_xlabel('x [cm]')
                ax_pix_xy.set_ylabel('y [cm]')
                # ax_pix_xy.set_aspect('equal')

                ax_pix_xz.set_xticklabels([])
                ax_pix_xz.set_ylabel('x [cm]')
                # ax_pix_xz.set_aspect('equal')
                      
                ax_pix_yz.set_xlabel('z [cm]')
                ax_pix_yz.set_ylabel('y [cm]')
                # ax_pix_yz.set_aspect('equal')

                #--------------------------------------------------------------

            else:

                #--------------------------------------------------------------

                # color map

                # colormap = plt.cm.jet
                colormap = plt.cm.viridis
                normalize = matplotlib.colors.Normalize(vmin=0, vmax=50)
                scalarmappable = matplotlib.cm.ScalarMappable(norm=normalize, cmap=colormap)

                #--------------------------------------------------------------

                # plot pixels in 3d

                ax_pix_xyz = fig.add_subplot(gs[1:, :], projection='3d')
                dq = np.empty(pix_tslr.shape)
                if len(pix_tslr) > 1:
                    dq = reset_threshold/pix_tslr
                sc_pix_xyz = ax_pix_xyz.scatter(
                    pix_z, pix_x, pix_y, alpha=1.0, s=dq/1000., c=dq,
                    cmap=colormap, norm=normalize)

                cb_pix_xyz = fig.colorbar(sc_pix_xyz, ax=ax_pix_xyz, shrink=0.8)
                cb_pix_xyz.set_label('[electrons/ns]')

                #--------------------------------------------------------------

                # labels of pixel plot

                ax_pix_xyz.set_title('qpix')

                ax_pix_xyz.set_xlabel('z [cm]')
                ax_pix_xyz.set_ylabel('x [cm]')
                ax_pix_xyz.set_zlabel('y [cm]')

                ax_pix_xyz.set_xlim(0, detector_length_z)
                ax_pix_xyz.set_ylim(0, detector_length_x)
                ax_pix_xyz.set_zlim(0, detector_length_y)

            #------------------------------------------------------------------

            # use tight layout
            fig.tight_layout()

            # display figure
            plt.show()

            # clear axes and figures
            plt.cla()
            plt.clf()
            plt.close()

            #------------------------------------------------------------------

            # prompt message
            msg = """
You may...

  (1) type in an event number to go to that event;
  (2) type in "3d" toggle between 3D and 2D Q-Pix view;
  (3) type in "exit", "quit", or "q" to terminate the program; or
  (4) type in anything else or press enter to go to the next event.

What will it be? """

            user_input = input(msg)

            try:
                value = int(user_input)
                idx = value
                if idx < 0 or idx >= number_events:
                    print("\nEvent number out of range! Exiting...\n")
                    break
            except ValueError:
                value = user_input.strip().lower()

                if value == "exit" or value == "quit" or value == "q":
                    print("\nExiting...\n")
                    break

                elif value == "3d" or value == "2d":
                    pix3d = not pix3d

                    if pix3d:
                        print("\nSwitching to 3D view...")
                    else:
                        print("\nSwitching to 2D view...")

                else:
                    idx += 1

            #------------------------------------------------------------------


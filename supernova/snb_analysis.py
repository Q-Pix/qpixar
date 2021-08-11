#!/usr/bin/env python

# -----------------------------------------------------------------------------
#  snb_analysis.py
#
#  Example for reading from ROOT files
#   * Author: Everybody is an author!
#   * Creation date: 2 August 2021
# -----------------------------------------------------------------------------

import sys
import numpy as np
import awkward as ak
import uproot

np.random.seed(2)

# input_dir = "/n/home02/jh/repos/qpixrtd/EXAMPLE/output"
input_dir = "/Volumes/seagate/work/qpix/supernova_test/s_bg_test"

file_array = [

    # buffer 0
    {
        # signal files
        # input_dir + "/SUPERNOVA_NEUTRINO_RTD.root" : 1,
        input_dir + "/SUPERNOVA_NEUTRINO_RTD_1000_events.root" : 1,
        # background files
        input_dir + "/SUPERNOVA_BACKGROUND_RTD_0000.root" : 0,
        input_dir + "/SUPERNOVA_BACKGROUND_RTD_0001.root" : 0,
        input_dir + "/SUPERNOVA_BACKGROUND_RTD_0002.root" : 0,
        input_dir + "/SUPERNOVA_BACKGROUND_RTD_0003.root" : 0,
        input_dir + "/SUPERNOVA_BACKGROUND_RTD_0004.root" : 0,
        input_dir + "/SUPERNOVA_BACKGROUND_RTD_0005.root" : 0,
        input_dir + "/SUPERNOVA_BACKGROUND_RTD_0006.root" : 0,
        # input_dir + "/SUPERNOVA_BACKGROUND_RTD_0007.root" : 0,
        # input_dir + "/SUPERNOVA_BACKGROUND_RTD_0008.root" : 0,
    },

    # # buffer 1
    # {
    #     # signal files
    #     input_dir + "/SUPERNOVA_NEUTRINO_RTD.root" : 1,
    #     input_dir + "/SUPERNOVA_NEUTRINO_RTD_1000_events.root" : 1,
    #     # background files
    #     input_dir + "/SUPERNOVA_BACKGROUND_RTD_0000.root" : 0,
    #     input_dir + "/SUPERNOVA_BACKGROUND_RTD_0001.root" : 0,
    #     input_dir + "/SUPERNOVA_BACKGROUND_RTD_0002.root" : 0,
    #     input_dir + "/SUPERNOVA_BACKGROUND_RTD_0003.root" : 0,
    #     input_dir + "/SUPERNOVA_BACKGROUND_RTD_0004.root" : 0,
    #     input_dir + "/SUPERNOVA_BACKGROUND_RTD_0005.root" : 0,
    #     input_dir + "/SUPERNOVA_BACKGROUND_RTD_0006.root" : 0,
    #     input_dir + "/SUPERNOVA_BACKGROUND_RTD_0007.root" : 0,
    # },

]

for buffer_idx in range(len(file_array)):

    print("Buffer index:", buffer_idx)
    files = file_array[buffer_idx]
    # print(files)
    # print(len(files))
    # print(files.items())

    signal_x = []
    signal_y = []
    signal_t = []

    background_x = []
    background_y = []
    background_t = []

    for file_path, signal_flag in files.items():
        print(file_path, signal_flag)

        with uproot.open(file_path) as f:

            #------------------------------------------------------------------
            # get metadata from ROOT file
            #------------------------------------------------------------------

            metadata = f['metadata']

            # get detector dimensions
            detector_length_x = metadata['detector_length_x'].array()[0]  # cm
            detector_length_y = metadata['detector_length_y'].array()[0]  # cm
            detector_length_z = metadata['detector_length_z'].array()[0]  # cm

            # get parameters used in Q_PIX_RTD
            drift_velocity = metadata['drift_velocity'].array()[0]  # cm/s
            longitudinal_diffusion = metadata['longitudinal_diffusion'].array()[0]  # cm^2/s
            transverse_diffusion = metadata['transverse_diffusion'].array()[0]  # cm^2/s
            electron_lifetime = metadata['electron_lifetime'].array()[0]  # s
            readout_dimensions = metadata['readout_dimensions'].array()[0]  # cm
            pixel_size = metadata['pixel_size'].array()[0]  # cm
            reset_threshold = metadata['reset_threshold'].array()[0]  # electrons
            sample_time = metadata['sample_time'].array()[0]  # s
            buffer_window = metadata['buffer_window'].array()[0]  # s
            dead_time = metadata['dead_time'].array()[0]  # s
            charge_loss = metadata['charge_loss'].array()[0]  # 0 is off, 1 is on

            #------------------------------------------------------------------
            # get event tree from ROOT file
            #------------------------------------------------------------------

            tree = f['event_tree']

            # list of branches that we want to access
            branches = [

                # event number
                'event',

                # MC particle information [Q_PIX_GEANT4]
                'particle_track_id', 'particle_pdg_code',
                'particle_mass', 'particle_initial_energy',

                # MC hit information [Q_PIX_GEANT4]
                'hit_energy_deposit', 'hit_track_id', 'hit_process_key',
                'hit_start_x', 'hit_start_y', 'hit_start_z', 'hit_start_t',
                'hit_end_x', 'hit_end_y', 'hit_end_z', 'hit_end_t',

                # pixel information [Q_PIX_RTD]
                'pixel_x', 'pixel_y', 'pixel_reset', 'pixel_tslr',
                'pixel_reset_truth_track_id', 'pixel_reset_truth_weight',

            ]

            #------------------------------------------------------------------
            # iterate through the event tree
            #------------------------------------------------------------------

            for arrays in tree.iterate(filter_name=branches):

                # get event number array
                event_array = arrays['event']

                # get MC particle arrays
                particle_track_id_array = arrays['particle_track_id']
                particle_pdg_code_array = arrays['particle_pdg_code']

                # get MC hit arrays
                hit_track_id_array = arrays['hit_track_id']
                hit_end_x_array = arrays['hit_end_x']
                hit_end_y_array = arrays['hit_end_y']
                hit_end_z_array = arrays['hit_end_z']
                hit_end_t_array = arrays['hit_end_t']
                hit_energy_deposit_array = arrays['hit_energy_deposit']
                hit_process_key_array = arrays['hit_process_key']

                # get pixel arrays
                pixel_x_array = arrays['pixel_x']
                pixel_y_array = arrays['pixel_y']
                pixel_reset_array = arrays['pixel_reset']
                pixel_tslr_array = arrays['pixel_tslr']

                # get pixel MC truth arrays
                pixel_reset_track_id_array = arrays['pixel_reset_truth_track_id']
                pixel_reset_weight_array = arrays['pixel_reset_truth_weight']

                # get number of events
                number_events = len(event_array)

                #--------------------------------------------------------------
                # extract spatial information from pixels and resets
                #--------------------------------------------------------------

                signal_idx = -1

                pixel_multiplicity = None

                pixel_x = None
                pixel_y = None
                pixel_reset = None
                pixel_tslr = None

                if signal_flag:

                    signal_idx = np.random.randint(number_events)

                    pixel_multiplicity = ak.count(pixel_reset_array[signal_idx], axis=1)
                    pixel_x = np.repeat(pixel_x_array[signal_idx].to_numpy(), pixel_multiplicity)
                    pixel_y = np.repeat(pixel_y_array[signal_idx].to_numpy(), pixel_multiplicity)
                    pixel_reset = ak.flatten(pixel_reset_array[signal_idx]).to_numpy()
                    pixel_tslr = ak.flatten(pixel_tslr_array[signal_idx]).to_numpy()

                else:

                    pixel_multiplicity = ak.count(pixel_reset_array, axis=2)
                    pixel_multiplicity = ak.flatten(pixel_multiplicity)

                    pixel_x = np.repeat(ak.flatten(pixel_x_array).to_numpy(), pixel_multiplicity)
                    pixel_y = np.repeat(ak.flatten(pixel_y_array).to_numpy(), pixel_multiplicity)
                    pixel_reset = ak.flatten(pixel_reset_array, axis=None).to_numpy()
                    pixel_tslr = ak.flatten(pixel_tslr_array, axis=None).to_numpy()

                """

                # loop over events
                for idx in range(number_events):

                    # get event number
                    event = event_array[idx]

                    # get pixel information for event
                    pixel_x = pixel_x_array[idx]
                    pixel_y = pixel_y_array[idx]
                    pixel_reset = pixel_reset_array[idx]
                    pixel_tslr = pixel_tslr_array[idx]
                    pixel_reset_track_id = pixel_reset_track_id_array[idx]
                    pixel_reset_weight = pixel_reset_weight_array[idx]

                    # get number of pixels in event
                    number_pixels = len(pixel_x)

                    # print('Event:', event)

                    # loop over pixels
                    for px in range(number_pixels):

                        # print the pixel coordinate
                        # print('  pixel (x, y): ({}, {})'.format(pixel_x[px], pixel_y[px]))

                        # print list of resets associated with this pixel
                        # print('  resets:', pixel_reset[px])

                        # print list of resets associated with this pixel
                        # print('  time since last reset:', pixel_tslr[px])

                        # get number of resets associated with this pixel
                        number_resets = len(pixel_reset[px])

                        # loop over resets
                        for rst in range(number_resets):

                            # get reset time
                            reset = pixel_reset[px][rst]

                            # get time since last reset
                            tslr = pixel_tslr[px][rst]

                            # get track IDs of MC particles responsible for this reset
                            mc_track_ids = pixel_reset_track_id[px][rst]
                            mc_weights = pixel_reset_weight[px][rst]

                            # get numbers of MC particles responsible for this reset
                            number_mc_particles = len(mc_track_ids)

                            # print reset, tslr, and z
                            # print('    reset time [ns]:', reset * 1e9,
                            #       '; time since last reset [ns]:', tslr * 1e9,
                            #       '; z [cm]:', reset * drift_velocity)

                            # print list of track IDs and weights
                            # print('      MC particle track IDs:', mc_track_ids)
                            # print('      MC weights:', mc_weights)

                            # # loop over MC particles
                            # for mcp in range(number_mc_particles):
                            #     mc_track_id = mc_track_ids[mcp]
                            #     mc_weight = mc_weights[mcp]

                    #----------------------------------------------------------
                    # extract spatial information from pixels and resets
                    #----------------------------------------------------------

                    # pixel array
                    pix_array = list(np.empty((1, 4)))

                    # iterate through pixels and fill pixel array
                    for px in range(number_pixels):

                        # get number of resets for this pixel
                        number_resets = len(pixel_reset[px])

                        # iterate through resets
                        for rst in range(number_resets):

                            # get reset time
                            reset = pixel_reset[px][rst]

                            # get time since last reset
                            tslr = pixel_tslr[px][rst]

                            # append to pixel array
                            pix_array.append([pixel_x[px], pixel_y[px], reset, tslr])

                    # convert list to array
                    if len(pix_array) > 1:
                        # ignore dummy element
                        pix_array = np.array(pix_array[1:], dtype=int)
                    else:
                        pix_array = np.array(pix_array, dtype=int)

                    # convert to physical units
                    pix_x = pix_array[:, 0] * pixel_size  # cm
                    pix_y = pix_array[:, 1] * pixel_size  # cm
                    pix_z = pix_array[:, 2] * drift_velocity  # cm
                    pix_tslr = pix_array[:, 3] * 1e9 # ns

                    """

                #--------------------------------------------------------------
                # check if this is a signal event
                #--------------------------------------------------------------

                if signal_flag:
                    # this is a signal event
                    signal_x.extend(pixel_x)
                    signal_y.extend(pixel_y)
                    signal_t.extend(pixel_reset)
                else:
                    # this is a backgroud event
                    background_x.extend(pixel_x)
                    background_y.extend(pixel_y)
                    background_t.extend(pixel_reset)

    #--------------------------------------------------------------------------
    # do things with pixels here
    #--------------------------------------------------------------------------

    # print("signal_x", np.array(signal_x))
    # print("signal_y", np.array(signal_y))
    # print("signal_t", np.array(signal_t))

    # print("background_t", np.array(background_t))

    # print("background_x.shape", np.array(background_x).shape)
    # print("background_y.shape", np.array(background_y).shape)
    # print("background_t.shape", np.array(background_t).shape)

    s_x = np.array(signal_x) * pixel_size
    s_y = np.array(signal_y) * pixel_size
    s_z = np.array(signal_t) * drift_velocity
    s_t = np.array(signal_t)

    bg_x = np.array(background_x) * pixel_size
    bg_y = np.array(background_y) * pixel_size
    bg_z = np.array(background_t) * drift_velocity
    bg_t = np.array(background_t)

    x = np.concatenate((s_x, bg_x))
    y = np.concatenate((s_y, bg_y))
    z = np.concatenate((s_z, bg_z))
    t = np.concatenate((s_t, bg_t))

    labels_true = np.concatenate(
        (np.zeros(s_t.shape, dtype=int), np.ones(bg_t.shape, dtype=int)),
        dtype=int)

    X = np.c_[x, y, z]

    from sklearn.cluster import DBSCAN
    from sklearn import metrics

    # run DBSCAN
    db = DBSCAN(eps=0.3, min_samples=3).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
          % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
          % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels))

    #--------------------------------------------------------------------------


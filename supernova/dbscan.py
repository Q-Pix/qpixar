#!/usr/bin/env python

# -----------------------------------------------------------------------------
#  dbscan.py
#
#  Example for reading from ROOT files
#   * Author: Everybody is an author!
#   * Creation date: 2 August 2021
# -----------------------------------------------------------------------------

from __future__ import print_function

import sys
import argparse
import numpy as np
import awkward as ak
import uproot

# parse arguments from command
parser = argparse.ArgumentParser(description="dbscan")
parser.add_argument("signal", type=int, help="index of signal files")
parser.add_argument("background", type=int, help="index of background files")
# parser.add_argument("--event", type=int, help="index of signal event")
parser.add_argument("--events", nargs="+", type=int,
                    help="indices of signal event")
parser.add_argument("-o", "--output", type=str, help="output file")

args = parser.parse_args()
signal = args.signal
background = args.background
# event = args.event
events = args.events
output = args.output

if signal < 0 or background < 0:
    raise ValueError("Signal and background file indices should be positive")
# if isinstance(event, int) and event < 0:
#     raise ValueError("Signal event index should be positive")
if events and np.less(events, 0).any():
    raise ValueError("Signal event indices should be positive")

signal = str(signal).zfill(6)
background = str(background).zfill(6)
events = np.unique(events)

print(signal, background, events)
# print(np.less(events, 0))
# print(np.less(events, 0).any())

# sys.exit()

# np.random.seed(2)

signal_dir = "/n/holystore01/LABS/guenette_lab/Lab/data/q-pix/supernova/isotropic/snb_timing/ve_cc"
background_dir = "/n/holystore01/LABS/guenette_lab/Lab/data/q-pix/supernova/production"

file_array = [

    # buffer 0
    {
        # signal files
        signal_dir + "/" + signal + "/ve_cc_rtd_slim_" + signal + ".root" : 1,
        # background files
        background_dir + "/" + background + "/Ar39_rtd_slim_"  + background + ".root" : 0,
        background_dir + "/" + background + "/Kr85_rtd_slim_"  + background + ".root" : 0,
        background_dir + "/" + background + "/Bi214_rtd_slim_" + background + ".root" : 0,
        background_dir + "/" + background + "/K40_rtd_slim_"   + background + ".root" : 0,
        background_dir + "/" + background + "/Rn222_rtd_slim_" + background + ".root" : 0,
        background_dir + "/" + background + "/Pb214_rtd_slim_" + background + ".root" : 0,
        background_dir + "/" + background + "/Co60_rtd_slim_"  + background + ".root" : 0,
        background_dir + "/" + background + "/K42_rtd_slim_"   + background + ".root" : 0,
        background_dir + "/" + background + "/Ar42_rtd_slim_"  + background + ".root" : 0,
        background_dir + "/" + background + "/Po210_rtd_slim_" + background + ".root" : 0,
    },

    # # buffer 1
    # {
    #     # signal files
    #     signal_dir + "/" + signal + "/ve_cc_rtd_slim_" + signal + ".root" : 1,
    #     # background files
    #     background_dir + "/" + background + "/Ar39_rtd_slim_"  + background + ".root" : 0,
    #     background_dir + "/" + background + "/Kr85_rtd_slim_"  + background + ".root" : 0,
    #     background_dir + "/" + background + "/Bi214_rtd_slim_" + background + ".root" : 0,
    #     background_dir + "/" + background + "/K40_rtd_slim_"   + background + ".root" : 0,
    #     background_dir + "/" + background + "/Rn222_rtd_slim_" + background + ".root" : 0,
    #     background_dir + "/" + background + "/Pb214_rtd_slim_" + background + ".root" : 0,
    #     background_dir + "/" + background + "/Co60_rtd_slim_"  + background + ".root" : 0,
    #     background_dir + "/" + background + "/K42_rtd_slim_"   + background + ".root" : 0,
    #     background_dir + "/" + background + "/Ar42_rtd_slim_"  + background + ".root" : 0,
    #     background_dir + "/" + background + "/Po210_rtd_slim_" + background + ".root" : 0,
    # },

]

for buffer_idx in range(len(file_array)):

    print("Buffer index:", buffer_idx)
    files = file_array[buffer_idx]
    # print(files)
    # print(len(files))
    # print(files.items())

    signal_event_idx = []

    neutrino_x = []
    neutrino_y = []
    neutrino_z = []
    neutrino_energy = []
    signal_energy_deposit = []

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

                # generator particle information [Q_PIX_GEANT4]
                'generator_initial_number_particles',
                'generator_initial_particle_pdg_code',
                'generator_initial_particle_x',
                'generator_initial_particle_y',
                'generator_initial_particle_z',
                'generator_initial_particle_t',
                'generator_initial_particle_px',
                'generator_initial_particle_py',
                'generator_initial_particle_pz',
                'generator_initial_particle_energy',
                'generator_initial_particle_mass',
                'generator_final_number_particles',
                'generator_final_particle_pdg_code',
                'generator_final_particle_x',
                'generator_final_particle_y',
                'generator_final_particle_z',
                'generator_final_particle_t',
                'generator_final_particle_px',
                'generator_final_particle_py',
                'generator_final_particle_pz',
                'generator_final_particle_energy',
                'generator_final_particle_mass',

                # # MC particle information [Q_PIX_GEANT4]
                # 'particle_track_id', 'particle_pdg_code',
                # 'particle_mass', 'particle_initial_energy',

                # MC hit information [Q_PIX_GEANT4]
                'energy_deposit',
                # 'hit_energy_deposit', 'hit_track_id', 'hit_process_key',
                # 'hit_start_x', 'hit_start_y', 'hit_start_z', 'hit_start_t',
                # 'hit_end_x', 'hit_end_y', 'hit_end_z', 'hit_end_t',

                # pixel information [Q_PIX_RTD]
                'pixel_x', 'pixel_y', 'pixel_reset', 'pixel_tslr',
                # 'pixel_reset_truth_track_id', 'pixel_reset_truth_weight',

            ]

            #------------------------------------------------------------------
            # iterate through the event tree
            #------------------------------------------------------------------

            # print("Iterating through event tree...")

            for arrays in tree.iterate(filter_name=branches):

                # get event number array
                event_array = arrays['event']

                # generator particle arrays
                generator_initial_number_particles_array = None
                generator_initial_particle_pdg_code_array = None
                generator_initial_particle_x_array = None
                generator_initial_particle_y_array = None
                generator_initial_particle_z_array = None
                generator_initial_particle_t_array = None
                generator_initial_particle_px_array = None
                generator_initial_particle_py_array = None
                generator_initial_particle_pz_array = None
                generator_initial_particle_energy_array = None
                generator_initial_particle_mass_array = None
                generator_final_number_particles_array = None
                generator_final_particle_pdg_code_array = None
                generator_final_particle_x_array = None
                generator_final_particle_y_array = None
                generator_final_particle_z_array = None
                generator_final_particle_t_array = None
                generator_final_particle_px_array = None
                generator_final_particle_py_array = None
                generator_final_particle_pz_array = None
                generator_final_particle_energy_array = None
                generator_final_particle_mass_array = None

                if signal_flag:
                    generator_initial_number_particles_array = arrays['generator_initial_number_particles']
                    generator_initial_particle_pdg_code_array = arrays['generator_initial_particle_pdg_code']
                    generator_initial_particle_x_array = arrays['generator_initial_particle_x']
                    generator_initial_particle_y_array = arrays['generator_initial_particle_y']
                    generator_initial_particle_z_array = arrays['generator_initial_particle_z']
                    generator_initial_particle_t_array = arrays['generator_initial_particle_t']
                    generator_initial_particle_px_array = arrays['generator_initial_particle_px']
                    generator_initial_particle_py_array = arrays['generator_initial_particle_py']
                    generator_initial_particle_pz_array = arrays['generator_initial_particle_pz']
                    generator_initial_particle_energy_array = arrays['generator_initial_particle_energy']
                    generator_initial_particle_mass_array = arrays['generator_initial_particle_mass']
                    generator_final_number_particles_array = arrays['generator_final_number_particles']
                    generator_final_particle_pdg_code_array = arrays['generator_final_particle_pdg_code']
                    generator_final_particle_x_array = arrays['generator_final_particle_x']
                    generator_final_particle_y_array = arrays['generator_final_particle_y']
                    generator_final_particle_z_array = arrays['generator_final_particle_z']
                    generator_final_particle_t_array = arrays['generator_final_particle_t']
                    generator_final_particle_px_array = arrays['generator_final_particle_px']
                    generator_final_particle_py_array = arrays['generator_final_particle_py']
                    generator_final_particle_pz_array = arrays['generator_final_particle_pz']
                    generator_final_particle_energy_array = arrays['generator_final_particle_energy']
                    generator_final_particle_mass_array = arrays['generator_final_particle_mass']

                # # get MC particle arrays
                # particle_track_id_array = arrays['particle_track_id']
                # particle_pdg_code_array = arrays['particle_pdg_code']

                # get energy deposit array
                energy_deposit_array = arrays['energy_deposit']

                # get MC hit arrays
                # hit_track_id_array = arrays['hit_track_id']
                # hit_end_x_array = arrays['hit_end_x']
                # hit_end_y_array = arrays['hit_end_y']
                # hit_end_z_array = arrays['hit_end_z']
                # hit_end_t_array = arrays['hit_end_t']
                # hit_energy_deposit_array = arrays['hit_energy_deposit']
                # hit_process_key_array = arrays['hit_process_key']

                # get pixel arrays
                pixel_x_array = arrays['pixel_x']
                pixel_y_array = arrays['pixel_y']
                pixel_reset_array = arrays['pixel_reset']
                pixel_tslr_array = arrays['pixel_tslr']

                # # get pixel MC truth arrays
                # pixel_reset_track_id_array = arrays['pixel_reset_truth_track_id']
                # pixel_reset_weight_array = arrays['pixel_reset_truth_weight']

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

                    # signal_idx = np.random.randint(number_events)
                    # signal_idx = event

                    for signal_idx in events:

                        if signal_idx >= number_events:
                            msg = "Event index out of range: signal index {} not in [0, {})".format(signal_idx, number_events)
                            raise ValueError(msg)

                        pixel_multiplicity = ak.count(pixel_reset_array[signal_idx], axis=1)
                        pixel_x = np.repeat(pixel_x_array[signal_idx].to_numpy(), pixel_multiplicity)
                        pixel_y = np.repeat(pixel_y_array[signal_idx].to_numpy(), pixel_multiplicity)
                        pixel_reset = ak.flatten(pixel_reset_array[signal_idx]).to_numpy()
                        pixel_tslr = ak.flatten(pixel_tslr_array[signal_idx]).to_numpy()

                        signal_event_idx.extend(np.repeat(signal_idx, pixel_reset.shape[0]))
                        signal_x.extend(pixel_x)
                        signal_y.extend(pixel_y)
                        signal_t.extend(pixel_reset)
                        signal_energy_deposit.append(energy_deposit_array[signal_idx])
                        neutrino_x.append(generator_initial_particle_x_array[signal_idx][generator_initial_particle_pdg_code_array[signal_idx] == 12][0])
                        neutrino_y.append(generator_initial_particle_y_array[signal_idx][generator_initial_particle_pdg_code_array[signal_idx] == 12][0])
                        neutrino_z.append(generator_initial_particle_z_array[signal_idx][generator_initial_particle_pdg_code_array[signal_idx] == 12][0])
                        neutrino_energy.append(generator_initial_particle_energy_array[signal_idx][generator_initial_particle_pdg_code_array[signal_idx] == 12][0])

                else:

                    pixel_multiplicity = ak.count(pixel_reset_array, axis=2)
                    pixel_multiplicity = ak.flatten(pixel_multiplicity)

                    pixel_x = np.repeat(ak.flatten(pixel_x_array).to_numpy(), pixel_multiplicity)
                    pixel_y = np.repeat(ak.flatten(pixel_y_array).to_numpy(), pixel_multiplicity)
                    pixel_reset = ak.flatten(pixel_reset_array, axis=None).to_numpy()
                    pixel_tslr = ak.flatten(pixel_tslr_array, axis=None).to_numpy()

                    background_x.extend(pixel_x)
                    background_y.extend(pixel_y)
                    background_t.extend(pixel_reset)

                #--------------------------------------------------------------
                # check if this is a signal event
                #--------------------------------------------------------------

                # if signal_flag:
                #     # this is a signal event
                #     signal_x.extend(pixel_x)
                #     signal_y.extend(pixel_y)
                #     signal_t.extend(pixel_reset)
                #     signal_energy_deposit.append(energy_deposit_array[signal_idx])
                #     neutrino_x.append(generator_initial_particle_x_array[signal_idx][generator_initial_particle_pdg_code_array[signal_idx] == 12][0])
                #     neutrino_y.append(generator_initial_particle_y_array[signal_idx][generator_initial_particle_pdg_code_array[signal_idx] == 12][0])
                #     neutrino_z.append(generator_initial_particle_z_array[signal_idx][generator_initial_particle_pdg_code_array[signal_idx] == 12][0])
                #     neutrino_energy.append(generator_initial_particle_energy_array[signal_idx][generator_initial_particle_pdg_code_array[signal_idx] == 12][0])
                # else:
                #     # this is a backgroud event
                #     background_x.extend(pixel_x)
                #     background_y.extend(pixel_y)
                #     background_t.extend(pixel_reset)

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

    s_idx = np.array(signal_event_idx)

    s_x = np.array(signal_x) * pixel_size
    s_y = np.array(signal_y) * pixel_size
    s_z = np.array(signal_t) * drift_velocity
    s_t = np.array(signal_t)

    bg_x = np.array(background_x) * pixel_size
    bg_y = np.array(background_y) * pixel_size
    bg_z = np.array(background_t) * drift_velocity
    bg_t = np.array(background_t)

    for idx in range(len(events)):

        signal_idx = events[idx]

        flag = s_idx == signal_idx

        x = np.concatenate((s_x[flag], bg_x))
        y = np.concatenate((s_y[flag], bg_y))
        z = np.concatenate((s_z[flag], bg_z))
        t = np.concatenate((s_t[flag], bg_t))

        # labels_true = np.concatenate(
        #     (np.zeros(s_t.shape, dtype=int), np.ones(bg_t.shape, dtype=int)),
        #     #(np.ones(s_t.shape, dtype=int), np.zeros(bg_t.shape, dtype=int)),
        #     dtype=int)

        labels_true = np.concatenate(
            (np.zeros(s_t[flag].shape, dtype=int), np.ones(bg_t.shape, dtype=int))).astype(int)

        X = np.c_[x, y, z]
        # print(x, y, z)
        # print(X)
        # print(signal_x, signal_y)
        # X = np.c_[np.array(signal_x), np.array(signal_y), z]

        print(X)
        print(X[:, 2])
        print(labels_true)
        print(X.shape)
        print(labels_true.shape)
        print((labels_true == 0).sum())

        drift_velocity  # 1.648e5 cm/s == 0.1648 cm/us
        resets_total = len(X)

        # sys.exit()

        from sklearn.cluster import DBSCAN
        from sklearn import metrics

        signal_event_idx_array = []
        neutrino_x_array = []
        neutrino_y_array = []
        neutrino_z_array = []
        neutrino_energy_array = []
        signal_energy_deposit_array = []
        eps_array = []
        min_samples_array = []

        signal_resets_total_array = []
        signal_resets_clustered_array = []
        resets_clustered_array = []
        resets_total_array = []

        number_clusters_array = []
        resets_unclustered_array = []

        completeness_array = []
        cleanliness_array = []

        # eps_values = np.linspace(0.05, 3, 60).round(decimals=2)
        eps_values = np.arange(0.05, 3+0.05, 0.05).round(decimals=2)
        min_samples_values = np.arange(2, 24+1, 1)

        # for testing
        # eps_values = np.arange(0.05, 1+0.05, 0.05)
        # min_samples_values = np.arange(2, 24+1, 1)

        for eps in eps_values:
            for min_samples in min_samples_values:

                # run DBSCAN
                # db = DBSCAN(eps=0.5, min_samples=5).fit(X)
                db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
                core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
                core_samples_mask[db.core_sample_indices_] = True
                labels = db.labels_

                n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise_ = list(labels).count(-1)

                # print('Estimated number of clusters: %d' % n_clusters_)
                # print('Estimated number of noise points: %d' % n_noise_)
                # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
                # print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
                # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
                # print("Adjusted Rand Index: %0.3f"
                #       % metrics.adjusted_rand_score(labels_true, labels))
                # print("Adjusted Mutual Information: %0.3f"
                #       % metrics.adjusted_mutual_info_score(labels_true, labels))
                # print("Silhouette Coefficient: %0.3f"
                #       % metrics.silhouette_score(X, labels))

                #------------------------------------------------------------------

                signal_resets_clustered = np.sum((labels_true == 0) & (labels > -1))

                completeness = 0
                cleanliness = 0

                signal_resets_total = np.sum(labels_true == 0)
                resets_clustered = np.sum(labels > -1)

                if signal_resets_total > 0:
                    completeness = float(signal_resets_clustered) / float(signal_resets_total)
                if resets_clustered > 0:
                    cleanliness = float(signal_resets_clustered) / float(resets_clustered)

                print('Signal event index:', signal_idx)
                print('Interaction vertex [cm]: (%s, %s, %s)' % (neutrino_x[idx], neutrino_y[idx], neutrino_z[idx]))
                print('Neutrino energy [MeV]:', neutrino_energy[idx])
                print('Signal energy deposit [MeV]:', signal_energy_deposit[idx])
                print('Total number of resets:', resets_total)

                print('eps, min_samples: %s, %s' % (eps, min_samples))

                print('Estimated number of clusters: %d' % n_clusters_)
                print('Estimated number of noise points: %d' % n_noise_)

                print('Completeness: %d / %d = %s' % (signal_resets_clustered, signal_resets_total, completeness))
                print('Cleanliness:  %d / %d = %s' % (signal_resets_clustered, resets_clustered, cleanliness))

                # eps, min_samples

                signal_event_idx_array.append(signal_idx)
                neutrino_x_array.append(neutrino_x[idx])
                neutrino_y_array.append(neutrino_y[idx])
                neutrino_z_array.append(neutrino_z[idx])
                neutrino_energy_array.append(neutrino_energy[idx])
                signal_energy_deposit_array.append(signal_energy_deposit[idx])
                eps_array.append(eps)
                min_samples_array.append(min_samples)

                signal_resets_total_array.append(signal_resets_total)
                signal_resets_clustered_array.append(signal_resets_clustered)
                resets_clustered_array.append(resets_clustered)
                resets_total_array.append(resets_total)
                completeness_array.append(completeness)
                cleanliness_array.append(cleanliness)

                number_clusters_array.append(n_clusters_)
                resets_unclustered_array.append(n_noise_)

                #------------------------------------------------------------------

        #--------------------------------------------------------------------------

        x = np.vstack([
            signal_event_idx_array,
            neutrino_x_array,
            neutrino_y_array,
            neutrino_z_array,
            neutrino_energy_array,
            signal_energy_deposit_array,
            eps_array,
            min_samples_array,
            signal_resets_total_array,
            signal_resets_clustered_array,
            resets_clustered_array,
            resets_total_array,
            completeness_array,
            cleanliness_array,
            number_clusters_array,
            resets_unclustered_array,
            ])

        print(x)
        print(x.T)

        with open("test.txt", "ab") as f:
            f.write(b"\n")
            np.savetxt(f, x.T)

        # signal idx | neutrino x | neutrino y | neutrino z | neutrino energy | energy deposited | eps | min_samples | total number of signal resets | number of signal resets clustered | number of resets clustered | total number of resets | completeness | cleanliness | number of clusters | number of resets not clustered

        #--------------------------------------------------------------------------


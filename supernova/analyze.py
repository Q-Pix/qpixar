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
bg_input_dir = "/Volumes/seagate/work/qpix/supernova/production"

file_array = [

    # buffer 0
    {
        # signal files
        # input_dir + "/SUPERNOVA_NEUTRINO_RTD.root" : 1,
        input_dir + "/SUPERNOVA_NEUTRINO_RTD_1000_events.root" : 1,
        # background files
        bg_input_dir + "/000001/Ar39_rtd_slim_000001.root"  : 0,
        bg_input_dir + "/000001/Kr85_rtd_slim_000001.root"  : 0,
        bg_input_dir + "/000001/Bi214_rtd_slim_000001.root" : 0,
        bg_input_dir + "/000001/K40_rtd_slim_000001.root"   : 0,
        bg_input_dir + "/000001/Rn222_rtd_slim_000001.root" : 0,
        bg_input_dir + "/000001/Pb214_rtd_slim_000001.root" : 0,
        bg_input_dir + "/000001/Co60_rtd_slim_000001.root"  : 0,
        bg_input_dir + "/000001/K42_rtd_slim_000001.root"   : 0,
        bg_input_dir + "/000001/Ar42_rtd_slim_000001.root"  : 0,
        bg_input_dir + "/000001/Po210_rtd_slim_000001.root" : 0,
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
                'generator_initial_particle_px',
                'generator_initial_particle_py',
                'generator_initial_particle_pz',
                'generator_initial_particle_energy',
                'generator_initial_particle_mass',
                'generator_initial_particle_t',
                'generator_final_number_particles',
                'generator_final_particle_pdg_code',
                'generator_final_particle_px',
                'generator_final_particle_py',
                'generator_final_particle_pz',
                'generator_final_particle_energy',
                'generator_final_particle_mass',
                'generator_final_particle_t',

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
                generator_initial_particle_px_array = None
                generator_initial_particle_py_array = None
                generator_initial_particle_pz_array = None
                generator_initial_particle_energy_array = None
                generator_initial_particle_mass_array = None
                generator_initial_particle_t_array = None
                generator_final_number_particles_array = None
                generator_final_particle_pdg_code_array = None
                generator_final_particle_px_array = None
                generator_final_particle_py_array = None
                generator_final_particle_pz_array = None
                generator_final_particle_energy_array = None
                generator_final_particle_mass_array = None
                generator_final_particle_t_array = None

                if signal_flag:
                    generator_initial_number_particles_array = arrays['generator_initial_number_particles']
                    generator_initial_particle_pdg_code_array = arrays['generator_initial_particle_pdg_code']
                    generator_initial_particle_px_array = arrays['generator_initial_particle_px']
                    generator_initial_particle_py_array = arrays['generator_initial_particle_py']
                    generator_initial_particle_pz_array = arrays['generator_initial_particle_pz']
                    generator_initial_particle_energy_array = arrays['generator_initial_particle_energy']
                    generator_initial_particle_mass_array = arrays['generator_initial_particle_mass']
                    generator_initial_particle_t_array = arrays['generator_initial_particle_t']
                    generator_final_number_particles_array = arrays['generator_final_number_particles']
                    generator_final_particle_pdg_code_array = arrays['generator_final_particle_pdg_code']
                    generator_final_particle_px_array = arrays['generator_final_particle_px']
                    generator_final_particle_py_array = arrays['generator_final_particle_py']
                    generator_final_particle_pz_array = arrays['generator_final_particle_pz']
                    generator_final_particle_energy_array = arrays['generator_final_particle_energy']
                    generator_final_particle_mass_array = arrays['generator_final_particle_mass']
                    generator_final_particle_t_array = arrays['generator_final_particle_t']

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

                #--------------------------------------------------------------
                # check if this is a signal event
                #--------------------------------------------------------------

                if signal_flag:
                    # this is a signal event
                    signal_x.extend(pixel_x)
                    signal_y.extend(pixel_y)
                    signal_t.extend(pixel_reset)
                    signal_energy_deposit.append(energy_deposit_array[signal_idx])
                    neutrino_energy.append(generator_initial_particle_energy_array[signal_idx][generator_initial_particle_pdg_code_array[signal_idx] == 12][0])
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
        #(np.ones(s_t.shape, dtype=int), np.zeros(bg_t.shape, dtype=int)),
        dtype=int)

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

            print('Estimated number of clusters: %d' % n_clusters_)
            print('Estimated number of noise points: %d' % n_noise_)
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

            print('Neutrino energy [MeV]:', neutrino_energy[0])
            print('Signal energy deposit [MeV]:', signal_energy_deposit[0])
            print('Total number of resets:', resets_total)

            print('eps, min_samples:', eps, min_samples)

            print('Completeness: %d / %d = %s' % (signal_resets_clustered, signal_resets_total, completeness))
            print('Cleanliness:  %d / %d = %s' % (signal_resets_clustered, resets_clustered, cleanliness))

            # eps, min_samples

            neutrino_energy_array.append(neutrino_energy[0])
            signal_energy_deposit_array.append(signal_energy_deposit[0])
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
        neutrino_energy_array,
        signal_energy_deposit_array,
        eps_array, min_samples_array,
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

    #--------------------------------------------------------------------------


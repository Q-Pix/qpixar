#!/usr/bin/env python

# -----------------------------------------------------------------------------
#  dbscan_chebyshev_trigger_efficiency.py
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

from sklearn.cluster import DBSCAN
from sklearn import metrics

# parse arguments from command
parser = argparse.ArgumentParser(description="dbscan")
parser.add_argument("signal", type=int, help="index of signal files")
parser.add_argument("background", type=int, help="index of background files")
# parser.add_argument("-s", "--signal", type=int, help="index of signal files",
#                     required=True)
# parser.add_argument("-b", "--background", type=int,
#                     help="index of background files", required=True)
parser.add_argument("-r", "--reaction", type=str, help="reaction: cc or es",
                    required=True)
parser.add_argument("--events", nargs="+", type=int,
                    help="indices of signal event", required=True)
parser.add_argument("-o", "--output", type=str, help="output file",
                    required=True)

args = parser.parse_args()
signal = args.signal
background = args.background
reaction = args.reaction
events = args.events
output = args.output

reactions = [ "ve_cc", "ve_es" ]

# check if reaction is valid
if reaction not in reactions:
    msg = """Reaction '{}' is not valid!
Valid reactions:
{}""".format(reaction, reactions)
    raise ValueError(msg)

if signal < 0 or background < 0:
    raise ValueError("Signal and background file indices should be positive")
if events and np.less(events, 0).any():
    raise ValueError("Signal event indices should be positive")

signal = str(signal).zfill(6)
background = str(background).zfill(6)
events = np.unique(events)

print(signal, background, events, output)

# sys.exit()

#signal_dir = "/n/holystore01/LABS/guenette_lab/Lab/data/q-pix/supernova/isotropic/snb_timing/" + reaction
#background_dir = "/n/holystore01/LABS/guenette_lab/Lab/data/q-pix/supernova/production"
signal_dir = "/n/holystore01/LABS/guenette_lab/Lab/data/q-pix/supernova/production/signal/" + reaction + "/garching"
background_dir = "/n/holystore01/LABS/guenette_lab/Lab/data/q-pix/supernova/production/backgrounds/radiogenic"

file_array = [

    # buffer 0
    {
        # signal files
        signal_dir + "/" + signal + "/" + reaction +"_rtd_slim_" + signal + ".root" : 1,
        # "/n/holystore01/LABS/guenette_lab/Lab/data/q-pix/supernova_test/rtd/nu_e/Nu_e-rtd-1053.root" : 1,
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

#------------------------------------------------------------------------------

with uproot.recreate(output, compression=uproot.LZ4(1)) as root_file:

    for buffer_idx in range(len(file_array)):

        print("Buffer index:", buffer_idx)
        files = file_array[buffer_idx]
        # print(files)
        # print(len(files))
        # print(files.items())

        signal_event_idx = []
        signal_event = []

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

                #--------------------------------------------------------------
                # get metadata from ROOT file
                #--------------------------------------------------------------

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

                #--------------------------------------------------------------
                # get event tree from ROOT file
                #--------------------------------------------------------------

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

                #--------------------------------------------------------------
                # iterate through the event tree
                #--------------------------------------------------------------

                # print("Iterating through event tree...")

                for arrays in tree.iterate(filter_name=branches):
                # for arrays in tree.iterate(filter_name=branches, step_size=1000):

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

                    #----------------------------------------------------------
                    # extract spatial information from pixels and resets
                    #----------------------------------------------------------

                    signal_idx = -1

                    pixel_multiplicity = None

                    pixel_x = None
                    pixel_y = None
                    pixel_reset = None
                    pixel_tslr = None

                    if signal_flag:

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
                            # signal_event.extend(np.repeat(event_array[signal_idx], pixel_reset.shape[0]))
                            signal_x.extend(pixel_x)
                            signal_y.extend(pixel_y)
                            signal_t.extend(pixel_reset)
                            signal_event.append(event_array[signal_idx])
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

                    #----------------------------------------------------------
                    # check if this is a signal event
                    #----------------------------------------------------------

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

        #----------------------------------------------------------------------
        # do things with pixels here
        #----------------------------------------------------------------------

        # print("signal_x", np.array(signal_x))
        # print("signal_y", np.array(signal_y))
        # print("signal_t", np.array(signal_t))

        # print("background_t", np.array(background_t))

        # print("background_x.shape", np.array(background_x).shape)
        # print("background_y.shape", np.array(background_y).shape)
        # print("background_t.shape", np.array(background_t).shape)

        s_idx = np.array(signal_event_idx)

        # don't convert pixel (x, y) to physical units
        s_x = np.array(signal_x) * pixel_size
        s_y = np.array(signal_y) * pixel_size
        s_z = np.array(signal_t) * drift_velocity
        s_t = np.array(signal_t)

        bg_x = np.array(background_x) * pixel_size
        bg_y = np.array(background_y) * pixel_size
        bg_z = np.array(background_t) * drift_velocity
        bg_t = np.array(background_t)

        # these will be populated and written to a ROOT file
        signal_event_idx_ = []
        signal_event_ = []
        neutrino_x_ = []
        neutrino_y_ = []
        neutrino_z_ = []
        neutrino_energy_ = []
        signal_energy_deposit_ = []
        signal_resets_total_ = []
        resets_total_ = []
        eps_ = []
        min_samples_ = []
        resets_clustered_ = []
        signal_resets_clustered_ = []
        completeness_ = []
        cleanliness_ = []
        number_clusters_ = []
        resets_not_clustered_ = []
        x_ = []
        y_ = []
        t_ = []
        label_signal_ = []
        label_signal_idx_ = []
        label_background_ = []
        label_background_idx_ = []

        label_ = []
        label_idx_ = []

        # bg_label_ = []
        # bg_label_idx_ = []

        # bg_x_ = []
        # bg_y_ = []
        # bg_t_ = []

        n_bg_resets = len(bg_t)
        background_resets_total_ = [ n_bg_resets ]

        #----------------------------------------------------------------------
        # run DBSCAN on background only
        #----------------------------------------------------------------------

        X = np.c_[bg_x, bg_y, bg_t]

        eps = 6e-6
        min_samples = 14

        #

        # run DBSCAN
        # db = DBSCAN(eps=0.5, min_samples=5).fit(X)
        # db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        Y = X.copy()
        Y[:, 0:2] /= 2.0  # scale the pixels by 2
        Y[:, 2] /= eps  # scale the drift time by eps
        db = DBSCAN(eps=1, min_samples=min_samples, metric='chebyshev').fit(Y)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        #

        label_empty = ak.Array([[0]])[ak.Array([[0]]) == 1]

        label_background = []
        label_background_idx = []

        for label in np.unique(labels):
            if label == -1:
                continue
            bg_index = np.argwhere(labels == label).flatten()
            print('label:', label)
            print('bg_index:', bg_index)
            label_background.extend([label]*len(bg_index))
            label_background_idx.extend(bg_index)

        label_.append(label_background)
        label_idx_.append(label_background_idx)

        label_ = ak.Array(label_)
        label_idx_ = ak.Array(label_idx_)

        if len(label_background) < 1:
            label_ = ak.copy(label_empty)
            label_idx_ = ak.copy(label_empty)

        """

        label_signal = []
        label_signal_idx = []
        label_background = []
        label_background_idx = []

        for label in np.unique(labels):
            if label == -1:
                continue
            bg_index = np.argwhere(bg_labels == label).flatten()
            signal_index = np.argwhere(signal_labels == label).flatten()
            print('label:', label)
            print('bg_index:', bg_index)
            print('signal_index:', signal_index)
            label_signal.extend([label]*len(signal_index))
            label_signal_idx.extend(signal_index)
            label_background.extend([label]*len(bg_index))
            label_background_idx.extend(bg_index)

        """

        #----------------------------------------------------------------------
        # write to background tree
        #----------------------------------------------------------------------

        data_buffer = {
            "_background"  : [ int(background) ],
            "_x"           : ak.Array([ bg_x ]),
            "_y"           : ak.Array([ bg_y ]),
            "_t"           : ak.Array([ bg_t ]),
            "_eps"         : [ eps ],
            "_min_samples" : [ min_samples ],
            "_label"       : label_,
            "_label_index" : label_idx_,
        }

        print("Writing data buffer to file...")

        print("root_file.keys():", root_file.keys())

        print(data_buffer)

        tree_name = "background"

        if any(tree_name+";" in s for s in root_file.keys()):
            print("Extending tree...")
            root_file[tree_name].extend(data_buffer)
            print("root_file.keys():", root_file.keys())
        else:
            print("Creating tree...")
            root_file[tree_name] = data_buffer
            print("root_file.keys():", root_file.keys())

        #----------------------------------------------------------------------

        for idx in range(len(events)):

            signal_idx = events[idx]

            print('------------------------------------------------------')
            print('Signal event index:', signal_idx)
            print('Signal event ID:', signal_event[idx])

            flag = s_idx == signal_idx

            # x = np.concatenate((s_x[flag], bg_x))
            # y = np.concatenate((s_y[flag], bg_y))
            # z = np.concatenate((s_z[flag], bg_z))
            # t = np.concatenate((s_t[flag], bg_t))

            x = np.concatenate((bg_x, s_x[flag]))
            y = np.concatenate((bg_y, s_y[flag]))
            z = np.concatenate((bg_z, s_z[flag]))
            t = np.concatenate((bg_t, s_t[flag]))

            # labels_true = np.concatenate(
            #     (np.zeros(s_t.shape, dtype=int), np.ones(bg_t.shape, dtype=int)),
            #     #(np.ones(s_t.shape, dtype=int), np.zeros(bg_t.shape, dtype=int)),
            #     dtype=int)

            # labels_true = np.concatenate(
            #     (np.zeros(s_t[flag].shape, dtype=int), np.ones(bg_t.shape, dtype=int))).astype(int)

            labels_true = np.concatenate(
                (np.ones(bg_t.shape, dtype=int), np.zeros(s_t[flag].shape, dtype=int))).astype(int)

            #------------------------------------------------------------------
            # https://stackoverflow.com/questions/45433863/multiple-eps-values-in-sklearn-dbscan
            # p = p * [1/eps1, 1/eps2, 1/eps3]
            # c = sklearn.cluster.DBSCAN(eps=1, metric="chebyshev", ...)
            #------------------------------------------------------------------
            X = np.c_[x, y, t]  # scale by x/eps1, y/eps2, z/eps3 (done below)
            # X = np.c_[x, y, z]
            # print(x, y, z)
            # print(X)
            # print(signal_x, signal_y)
            # X = np.c_[np.array(signal_x), np.array(signal_y), z]

            # print(X)
            # print(X[:, 2])
            # print(labels_true)
            # print(X.shape)
            # print(labels_true.shape)
            # print((labels_true == 0).sum())

            drift_velocity  # 1.648e5 cm/s == 0.1648 cm/us
            resets_total = len(X)
            signal_resets_total = np.sum(labels_true == 0)

            # sys.exit()

            # from sklearn.cluster import DBSCAN
            # from sklearn import metrics

            # initial DBSCAN

            """

            # xy = DBSCAN(eps=0.5, min_samples=1).fit(X[:, 0:2])
            # xy = DBSCAN(eps=1, min_samples=1, metric='chebyshev').fit(X[:, 0:2]/0.5)
            xy = DBSCAN(eps=1.5, min_samples=1).fit(X[:, 0:2])
            xy_labels = np.unique(xy.labels_)

            # print(xy_labels)
            # print(X.shape)

            # X_ = np.ma.masked_where(xy.labels_ == 0, X)

            Y = np.c_[xy.labels_*1e3, t]
            flag = xy.labels_ > -1

            # db = DBSCAN(eps=2.5, min_samples=13).fit(Y)
            # labels = np.unique(db.labels_)

            # print(Y)
            # print(labels)

            #

            # signal_event_idx_array = []
            # signal_event_array = []
            # neutrino_x_array = []
            # neutrino_y_array = []
            # neutrino_z_array = []
            # neutrino_energy_array = []
            # signal_energy_deposit_array = []

            # eps_array = []
            # min_samples_array = []

            # signal_resets_total_array = []
            # resets_total_array = []

            # signal_resets_clustered_array = []
            # resets_clustered_array = []

            # number_clusters_array = []
            # resets_not_clustered_array = []

            # completeness_array = []
            # cleanliness_array = []

            # run DBSCAN
            Z = Y[flag].copy()
            db = DBSCAN(eps=eps, min_samples=min_samples, metric='chebyshev').fit(Z)
            # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            # core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_

            """

            Y = X.copy()
            Y[:, 0:2] /= 2.0  # scale the pixels by 2
            Y[:, 2] /= eps  # scale the drift time by eps
            db = DBSCAN(eps=1, min_samples=min_samples, metric='chebyshev').fit(Y)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_

            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise_ = list(labels).count(-1)

            #--------------------------------------------------------------

            signal_resets_clustered = 0
            resets_clustered = 0

            completeness = []
            cleanliness = []

            for label in set(labels):
                if label == -1:
                    continue
                signal_resets_clustered += np.sum((labels_true == 0) & (labels == label))
                resets_clustered += np.sum(labels == label)

                signal_resets_clustered_value = np.sum((labels_true == 0) & (labels == label))
                resets_clustered_value = np.sum(labels == label)
                completeness_value = -1
                cleanliness_value = -1
                if signal_resets_total > 0:
                    completeness_value = float(signal_resets_clustered_value) / float(signal_resets_total)
                if resets_clustered_ > 0:
                    cleanliness_value = float(signal_resets_clustered_value) / float(resets_clustered_value)

                # signal_resets_clustered.append(signal_resets_clustered_value)
                # resets_clustered.append(resets_clustered_value)

                completeness.append(completeness_value)
                cleanliness.append(cleanliness_value)

            # # signal_resets_clustered = np.sum((labels_true == 0) & (labels > -1))
            # signal_resets_clustered = np.sum((labels_true[flag] == 0) & (labels > -1))
            # resets_clustered = np.sum(labels > -1)

            # completeness = -1
            # cleanliness = -1

            # if signal_resets_total > 0:
            #     completeness = float(signal_resets_clustered) / float(signal_resets_total)
            # if resets_clustered > 0:
            #     cleanliness = float(signal_resets_clustered) / float(resets_clustered)

            # print('------------------------------------------------------')
            # print('Signal event index:', signal_idx)
            # print('Signal event ID:', signal_event[idx])
            # print('Interaction vertex [cm]: (%s, %s, %s)' % (neutrino_x[idx], neutrino_y[idx], neutrino_z[idx]))
            # print('Neutrino energy [MeV]:', neutrino_energy[idx])
            # print('Signal energy deposit [MeV]:', signal_energy_deposit[idx])
            # print('Total number of resets:', resets_total)

            # print('eps, min_samples: %s, %s' % (eps, min_samples))

            # print('Estimated number of clusters: %d' % n_clusters_)
            # print('Estimated number of noise points: %d' % n_noise_)

            # # print('Completeness: %d / %d = %s' % (signal_resets_clustered, signal_resets_total, completeness))
            # # print('Cleanliness:  %d / %d = %s' % (signal_resets_clustered, resets_clustered, cleanliness))

            # print('Completeness: %s / %s = %s' % (np.array(signal_resets_clustered), np.array(signal_resets_total), np.array(completeness)))
            # print('Cleanliness:  %s / %s = %s' % (np.array(signal_resets_clustered), np.array(resets_clustered), np.array(cleanliness)))

            # h, edges = np.histogram(labels, bins=np.append(np.unique(labels), labels[-1]+1))
            # h, edges = np.histogram(labels, bins=np.unique(np.append(labels, np.unique(labels)[-1]+1)))
            # print(np.c_[h, edges[:-1]], h.sum(), np.sum(labels > -1))
            # print(Y[flag])
            # print(np.ma.mask_rows(np.ma.masked_where(Y < 0, Y)))

            # eps, min_samples

            # signal_event_idx_array.append(signal_idx)
            # signal_event_array.append(signal_event[idx])
            # neutrino_x_array.append(neutrino_x[idx])
            # neutrino_y_array.append(neutrino_y[idx])
            # neutrino_z_array.append(neutrino_z[idx])
            # neutrino_energy_array.append(neutrino_energy[idx])
            # signal_energy_deposit_array.append(signal_energy_deposit[idx])
            # eps_array.append(eps)
            # min_samples_array.append(min_samples)

            # signal_resets_total_array.append(signal_resets_total)
            # resets_total_array.append(resets_total)

            # signal_resets_clustered_array.append(signal_resets_clustered)
            # resets_clustered_array.append(resets_clustered)
            # completeness_array.append(completeness)
            # cleanliness_array.append(cleanliness)

            # print(n_clusters_, len(signal_resets_clustered), len(resets_clustered))

            # eps_array.extend([eps]*n_clusters_)
            # min_samples_array.extend([min_samples]*n_clusters_)
            # signal_resets_clustered_array.extend(signal_resets_clustered)
            # resets_clustered_array.extend(resets_clustered)
            # completeness_array.extend(completeness)
            # cleanliness_array.extend(cleanliness)

            # number_clusters_array.append(n_clusters_)
            # resets_not_clustered_array.append(n_noise_)

            #------------------------------------------------------------------

            bg_labels = labels[:n_bg_resets]
            signal_labels = labels[n_bg_resets:]

            # print('background resets', np.sum((1-labels_true) == 0))
            # print('background resets', n_bg_resets)
            # print('signal resets:', np.sum((1-labels_true) == 1))

            # print('bg:', (1-labels_true)[:n_bg_resets], len((1-labels_true)[:n_bg_resets]))
            # print('s:', (1-labels_true)[n_bg_resets:], len((1-labels_true)[n_bg_resets:]))

            # print('bg_labels:', bg_labels)
            # print('signal_labels:', signal_labels)

            # label_signal_ = []
            # label_signal_idx_ = []
            # label_background_ = []
            # label_background_idx_ = []

            label_signal = []
            label_signal_idx = []
            label_background = []
            label_background_idx = []

            for label in np.unique(labels):
                if label == -1:
                    continue
                bg_index = np.argwhere(bg_labels == label).flatten()
                signal_index = np.argwhere(signal_labels == label).flatten()
                print('label:', label)
                print('bg_index:', bg_index)
                print('signal_index:', signal_index)
                label_signal.extend([label]*len(signal_index))
                label_signal_idx.extend(signal_index)
                label_background.extend([label]*len(bg_index))
                label_background_idx.extend(bg_index)

            #------------------------------------------------------------------

            signal_event_idx_.append(signal_idx)
            signal_event_.append(signal_event[idx])
            neutrino_x_.append(neutrino_x[idx])
            neutrino_y_.append(neutrino_y[idx])
            neutrino_z_.append(neutrino_z[idx])
            neutrino_energy_.append(neutrino_energy[idx])
            signal_energy_deposit_.append(signal_energy_deposit[idx])
            signal_resets_total_.append(signal_resets_total)
            resets_total_.append(resets_total)

            eps_.append(eps)
            min_samples_.append(min_samples)
            resets_clustered_.append(resets_clustered)
            signal_resets_clustered_.append(signal_resets_clustered)
            completeness_.append(completeness)
            cleanliness_.append(cleanliness)
            number_clusters_.append(n_clusters_)
            resets_not_clustered_.append(n_noise_)
            x_.append(x[n_bg_resets:])
            y_.append(y[n_bg_resets:])
            t_.append(t[n_bg_resets:])
            # label_.append(labels)
            # label_signal_.append(1-labels_true)
            # label_signal_.append(np.array(label_signal, dtype=int))
            # label_signal_idx_.append(np.array(label_signal_idx, dtype=int))
            # label_background_.append(np.array(label_background, dtype=int))
            # label_background_idx_.append(np.array(label_background_idx, dtype=int))
            label_signal_.append(label_signal)
            label_signal_idx_.append(label_signal_idx)
            label_background_.append(label_background)
            label_background_idx_.append(label_background_idx)

            label_signal_ = ak.Array(label_signal_)
            label_signal_idx_ = ak.Array(label_signal_idx_)
            label_background_ = ak.Array(label_background_)
            label_background_idx_ = ak.Array(label_background_idx_)

            if len(label_signal) < 1:
                label_signal_ = ak.copy(label_empty)
                label_signal_idx_ = ak.copy(label_empty)

            if len(label_background) < 1:
                label_background_ = ak.copy(label_empty)
                label_background_idx_ = ak.copy(label_empty)

            #------------------------------------------------------------------

            #------------------------------------------------------------------

            data_buffer = {
                "_signal"                  : [ int(signal) ],
                "_background"              : [ int(background) ],
                "_event"                   : signal_event_,
                "_neutrino_x"              : neutrino_x_,
                "_neutrino_y"              : neutrino_y_,
                "_neutrino_z"              : neutrino_z_,
                "_neutrino_energy"         : neutrino_energy_,
                "_neutrino_energy_deposit" : signal_energy_deposit_,
                "_signal_resets_total"     : signal_resets_total_,
                "_resets_total"            : resets_total_,
                "_eps"                     : eps_,
                "_min_samples"             : min_samples_,
                "_resets_clustered"        : resets_clustered_,
                "_signal_resets_clustered" : signal_resets_clustered_,
                "_number_clusters"         : number_clusters_,
                "_resets_not_clustered"    : resets_not_clustered_,
                "_x"                       : ak.Array(x_),
                "_y"                       : ak.Array(y_),
                "_t"                       : ak.Array(t_),
                # "_label"                   : ak.Array(label_),
                "_label_signal"            : label_signal_,
                "_label_signal_index"      : label_signal_idx_,
                "_label_background"        : label_background_,
                "_label_background_index"  : label_background_idx_,
                # "_completeness"            : completeness_,
                # "_cleanliness"             : cleanliness_,
            }

            print("Writing data buffer to file...")

            print("root_file.keys():", root_file.keys())

            print(data_buffer)

            tree_name = "tree"

            if any(tree_name+";" in s for s in root_file.keys()):
                print("Extending tree...")
                root_file[tree_name].extend(data_buffer)
                print("root_file.keys():", root_file.keys())
            else:
                print("Creating tree...")
                root_file[tree_name] = data_buffer
                print("root_file.keys():", root_file.keys())

            signal_event_idx_ = []
            signal_event_ = []
            neutrino_x_ = []
            neutrino_y_ = []
            neutrino_z_ = []
            neutrino_energy_ = []
            signal_energy_deposit_ = []
            signal_resets_total_ = []
            resets_total_ = []
            eps_ = []
            min_samples_ = []
            resets_clustered_ = []
            signal_resets_clustered_ = []
            completeness_ = []
            cleanliness_ = []
            number_clusters_ = []
            resets_not_clustered_ = []
            x_ = []
            y_ = []
            t_ = []
            label_signal_ = []
            label_signal_idx_ = []
            label_background_ = []
            label_background_idx_ = []

            label_ = []
            label_idx_ = []


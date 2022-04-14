#!/usr/bin/env python

# -----------------------------------------------------------------------------
#  dEdx.py
#
#  Example for computing the dE/dx from GEANT4 truth information
#   * Author: Everybody is an author!
#   * Creation date: 13 April 2022
# -----------------------------------------------------------------------------

import argparse
import numpy as np
import awkward as ak
import uproot

# parse arguments from command
parser = argparse.ArgumentParser(description='example program')
parser.add_argument('input', type=str, default=None,
                    help='path to input ROOT file')
parser.add_argument('output', type=str, default=None,
                    help='path to output ROOT file')
parser.add_argument('--plot', action='store_false', help='plot')
parser.add_argument('--n', type=int,
                    help='maximum number of events to process')

# parse arguments
args = parser.parse_args()
input_path = str(args.input)
output_path = str(args.output)

# lists to be written to ROOT file
event_ = []
pdg_code_ = []
kinetic_energy_ = []
mass_ = []
energy_deposit_ = []
dE_ = []
dx_ = []
residual_range_ = []

# event counter
event_counter = 0

# open the ROOT file
with uproot.open(input_path) as f:

    #--------------------------------------------------------------------------
    # get metadata from ROOT file
    #--------------------------------------------------------------------------

    metadata = f['metadata']

    # get detector dimensions
    detector_length_x = metadata['detector_length_x'].array()[0]  # cm
    detector_length_y = metadata['detector_length_y'].array()[0]  # cm
    detector_length_z = metadata['detector_length_z'].array()[0]  # cm

    #--------------------------------------------------------------------------
    # get event tree from ROOT file
    #--------------------------------------------------------------------------

    tree = f['event_tree']

    # list of branches that we want to access
    branches = [

        # event number
        'event',

        # MC particle information [Q_PIX_GEANT4]
        'number_particles', 'particle_track_id', 'particle_parent_track_id',
        'particle_pdg_code', 'particle_mass',
        'particle_initial_x', 'particle_initial_y', 'particle_initial_z',
        'particle_initial_t',
        'particle_initial_px', 'particle_initial_py', 'particle_initial_pz',
        'particle_initial_energy',

        # MC hit information [Q_PIX_GEANT4]
        'hit_energy_deposit', 'hit_track_id', 'hit_process_key',
        'hit_start_x', 'hit_start_y', 'hit_start_z', 'hit_start_t',
        'hit_end_x', 'hit_end_y', 'hit_end_z', 'hit_end_t',

        # MC energy deposited [Q_PIX_GEANT4]
        'energy_deposit',

    ]

    #--------------------------------------------------------------------------
    # iterate through the event tree
    #--------------------------------------------------------------------------

    for arrays in tree.iterate(filter_name=branches):

        # get event number array
        event_array = arrays['event']

        # get MC particle arrays
        number_particles_array = arrays['number_particles']
        particle_track_id_array = arrays['particle_track_id']
        particle_parent_track_id_array = arrays['particle_parent_track_id']
        particle_pdg_code_array = arrays['particle_pdg_code']
        particle_mass_array = arrays['particle_mass']
        particle_initial_x_array = arrays['particle_initial_x']
        particle_initial_y_array = arrays['particle_initial_y']
        particle_initial_z_array = arrays['particle_initial_z']
        particle_initial_t_array = arrays['particle_initial_t']
        particle_initial_px_array = arrays['particle_initial_px']
        particle_initial_py_array = arrays['particle_initial_py']
        particle_initial_pz_array = arrays['particle_initial_pz']
        particle_initial_energy_array = arrays['particle_initial_energy']

        # get MC energy deposited arrays
        energy_deposit_array = arrays['energy_deposit']

        # get MC hit arrays
        hit_track_id_array = arrays['hit_track_id']
        hit_start_x_array = arrays['hit_start_x']
        hit_start_y_array = arrays['hit_start_y']
        hit_start_z_array = arrays['hit_start_z']
        hit_start_t_array = arrays['hit_start_t']
        hit_end_x_array = arrays['hit_end_x']
        hit_end_y_array = arrays['hit_end_y']
        hit_end_z_array = arrays['hit_end_z']
        hit_end_t_array = arrays['hit_end_t']
        hit_energy_deposit_array = arrays['hit_energy_deposit']
        hit_process_key_array = arrays['hit_process_key']

        # get number of events
        number_events = len(event_array)

        # loop over events
        for idx in range(number_events):

            if args.n and event_counter >= args.n:
                break

            # increase event counter
            event_counter += 1

            # get event number
            event = event_array[idx]

            print('Event:', event)

            # get MC particle information for event
            number_particles = number_particles_array[idx]
            particle_track_id = particle_track_id_array[idx]
            particle_parent_track_id = particle_parent_track_id_array[idx]
            particle_pdg_code = particle_pdg_code_array[idx]
            particle_mass = particle_mass_array[idx]
            particle_initial_x = particle_initial_x_array[idx]
            particle_initial_y = particle_initial_y_array[idx]
            particle_initial_z = particle_initial_z_array[idx]
            particle_initial_t = particle_initial_t_array[idx]
            particle_initial_px = particle_initial_px_array[idx]
            particle_initial_py = particle_initial_py_array[idx]
            particle_initial_pz = particle_initial_pz_array[idx]
            particle_initial_energy = particle_initial_energy_array[idx]

            # get MC energy deposited information for event
            # this is the total energy deposited in this event
            energy_deposit = energy_deposit_array[idx]

            # get MC hit information for event
            hit_track_id = hit_track_id_array[idx]
            hit_start_x = hit_start_x_array[idx]
            hit_start_y = hit_start_y_array[idx]
            hit_start_z = hit_start_z_array[idx]
            hit_start_t = hit_start_t_array[idx]
            hit_end_x = hit_end_x_array[idx]
            hit_end_y = hit_end_y_array[idx]
            hit_end_z = hit_end_z_array[idx]
            hit_end_t = hit_end_t_array[idx]
            hit_energy_deposit = hit_energy_deposit_array[idx]
            hit_process_key = hit_process_key_array[idx]

            # loop over all MC particles in the event
            for p_idx in range(number_particles):

                # primary particles have a parent track ID of 0
                # skip particles that aren't a primary particle
                if particle_parent_track_id[p_idx] != 0:
                    continue

                # get track ID of particle
                track_id = particle_track_id[p_idx]

                # boolean array of hits from this particle
                flag = (hit_track_id == track_id)

                # arrays of hits
                hit_x0 = hit_start_x[flag]
                hit_y0 = hit_start_y[flag]
                hit_z0 = hit_start_z[flag]
                hit_t0 = hit_start_t[flag]
                hit_x1 = hit_end_x[flag]
                hit_y1 = hit_end_y[flag]
                hit_z1 = hit_end_z[flag]
                hit_t1 = hit_end_t[flag]
                dE = hit_energy_deposit[flag]
                process = hit_process_key[flag]

                # sort hits by time
                sort = np.argsort(hit_t0)
                hit_x0 = hit_x0[sort]
                hit_y0 = hit_y0[sort]
                hit_z0 = hit_z0[sort]
                hit_t0 = hit_t0[sort]
                hit_x1 = hit_x1[sort]
                hit_y1 = hit_y1[sort]
                hit_z1 = hit_z1[sort]
                hit_t1 = hit_t1[sort]
                dE = dE[sort]
                process = process[sort]

                # calculate step sizes
                dx = hit_x1 - hit_x0
                dy = hit_y1 - hit_y0
                dz = hit_z1 - hit_z0
                dt = hit_t1 - hit_t0
                dr = np.sqrt(dx*dx + dy*dy + dz*dz)

                # get residual range
                rr = np.cumsum(dr[::-1])[::-1]

                # dE/dx in MeV/cm
                dEdx = dE/dr  # MeV/cm

                # append to lists for writing to output ROOT file
                event_.append(event)
                pdg_code_.append(particle_pdg_code[p_idx])
                kinetic_energy_.append(particle_initial_energy[p_idx] - particle_mass[p_idx])
                mass_.append(particle_mass[p_idx])
                energy_deposit_.append(energy_deposit)
                dE_.append(dE)
                dx_.append(dr)
                residual_range_.append(rr)

# write to output ROOT file
with uproot.recreate(output_path, compression=uproot.LZ4(1)) as root_file:

    root_file['tree'] = ak.zip({
        'event'          : event_,
        'pdg_code'       : pdg_code_,
        'kinetic_energy' : kinetic_energy_,
        'mass'           : mass_,
        'energy_deposit' : energy_deposit_,
        'dE'             : dE_,
        'dx'             : dx_,
        'residual_range' : residual_range_,
    })

if args.plot:

    import matplotlib as mpl
    # matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import AutoMinorLocator, LogLocator
    from matplotlib import rc
    rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
    rc('text', usetex=True)

    fig = plt.figure(figsize=(16, 5))
    gs = fig.add_gridspec(nrows=1, ncols=2)

    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[:, 1])

    axes = [ ax0, ax1 ]

    energy_deposit = ak.Array(energy_deposit_)
    residual_range = ak.Array(residual_range_)
    dE = ak.Array(dE_)
    dx = ak.Array(dx_)

    #--------------------------------------------------------------------------
    # this is our quick-and-dirty attempt at analyzing only stopping particles
    #--------------------------------------------------------------------------
    flag = ak.sum(dE, axis=1) / energy_deposit > 0.99
    #--------------------------------------------------------------------------

    residual_range = ak.flatten(residual_range[flag])
    dE = ak.flatten(dE[flag])
    dx = ak.flatten(dx[flag])

    ax0.scatter(residual_range, dE/dx, marker='.', s=1)
    ax0.set_xlim(left=0, right=25)
    ax0.set_ylim(bottom=0, top=50)

    ax0.xaxis.set_minor_locator(AutoMinorLocator())
    ax0.yaxis.set_minor_locator(AutoMinorLocator())
    ax0.set_title('linear--linear', fontsize=20)

    ax1.scatter(residual_range, dE/dx, marker='.', s=1)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_xlim(right=25)
    ax1.set_title('log--log', fontsize=20)

    ax1.xaxis.set_major_locator(LogLocator(base=10, numticks=10))

    for ax in axes:

        ax.yaxis.offsetText.set_fontsize(16)
        ax.tick_params(axis='both', which='major', labelsize=20)

        ax.grid(True, which='both', axis='both', color='k', linestyle=':',
                linewidth=1, alpha=0.2)

        ax.set_ylabel('d$E$/d$x$ [MeV/cm]', horizontalalignment='right',
                      y=1.0, fontsize=20)
        ax.set_xlabel('residual range [cm]', horizontalalignment='right',
                      x=1.0, fontsize=20)

    plt.tight_layout()
    plt.show()


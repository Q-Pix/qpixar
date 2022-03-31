#!/usr/bin/env python

# -----------------------------------------------------------------------------
#  root_to_csv_byevent.py
#
#  Example for reading from ROOT file
#   * Author: Dave Elofson
#   * Creation date: 30 March 2022
# -----------------------------------------------------------------------------

import argparse
import numpy as np
import uproot
import pandas as pd
import os.path
from datetime import datetime

# parse arguments from command
parser = argparse.ArgumentParser(description="pixel event display")
parser.add_argument("file", type=str, default=None,
                    help="path to ROOT file")
#parser.add_argument("event", type=int, default=None,
#					help="number of event for processing")

args = parser.parse_args()
file_path = str(args.file)
#nEvent = int(args.event)
pwd = os.path.dirname(file_path)
base = os.path.basename(file_path)
file_name = os.path.splitext(base)[0]
now = datetime.now()
dt_string = now.strftime("%Y-%m-%d_%H%M%S")
newdirs = str('%s/%s_%s'%(pwd,file_name,dt_string))
os.makedirs(newdirs)

reset_data = pd.DataFrame(columns=['event','pixel_x','pixel_y','reset_time','TSLR','nMCParticles','MC_TrackIDs','MC_Weights'])
G4_data = pd.DataFrame(columns=['event','xi','xf','yi','yf','zi','zf','ti','tf','E','ParticleID','PDG'])

with uproot.open(file_path) as f:

	#--------------------------------------------------------------------------
	# get metadata from ROOT file
	#--------------------------------------------------------------------------

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

	#--------------------------------------------------------------------------
	# get event tree from ROOT file
	#--------------------------------------------------------------------------

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

	#--------------------------------------------------------------------------
	# iterate through the event tree
	#--------------------------------------------------------------------------

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
		hit_start_x_array = arrays['hit_start_x']
		hit_start_y_array = arrays['hit_start_y']
		hit_start_z_array = arrays['hit_start_z']
		hit_start_t_array = arrays['hit_start_t']
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

		# loop over events
		#for idx in range(number_events):
		for idx in np.arange(0,number_events):

			reset_data = pd.DataFrame(columns=['event','pixel_x','pixel_y','reset_time','TSLR','nMCParticles','MC_TrackIDs','MC_Weights'])
			G4_data = pd.DataFrame(columns=['event','xi','xf','yi','yf','zi','zf','ti','tf','E','ParticleID','PDG'])
			
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

			print('Event:', event)

			# loop over pixels
			for px in range(number_pixels):

				# print the pixel coordinate
				print('  (x, y): ({}, {})'.format(pixel_x[px], pixel_y[px]))

				# print list of resets associated with this pixel
				print('  resets:', pixel_reset[px])

				# print list of resets associated with this pixel
				print('  time since last reset:', pixel_tslr[px])

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

                    ## print reset, tslr, and z
                    #print('    reset time [ns]:', reset * 1e9,
                    #      '; time since last reset [ns]:', tslr * 1e9,
                    #      '; z [cm]:', reset * drift_velocity)

                    ## print list of track IDs and weights
					#print('      MC particle track IDs:', mc_track_ids)
					#print('      MC weights:', mc_weights)

					reset_series = pd.Series([idx,pixel_x[px],pixel_y[px],reset,tslr,number_mc_particles,mc_track_ids,mc_weights],index=reset_data.columns)
					reset_data = reset_data.append(reset_series, ignore_index=True)

			# loop over all energy deposits
			edep_x = hit_end_x_array[idx]
			edep_y = hit_end_y_array[idx]
			edep_z = hit_end_z_array[idx]
			edep_t = hit_end_t_array[idx]
			edep_xi = hit_start_x_array[idx]
			edep_yi = hit_start_y_array[idx]
			edep_zi = hit_start_z_array[idx]
			edep_ti = hit_start_t_array[idx]
			edep_E = hit_energy_deposit_array[idx]
			edep_PID = hit_track_id_array[idx]
			edep_PDG = particle_pdg_code_array[idx] 

			for edep in range(len(edep_x)):
				edep_data = pd.Series([event,edep_xi[edep],edep_x[edep],edep_yi[edep],edep_y[edep],edep_zi[edep],edep_z[edep],edep_ti[edep],edep_t[edep],edep_E[edep],edep_PID[edep],edep_PDG[int(edep_PID[edep]-1)]],index=G4_data.columns)
				G4_data = G4_data.append(edep_data,ignore_index=True)


			#------------------------------------------------------------------
			# extract spatial information from pixels and resets
			#------------------------------------------------------------------

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

			#------------------------------------------------------------------

			# do things with pixels here

			# Compile event data into Resets and G4 files
			out_file1 = str('%s/resets_E%d.txt'%(newdirs,idx))
			out_file2 = str('%s/G4_E%d.txt'%(newdirs,idx))
			reset_data.to_csv(out_file1)
			G4_data.to_csv(out_file2)

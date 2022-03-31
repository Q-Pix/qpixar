#!/usr/bin/env python

# -----------------------------------------------------------------------------
#  recombine_sample.py
#
#  Used to recombine many separate events into a single sample after root_to_csv_byevent.py
#   * Author: Dave Elofson
#   * Creation date: 29 March 2022
# -----------------------------------------------------------------------------

import argparse
import numpy as np
import pandas as pd
import os, os.path


# parse arguments from command
parser = argparse.ArgumentParser(description="pixel event display")
parser.add_argument("pwd", type=str, default=None,
                    help="path to directory with separate files")
parser.add_argument("nevents", type=int, default=None,
                    help="number of events in samples")

args = parser.parse_args()
pwd = str(args.pwd)

nEvents = int(args.nevents)

resets = pd.read_csv(str('%s/resets_E0.txt'%(pwd)),index_col=0)
g4 = pd.read_csv(str('%s/G4_E0.txt'%(pwd)),index_col=0)

for event in np.arange(1,nEvents):
	filename1 = str('%s/resets_E%d.txt'%(pwd,event))
	df1 = pd.read_csv(filename1)
	resets = pd.concat([resets,df1],ignore_index=True)

	filename2 = str('%s/G4_E%d.txt'%(pwd,event))
	df2 = pd.read_csv(filename2)
	g4 = pd.concat([g4,df2],ignore_index=True)

resets.to_csv(str('%s/resets_output.txt'%pwd))
g4.to_csv(str('%s/g4_output.txt'%pwd))


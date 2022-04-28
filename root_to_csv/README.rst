This attachment of qpixar is a way to convert the RTD output files to .txt files with the desired information.

In order to convert the root data to a dataframe in .txt format, run

``$ python root_to_csv.py <input file> <output file>``

This will create a folder in the same directory as your input file, labeled with the date and time of creation, and will put all output files into the folder. 

Then you will need to run recombine_sample.py to recombine all the separate events back into a single sample for use with analysis software. This can be done by running

$ python recombine_samples.py <pwd to directory with separate event files> <total number of events>

This will create two output files: resets_output.txt and g4_output.txt, in the same new directory created by root_to_csv.py. These are the two files that will be used for the analysis work.

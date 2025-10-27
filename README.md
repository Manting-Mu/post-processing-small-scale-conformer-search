# post-processing-small-scale-conformer-search
A pythons script that reads in all the conformers generated via alternative softwares, such as openbable, rdkit or PCmodel, and clusters the structures to identify the most representing structure from each of the cluster for further small scale modelling. Note, this is an adopted version from the main Conformer Study script.

Usage:
do 'conda env create -f environment.yml' to set up the python environment.

to use the python script, open with 'vi' or text editor find the line:
"# Load the SDF file
sdf_file = "clevin.sdf"  # Replace with the path to your SDF file"

edit here to the path of the sdf file, this is the combined sdf with all conformers, see example 'exampl.sdf.zip' which contains all the conformers from the 'data.zip' files.




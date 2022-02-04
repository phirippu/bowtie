# The bowtie module
This is a module to perform the Van Allen bow-tie analysis of a channel of a charged particle instrument.
The details of the analysis are at https://www.utupub.fi/handle/10024/152846 p.53-57 and in references therein.

The module calculates the effective energies and geometric factors given the channel energy response as the main input.

# A testbench
A concise testbench is provided in *bowtie_test.py* to check the functionality of the module

# A sample code
A sample code for the bow-tie analysis of the BepiColombo/SIXS-P response is provided in *bowtie_calc_np.py*. 
The script shows the basic usage of the module applied to the BepiColombo/SIXS-P instrument.

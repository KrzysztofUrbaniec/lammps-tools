## Motivation
This is a part of a personal project I did at the university. The project aimed to reproduce the results of Molecular Dynamics simulations with polarizable and non-polarizable force fields from an article on the dynamical properties of Ionic Liquids (Goloviznina et al., **2019**).

## Project Structure
The code in the repository consists of a few tools I created to compute necessary quantities, namely, the molecular Center-of-Mass (COM), the Mean-Square-Displacement (MSD) and the ionic diffusion coefficients. 

The structure of the tools is based on a class called DumpFileLoader, which is used to load the data from the simulation output (LAMMPS dump file) into the memory. Objects of this class are utilized later in various functions to facilitate data processing and computations. 

All source files are located in the /src directory. In the root directory there is also a file "LoaderDemo.ipynb", which contains the description of particular functions and examples of their application.

## License
MIT License.

## Citation

Goloviznina, K.; Lopes, J. N. C.; Gomes, M. C.; Pádua, A. A. H. Transferable, Polarizable Force Field for Ionic Liquids, *J. Chem. Theory Comput.*, **2019**, 15, 5858-5871
## Motivation
This is a part of a personal project I did in the university. The project aimed to reproduce the results of Molecular Dynamics simulation from an article on the dynamics of Ionic Liquids investigated with polarizable and non-polarizable force fields.

## Project Structure
The code in the repository consists of a few tools I created to compute necessary quantities, namely, the molecular Center-of-Mass (COM), the Mean-Square-Displacement (MSD) and the ionic diffusion coefficients. \

The structure of the tools is based on a class called DumpFileLoader, which is used to load the data from the simulation output (LAMMPS dump file) into the memory. \

Objects of this class are utilized later in various functions to facilitate data processing and computations. 

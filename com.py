from misc import print_progress_bar, ELEMENT_TO_MASS_MAP
from DumpFileLoader import DumpFileLoader
import numpy as np
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_COM(moltype, mol_geom_array, timestep, atom_mass_map, element_present):
    mol_geom_com = np.zeros(shape=(len(mol_geom_array), 3))
    
    total_mass = 0
    for i, atom in enumerate(mol_geom_array):
        if element_present:
            atom_mass = atom_mass_map[moltype.data_dict['element'][timestep][i]]
        else:
            atom_mass = atom_mass_map[moltype.data_dict['type'][timestep][i]]
        total_mass += atom_mass
        mol_geom_com[i] = atom * atom_mass
    return mol_geom_com.sum(axis=0) / total_mass
    
def convert_to_com(dump_obj : DumpFileLoader, type_to_mass_map=None):

    ### CHECKS ###

    recognizes_mol_types = True
    if len(dump_obj.molecule_types) == 0:
        print(f'WARNING: COM is being computed for all molecules in the system, with no distinction between molecule type. \
              If this behavior is not expected, consider using recognize_molecules() method from DumpFileLoader object.')
        recognizes_mol_types = False

    data_dict_keys = dump_obj.data_dict.keys()
    # Find names of coordinates in the dict and assign them to variables, to be able to index the dictionary later
    if 'xu' not in data_dict_keys or 'yu' not in data_dict_keys or 'zu' not in data_dict_keys:
        print("WARNING: COM is being computed using coordinates not recognized as unwrapped. This method doesn't check if periodic boundary conditions \
              were applied nor does it convert wrapped coordinates to unwrapped ones.")

    atom_mass_map = None            
    element_present = 'element' in dump_obj.data_dict.keys()
    
    if element_present and type_to_mass_map is None:
        atom_mass_map = ELEMENT_TO_MASS_MAP
    elif element_present and type_to_mass_map is not None:
        atom_mass_map = type_to_mass_map
        
    if not element_present and type_to_mass_map is None:
        print('Element not provided in dump file and type to mass map is not defined')
        raise Exception('No rule for atom to mass mapping.')
    if not element_present and type_to_mass_map is not None:
        atom_mass_map = type_to_mass_map
    
    ### END CHECKS ###

    # Modify for case, where there are nmols in None (COM computed over all molecules)

    for moltype in dump_obj.molecule_types:
        logging.debug(moltype.data_dict.keys())
        
        moltype.data_dict['COM_x'] = np.empty(shape=(0,moltype.nummols))
        moltype.data_dict['COM_y'] = np.empty(shape=(0,moltype.nummols))
        moltype.data_dict['COM_z'] = np.empty(shape=(0,moltype.nummols))

        print(f'Computing COM for molecules of type: {moltype.name}')
        
        for timestep_idx in range(len(dump_obj.timesteps)):
        
            # Just for now let's assume that coordinates are necessarily unwrapped
            x = moltype.data_dict['xu'][timestep_idx][:,np.newaxis]
            y = moltype.data_dict['yu'][timestep_idx][:,np.newaxis]
            z = moltype.data_dict['zu'][timestep_idx][:,np.newaxis]

            # This array contains sets of three atomic coordinates
            # Its dimensions are (n_atoms x 3), each row corresponds to one atom
            mol_geom_array = np.concatenate((x,y,z), axis=1)

            # The check if dump file contained info about molecule id
            # If it didn't contain it, default to calculate com over
            # all atoms or print message, that it can't be computed

            start_mol, end_mol = (moltype.data_dict['mol'][timestep_idx].min().astype(np.int32),
                                moltype.data_dict['mol'][timestep_idx].max().astype(np.int32))
            com_array = np.zeros(shape=(moltype.nummols,3))
            for i in range(start_mol, end_mol+1):
                mol_mask = np.where(moltype.data_dict['mol'][timestep_idx] == i)[0] # Mask for atoms belonging to specific molecule
                molecule_com = compute_COM(moltype, mol_geom_array[mol_mask], timestep_idx, atom_mass_map, element_present)
                com_array[i-start_mol,:] = molecule_com
                
            # print_progress_bar(iteration=timestep_idx+1, total=len(dump_obj.timesteps))
            
            moltype.data_dict['COM_x'] = np.r_[moltype.data_dict['COM_x'], com_array[:,0][np.newaxis,:]]
            moltype.data_dict['COM_y'] = np.r_[moltype.data_dict['COM_y'], com_array[:,1][np.newaxis,:]]
            moltype.data_dict['COM_z'] = np.r_[moltype.data_dict['COM_z'], com_array[:,2][np.newaxis,:]]
        print('')
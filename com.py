from misc import print_progress_bar, ELEMENT_TO_MASS_MAP
from DumpFileLoader import DumpFileLoader
import numpy as np
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_COM(moltype: DumpFileLoader.MoleculeType, mol_geom_array: np.array, timestep: int, atom_mass_map: dict, element_present: bool) -> np.array:
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
    
def convert_to_com(dump_obj: DumpFileLoader, type_to_mass_map: dict = None) -> None: 
    '''Convert atomic coordinates to molecular center-of-mass for defined types of molecules. Requires definitions of molecules (created through recognize_molecules() from DumpDataLoader) and
    presence of 'mol' property in dump data file.
    
    Parameters:
    ----------------------
    :param dump_obj: DumpDataLoader object
    :param type_to_mass_map: Dictionary specifying rule for mapping between atom types and atom masses. If it's not provided, then the function tries to map using element-based masses. 
    If 'element' property was not present in the dump file and type_to_mass_map is not specified, an error is raised.

    Returns: None, molecule type objects are modified inplace.
    '''

    ### CHECKS ###

    if 'mol' not in dump_obj.data_dict.keys():
        raise Exception('Dump object does not contain molecule id. It is therefore not possible to recognize, which atoms belong to which molecules.')

    if len(dump_obj.molecule_types) == 0:
        raise Exception('No molecule was defined.')

    data_dict_keys = dump_obj.data_dict.keys()
    # Find names of coordinates in the dict and assign them to variables, to be able to index the dictionary later
    if 'xu' not in data_dict_keys or 'yu' not in data_dict_keys or 'zu' not in data_dict_keys:
        print("WARNING: COM is being computed using at least one coordinate not recognized as unwrapped. This method doesn't check if periodic boundary conditions \
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

    for moltype in dump_obj.molecule_types:
        # logging.debug(moltype.data_dict.keys())
        
        moltype.data_dict['COM_x'] = np.empty(shape=(0,moltype.nummols))
        moltype.data_dict['COM_y'] = np.empty(shape=(0,moltype.nummols))
        moltype.data_dict['COM_z'] = np.empty(shape=(0,moltype.nummols))

        print(f'Computing COM for molecules of type: {moltype.name}')
        
        for timestep_idx in range(len(dump_obj.timesteps)):
            
            # Now it assumes that the simulation is 3D
            x = recognize_coordinate(moltype, timestep_idx, ['x','xs','xu'])
            y = recognize_coordinate(moltype, timestep_idx, ['y','ys','yu'])
            z = recognize_coordinate(moltype, timestep_idx, ['z','zs','zu'])

            # This array contains sets of three atomic coordinates
            # Its dimensions are (n_atoms x 3), each row corresponds to one atom
            mol_geom_array = np.concatenate((x,y,z), axis=1)

            start_mol, end_mol = (moltype.data_dict['mol'][timestep_idx].min().astype(np.int32),
                                moltype.data_dict['mol'][timestep_idx].max().astype(np.int32))
            com_array = np.zeros(shape=(moltype.nummols,3))
            for i in range(start_mol, end_mol+1):
                mol_mask = np.where(moltype.data_dict['mol'][timestep_idx] == i)[0] # Mask for atoms belonging to specific molecule
                molecule_com = compute_COM(moltype, mol_geom_array[mol_mask], timestep_idx, atom_mass_map, element_present)
                com_array[i-start_mol,:] = molecule_com
                
            print_progress_bar(iteration=timestep_idx+1, total=len(dump_obj.timesteps))
            
            moltype.data_dict['COM_x'] = np.r_[moltype.data_dict['COM_x'], com_array[:,0][np.newaxis,:]]
            moltype.data_dict['COM_y'] = np.r_[moltype.data_dict['COM_y'], com_array[:,1][np.newaxis,:]]
            moltype.data_dict['COM_z'] = np.r_[moltype.data_dict['COM_z'], com_array[:,2][np.newaxis,:]]
        print('')

def recognize_coordinate(moltype: DumpFileLoader.MoleculeType, timestep_idx: int, target_coord_names: list) -> np.array:
    coordinate = [coordinate for coordinate in moltype.data_dict.keys() if coordinate in target_coord_names][0] # Check, which coordinate was in dump file (xu,xs,x...)
    return moltype.data_dict[coordinate][timestep_idx][:,np.newaxis]
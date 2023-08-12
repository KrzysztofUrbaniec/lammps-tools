import pandas as pd
import numpy as np
from io import StringIO
import pickle
import sys
import logging
from misc import print_progress_bar
 
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class DumpFileLoader:
    '''
    Class for handling LAMMPS dump files, currently for atom and custom dump styles.
    '''
    
    class MoleculeType:
        
        def __init__(self, name, atom_types, timesteps):
            self.data_dict = {}
            self.name = name
            self.atom_types = atom_types
            self.nummols = None
            self.timesteps = timesteps

    def __init__(self, input_file, _sort_by_id=True):
        self.data_dict = {}
        self.keywords = []
        self._int_data_types = ['id', 'mol', 'proc', 'procp1',
                               'type', 'ix', 'iy', 'iz']
        self.natoms = None
        self.nmols = None
        self.timesteps = []
        self.molecule_types = []
        self._sort_by_id = _sort_by_id
        self._read_dump_data(input_file)
        
    def _read_dump_data(self, ifile):
        with open(ifile) as file:
            contents = file.readlines()
            
        natoms = int(contents[3])
        self.natoms = natoms
        self.keywords = contents[8].split()[2:]
        self._initialize_data_dict() # Create empty arrays in the dict to be filled with data later
        
        data_encountered = False
        
        i = 0
        print('Reading data...')
        while i < len(contents):
            line = contents[i]

            if line.startswith('ITEM: TIMESTEP'):
                data_encountered = False
                self.timesteps.append(int(contents[i+1])) # Collect timestep
            
            if data_encountered:
                partial_data = contents[i:i + self.natoms]
                partial_data_table = pd.read_table(StringIO(' '.join(partial_data)),
                                            sep=r'\s+',names=self.keywords)
                
                if self._sort_by_id:
                    partial_data_table.sort_values(by=['id'], axis='rows', inplace=True)
                
                i += self.natoms - 1 # Jump to next segment
                
                # Fill dictionary with data
                for keyword, data_array in self.data_dict.items():
                    data_to_append = partial_data_table[keyword].to_numpy()[np.newaxis,:]
                    self.data_dict[keyword] = np.r_[data_array, data_to_append]

            if line.startswith('ITEM: ATOMS'):
                data_encountered = True
                
            i += 1
            print_progress_bar(i, len(contents))
            
        if 'mol' in self.keywords:
            self.nmols = len(np.unique(self.data_dict['mol'][0]))
        print('\nDone!')
        
    def _initialize_data_dict(self):
        for keyword in self.keywords:
            if keyword in self._int_data_types:
                self.data_dict[keyword] = np.empty(shape=(0,self.natoms), dtype=int)
            else:
                self.data_dict[keyword] = np.empty(shape=(0,self.natoms))
                
    def get_property(self, name: str) -> np.array:
        '''Get an array of selected property values collected from dump file and grouped by timestep.
        If the array is A, then element A(i,j) gives value of selected property for j-th atom at i-th timestep.
        
        Parameters:
        ------------------------
        :param name: name of the property form the dump file
        
        Returns: An array with selected property values
        '''
        
        if name in self.data_dict.keys():
            return self.data_dict[name]
        else:
            print("Entered property doesn't exist.")
            return None

    def get_molecule_type(self, molname: str) -> MoleculeType: 
        '''Get molecule type using previously defined molecule name.
        
        Parameters:
        ------------------------
        :param molname: Molecule name corresponding to specific molecule type.

        Returns: MoleculeType object
        '''

        for mol in self.molecule_types:
            if mol.name == molname:
                return mol
        print('Molecule type with given name is not defined.')

    def get_coordinates_array(self) -> dict:
        '''Get arrays of coordinates loaded from the dump file grouped by timestep.'''

        # Select only coordinates but allow for different types
        target_coord_names = ['x', 'y', 'z', 'xs', 'ys', 'zs', 'xu', 'yu', 'zu']
        names_found = sorted([coordinate for coordinate in self.data_dict.keys() if coordinate in target_coord_names])
        coordinates_dict = self._populate_timestep_grouped_arrays(names_found)
        return coordinates_dict
    
    def get_custom_array(self, property_names: list) -> dict:
        '''Get arrays consisting of selected properties and grouped by timestep.
        
        Parameters:
        ------------------------
        :param property_names: A list containing properties that should be assembled into an array. The order of properties in the list corresponds to the order of columns in the array
        '''

        if not isinstance(property_names, list): raise TypeError('property_names accepts only lists.')
        property_dict = self._populate_timestep_grouped_arrays(property_names)
        return property_dict
    
    def _populate_timestep_grouped_arrays(self, items):
        item_dict = {}
        for timestep_idx, timestep in enumerate(self.timesteps):
            array = np.empty(shape=(self.natoms,0))
            for item in items:
                array = np.concatenate((array, self.data_dict[item][timestep_idx][:,np.newaxis]), axis=1)
            item_dict[timestep] = array
        return item_dict

    def recognize_molecules(self, mols_by_atom_types: list, molnames: list = None) -> None:
        '''Recognize molecule types using provided atom types. Works only if each molecule is defined by separate set of atom types, not overlapping with other molecules.
        Atom types are paired with molecule names in the same order as they appear in arguments of the method. Each molecule type is stored as an object of separate class.
        These objects can be requested by user with get_molecule_type() method.

        Parameters:
        ------------------------
        :param mol_by_atom_types: List-of-lists or tuple-of-lists containing sets of atom types defining specific molecules
        :param molnames: List or tuple containing set of molecule names

        Returns: None, dump object is modified inplace 
        '''

        ### START CHECKS
        # If no molnames, initialize empty list
        if molnames is None: 
            molnames = []
    
        # Check if all molnames are unique
        if len(molnames) != len(set(molnames)):
            raise Exception('All molnames must be unique.')

        # If this function had been used before and is used once again, erase previously created keys
        if len(self.molecule_types) != 0:
            self.molecule_types = []

        # Check consistence of data types 
        if not isinstance(molnames, (list,tuple)) and molnames is not None:
            raise TypeError('"molnames" accepts only lists or tuples.')
        if not isinstance(mols_by_atom_types, (tuple, list)):
            raise TypeError('"mols_by_atom_types" accepts only lists or tuples.')
        
        self.molnames = molnames
        
        # Check the number of molnames compared to number of provided sets of atom types 
        if len(molnames) != len(mols_by_atom_types):
            raise Exception('Number of provided atom names does not match the number of provided sets of atom types.')
        ### END CHECKS

        # Create new MoleculeType objects and populate them with selected coordinates
        for i, molname in enumerate(self.molnames):
            atom_types = mols_by_atom_types[i]
            moltype = self.MoleculeType(molname, atom_types, self.timesteps)
            self.molecule_types.append(moltype)
            for keyword in self.data_dict.keys():
                atom_types_mask = np.isin(self.data_dict['type'][0], atom_types)
                moltype.data_dict[keyword] = self.data_dict[keyword][:,atom_types_mask]
            if 'mol' in self.data_dict.keys():
                moltype.nummols = len(np.unique(moltype.data_dict['mol'][0]))
            
    def save_dump_state(self, filename: str) -> None:
        '''Serialize dump object and save it to a .pkl file.
        
        Parameters:
        ------------------------
        :param filename: Name of the file
        
        Returns: None
        '''
        if not filename.endswith('.pkl'):
            filename = filename + '.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
            
    def load_saved_state(filepath: str):
        '''Deserialize dump object and load previously saved state. The method is provided with
        class for convenience, but it can load any serialized .pkl object and is not restriced to dump objects.
        
        Parameters:
        ------------------------
        :param filepath: Path to the file
        
        Returns: Deserialized .pkl object
        '''
        
        with open(filepath, 'rb') as file:
            return pickle.load(file)
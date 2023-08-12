import numpy as np
import statsmodels.api as sm
from DumpFileLoader import DumpFileLoader
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_MSD(moltype_obj: DumpFileLoader.MoleculeType, n_origin=None, step_origin=10, type='com'):
    # ----------------------------- 
    # Curently allows to compute MSD for center-of-mass of molecules
    # -----------------------------

    n_steps = len(moltype_obj.timesteps) // 2
    if n_origin is None:
        n_origin = len(moltype_obj.timesteps) // 2
    # elif n_origin > len(moltype_obj.timesteps) // 2:
    #     raise ValueError('n_origin cannot exceed half the number of timesteps ')

    origins = np.arange(0,n_origin,step_origin,dtype=int)

    MSD = np.zeros(shape=(n_steps, len(origins)))

    for i, step_origin in enumerate(origins):
        x_origin = moltype_obj.data_dict['COM_x'][step_origin][:,np.newaxis]
        y_origin = moltype_obj.data_dict['COM_y'][step_origin][:,np.newaxis]
        z_origin = moltype_obj.data_dict['COM_z'][step_origin][:,np.newaxis]

        coords_origin =  np.concatenate((x_origin, y_origin, z_origin), axis=1)

        for k, step in enumerate(range(step_origin+1,step_origin+n_steps+1)):
            x = moltype_obj.data_dict['COM_x'][step][:,np.newaxis]
            y = moltype_obj.data_dict['COM_y'][step][:,np.newaxis]
            z = moltype_obj.data_dict['COM_z'][step][:,np.newaxis]

            coords =  np.concatenate((x, y, z), axis=1)
            d = coords - coords_origin
            d_sq = d ** 2
            d_sq_summed_atoms = d_sq.mean(axis=0)
            MSD_values = d_sq_summed_atoms.sum()
            MSD[k,i] = MSD_values
    
    return MSD

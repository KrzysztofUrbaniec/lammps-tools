import numpy as np
import statsmodels.api as sm
from DumpFileLoader import DumpFileLoader

def compute_MSD(moltype_obj: DumpFileLoader.MoleculeType, n_origin=None, type='com'):
    # ----------------------------- 
    # Curently allows to compute MSD for center-of-mass of molecules
    # -----------------------------

    n_steps = len(moltype_obj.timesteps) // 2
    if n_origin is None:
        n_origin = len(moltype_obj.timesteps) // 2
    # elif n_origin > len(moltype_obj.timesteps) // 2:
    #     raise ValueError('n_origin cannot exceed half the number of timesteps ')

    MSD = np.zeros(shape=(n_steps, n_origin))

    # Currently origins are all timesteps from the beginning of the simulation up to n_origin
    origins = moltype_obj.timesteps[:n_origin]

    for i in range(len(origins)):
        x_origin = moltype_obj.data_dict['COM_x'][i][:,np.newaxis]
        y_origin = moltype_obj.data_dict['COM_y'][i][:,np.newaxis]
        z_origin = moltype_obj.data_dict['COM_z'][i][:,np.newaxis]

        coords_origin =  np.concatenate((x_origin, y_origin, z_origin), axis=1)

        for k, step in enumerate(range(i+1,i+n_steps+1)):
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



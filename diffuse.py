import numpy as np
import statsmodels.api as sm
from DumpFileLoader import DumpFileLoader
import logging
import sys
from sklearn.linear_model import LinearRegression

logging.basicConfig(stream=sys.stdout, level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_MSD(moltype_obj: DumpFileLoader.MoleculeType, as_key=True, step_origin=1) -> (np.array,None):
    '''Compute MSD of for a collection of molecules. Currently it uses molecular center-of-mass for computations. 
    The function uses a number of steps equal to half the number of timesteps, in which the data was collected.
    
    Parameters:
    ----------------------------
    :param moltype_obj: MoleculeType object returned by get_molecule_type() method from DumpFileLoader class
    :param as_key: Flag used to specify, whether an array of MSD should be returned or stored as a key-value pair in moltype_obj data dictionary
    :param step_origin: Specifies, which timesteps provide the reference coordinates for MSD computation. For example if step_origin=1, then coordinates from
    the beginning up to half the number of timesteps are considered as reference ones. If step_origin equals half the number of timesteps, then coordinates from only the first timestep are used.

    Returns: 2-D array with dimensions (n_molecules x (number_of_timesteps / 2) / step_origin) filled with MSD for different molecules and different time origins.
    '''
   
    # ----------------------------- 
    # Curently allows to compute MSD for center-of-mass of molecules
    # -----------------------------

    n_steps = len(moltype_obj.timesteps) // 2
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

            coords =  np.concatenate((x, y, z), axis=1) # Consider using np.column_stack
            d = coords - coords_origin
            d_sq = d ** 2
            d_sq_summed_atoms = d_sq.mean(axis=0)
            MSD_values = d_sq_summed_atoms.sum()
            MSD[k,i] = MSD_values
    
    if as_key:
        moltype_obj.data_dict['MSD'] = MSD
    else:
        return MSD    


def calculate_diffusion_coefficient(moltype_obj: DumpFileLoader.MoleculeType, start_time: int = None, end_time: int = None, find_best_interval_flag: bool = True) -> tuple:
    '''Calculate diffusion coefficient using Einstein approach.
    
    Parameters:
    -----------------------
    :param moltype_obj: MoleculeType object returned by get_molecule_type() method from DumpFileLoader class
    :param start_time: The beginning of the time interval, for which the slope of MSD vs. time is estimated. If None, then start_time is equal to the beginning of the simulation time
    :param end_time: The end of the time interval, for which the slope of MSD vs. time is estimated. If None, then end_time is equal to the last timestep, for which MSD was calculated
    :param find_best_interval_flag: If True, apply a procedure for estimation of the best time interval based on log MSD vs. log time plot slope
    
    Reutrns: A tuple containing estimated diffusion coefficient, estimated slope of MSD vs. time and error of fit of the slope
    '''

    MSD = moltype_obj.data_dict['MSD'].mean(axis=1)
    dump_freq = moltype_obj.timesteps[1] 
    time = np.arange(1, MSD.shape[0]+1) * dump_freq

    if find_best_interval_flag:
        # Find the best interval based on log-log plot slope
        start_time, end_time = find_best_interval(moltype_obj)
    else:
        # If both start_time and end_time are None, use entire time range (usually not a good idea)
        if end_time is None: end_time = len(time) + 1
        if start_time is None: start_time = 0
    
    time_interval = time[start_time:end_time]
    MSD_interval = MSD[start_time:end_time]

    # Build ordinary least squares model
    time_interval_stats = sm.add_constant(time_interval)
    model = sm.OLS(MSD_interval, time_interval_stats)
    results = model.fit()
    
    slope = results.params[1]
    slope_fit_error = results.bse[1]
    diffusion_coefficient = slope / 6 * 1e15 * 1e-20 # m^2/s
    return (diffusion_coefficient, slope, slope_fit_error, start_time, end_time)

def find_best_interval(moltype_obj):
    MSD = moltype_obj.data_dict['MSD'].mean(axis=1)
    dump_freq = moltype_obj.timesteps[1] 
    time = np.arange(1, MSD.shape[0]+1) * dump_freq

    MSD_log = np.log10(MSD)
    time_log = np.log10(time)

    # Find best interval for slope estimation
    scores = np.zeros(shape=((time[-5] - time[5]) // 10000,4)) # This is error-prone
    for idx, start_time in enumerate(range(time[5],time[-5],10000)):
        mask = np.where((time > start_time) & (time < start_time + 1e6))[0]
        start = mask.min()
        end = mask.max()
        
        time_log_stats = sm.add_constant(time_log[start:end])
        model = sm.OLS(MSD_log[start:end], time_log_stats)
        results = model.fit()

        scores[idx,0] = start
        scores[idx,1] = end
        scores[idx,2] = results.params[1] # Slope
        scores[idx,3] = results.bse[1] # Standard error of slope estimation

    # Find best score (slope of log-log plot close to 1)
    scores_sorted = scores[np.argsort(scores[:,2])] # Sort scores by slope
    closest_to_one_idx = np.argmin(np.abs((scores_sorted[:,2] - 1))) # Find index of the slope closest to 1
    best_time_interval = scores_sorted[closest_to_one_idx]
    
    start_time = int(best_time_interval[0])
    end_time = int(best_time_interval[1])

    return (start_time, end_time)
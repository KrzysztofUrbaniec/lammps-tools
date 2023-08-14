import numpy as np
import statsmodels.api as sm
from DumpFileLoader import DumpFileLoader
import logging
import sys
from sklearn.linear_model import LinearRegression

logging.basicConfig(stream=sys.stdout, level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_MSD(moltype_obj: DumpFileLoader.MoleculeType, n_origin=None, as_key=True, step_origin=10, type='com'):
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
    
    if as_key:
        moltype_obj.data_dict['MSD'] = MSD
    else:
        return MSD    


def calculate_diffusion_coefficient(moltype_obj, start_time=None, end_time=None, find_best_interval_flag=True):
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
    return (diffusion_coefficient, slope, slope_fit_error)

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

    # Find best score (slope of log-log plot closes to 1)
    scores_sorted = scores[np.argsort(scores[:,2])] # Sort scores by slope
    closest_to_one_idx = np.argmin(np.abs((scores_sorted[:,2] - 1))) # Find index of the slope closest to 1
    best_time_interval = scores_sorted[closest_to_one_idx]
    
    start_time = int(best_time_interval[0])
    end_time = int(best_time_interval[1])

    return (start_time, end_time)
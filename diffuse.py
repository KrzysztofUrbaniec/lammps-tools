import numpy as np
import statsmodels.api as sm

from DumpFileLoader import DumpFileLoader

def compute_MSD(moltype_obj: DumpFileLoader.MoleculeType, as_key=True, origin_step_interval=1) -> np.array:
    '''Compute MSD (mean-square-displacement) for a collection of molecules. Currently it uses molecular center-of-mass for computations. 
    
    The code computes MSD a number of times, each time using a fixed number of steps equal to the half of the length of simulation time. 
    Each computation starts with a time origin, which serves as a reference point, determining, where particular time intervals should begin.
    This results in m vectors, where m is the number of time origins.

    For example, if the number of timesteps in a simulation is equal to 20, the fixed number of steps is equal to 10. 
    MSD vector is computed with respect to 0th timestep (origin) and has 10 elements corresponding to 10 timesteps after the origin (timesteps 1-10).
    Then, next time origin is selected and the procedure is repeated.
    
    Parameters:
    ----------------------------
    :param moltype_obj: MoleculeType object returned by get_molecule_type() method from DumpFileLoader class

    :param as_key: Flag used to specify, whether an array of MSD should be returned as a separate object or stored as a key-value pair in moltype_obj data dictionary

    :param origin_step_interval: Specifies, which timesteps provide the reference coordinates for MSD computation (how large is the step between successive time origins).
    For example if origin_step_interval=10, them timesteps 0, 10, 20... will be considered as the reference ones.

    Returns: 2-D array with MSD for different time origins (columns) and its evolution in time relative to particular origin (rows).
    '''
   
    # ----------------------------- 
    # Curently allows to compute MSD for center-of-mass of molecules. 
    # -----------------------------

    # Determine the number of time steps, during which the MSD is computed (n_steps) and the number of time origins (n_origins)
    # Time origin is a timestep, which serves as the first timestep in a particular time interval
    # The number of steps indicates, how many data points will be obtained from each interval (n_steps is set to be equal to half the simulation time)

    n_steps = len(moltype_obj.timesteps) // 2
    n_origin = len(moltype_obj.timesteps) // 2
  
    # Arange an array of indices indicating time origins
    origins = np.arange(0, n_origin, origin_step_interval, dtype=int)

    # n x k matrix, where n = step relative to particular initial timestep, k = initial timestep (reference)
    # For example: element in the first second row and the first columns of the array indicates the MSD computed with reference to 1-st time
    # origin (i.e. the beginning of the simulation time) and during the second timestep after the origin
    # (i.e. the third timestep, during which the data was collected)
    MSD = np.zeros(shape=(n_steps, len(origins)))

    for i, timestep_origin in enumerate(origins):

        # Get the x,y,z coordinates of molecular center-of-masses for initial time origin
        x_origin = moltype_obj.data_dict['COM_x'][timestep_origin][:,np.newaxis]
        y_origin = moltype_obj.data_dict['COM_y'][timestep_origin][:,np.newaxis]
        z_origin = moltype_obj.data_dict['COM_z'][timestep_origin][:,np.newaxis]

        coords_origin =  np.concatenate((x_origin, y_origin, z_origin), axis=1)

        # Get the interval spanning half the simulation time 
        next_timestep = timestep_origin + 1
        last_timestep_in_range = timestep_origin + n_steps + 1
        for k, step in enumerate(range(next_timestep, last_timestep_in_range)):

            # Get the x,y,z coordinates of molecular center-of-masses for "step" time origin
            x = moltype_obj.data_dict['COM_x'][step][:,np.newaxis]
            y = moltype_obj.data_dict['COM_y'][step][:,np.newaxis]
            z = moltype_obj.data_dict['COM_z'][step][:,np.newaxis]

            # Compute MSD
            coords =  np.concatenate((x, y, z), axis=1)  # n x 3 matrix of center of mass coordinates
            distance_squared = (coords - coords_origin) ** 2 # Compute square of the distance traveled by a molecule along particular axis
            MSD_components = distance_squared.mean(axis=0) # Compute mean displacement along particular axis
            MSD_values = MSD_components.sum() # Add x,y,z components to obtain total MSD

            # Store MSD
            MSD[k,i] = MSD_values
    
    if as_key is True:
        moltype_obj.data_dict['MSD'] = MSD
    else:
        return MSD    


def compute_diffusion_coefficient(moltype_obj: DumpFileLoader.MoleculeType, start_time: int = None, end_time: int = None) -> tuple:
    '''Compute diffusion coefficient using Einstein approach. The result is returned in units of m^2/s
    
    Parameters:
    -----------------------
    :param moltype_obj: MoleculeType object returned by get_molecule_type() method from DumpFileLoader class

    :param start_time: The beginning of the time interval, for which the slope of MSD vs. time is estimated. If None, start_time is equal to the beginning of the simulation time
    
    :param end_time: The end of the time interval, for which the slope of MSD vs. time is estimated. If None, end_time is equal to the last timestep, for which MSD was computed
    
    Reutrns: A tuple, which elements are: estimated diffusion coefficient, estimated slope of MSD vs. time and the standard error error of the slope
    '''

    # Compute the mean with respect to particular referece timesteps (origins)
    MSD = moltype_obj.data_dict['MSD'].mean(axis=1)
    dump_freq = moltype_obj.timesteps[1] # How frequently molecular data was saved, i.e. the length of a timestep
    time = np.arange(1, MSD.shape[0] + 1) * dump_freq
  
    if end_time is None: end_time = len(time)
    if start_time is None: start_time = 0
    
    time_interval = time[start_time:end_time + 1]
    MSD_interval = MSD[start_time:end_time + 1]

    # Build ordinary least squares model estimating the slope of the MSD curve in given time interval
    time_interval_stats = sm.add_constant(time_interval)
    model = sm.OLS(MSD_interval, time_interval_stats).fit()
    
    # Extract slope and standard error of the slope
    slope = model.params[1]
    slope_fit_error = model.bse[1]
    diffusion_coefficient = slope / 6 * 1e15 * 1e-20 # The constants convert the results to m^2/s

    return (diffusion_coefficient, slope, slope_fit_error)
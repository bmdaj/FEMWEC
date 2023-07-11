import numpy as np

def init_dir(str):

    import os
    import datetime

    now = datetime.datetime.now()
    today = now.strftime("%Y-%m-%d_%H-%M-%S"+str)
    directory = f"{today}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    return directory, today

def write_line(f):
    f.write("--------------------------------------------- \n")

def write_general_info(f, sim):
    write_line(f)
    f.write("Simulation domain data: \n")
    write_line(f)
    f.write("Scaling: {:.6e}  \n".format(sim.scaling))
    f.write("Number of elements in Y: {:.6e}  \n".format(sim.nElY_EM))
    f.write("Number of elements in X: {:.6e}  \n".format(sim.nElX_EM))
    write_line(f)
    f.write("Material parameters: \n")
    write_line(f)
    f.write("Metal refractive index: {:.6e}  \n".format(sim.n_metal))
    f.write("Metal extinction coefficient: {:.6e}  \n".format(sim.k_r))
    f.write("Metal heat conductivity: {:.6e}  \n".format(sim.k_metal))
    f.write("Waveguide refractive index: {:.6e}  \n".format(sim.n_wg))
    f.write("Waveguide heat conductivity: {:.6e}  \n".format(sim.k_wg))
    f.write("Cladding refractive index: {:.6e}  \n".format(sim.n_clad))
    f.write("Cladding heat conductivity: {:.6e}  \n".format(sim.k_clad))
    f.write("Wavelength: {:.6e}  \n".format(sim.wavelength))
    f.write("Effective index for unheated device: {:.6e}  \n".format(sim.delta))
    f.write("Effective index for heated device: {:.6e}  \n".format(sim.deltaT))
    f.write("Effective index change: {:.6e}  \n".format(sim.deltaT - sim.delta))
    f.write("Length of waveguide: {:.6e}  \n".format(sim.dz))
    phase_shift = (sim.deltaT-sim.delta)*(2*np.pi*sim.dz)/(sim.wavelength*sim.scaling)
    f.write("Expected phase-shift: {:.6e}  \n".format(phase_shift))
    f.write("Relative permeability: {:.6e}  \n".format(sim.mu))
    write_line(f)

def create_logfile_optimization(sim,idx_RHS_EM = None, val_RHS_EM=None, idx_RHS_heat = None, val_RHS_heat=None):
    import numpy as np
    #directory, today = init_dir("_opt")
    directory = sim.directory_opt
    today = sim.today

    logfile = f"{directory}/opt_logfile_{today}.txt"
    with open(logfile, "w") as f:
        f.write(f"Optimization log data for {today}:\n")
        write_general_info(f, sim)
        f.write("Topology Optimization parameters: \n")
        write_line(f)
        f.write(f"FOM type: {sim.FOM_type}  \n")
        f.write("Threshold level, eta: {:.6e}  \n".format(sim.eta))
        f.write("Threshold strength, beta: {:.6e}  \n".format(sim.beta))
        f.write("Filter radius: {:.6e}  \n".format(sim.fR))
        f.write(f"Optimization algorithm: {sim.algorithm}  \n")
        f.write("Maximum number of iterations: {:.6e}  \n".format(sim.maxItr))
        f.write("Maximum tolerance: {:.6e}  \n".format(sim.tol))
        f.write(f"Continuation scheme: {sim.continuation_scheme}  \n")
        f.write(f"Eliminate excitation: {sim.eliminate_excitation}  \n")
        np.save(f"{directory}/dVini.npy", sim.dVini)
        np.save(f"{directory}/dVs.npy", sim.dVs)
        write_line(f)
        f.write("Constraints: \n")

        write_line(f)
        f.write(f"Volume constraint: {sim.volume_constraint}  \n")
        f.write("Volume constraint value: {:.6e}  \n".format(sim.vol_cons_val))
        f.write(f"Heating constraint: {sim.heating_constraint}  \n")
        f.write("Heating constraint value: {:.6e}  \n".format(sim.heat_cons_val))
        write_line(f)
        f.write("Right-hand-side: \n")
        write_line(f)
        if idx_RHS_EM.any() != None:
            for i in range(idx_RHS_EM.shape[0]):
                f.write(f"Right-hand-side EM index {i}:"+"{:.6e} \n".format(idx_RHS_EM[i]))
                f.write(f"Value EM {i}: :"+"{:.6e}".format(val_RHS_EM[i].real)+"+{:.6e}j \n".format(val_RHS_EM[i].imag))
        else:
            f.write("Right-hand-side EM: None\n")
        if idx_RHS_heat.any() != None:
                f.write("Value heat: {:.6e} \n".format(val_RHS_heat[0,0]))
        else:
            f.write("Right-hand-side heat: None\n")
        write_line(f)
        write_line(f)
        f.write("Optimization results: \n")
        write_line(f)
        f.write("Initial FOM: {:.6e} \n".format(sim.FOM_list[0]))
        f.write("Final FOM: {:.6e} \n".format(sim.FOM_list[-1]))
        f.write("Initial FOM constraint 1: {:.6e} \n".format(sim.constraint_1[0]))
        f.write("Final FOM constraint 1: {:.6e} \n".format(sim.constraint_1[-1]))
        f.write("Initial FOM constraint 2: {:.6e} \n".format(sim.constraint_2[0]))
        f.write("Final FOM constraint 2: {:.6e} \n".format(sim.constraint_2[-1]))
        f.write("Initial FOM constraint 2: {:.6e} \n".format(sim.constraint_3[0]))
        f.write("Final FOM constraint 2: {:.6e} \n".format(sim.constraint_3[-1]))
        sim.iteration_history(minmax=True, save=True, dir=directory)
        sim.plot_material_interpolation(save=True, dir=directory)

def create_logfile_sweep(sim, idx_RHS_EM, val_RHS_EM, idx_RHS_heat, val_RHS_heat, neff, neff0, neffT, values, sweep_text, neff_heat=None, values_heat=None):
    import numpy as np
    from plot import plot_propagation_response_coupled, plot_propagation_response
    #directory, today= init_dir("_sweep")
    directory = sim.directory_sweep
    today = sim.today
    sim.plot_material_interpolation(save=True, dir=directory)
    logfile = f"{directory}/sweep_{sweep_text}_logfile_{today}.txt"
    np.save(f"{directory}/neff_{sweep_text}.npy", neff)
    np.save(f"{directory}/values_{sweep_text}.npy", values)
    np.save(f"{directory}/dVs.npy", sim.dVs)
    with open(logfile, "w") as f:
        f.write(f"Sweep log data for {today}:\n")
        write_line(f)
        write_general_info(f,sim)
        write_line(f)
        f.write("Right-hand-side: \n")
        write_line(f)
        if idx_RHS_EM.any() != None:
            for i in range(idx_RHS_EM.shape[0]):
                f.write(f"Right-hand-side EM index {i}:"+"{:.6e} \n".format(idx_RHS_EM[i]))
                f.write(f"Value EM {i}: :"+"{:.6e}".format(val_RHS_EM[i].real)+"+{:.6e}j \n".format(val_RHS_EM[i].imag))
        else:
            f.write("Right-hand-side EM: None\n")
        if idx_RHS_heat.any() != None:
                f.write("Value heat: {:.6e} \n".format(val_RHS_heat[0,0]))
        else:
            f.write("Right-hand-side heat: None\n")
        write_line(f)
        write_line(f)
        #f.write("Rest of parameters are the same as for optimization. \n")
        write_line(f)
        f.write("Minimum effective index value: {:.6e} \n".format(neff[0]))
        f.write("Maximum effective index value: {:.6e} \n".format(neff[-1]))
        from scipy.signal import argrelextrema
        if sweep_text == "total":
            maxima_index = argrelextrema(np.array(values), np.greater)
            maximima_beta = neff[maxima_index]
            maxima_index_heat = argrelextrema(np.array(values_heat), np.greater)
            maximima_beta_heat = neff_heat[maxima_index_heat]
            for i in range(len(maximima_beta)):
                f.write(f"Unheated device effective index for max. {i}"+": {:.6e} \n".format(maximima_beta[i]))
            for i in range(len(maximima_beta_heat)):
                f.write(f"Heated device effective index for max. {i}"+": {:.6e} \n".format(maximima_beta_heat[i]))
            plot_propagation_response_coupled(neff, neff0, neffT, np.array(values), np.array(values_heat), save=True, dir=directory)

        if sweep_text == "partial_heat":
            maxima_index_heat = argrelextrema(np.array(values_heat), np.greater)
            maximima_beta_heat = neff_heat[maxima_index_heat]
            for i in range(len(maximima_beta_heat)):
                f.write(f"Heated device effective index for max. {i}"+": {:.6e} \n".format(maximima_beta_heat[i]))
            plot_propagation_response(neff, neff0, neffT, np.array(values_heat), save=True, dir=directory)

    
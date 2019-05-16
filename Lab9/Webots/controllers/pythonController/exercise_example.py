"""Exercise example"""

import numpy as np
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters
import plot_results 


def exercise_example(world, timestep, reset):
    """Exercise example"""
    # Parameters
    n_joints = 10
    """parameter_set = [
        SimulationParameters(
            simulation_duration=10,
            drive=drive,
            amplitude_gradient=[0.05, 0.02], # [1, 2, 3],
            phase_lag=np.pi/5,#np.zeros(n_joints),
            turn=0,
            # ...
        )
        for drive in np.linspace(1, 2, 2)
        # for amplitudes in ...
        # for ...
    ]"""
    parameter_set = [SimulationParameters(
        drive=5,
        amplitude=[0.25,0.25], # head, tail 
        phase_lag=0.5, # total phase lag of 2 pi along the body 2*np.pi/10
        turn=None, 
        simulation_duration = 30,
        limb_spine_phase_lag=0
        )
        for drive in [5] # Implement walking and swimming 
    ]

    # Grid search
    for simulation_i, parameters in enumerate(parameter_set):
        reset.reset()
        run_simulation(
            world,
            parameters,
            timestep,
            int(1000*parameters.simulation_duration/timestep),
            logs="./logs/example/simulation_{}.npz".format(simulation_i)
        )
        plot_results.plot_simulation(plot=True, path="./logs/example/simulation_{}.npz".format(simulation_i)) 


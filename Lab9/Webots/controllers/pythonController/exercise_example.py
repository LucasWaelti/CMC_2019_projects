"""Exercise example"""

import numpy as np
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters


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
        drive=2.5,
        amplitude=0.001, # slope
        phase_lag=2*np.pi/10,
        turn=None,
        couplingBody=10, 
        couplingLeg=30,
        rate=20,
        cRBody = [0.025, 0.005]
    )]

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


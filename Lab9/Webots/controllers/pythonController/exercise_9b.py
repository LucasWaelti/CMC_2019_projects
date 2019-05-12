"""Exercise 9b"""

import numpy as np
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters
import plot_results 


def exercise_9b(world, timestep, reset):
    """Exercise 9b"""

    """
        Influence of: 
            - phase lag
            - amplitude
        on: 
            - energy (integral of [torque * joint's speed])

        -> Run grid search on those parameters. For each run, evaluate: 
            - minimal energy (absolute value)
            - maximal energy (absolute value)
            - average energy
        -> Display 2D plot of results of the grid search. 
    """
    # Parameters
    n_joints = 10
    amplitudes = np.arange(0.01, 0.1, 0.05) 
    phase_lags = np.arange(0.1,1,0.1) 
    parameter_set = [SimulationParameters(
        drive=drive,
        amplitude=[0.08,0.2], # head, tail 
        phase_lag=2*np.pi/10,
        turn=None,
        couplingBody=10, 
        couplingLeg=30,
        rate=20,
        cRBody = [0.025, 0.005],
        simulation_duration = 1
        )
        for drive in [2.5,5] # Implement walking and swimming 
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


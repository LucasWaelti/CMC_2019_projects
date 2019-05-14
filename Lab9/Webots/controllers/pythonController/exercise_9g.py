"""Exercise 9g"""

# from run_simulation import run_simulation
from simulation_parameters import SimulationParameters
from run_simulation import run_simulation
import numpy as np 

def exercise_9g(world, timestep, reset):
    """Exercise 9g"""
    
    # Parameters
    n_joints = 10

    parameters = SimulationParameters(
        drive=2.5,
        amplitude=[0.16,0.2], # head, tail 
        phase_lag=2*np.pi/10, # total phase lag of 2 pi along the body -> but swim BACKWARD!! 
        turn=None, 
        couplingBody=10, 
        couplingLeg=30,
        rate=20,
        simulation_duration = 30
    )
    reset.reset()
    run_simulation(
        world,
        parameters,
        timestep,
        int(1000*parameters.simulation_duration/timestep),
        logs="./logs/9g/simulation.npz",
        transition=True # Transition between walk and swim 
    )


"""Exercise 9d"""

import numpy as np
from matplotlib import pyplot as plt 
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters
import plot_results 


def exercise_9d1(world, timestep, reset):
    """Exercise 9d1"""

    '''
        Modulate drive to generate turning. 

        -> plot GPS trajectory to show turning results. 
        -> plot spine angles. 
    '''
    # Parameters
    n_joints = 10

    parameters = SimulationParameters(
        drive=5,
        amplitude=[0.16,0.2], # head, tail 
        phase_lag=2*np.pi/10, # total phase lag of 2 pi along the body 
        turn=-0.5, # turn left 
        couplingBody=10, 
        couplingLeg=30,
        rate=20,
        simulation_duration = 10
    )
    reset.reset()
    run_simulation(
        world,
        parameters,
        timestep,
        int(1000*parameters.simulation_duration/timestep),
        logs="./logs/9d1/simulation.npz"
    )

def plot_9d1():
    with np.load("./logs/9d1/simulation.npz") as data:
        # Load stored data
        timestep = float(data["timestep"])
        link_data = data["links"][:, 0, :] # Only get the head position 
        joints = data["joints"] # (position, velocity, command, torque, torque_fb, output)
        network_state = data["network_state"]
        network_output = data["network_output"]
        times = np.arange(0, timestep*np.shape(link_data)[0], timestep)

        #plt.figure('9d1_positions')
        #plot_results.plot_positions(times, link_data)

        # Display the GPS trajectory of the head 
        plt.figure('9d1_trajectory')
        plt.title('Head trajectory when turning left')
        plot_results.plot_trajectory(link_data)
        plt.savefig("./logs/9d1/turning_left_trajectory.jpg")
        plt.tight_layout() 

        # Display the spine angles 
        plot_results.plot_network_output(times[:500], network_output[:500,:-4])
        plt.plot([0,4],[0,0],color='b',linestyle='--')
        plt.savefig("./logs/9d1/turning_left_spine_angles.jpg")
        plt.tight_layout() 
        plt.show() 

def exercise_9d2(world, timestep, reset):
    """Exercise 9d2"""

    '''
        Modify phase to generate backward swimming. 

        -> plot GPS trajectory to show turning results. 
        -> plot spine angles. 
    '''
    # Parameters
    n_joints = 10

    parameters = SimulationParameters(
        drive=5,
        amplitude=[0.16,0.2], # head, tail 
        phase_lag=-2*np.pi/10, # total phase lag of 2 pi along the body -> but swim BACKWARD!! 
        turn=None, 
        couplingBody=10, 
        couplingLeg=30,
        rate=20,
        simulation_duration = 10
    )
    reset.reset()
    run_simulation(
        world,
        parameters,
        timestep,
        int(1000*parameters.simulation_duration/timestep),
        logs="./logs/9d2/simulation.npz"
    )

def plot_9d2():
    with np.load("./logs/9d2/simulation.npz") as data:
        # Load stored data
        timestep = float(data["timestep"])
        link_data = data["links"][:, 0, :] # Only get the head position 
        joints = data["joints"] # (position, velocity, command, torque, torque_fb, output)
        network_state = data["network_state"]
        network_output = data["network_output"]
        times = np.arange(0, timestep*np.shape(link_data)[0], timestep)

        #plt.figure('9d1_positions')
        #plot_results.plot_positions(times, link_data)

        # Display the GPS trajectory of the head 
        plt.figure('9d2_trajectory')
        plt.title('Head trajectory when swimming backward')
        plot_results.plot_trajectory(link_data)
        plt.savefig("./logs/9d2/backward_trajectory.jpg")
        plt.tight_layout() 

        # Display the spine angles 
        plot_results.plot_network_output(times[:500], network_output[:500,:-4])
        plt.savefig("./logs/9d2/backward_spine_angles.jpg")
        plt.tight_layout() 
        plt.show() 

if __name__ == "__main__":
    plot_9d1() 
    plot_9d2() 
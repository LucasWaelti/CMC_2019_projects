"""Exercise 9b"""

import numpy as np
from matplotlib import pyplot as plt 
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters
import plot_results 

sim_time= 30

def get_search_space():
    num_trials = 5
    amplitudes = np.linspace(0.12, 0.4, num=num_trials) 
    phase_lags = np.linspace(0.01,2,num=num_trials)  
    return num_trials,amplitudes,phase_lags 

def exercise_9b(world, timestep, reset):
    """Exercise 9b"""

    """
        Influence of: 
            - phase lag
            - amplitude
        on: 
            - speed (along the x axis)
            - energy (integral of [torque * joint's speed])

        -> Run grid search on those parameters. For each run, evaluate: 
            - maximal energy (absolute value)
        -> Display 2D plot of results of the grid search. 
    """
    # Parameters
    n_joints = 10

    # Search space definition
    num_trials,amplitudes,phase_lags = get_search_space()

    # Grid search 
    simulation_i = -1
    for amplitude in amplitudes:
        for phase_lag in phase_lags:
            simulation_i += 1
            parameters = SimulationParameters(
                drive=5,
                amplitude=[amplitude,amplitude], # head, tail 
                phase_lag=phase_lag,
                turn=None,
                couplingBody=10, 
                couplingLeg=30,
                rate=20,
                cRBody = [0.025, 0.005],
                simulation_duration = sim_time
            )

            reset.reset()
            run_simulation(
                world,
                parameters,
                timestep,
                int(1000*parameters.simulation_duration/timestep),
                logs="./logs/9b/simulation_{}.npz".format(simulation_i)
            )
    
    # Evaluate the grid search 
    compute_energy_speed() 

def compute_energy_speed():
    n_joints = 10

    _,amplitudes,phase_lags = get_search_space() 

    max_energy = []
    max_speed = []
    energy_labels = ['amplitude','phase_lag','energy']
    speed_labels = ['amplitude','phase_lag','speed']

    simulation_i = -1
    for amplitude in amplitudes:
        for phase_lag in phase_lags:
            simulation_i += 1
            # Load data
            with np.load("./logs/9b/simulation_{}.npz".format(simulation_i)) as data:
                # Load stored data
                timestep = float(data["timestep"])
                link_data = data["links"][:, 0, :] # Only get the head position 
                joints = data["joints"] # (position, velocity, command, torque, torque_fb, output)

                # Compute energy derivative
                d_energy = np.array([joints[:,i,1]*joints[:,i,4] for i in range(n_joints)]).transpose() # shape (250,10) 
                # Integrate the energy
                energy = np.zeros_like(d_energy)
                for j in range(d_energy.shape[0]):
                    energy[j] = np.abs(np.sum(d_energy[0:j,:],axis=0))
                # Return the maximal energy 
                max_energy.append([amplitude,phase_lag,np.max(np.max(energy))])

                # Compute the speed
                # speed = np.array([(link_data[i+1,:]-link_data[i,:])/timestep for i in range(link_data.shape[0]-1)])
                speed = (link.data[-1,0] - link.data[0,0])/sim_time
                max_speed.append([amplitude,phase_lag,speed])

    max_energy = np.array(max_energy)
    plt.figure()
    plot_results.plot_2d(max_energy,energy_labels,n_data=len(max_energy))
    plt.savefig("./logs/9b/energy.jpg") 

    max_speed = np.array(max_speed)
    plt.figure()
    plot_results.plot_2d(max_speed,speed_labels,n_data=len(max_speed))
    plt.savefig("./logs/9b/speed.jpg") 

    plt.show() 

if __name__ == "__main__":
    compute_energy_speed()


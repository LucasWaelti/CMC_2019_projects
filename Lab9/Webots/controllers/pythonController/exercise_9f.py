"""Exercise 9f"""

import numpy as np
from matplotlib import pyplot as plt 
from run_simulation import run_simulation
from simulation_parameters import SimulationParameters

def get_phase_search_space():
    num_trials = 20
    phases = np.linspace(-np.pi, np.pi, num=num_trials)   
    return num_trials,phases

def get_amplitude_search_space():
    num_trials = 20
    amplitudes = np.linspace(0, 0.3, num=num_trials)   
    return num_trials,amplitudes

def exercise_9f(world, timestep, reset):
    '''
        Choose which grid search to run 
    '''
    #exercise_9f1(world, timestep, reset)
    exercise_9f2(world, timestep, reset)

def exercise_9f1(world, timestep, reset):
    """Exercise 9f"""

    """
        I) Influence of: 
            - Limb spine phase 
        on: 
            - walking speed (along the x axis)

        -> Set nominal amplitude radius to 0.3 [rad]. 
        -> show plot showing the effect of phase offset. 

    """
    # Parameters
    n_joints = 10
    _, phase_set = get_phase_search_space()
    for i,phase in enumerate(phase_set):
        parameters = SimulationParameters(
            drive=2.5,
            amplitude=[0.15,0.15], # head, tail 
            phase_lag=2*np.pi/10, # total phase lag of 2 pi along the body
            turn=None, 
            couplingBody=10, 
            couplingLeg=30,
            rate=20,
            simulation_duration = 10,
            limb_spine_phase_lag=phase
        )
        reset.reset()
        run_simulation(
            world,
            parameters,
            timestep,
            int(1000*parameters.simulation_duration/timestep),
            logs="./logs/9f1/simulation_{}.npz".format(i) 
        )


def exercise_9f2(world, timestep, reset):
    """Exercise 9f"""

    '''
        II) Influence of: 
            - Body oscillation amplitude
        on: 
            - walking speed (along the x axis)

        -> Set nominal radius to best value found previously. 
        -> show plot showing the effect of the amplitude. 
    '''
    # TODO 
    # Parameters
    n_joints = 10
    _, amplitudes = get_amplitude_search_space()
    for i,amplitude in enumerate(amplitudes):
        parameters = SimulationParameters(
            drive=2.5,
            amplitude=[amplitude,amplitude], # head, tail 
            phase_lag=2*np.pi/10, # total phase lag of 2 pi along the body 
            turn=None, 
            couplingBody=10, 
            couplingLeg=30,
            rate=20,
            simulation_duration=10,
            limb_spine_phase_lag=0 
        )
        reset.reset()
        run_simulation(
            world,
            parameters,
            timestep,
            int(1000*parameters.simulation_duration/timestep),
            logs="./logs/9f2/simulation_{}.npz".format(i) 
        )

    


def compute_speed_9f1():
    n_joints = 10

    _,phase_set = get_phase_search_space() 

    max_speed = []
    speed_labels = ['Rhead','Rtail','speed'] 

    for simulation_i,phase in enumerate(phase_set):
        # Load data
        with np.load("./logs/9f1/simulation_{}.npz".format(simulation_i)) as data:
            # Load stored data
            timestep = float(data["timestep"])
            link_data = data["links"][:, 0, :] # Only get the head position 
            joints = data["joints"] # (position, velocity, command, torque, torque_fb, output)

            # Compute the speed
            speed = np.array([(link_data[i+1,:]-link_data[i,:])/timestep for i in range(link_data.shape[0]-1)]) 
            max_speed.append([phase,np.max(speed[:,0],axis=0)]) # np.mean(speed[:,0],axis=0)

    max_speed = np.array(max_speed)
    plt.figure()
    plt.title('Speed for different limb spine phase lags')
    plt.plot(max_speed[:,0],max_speed[:,1])
    plt.xlabel('Phase lag [rad]')
    plt.ylabel('Top speed [m/s]') 
    plt.savefig("./logs/9f1/speed.jpg") 
    plt.tight_layout()
    plt.show() 

def compute_speed_9f2():
    n_joints = 10

    _,amplitudes = get_amplitude_search_space() 

    max_speed = []

    for simulation_i,amplitude in enumerate(amplitudes):
        # Load data
        with np.load("./logs/9f2/simulation_{}.npz".format(simulation_i)) as data:
            # Load stored data
            timestep = float(data["timestep"])
            link_data = data["links"][:, 0, :] # Only get the head position 
            joints = data["joints"] # (position, velocity, command, torque, torque_fb, output)

            # Compute the speed
            speed = np.array([(link_data[i+1,:]-link_data[i,:])/timestep for i in range(link_data.shape[0]-1)]) 
            max_speed.append([amplitude,np.max(speed[:,0],axis=0)]) # np.mean(speed[:,0],axis=0)

    max_speed = np.array(max_speed)
    plt.figure()
    plt.title('Speed for different amplitudes')
    plt.plot(max_speed[:,0],max_speed[:,1])
    plt.xlabel('amplitude [rad]')
    plt.ylabel('Top speed [m/s]') 
    plt.savefig("./logs/9f2/speed.jpg") 
    plt.tight_layout()
    plt.show() 

if __name__ == "__main__":
    #compute_speed_9f1() 
    compute_speed_9f2() 
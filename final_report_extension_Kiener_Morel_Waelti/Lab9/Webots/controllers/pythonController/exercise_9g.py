"""Exercise 9g"""

# from run_simulation import run_simulation
from simulation_parameters import SimulationParameters
from run_simulation import run_simulation
import plot_results 
import numpy as np 
from matplotlib import pyplot as plt 

def exercise_9g(world, timestep, reset):
    """Exercise 9g"""

    '''
        Implement walking to swimming transition. 
        A new optional argument was added to run_simulation(..., transition=False). 
        By setting transition=True, run_simulation() will monitor the robot's x position 
        and modify the drive level accordingly. 

        Plots:
            - spine angles
            - limb angles
            - GPS x component
        Video:
            - Add a video of the switch in the behaviour 
    '''
    
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

def plot_9g():
    with np.load("./logs/9g/simulation.npz") as data: 
        # Load stored data
        timestep = float(data["timestep"])
        link_data = data["links"][:, 0, :] # Only get the head position 
        joints = data["joints"] # (position, velocity, command, torque, torque_fb, output)
        times = np.arange(0, timestep*np.shape(link_data)[0], timestep)

    # Plot the spine angles 
    plt.figure() 
    plt.title('Spine angles, walking/swimming transition')
    for i in range(10): 
        plt.plot(times[2000:2800], joints[2000:2800,i,0],label=i)
    plt.xlabel('Time [s]')
    plt.ylabel('angle [rad]') 
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("./logs/9g/spine_angles.jpg") 
    

    # Plot the leg angles 
    plt.figure() 
    plt.title('Limb angles')
    for i in range(10,14): 
        plt.plot(times, joints[:,i,0],label=i)
    plt.xlabel('Time [s]')
    plt.ylabel('angle [rad]') 
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("./logs/9g/leg_angles.jpg") 
    

    plt.figure()
    plt.title('GPS x coordinate')
    plt.plot(times,link_data[:,0])
    for i,link in enumerate(link_data[:,0]):
        if link > 1.2:
            break 
    plt.scatter(times[i],[1.2],color='r',label='Switch location') 
    plt.xlabel('time [s]')
    plt.ylabel('x [m]') 
    plt.tight_layout() 
    plt.grid()
    plt.legend() 
    plt.savefig("./logs/9g/x_gps.jpg") 

    plt.show() 

if __name__ == "__main__":
    plot_9g() 


"""Plot results"""

import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from cmc_robot import ExperimentLogger
from save_figures import save_figures
from parse_args import save_plots


def plot_positions(times, link_data):
    """Plot positions"""
    for i, data in enumerate(link_data.T):
        plt.plot(times, data, label=["x", "y", "z"][i])
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Distance [m]")
    plt.grid(True)


def plot_trajectory(link_data):
    """Plot positions"""
    plt.plot(link_data[:, 0], link_data[:, 2])
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")
    plt.axis("equal")
    plt.grid(True)


def plot_2d(results, labels, n_data=300, log=False, cmap=None):
    """Plot result

    results - The results are given as a 2d array of dimensions [N, 3].

    labels - The labels should be a list of three string for the xlabel, the
    ylabel and zlabel (in that order).

    n_data - Represents the number of points used along x and y to draw the plot

    log - Set log to True for logarithmic scale.

    cmap - You can set the color palette with cmap. For example,
    set cmap='nipy_spectral' for high constrast results.

    """
    xnew = np.linspace(min(results[:, 0]), max(results[:, 0]), n_data)
    ynew = np.linspace(min(results[:, 1]), max(results[:, 1]), n_data)
    grid_x, grid_y = np.meshgrid(xnew, ynew)
    results_interp = griddata(
        (results[:, 0], results[:, 1]), results[:, 2],
        (grid_x, grid_y),
        method='linear'  # nearest, cubic
    )
    extent = (
        min(xnew), max(xnew),
        min(ynew), max(ynew)
    )
    plt.plot(results[:, 0], results[:, 1], "r.")
    imgplot = plt.imshow(
        results_interp,
        extent=extent,
        aspect='auto',
        origin='lower',
        interpolation="none",
        norm=LogNorm() if log else None
    )
    if cmap is not None:
        imgplot.set_cmap(cmap)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    cbar = plt.colorbar()
    cbar.set_label(labels[2])

def plot_network_state(times, network_state):
    n_phases = int(network_state.shape[1]/2) # 24 phases and 24 amplitudes
    phases = network_state[:,0:n_phases]
    amplitudes = network_state[:,n_phases:]

    plt.figure('Network state: phase')
    plt.title('Network state: phase')
    for i in range(n_phases):
        plt.plot(times, phases[:,i],label=i)
    plt.xlabel('time [s]')
    plt.ylabel('phase [rad]')
    plt.tight_layout()
    plt.legend()
    plt.grid()

    plt.figure('Network state: amplitude')
    plt.title('Network state: amplitude')
    for i in range(n_phases):
        plt.plot(times, amplitudes[:,i],label=i)
    plt.xlabel('time [s]')
    plt.ylabel('amplitude []')
    plt.legend()
    plt.tight_layout()
    plt.grid()

def plot_network_output(times, output):
    plt.figure('Network output')
    plt.title('Network output')
    for i in range(output.shape[1]):
        plt.plot(times, output[:,i],label=i)
    plt.xlabel('time [s]')
    plt.ylabel('angle [rad]')
    plt.tight_layout()
    plt.legend()
    plt.grid()

def plot_torques(times, torques):
    plt.figure('Torques')
    plt.title('Torques') 
    for i in range(torques.shape[1]):
        plt.plot(times,torques[:,i], label=i)
    plt.xlabel('time [s]')
    plt.ylabel('Force [N]')
    plt.tight_layout()
    plt.legend()
    plt.grid()

def plot_simulation(plot=True, path='logs/example/simulation_0.npz'):
    # Load data
    with np.load(path) as data:
        print('Loaded data:',data)
        timestep = float(data["timestep"])
        links = data["links"]
        joints = data["joints"] # (position, velocity, command, torque, torque_fb, output)
        network = data["network"] # UNUSED - oly contains zero values
        network_state = data["network_state"]
        network_output = data["network_output"]
        link_data = data["links"][:, 0, :] # Only get the head position 
        joints_data = data["joints"]
    times = np.arange(0, timestep*np.shape(link_data)[0], timestep)

    print('Retrieved data from log:')
    print('Links:',links.shape)
    print('joints:',joints.shape) 
    #print('network:',network.shape) 
    print('network_state:',network_state.shape) 
    print('network_output:',network_output.shape) 
    print('link_data:',link_data.shape) 
    print('joints_data:',joints_data.shape) 

    # Plot data
    plt.figure("Positions")
    plot_positions(times, link_data)
    plt.figure("Trajectories")
    plot_trajectory(link_data) 
    plot_network_state(times,network_state)
    plot_network_output(times, network_output)
    plot_torques(times, joints[:,:,4]) # 4: torque feedback (check experiment_logger.py)

    # Show plots
    if plot:
        plt.show()
    else:
        save_figures()

def main(plot=True, path='logs/example/simulation_0.npz'):
    """Main - use plot_simulation instead""" 
    # Load data
    with np.load(path) as data:
        print('Loaded data:',data)
        timestep = float(data["timestep"])
        amplitude = data["amplitude"]
        phase_lag = data["phase_lag"]
        link_data = data["links"][:, 0, :]
        joints_data = data["joints"]
    times = np.arange(0, timestep*np.shape(link_data)[0], timestep) 

    # Plot data
    plt.figure("Positions")
    plot_positions(times, link_data)
    plt.figure("Trajectories")
    plot_trajectory(link_data)

    # Show plots
    if plot:
        plt.show()
    else:
        save_figures()


if __name__ == '__main__':
    main(plot=not save_plots())


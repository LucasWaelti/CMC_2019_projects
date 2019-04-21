"""Run network without Webots"""

import time
import numpy as np
import matplotlib.pyplot as plt
from network import SalamanderNetwork
from save_figures import save_figures
from parse_args import save_plots

def plotLogData(time,log,what='phase',figure_name='log'):
    _ = plt.figure()
    plt.plot(time,log)
    plt.title(figure_name)
    plt.grid(True)
    plt.xlabel("time [s]")
    plt.ylabel(what)


def main(plot=True):
    """Main - Run network without Webots and plot results"""
    # Simulation setup
    timestep = 1e-3
    times = np.arange(0, 2, timestep)
    freqs = np.ones(20) # 20 amplitudes to be specified
    amplitudes = [1,1]#np.ones(20) #[1, 1] # 20 amplitudes to be specified
    phase_lag = 2*np.pi/10 # Single scalar
    turn = 0 # Will be used to modify set_parameters from AmplitudeEquation in network.py 

    network = SalamanderNetwork(timestep, freqs, amplitudes, phase_lag, turn)

    # Logs
    phases_log = np.zeros([
        len(times),
        len(network.phase_equation.phases)
    ])
    phases_log[0, :] = network.phase_equation.phases
    amplitudes_log = np.zeros([
        len(times),
        len(network.amplitude_equation.amplitudes)
    ])
    amplitudes_log[0, :] = network.amplitude_equation.amplitudes
    outputs_log = np.zeros([
        len(times),
        len(network.get_motor_position_output())
    ])
    outputs_log[0, :] = network.get_motor_position_output()

    # Simulation
    tic = time.time()
    for i, _ in enumerate(times[1:]):
        network.step()
        phases_log[i+1, :] = network.phase_equation.phases
        amplitudes_log[i+1, :] = network.amplitude_equation.amplitudes
        outputs_log[i+1, :] = network.get_motor_position_output()
    toc = time.time()

    # Simulation information
    print("Time to run simulation for {} steps: {} [s]".format(
        len(times),
        toc - tic
    ))

    plotLogData(times,phases_log,what='phase',figure_name='phases_log')
    plotLogData(times,amplitudes_log,what='amplitude',figure_name='amplitudes_log')
    plotLogData(times,outputs_log,what='output',figure_name='outputs_log')

    # Show plots
    if plot:
        plt.show()
    else:
        save_figures()


if __name__ == '__main__':
    main(plot=not save_plots())


"""Run network without Webots"""

import time
import numpy as np
import matplotlib.pyplot as plt
import cmc_pylog as pylog
from network import SalamanderNetwork
from save_figures import save_figures
from parse_args import save_plots
from simulation_parameters import SimulationParameters

import plot_results 


def run_network(duration, update=False, drive=0):
    """Run network without Webots and plot results"""
    # Simulation setup
    timestep = 5e-3
    times = np.arange(0, duration, timestep)
    n_iterations = len(times)
    parameters = SimulationParameters(
        drive=2.5,
        amplitude=[0.1,0.1], # head, tail 
        phase_lag=2*np.pi/8,
        turn=None,
        couplingBody=10, 
        couplingLeg=30,
        rate=20, 
        limb_spine_phase_lag=1 
    )
    network = SalamanderNetwork(timestep, parameters)
    osc_left = np.arange(10)
    osc_right = np.arange(10, 20)
    osc_legs = np.arange(20, 24)

    # Logs
    phases_log = np.zeros([
        n_iterations,
        len(network.state.phases)
    ])
    phases_log[0, :] = network.state.phases
    amplitudes_log = np.zeros([
        n_iterations,
        len(network.state.amplitudes)
    ])
    amplitudes_log[0, :] = network.state.amplitudes
    freqs_log = np.zeros([
        n_iterations,
        len(network.parameters.freqs)
    ])
    freqs_log[0, :] = network.parameters.freqs
    outputs_log = np.zeros([
        n_iterations,
        len(network.get_motor_position_output())
    ])
    outputs_log[0, :] = network.get_motor_position_output()

    # Run network ODE and log data
    tic = time.time()
    for i, _ in enumerate(times[1:]):
        if update:
            network.parameters.update(
                SimulationParameters(
                    # amplitude_gradient=None,
                    # phase_lag=None
                )
            )
        network.step()
        phases_log[i+1, :] = network.state.phases
        amplitudes_log[i+1, :] = network.state.amplitudes
        outputs_log[i+1, :] = network.get_motor_position_output()
        freqs_log[i+1, :] = network.parameters.freqs
    toc = time.time()

    # Network performance
    pylog.info("Time to run simulation for {} steps: {} [s]".format(
        n_iterations,
        toc - tic
    ))

    # Implement plots of network results
    #pylog.warning("Implement plots")
    plotLogData(times, phases_log, what='phase', figure_name='phase log')
    plotLogData(times, amplitudes_log, what='amplitude', figure_name='amplitude log')
    plotLogData(times, outputs_log[:,0:14], what='output', figure_name='output log')


def plotLogData(time,log,what='phase',figure_name='phase log'):
    _ = plt.figure()
    for i in range(log.shape[1]):
        plt.plot(time,log[:,i],label=str(i))
    plt.title(figure_name)
    plt.grid(True)
    plt.xlabel("time [s]")
    plt.ylabel(what)
    plt.legend() 
    
    
def main(plot):
    """Main"""

    run_network(duration=5)

    # Show plots
    if plot:
        plt.show()
    else:
        save_figures()


if __name__ == '__main__':
    main(plot=not save_plots())

